# Tata Consumer Products Ltd. (TATACONSUM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1176.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 246 |
| ALERT1 | 158 |
| ALERT2 | 155 |
| ALERT2_SKIP | 81 |
| ALERT3 | 459 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 219 |
| PARTIAL | 6 |
| TARGET_HIT | 9 |
| STOP_HIT | 214 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 227 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 52 / 175
- **Target hits / Stop hits / Partials:** 9 / 212 / 6
- **Avg / median % per leg:** 0.10% / -0.56%
- **Sum % (uncompounded):** 21.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 120 | 19 | 15.8% | 3 | 117 | 0 | -0.24% | -28.3% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.91% | -1.8% |
| BUY @ 3rd Alert (retest2) | 118 | 19 | 16.1% | 3 | 115 | 0 | -0.22% | -26.4% |
| SELL (all) | 107 | 33 | 30.8% | 6 | 95 | 6 | 0.47% | 50.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 107 | 33 | 30.8% | 6 | 95 | 6 | 0.47% | 50.0% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.91% | -1.8% |
| retest2 (combined) | 225 | 52 | 23.1% | 9 | 210 | 6 | 0.10% | 23.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 15:15:00 | 777.34 | 778.71 | 778.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 09:15:00 | 771.56 | 776.28 | 777.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-18 09:15:00 | 770.28 | 770.19 | 772.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 09:15:00 | 770.28 | 770.19 | 772.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 09:15:00 | 770.28 | 770.19 | 772.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-18 14:00:00 | 768.45 | 770.80 | 772.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-24 10:15:00 | 763.71 | 760.22 | 759.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 10:15:00 | 763.71 | 760.22 | 759.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-25 09:15:00 | 766.97 | 762.38 | 761.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 10:15:00 | 780.94 | 782.16 | 778.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-30 11:00:00 | 780.94 | 782.16 | 778.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 11:15:00 | 781.73 | 782.07 | 778.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 12:00:00 | 781.73 | 782.07 | 778.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 783.36 | 782.32 | 780.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 13:45:00 | 784.89 | 783.11 | 781.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-01 09:30:00 | 786.13 | 785.44 | 782.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-01 14:45:00 | 785.49 | 786.58 | 784.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 09:15:00 | 787.71 | 785.72 | 784.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 09:15:00 | 785.49 | 785.67 | 784.49 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-06-05 15:15:00 | 782.77 | 784.91 | 785.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2023-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-05 15:15:00 | 782.77 | 784.91 | 785.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-06 10:15:00 | 779.12 | 783.49 | 784.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-06 13:15:00 | 781.29 | 781.08 | 782.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-06 14:00:00 | 781.29 | 781.08 | 782.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 14:15:00 | 781.88 | 781.24 | 782.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-06 14:45:00 | 781.78 | 781.24 | 782.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 15:15:00 | 782.77 | 781.55 | 782.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-07 09:15:00 | 789.68 | 781.55 | 782.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2023-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 09:15:00 | 800.80 | 785.40 | 784.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 11:15:00 | 803.02 | 791.61 | 787.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 11:15:00 | 803.27 | 804.65 | 797.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-08 11:45:00 | 802.72 | 804.65 | 797.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 14:15:00 | 797.49 | 802.55 | 798.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 15:00:00 | 797.49 | 802.55 | 798.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 15:15:00 | 796.15 | 801.27 | 798.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 09:15:00 | 800.06 | 801.27 | 798.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 10:15:00 | 795.36 | 799.70 | 798.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 11:00:00 | 795.36 | 799.70 | 798.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 11:15:00 | 795.66 | 798.89 | 797.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 12:00:00 | 795.66 | 798.89 | 797.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 12:15:00 | 791.86 | 797.49 | 797.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 13:00:00 | 791.86 | 797.49 | 797.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2023-06-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 13:15:00 | 790.13 | 796.02 | 796.69 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 10:15:00 | 801.73 | 795.88 | 795.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 11:15:00 | 808.30 | 798.36 | 796.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-16 10:15:00 | 843.32 | 849.08 | 839.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-16 11:00:00 | 843.32 | 849.08 | 839.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 13:15:00 | 847.27 | 851.08 | 847.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 14:00:00 | 847.27 | 851.08 | 847.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 14:15:00 | 847.66 | 850.40 | 847.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 14:45:00 | 846.43 | 850.40 | 847.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 15:15:00 | 845.98 | 849.51 | 847.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 09:15:00 | 840.55 | 849.51 | 847.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 09:15:00 | 844.70 | 848.55 | 846.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 09:30:00 | 843.86 | 848.55 | 846.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 10:15:00 | 839.52 | 846.74 | 846.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 10:30:00 | 839.32 | 846.74 | 846.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2023-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 11:15:00 | 839.81 | 845.36 | 845.56 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-20 13:15:00 | 847.32 | 845.77 | 845.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-20 14:15:00 | 849.15 | 846.45 | 846.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-21 10:15:00 | 847.02 | 849.59 | 847.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-21 10:15:00 | 847.02 | 849.59 | 847.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 10:15:00 | 847.02 | 849.59 | 847.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-21 11:00:00 | 847.02 | 849.59 | 847.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 11:15:00 | 849.54 | 849.58 | 847.93 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-06-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 10:15:00 | 840.45 | 846.44 | 847.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 11:15:00 | 837.69 | 844.69 | 846.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-23 12:15:00 | 834.63 | 834.14 | 838.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-23 12:30:00 | 835.22 | 834.14 | 838.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 850.43 | 836.66 | 838.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 09:30:00 | 851.71 | 836.66 | 838.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 10:15:00 | 847.91 | 838.91 | 839.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-26 11:15:00 | 845.05 | 838.91 | 839.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-26 11:15:00 | 843.81 | 839.89 | 839.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2023-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 11:15:00 | 843.81 | 839.89 | 839.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-26 13:15:00 | 849.39 | 842.83 | 841.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-27 10:15:00 | 847.07 | 847.35 | 844.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-27 11:00:00 | 847.07 | 847.35 | 844.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 12:15:00 | 845.19 | 846.64 | 844.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-28 11:15:00 | 846.77 | 845.67 | 844.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-28 11:45:00 | 847.02 | 846.03 | 844.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-28 13:00:00 | 847.47 | 846.32 | 845.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-30 09:15:00 | 842.03 | 845.38 | 845.08 | SL hit (close<static) qty=1.00 sl=843.76 alert=retest2 |

### Cycle 11 — SELL (started 2023-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 10:15:00 | 842.33 | 844.77 | 844.83 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 11:15:00 | 849.00 | 845.62 | 845.21 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 11:15:00 | 844.70 | 846.63 | 846.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-05 09:15:00 | 838.48 | 844.47 | 845.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 822.08 | 819.36 | 823.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-11 10:00:00 | 822.08 | 819.36 | 823.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 10:15:00 | 821.79 | 819.85 | 823.79 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 09:15:00 | 832.06 | 826.07 | 825.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-13 09:15:00 | 835.86 | 832.07 | 829.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 11:15:00 | 832.80 | 832.92 | 830.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-13 12:00:00 | 832.80 | 832.92 | 830.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 12:15:00 | 832.60 | 832.85 | 830.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-14 10:15:00 | 833.84 | 830.05 | 829.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-14 12:00:00 | 836.60 | 831.40 | 830.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-21 15:15:00 | 841.79 | 850.47 | 850.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2023-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 15:15:00 | 841.79 | 850.47 | 850.70 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 13:15:00 | 855.32 | 851.11 | 850.80 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 09:15:00 | 847.37 | 850.27 | 850.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-25 10:15:00 | 846.48 | 849.51 | 850.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-25 11:15:00 | 852.90 | 850.19 | 850.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-25 11:15:00 | 852.90 | 850.19 | 850.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 11:15:00 | 852.90 | 850.19 | 850.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-25 12:00:00 | 852.90 | 850.19 | 850.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2023-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 12:15:00 | 853.39 | 850.83 | 850.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 14:15:00 | 858.28 | 852.89 | 851.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-26 15:15:00 | 859.42 | 860.48 | 857.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-27 09:15:00 | 854.97 | 860.48 | 857.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 850.43 | 858.47 | 856.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 09:45:00 | 847.17 | 858.47 | 856.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 10:15:00 | 850.23 | 856.82 | 856.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 10:45:00 | 850.97 | 856.82 | 856.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2023-07-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 11:15:00 | 842.28 | 853.91 | 854.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 09:15:00 | 829.19 | 843.20 | 845.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-07 11:15:00 | 825.44 | 824.17 | 827.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-07 12:00:00 | 825.44 | 824.17 | 827.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 12:15:00 | 828.95 | 825.13 | 827.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-07 12:45:00 | 828.90 | 825.13 | 827.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 13:15:00 | 826.72 | 825.45 | 827.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-07 13:30:00 | 829.49 | 825.45 | 827.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 14:15:00 | 828.16 | 825.99 | 827.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-07 15:00:00 | 828.16 | 825.99 | 827.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 15:15:00 | 827.96 | 826.38 | 827.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-08 09:15:00 | 828.40 | 826.38 | 827.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 09:15:00 | 825.59 | 826.22 | 827.50 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2023-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 12:15:00 | 841.05 | 830.12 | 829.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-09 12:15:00 | 842.82 | 837.89 | 834.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 10:15:00 | 842.43 | 842.87 | 838.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-10 10:15:00 | 842.43 | 842.87 | 838.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 10:15:00 | 842.43 | 842.87 | 838.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 10:30:00 | 839.37 | 842.87 | 838.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 14:15:00 | 841.00 | 842.37 | 839.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 14:45:00 | 839.37 | 842.37 | 839.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 15:15:00 | 839.27 | 841.75 | 839.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 09:15:00 | 834.97 | 841.75 | 839.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 09:15:00 | 830.38 | 839.48 | 838.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 10:00:00 | 830.38 | 839.48 | 838.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2023-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 10:15:00 | 831.02 | 837.79 | 838.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 14:15:00 | 826.18 | 832.77 | 835.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-14 13:15:00 | 830.68 | 829.39 | 832.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-14 14:00:00 | 830.68 | 829.39 | 832.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 14:15:00 | 834.77 | 830.47 | 832.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-14 15:00:00 | 834.77 | 830.47 | 832.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 15:15:00 | 834.63 | 831.30 | 832.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-16 10:30:00 | 830.92 | 830.65 | 832.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-17 15:15:00 | 834.53 | 831.17 | 830.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2023-08-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 15:15:00 | 834.53 | 831.17 | 830.86 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 09:15:00 | 827.96 | 830.53 | 830.60 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 12:15:00 | 833.69 | 831.00 | 830.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-18 13:15:00 | 836.01 | 832.00 | 831.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-18 14:15:00 | 831.22 | 831.85 | 831.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 14:15:00 | 831.22 | 831.85 | 831.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 14:15:00 | 831.22 | 831.85 | 831.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 15:00:00 | 831.22 | 831.85 | 831.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 15:15:00 | 833.05 | 832.09 | 831.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-21 09:15:00 | 834.48 | 832.09 | 831.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-21 10:15:00 | 834.23 | 831.87 | 831.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-21 11:15:00 | 834.13 | 832.04 | 831.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-21 12:15:00 | 835.56 | 831.98 | 831.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 12:15:00 | 837.79 | 833.14 | 832.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-21 15:00:00 | 838.03 | 834.50 | 832.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-22 09:15:00 | 838.58 | 835.06 | 833.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-22 10:15:00 | 838.33 | 834.97 | 833.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-22 10:45:00 | 839.07 | 835.85 | 833.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 15:15:00 | 836.11 | 837.23 | 835.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 09:15:00 | 833.14 | 837.23 | 835.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 09:15:00 | 835.32 | 836.85 | 835.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 09:45:00 | 831.56 | 836.85 | 835.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 10:15:00 | 833.69 | 836.21 | 835.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 10:30:00 | 833.74 | 836.21 | 835.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 11:15:00 | 833.93 | 835.76 | 835.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 11:45:00 | 835.32 | 835.76 | 835.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 12:15:00 | 832.90 | 835.19 | 835.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 13:00:00 | 832.90 | 835.19 | 835.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-08-23 13:15:00 | 833.59 | 834.87 | 834.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2023-08-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 13:15:00 | 833.59 | 834.87 | 834.88 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-08-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-24 09:15:00 | 840.31 | 835.61 | 835.18 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-08-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 14:15:00 | 832.85 | 834.78 | 834.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 09:15:00 | 828.45 | 833.49 | 834.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 12:15:00 | 829.69 | 825.92 | 828.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 12:15:00 | 829.69 | 825.92 | 828.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 12:15:00 | 829.69 | 825.92 | 828.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 12:45:00 | 829.69 | 825.92 | 828.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 13:15:00 | 829.93 | 826.72 | 828.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-29 09:30:00 | 828.65 | 827.98 | 828.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-29 10:15:00 | 832.16 | 828.81 | 829.02 | SL hit (close>static) qty=1.00 sl=831.47 alert=retest2 |

### Cycle 28 — BUY (started 2023-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 09:15:00 | 834.87 | 829.37 | 829.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 10:15:00 | 836.80 | 830.85 | 829.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 09:15:00 | 833.29 | 834.76 | 832.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 09:15:00 | 833.29 | 834.76 | 832.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 833.29 | 834.76 | 832.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 09:30:00 | 833.29 | 834.76 | 832.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 10:15:00 | 828.16 | 833.44 | 832.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 11:00:00 | 828.16 | 833.44 | 832.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 11:15:00 | 828.70 | 832.49 | 831.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 11:30:00 | 829.44 | 832.49 | 831.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2023-08-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 13:15:00 | 828.06 | 831.01 | 831.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 14:15:00 | 825.54 | 829.92 | 830.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 09:15:00 | 830.08 | 829.44 | 830.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 09:15:00 | 830.08 | 829.44 | 830.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 09:15:00 | 830.08 | 829.44 | 830.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 09:30:00 | 831.37 | 829.44 | 830.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 10:15:00 | 832.16 | 829.98 | 830.54 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 12:15:00 | 832.80 | 830.87 | 830.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 13:15:00 | 834.18 | 831.53 | 831.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-01 15:15:00 | 831.86 | 831.87 | 831.40 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 09:15:00 | 835.86 | 831.87 | 831.40 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 829.00 | 831.30 | 831.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-09-04 09:15:00 | 829.00 | 831.30 | 831.18 | SL hit (close<ema400) qty=1.00 sl=831.18 alert=retest1 |

### Cycle 31 — SELL (started 2023-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 10:15:00 | 829.39 | 830.92 | 831.02 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 12:15:00 | 833.64 | 831.03 | 831.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 14:15:00 | 836.11 | 832.37 | 831.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 14:15:00 | 835.81 | 836.12 | 834.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-05 14:45:00 | 836.50 | 836.12 | 834.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 13:15:00 | 837.59 | 837.61 | 835.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 13:30:00 | 837.74 | 837.61 | 835.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 14:15:00 | 870.09 | 844.11 | 839.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 14:15:00 | 873.15 | 861.37 | 855.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-15 10:00:00 | 871.37 | 871.91 | 866.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-18 09:30:00 | 873.54 | 867.35 | 866.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-18 11:45:00 | 873.30 | 868.28 | 867.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 860.75 | 867.11 | 867.04 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-09-20 10:15:00 | 859.47 | 865.59 | 866.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 10:15:00 | 859.47 | 865.59 | 866.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 09:15:00 | 854.38 | 859.79 | 862.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 12:15:00 | 861.10 | 859.63 | 861.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 12:15:00 | 861.10 | 859.63 | 861.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 12:15:00 | 861.10 | 859.63 | 861.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 12:45:00 | 860.36 | 859.63 | 861.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 13:15:00 | 861.05 | 859.91 | 861.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 14:15:00 | 863.07 | 859.91 | 861.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 14:15:00 | 857.44 | 859.42 | 861.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 14:30:00 | 857.94 | 859.42 | 861.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 15:15:00 | 859.42 | 859.42 | 861.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 09:15:00 | 863.76 | 859.42 | 861.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 860.70 | 859.67 | 861.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 09:45:00 | 862.87 | 859.67 | 861.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2023-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 10:15:00 | 873.39 | 862.42 | 862.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 10:15:00 | 879.07 | 870.09 | 866.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 14:15:00 | 883.57 | 887.04 | 880.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-26 15:00:00 | 883.57 | 887.04 | 880.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 15:15:00 | 882.04 | 886.04 | 880.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 09:15:00 | 882.09 | 886.04 | 880.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 880.56 | 884.95 | 880.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 10:15:00 | 878.53 | 884.95 | 880.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 10:15:00 | 882.73 | 884.50 | 881.08 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-09-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 10:15:00 | 874.78 | 879.31 | 879.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 11:15:00 | 872.46 | 877.94 | 879.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 12:15:00 | 870.83 | 869.98 | 873.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-29 13:00:00 | 870.83 | 869.98 | 873.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 863.66 | 854.01 | 856.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 10:00:00 | 863.66 | 854.01 | 856.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 10:15:00 | 863.32 | 855.87 | 856.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-06 11:30:00 | 861.39 | 857.04 | 857.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-06 12:15:00 | 860.21 | 857.04 | 857.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-06 12:15:00 | 861.59 | 857.95 | 857.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2023-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 12:15:00 | 861.59 | 857.95 | 857.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 13:15:00 | 863.22 | 859.01 | 858.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 14:15:00 | 877.10 | 879.16 | 874.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-11 15:00:00 | 877.10 | 879.16 | 874.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 09:15:00 | 874.09 | 877.66 | 874.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 10:00:00 | 874.09 | 877.66 | 874.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 10:15:00 | 878.18 | 877.76 | 875.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 10:30:00 | 874.43 | 877.76 | 875.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 892.56 | 882.60 | 878.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 10:15:00 | 900.16 | 882.60 | 878.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 12:00:00 | 898.14 | 888.46 | 882.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-18 11:15:00 | 889.15 | 897.08 | 897.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2023-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 11:15:00 | 889.15 | 897.08 | 897.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 12:15:00 | 885.49 | 894.76 | 896.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 12:15:00 | 885.49 | 883.89 | 888.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-19 12:45:00 | 885.20 | 883.89 | 888.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 13:15:00 | 884.01 | 883.92 | 888.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 13:30:00 | 885.44 | 883.92 | 888.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 880.85 | 883.36 | 887.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 09:30:00 | 882.23 | 883.36 | 887.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 14:15:00 | 889.20 | 883.36 | 885.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 15:00:00 | 889.20 | 883.36 | 885.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 15:15:00 | 888.90 | 884.47 | 885.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-23 09:15:00 | 893.84 | 884.47 | 885.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 10:15:00 | 882.73 | 884.08 | 885.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 12:00:00 | 879.76 | 883.22 | 884.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 13:15:00 | 879.22 | 882.66 | 884.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 14:30:00 | 878.53 | 879.73 | 882.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-25 10:00:00 | 875.62 | 877.51 | 881.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 10:15:00 | 878.09 | 877.63 | 880.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-25 10:30:00 | 880.46 | 877.63 | 880.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 11:15:00 | 878.63 | 877.83 | 880.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-25 11:45:00 | 880.01 | 877.83 | 880.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 13:15:00 | 881.54 | 877.92 | 880.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-25 14:00:00 | 881.54 | 877.92 | 880.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 14:15:00 | 878.48 | 878.03 | 880.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-26 09:15:00 | 868.50 | 878.11 | 879.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-27 13:15:00 | 879.42 | 876.06 | 875.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2023-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 13:15:00 | 879.42 | 876.06 | 875.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 14:15:00 | 883.37 | 879.60 | 878.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 13:15:00 | 892.56 | 894.55 | 888.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-01 14:00:00 | 892.56 | 894.55 | 888.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 905.25 | 906.62 | 903.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 09:30:00 | 904.66 | 906.62 | 903.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 13:15:00 | 907.72 | 909.47 | 907.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 13:45:00 | 908.36 | 909.47 | 907.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 14:15:00 | 905.94 | 908.76 | 907.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 14:45:00 | 904.56 | 908.76 | 907.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 15:15:00 | 903.62 | 907.73 | 907.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 09:15:00 | 894.93 | 907.73 | 907.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2023-11-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 09:15:00 | 890.73 | 904.33 | 905.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-10 09:15:00 | 887.32 | 894.08 | 898.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 10:15:00 | 895.12 | 894.29 | 898.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-10 11:00:00 | 895.12 | 894.29 | 898.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 11:15:00 | 898.38 | 895.11 | 898.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 11:30:00 | 898.19 | 895.11 | 898.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 13:15:00 | 897.35 | 895.83 | 898.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 13:30:00 | 897.45 | 895.83 | 898.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 14:15:00 | 901.59 | 896.98 | 898.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 14:45:00 | 903.62 | 896.98 | 898.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 15:15:00 | 903.12 | 898.21 | 899.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-12 18:15:00 | 906.63 | 898.21 | 899.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 902.78 | 899.12 | 899.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 09:15:00 | 896.70 | 899.12 | 899.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-13 11:15:00 | 906.83 | 900.75 | 900.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2023-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 11:15:00 | 906.83 | 900.75 | 900.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-13 12:15:00 | 907.72 | 902.14 | 900.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 12:15:00 | 913.15 | 914.59 | 910.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-16 12:45:00 | 913.30 | 914.59 | 910.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 14:15:00 | 907.62 | 912.67 | 910.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-16 15:00:00 | 907.62 | 912.67 | 910.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 15:15:00 | 909.20 | 911.98 | 910.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-17 09:15:00 | 916.56 | 911.98 | 910.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-22 12:15:00 | 914.24 | 917.05 | 917.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2023-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 12:15:00 | 914.24 | 917.05 | 917.15 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-11-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 09:15:00 | 919.22 | 917.15 | 917.08 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 12:15:00 | 913.00 | 917.62 | 917.78 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 12:15:00 | 920.41 | 918.08 | 917.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 13:15:00 | 920.71 | 918.61 | 918.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 13:15:00 | 923.47 | 923.86 | 922.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-30 14:15:00 | 923.13 | 923.86 | 922.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 14:15:00 | 928.56 | 924.80 | 922.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-01 09:15:00 | 933.94 | 925.53 | 923.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-01 10:30:00 | 930.29 | 927.91 | 924.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-08 13:15:00 | 935.27 | 940.81 | 940.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2023-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 13:15:00 | 935.27 | 940.81 | 940.83 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 10:15:00 | 942.73 | 939.68 | 939.42 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2023-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 14:15:00 | 937.35 | 939.53 | 939.54 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2023-12-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 13:15:00 | 943.28 | 939.73 | 939.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 14:15:00 | 945.20 | 940.82 | 940.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-14 10:15:00 | 942.39 | 942.40 | 941.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 10:15:00 | 942.39 | 942.40 | 941.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 10:15:00 | 942.39 | 942.40 | 941.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-14 10:30:00 | 942.14 | 942.40 | 941.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 11:15:00 | 941.60 | 942.24 | 941.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-14 12:00:00 | 941.60 | 942.24 | 941.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 12:15:00 | 938.49 | 941.49 | 940.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-14 12:30:00 | 937.94 | 941.49 | 940.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 13:15:00 | 937.94 | 940.78 | 940.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-14 14:00:00 | 937.94 | 940.78 | 940.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 09:15:00 | 944.46 | 941.53 | 940.98 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2023-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-18 14:15:00 | 938.44 | 940.91 | 941.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-18 15:15:00 | 935.37 | 939.80 | 940.52 | Break + close below crossover candle low |

### Cycle 50 — BUY (started 2023-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 09:15:00 | 947.67 | 941.38 | 941.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-19 13:15:00 | 965.01 | 953.60 | 947.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 14:15:00 | 962.73 | 968.84 | 960.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-20 15:00:00 | 962.73 | 968.84 | 960.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 09:15:00 | 969.30 | 968.17 | 961.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 14:15:00 | 970.93 | 967.36 | 963.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-22 09:15:00 | 971.92 | 966.85 | 963.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-12-29 10:15:00 | 1068.02 | 1035.96 | 1018.67 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2024-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 15:15:00 | 1088.57 | 1094.99 | 1095.35 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 09:15:00 | 1101.31 | 1096.25 | 1095.90 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 10:15:00 | 1090.45 | 1096.03 | 1096.48 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-01-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 15:15:00 | 1101.21 | 1097.24 | 1096.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-11 09:15:00 | 1102.74 | 1098.34 | 1097.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-15 12:15:00 | 1128.52 | 1129.38 | 1120.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-15 13:00:00 | 1128.52 | 1129.38 | 1120.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 14:15:00 | 1129.46 | 1135.82 | 1130.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 15:00:00 | 1129.46 | 1135.82 | 1130.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 15:15:00 | 1131.93 | 1135.04 | 1130.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 09:15:00 | 1122.35 | 1135.04 | 1130.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 1123.98 | 1132.83 | 1129.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 10:30:00 | 1126.94 | 1132.76 | 1130.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-18 09:15:00 | 1112.62 | 1126.39 | 1128.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2024-01-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 09:15:00 | 1112.62 | 1126.39 | 1128.18 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-01-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 15:15:00 | 1129.81 | 1128.39 | 1128.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 09:15:00 | 1139.59 | 1130.63 | 1129.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 12:15:00 | 1141.76 | 1142.22 | 1138.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-20 12:30:00 | 1143.34 | 1142.22 | 1138.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 13:15:00 | 1135.88 | 1140.95 | 1137.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 13:45:00 | 1136.87 | 1140.95 | 1137.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 14:15:00 | 1137.61 | 1140.28 | 1137.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 14:30:00 | 1139.04 | 1140.28 | 1137.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 15:15:00 | 1136.87 | 1139.60 | 1137.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-23 09:15:00 | 1145.81 | 1139.60 | 1137.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-23 12:15:00 | 1129.46 | 1138.08 | 1137.79 | SL hit (close<static) qty=1.00 sl=1134.00 alert=retest2 |

### Cycle 57 — SELL (started 2024-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 13:15:00 | 1130.20 | 1136.50 | 1137.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-24 09:15:00 | 1124.62 | 1132.71 | 1135.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 10:15:00 | 1134.79 | 1133.12 | 1135.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 10:15:00 | 1134.79 | 1133.12 | 1135.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 10:15:00 | 1134.79 | 1133.12 | 1135.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 10:30:00 | 1132.08 | 1133.12 | 1135.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 11:15:00 | 1133.96 | 1133.29 | 1134.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 11:45:00 | 1136.33 | 1133.29 | 1134.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 12:15:00 | 1136.18 | 1133.87 | 1135.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 13:00:00 | 1136.18 | 1133.87 | 1135.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 13:15:00 | 1142.40 | 1135.57 | 1135.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 14:00:00 | 1142.40 | 1135.57 | 1135.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2024-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 14:15:00 | 1144.97 | 1137.45 | 1136.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-24 15:15:00 | 1145.76 | 1139.11 | 1137.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-25 09:15:00 | 1137.07 | 1138.71 | 1137.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-25 09:15:00 | 1137.07 | 1138.71 | 1137.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 1137.07 | 1138.71 | 1137.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 10:00:00 | 1137.07 | 1138.71 | 1137.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 10:15:00 | 1135.44 | 1138.05 | 1137.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 10:30:00 | 1137.26 | 1138.05 | 1137.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 11:15:00 | 1135.98 | 1137.64 | 1137.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 12:15:00 | 1134.45 | 1137.64 | 1137.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 12:15:00 | 1134.40 | 1136.99 | 1136.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 12:30:00 | 1134.45 | 1136.99 | 1136.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2024-01-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 13:15:00 | 1125.95 | 1134.78 | 1135.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 14:15:00 | 1120.92 | 1132.01 | 1134.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-29 10:15:00 | 1131.63 | 1129.89 | 1132.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-29 10:15:00 | 1131.63 | 1129.89 | 1132.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 10:15:00 | 1131.63 | 1129.89 | 1132.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 11:00:00 | 1131.63 | 1129.89 | 1132.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 11:15:00 | 1127.49 | 1129.41 | 1132.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-29 13:15:00 | 1124.23 | 1129.10 | 1131.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-30 10:30:00 | 1123.88 | 1127.64 | 1130.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-02 09:15:00 | 1126.00 | 1113.93 | 1113.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2024-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 09:15:00 | 1126.00 | 1113.93 | 1113.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-02 13:15:00 | 1151.19 | 1126.05 | 1119.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-05 11:15:00 | 1140.33 | 1140.69 | 1130.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-05 11:30:00 | 1146.80 | 1140.69 | 1130.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 14:15:00 | 1135.98 | 1140.74 | 1133.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 14:45:00 | 1133.41 | 1140.74 | 1133.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 1138.01 | 1139.86 | 1134.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-06 10:45:00 | 1148.72 | 1140.47 | 1134.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-06 11:15:00 | 1148.72 | 1140.47 | 1134.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-07 11:45:00 | 1145.71 | 1146.76 | 1141.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-07 12:15:00 | 1147.34 | 1146.76 | 1141.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 09:15:00 | 1126.05 | 1145.77 | 1143.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-02-08 09:15:00 | 1126.05 | 1145.77 | 1143.69 | SL hit (close<static) qty=1.00 sl=1128.08 alert=retest2 |

### Cycle 61 — SELL (started 2024-02-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 10:15:00 | 1121.07 | 1140.83 | 1141.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 11:15:00 | 1115.98 | 1135.86 | 1139.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 09:15:00 | 1112.97 | 1112.54 | 1118.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 09:15:00 | 1112.97 | 1112.54 | 1118.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 09:15:00 | 1112.97 | 1112.54 | 1118.12 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-02-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 09:15:00 | 1126.35 | 1119.08 | 1118.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 11:15:00 | 1132.72 | 1124.11 | 1121.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-20 09:15:00 | 1126.00 | 1136.40 | 1132.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 09:15:00 | 1126.00 | 1136.40 | 1132.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 1126.00 | 1136.40 | 1132.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 10:00:00 | 1126.00 | 1136.40 | 1132.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 10:15:00 | 1131.39 | 1135.39 | 1132.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 11:30:00 | 1134.40 | 1136.18 | 1132.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-22 11:00:00 | 1134.25 | 1141.38 | 1139.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-22 12:15:00 | 1132.03 | 1138.16 | 1138.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2024-02-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 12:15:00 | 1132.03 | 1138.16 | 1138.56 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-02-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 14:15:00 | 1144.87 | 1139.87 | 1139.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 10:15:00 | 1147.09 | 1141.99 | 1140.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-23 15:15:00 | 1144.38 | 1144.79 | 1142.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-26 09:15:00 | 1141.41 | 1144.79 | 1142.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 1137.96 | 1143.42 | 1142.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-26 11:30:00 | 1147.14 | 1144.37 | 1142.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-29 09:15:00 | 1153.81 | 1158.83 | 1158.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-02-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-29 09:15:00 | 1153.81 | 1158.83 | 1158.92 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-02-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 10:15:00 | 1166.95 | 1160.46 | 1159.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-29 11:15:00 | 1170.20 | 1162.41 | 1160.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 09:15:00 | 1182.11 | 1186.60 | 1180.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 09:15:00 | 1182.11 | 1186.60 | 1180.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 1182.11 | 1186.60 | 1180.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 09:45:00 | 1183.98 | 1186.60 | 1180.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 10:15:00 | 1192.87 | 1187.85 | 1181.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 10:30:00 | 1189.42 | 1187.85 | 1181.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 14:15:00 | 1187.89 | 1187.66 | 1183.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 14:45:00 | 1184.23 | 1187.66 | 1183.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 10:15:00 | 1185.27 | 1186.86 | 1184.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 10:45:00 | 1185.02 | 1186.86 | 1184.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 11:15:00 | 1192.97 | 1188.09 | 1184.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-06 13:30:00 | 1197.86 | 1191.10 | 1187.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-06 14:00:00 | 1204.03 | 1191.10 | 1187.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-11 11:15:00 | 1200.73 | 1217.28 | 1208.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-12 10:15:00 | 1199.00 | 1206.58 | 1206.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2024-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 10:15:00 | 1199.00 | 1206.58 | 1206.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 09:15:00 | 1189.17 | 1200.10 | 1203.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 11:15:00 | 1174.90 | 1174.19 | 1184.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 11:30:00 | 1174.11 | 1174.19 | 1184.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 14:15:00 | 1185.22 | 1177.31 | 1183.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 15:00:00 | 1185.22 | 1177.31 | 1183.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 15:15:00 | 1188.03 | 1179.46 | 1183.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 09:15:00 | 1185.12 | 1179.46 | 1183.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 12:15:00 | 1177.81 | 1174.41 | 1179.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 13:00:00 | 1177.81 | 1174.41 | 1179.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 13:15:00 | 1181.76 | 1175.88 | 1179.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 13:30:00 | 1183.34 | 1175.88 | 1179.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 14:15:00 | 1198.26 | 1180.35 | 1181.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 15:00:00 | 1198.26 | 1180.35 | 1181.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2024-03-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 15:15:00 | 1195.14 | 1183.31 | 1182.54 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 11:15:00 | 1175.19 | 1180.90 | 1181.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-18 14:15:00 | 1172.28 | 1177.22 | 1179.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 09:15:00 | 1119.44 | 1116.45 | 1132.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-21 10:00:00 | 1119.44 | 1116.45 | 1132.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 09:15:00 | 1112.62 | 1117.78 | 1125.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-22 11:15:00 | 1109.95 | 1116.48 | 1124.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-22 12:00:00 | 1109.61 | 1115.11 | 1122.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-22 15:00:00 | 1108.57 | 1113.04 | 1120.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-02 14:15:00 | 1118.50 | 1089.99 | 1087.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2024-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-02 14:15:00 | 1118.50 | 1089.99 | 1087.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 09:15:00 | 1120.37 | 1106.04 | 1103.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-09 10:15:00 | 1112.52 | 1116.66 | 1111.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 10:15:00 | 1112.52 | 1116.66 | 1111.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 10:15:00 | 1112.52 | 1116.66 | 1111.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 10:45:00 | 1113.81 | 1116.66 | 1111.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 13:15:00 | 1109.46 | 1114.60 | 1112.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 14:00:00 | 1109.46 | 1114.60 | 1112.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 14:15:00 | 1111.78 | 1114.04 | 1112.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-10 09:30:00 | 1115.63 | 1113.16 | 1111.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-10 11:00:00 | 1114.84 | 1113.50 | 1112.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-10 12:30:00 | 1117.71 | 1114.97 | 1113.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-15 10:45:00 | 1115.34 | 1123.47 | 1122.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-15 11:15:00 | 1113.16 | 1121.41 | 1121.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2024-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 11:15:00 | 1113.16 | 1121.41 | 1121.46 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 15:15:00 | 1123.04 | 1119.78 | 1119.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 09:15:00 | 1128.72 | 1121.57 | 1120.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-18 13:15:00 | 1130.00 | 1130.31 | 1125.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 13:15:00 | 1130.00 | 1130.31 | 1125.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 13:15:00 | 1130.00 | 1130.31 | 1125.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 13:30:00 | 1123.58 | 1130.31 | 1125.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 14:15:00 | 1120.32 | 1128.31 | 1125.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 14:45:00 | 1122.05 | 1128.31 | 1125.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 15:15:00 | 1126.00 | 1127.85 | 1125.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-19 09:15:00 | 1107.39 | 1127.85 | 1125.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2024-04-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 09:15:00 | 1101.21 | 1122.52 | 1122.94 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-04-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-19 13:15:00 | 1133.96 | 1124.01 | 1123.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 09:15:00 | 1141.31 | 1127.42 | 1124.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 10:15:00 | 1144.08 | 1148.41 | 1140.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 10:15:00 | 1144.08 | 1148.41 | 1140.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 10:15:00 | 1144.08 | 1148.41 | 1140.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 10:45:00 | 1141.36 | 1148.41 | 1140.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 09:15:00 | 1098.69 | 1141.78 | 1141.08 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2024-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 10:15:00 | 1101.36 | 1133.70 | 1137.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 15:15:00 | 1095.38 | 1111.77 | 1124.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 14:15:00 | 1093.01 | 1092.68 | 1106.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-25 14:45:00 | 1094.35 | 1092.68 | 1106.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 1090.84 | 1088.88 | 1095.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-29 11:30:00 | 1081.21 | 1087.14 | 1093.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-30 13:15:00 | 1098.20 | 1091.02 | 1091.59 | SL hit (close>static) qty=1.00 sl=1096.77 alert=retest2 |

### Cycle 76 — BUY (started 2024-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 15:15:00 | 1093.95 | 1092.32 | 1092.13 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 09:15:00 | 1077.46 | 1089.35 | 1090.79 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-05-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 11:15:00 | 1092.13 | 1087.97 | 1087.95 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-05-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 13:15:00 | 1083.98 | 1087.26 | 1087.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 14:15:00 | 1081.51 | 1086.11 | 1087.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 10:15:00 | 1087.83 | 1085.28 | 1086.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-06 10:15:00 | 1087.83 | 1085.28 | 1086.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 1087.83 | 1085.28 | 1086.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 11:00:00 | 1087.83 | 1085.28 | 1086.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 11:15:00 | 1088.03 | 1085.83 | 1086.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 11:45:00 | 1089.80 | 1085.83 | 1086.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 12:15:00 | 1083.53 | 1085.37 | 1086.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 13:15:00 | 1082.00 | 1085.37 | 1086.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 14:45:00 | 1082.35 | 1084.98 | 1085.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-07 09:15:00 | 1100.32 | 1087.97 | 1087.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-07 09:15:00 | 1100.32 | 1087.97 | 1087.08 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 12:15:00 | 1082.30 | 1086.68 | 1086.75 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 09:15:00 | 1097.26 | 1088.54 | 1087.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 10:15:00 | 1100.82 | 1091.00 | 1088.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 09:15:00 | 1086.79 | 1095.26 | 1092.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-09 09:15:00 | 1086.79 | 1095.26 | 1092.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 09:15:00 | 1086.79 | 1095.26 | 1092.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 09:45:00 | 1087.04 | 1095.26 | 1092.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 10:15:00 | 1078.99 | 1092.00 | 1091.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 11:00:00 | 1078.99 | 1092.00 | 1091.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2024-05-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 11:15:00 | 1079.63 | 1089.53 | 1090.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 14:15:00 | 1073.36 | 1082.69 | 1086.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 14:15:00 | 1076.91 | 1075.69 | 1080.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 14:15:00 | 1076.91 | 1075.69 | 1080.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 14:15:00 | 1076.91 | 1075.69 | 1080.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 15:00:00 | 1076.91 | 1075.69 | 1080.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 12:15:00 | 1079.78 | 1074.15 | 1077.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 13:00:00 | 1079.78 | 1074.15 | 1077.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 13:15:00 | 1081.56 | 1075.63 | 1077.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 13:45:00 | 1083.04 | 1075.63 | 1077.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 14:15:00 | 1082.35 | 1076.97 | 1078.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 15:00:00 | 1082.35 | 1076.97 | 1078.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 10:15:00 | 1078.35 | 1077.27 | 1078.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 11:00:00 | 1078.35 | 1077.27 | 1078.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 11:15:00 | 1081.95 | 1078.21 | 1078.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 11:30:00 | 1083.29 | 1078.21 | 1078.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 1081.41 | 1078.85 | 1078.73 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2024-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 14:15:00 | 1071.93 | 1077.36 | 1078.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 09:15:00 | 1061.80 | 1073.50 | 1076.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 11:15:00 | 1066.74 | 1059.89 | 1065.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 11:15:00 | 1066.74 | 1059.89 | 1065.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 1066.74 | 1059.89 | 1065.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 12:00:00 | 1066.74 | 1059.89 | 1065.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 1065.26 | 1060.96 | 1065.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 13:30:00 | 1059.63 | 1060.42 | 1065.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-16 14:15:00 | 1086.50 | 1065.64 | 1067.11 | SL hit (close>static) qty=1.00 sl=1068.62 alert=retest2 |

### Cycle 86 — BUY (started 2024-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 15:15:00 | 1084.87 | 1069.48 | 1068.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 10:15:00 | 1092.62 | 1083.98 | 1080.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 10:15:00 | 1094.79 | 1097.23 | 1090.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 10:15:00 | 1094.79 | 1097.23 | 1090.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 1094.79 | 1097.23 | 1090.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:30:00 | 1092.42 | 1097.23 | 1090.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 1089.51 | 1099.31 | 1095.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:00:00 | 1089.51 | 1099.31 | 1095.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 1088.96 | 1097.24 | 1094.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 11:00:00 | 1088.96 | 1097.24 | 1094.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2024-05-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 13:15:00 | 1087.48 | 1093.09 | 1093.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 14:15:00 | 1084.52 | 1091.38 | 1092.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 12:15:00 | 1080.52 | 1078.34 | 1082.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-28 13:00:00 | 1080.52 | 1078.34 | 1082.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 1077.85 | 1078.24 | 1082.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 09:15:00 | 1072.47 | 1079.65 | 1082.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 09:15:00 | 1062.79 | 1056.90 | 1056.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 09:15:00 | 1062.79 | 1056.90 | 1056.71 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 1040.27 | 1053.57 | 1055.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 1038.10 | 1050.48 | 1053.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 12:15:00 | 1055.04 | 1051.39 | 1053.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 12:15:00 | 1055.04 | 1051.39 | 1053.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 1055.04 | 1051.39 | 1053.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:45:00 | 1053.65 | 1051.39 | 1053.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 13:15:00 | 1068.22 | 1054.76 | 1055.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 14:00:00 | 1068.22 | 1054.76 | 1055.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2024-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 14:15:00 | 1076.52 | 1059.11 | 1057.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 09:15:00 | 1136.87 | 1077.00 | 1065.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 12:15:00 | 1122.89 | 1123.54 | 1113.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 13:00:00 | 1122.89 | 1123.54 | 1113.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 09:15:00 | 1121.11 | 1122.25 | 1115.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 09:45:00 | 1117.02 | 1122.25 | 1115.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 1118.70 | 1123.00 | 1118.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:30:00 | 1118.00 | 1123.00 | 1118.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 1116.13 | 1121.63 | 1118.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:15:00 | 1121.16 | 1121.63 | 1118.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 1122.60 | 1121.82 | 1118.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 10:15:00 | 1125.81 | 1121.82 | 1118.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 13:15:00 | 1112.77 | 1119.12 | 1119.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 13:15:00 | 1112.77 | 1119.12 | 1119.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-13 09:15:00 | 1092.67 | 1111.48 | 1116.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-14 09:15:00 | 1107.68 | 1104.15 | 1109.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-14 10:15:00 | 1107.19 | 1104.15 | 1109.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 10:15:00 | 1106.84 | 1104.69 | 1108.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-14 11:45:00 | 1103.68 | 1104.20 | 1108.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-18 10:00:00 | 1104.18 | 1102.62 | 1105.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-18 11:30:00 | 1105.76 | 1103.99 | 1105.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-18 12:15:00 | 1106.10 | 1103.99 | 1105.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 12:15:00 | 1110.65 | 1105.32 | 1106.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 13:00:00 | 1110.65 | 1105.32 | 1106.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 13:15:00 | 1110.30 | 1106.32 | 1106.71 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-18 14:15:00 | 1113.21 | 1107.70 | 1107.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2024-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 14:15:00 | 1113.21 | 1107.70 | 1107.30 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 09:15:00 | 1099.98 | 1106.86 | 1107.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 14:15:00 | 1091.83 | 1101.17 | 1103.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 11:15:00 | 1080.17 | 1078.81 | 1084.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-24 11:45:00 | 1080.52 | 1078.81 | 1084.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 1087.38 | 1080.53 | 1084.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 13:00:00 | 1087.38 | 1080.53 | 1084.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 13:15:00 | 1087.98 | 1082.02 | 1085.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 14:00:00 | 1087.98 | 1082.02 | 1085.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 15:15:00 | 1090.40 | 1084.81 | 1085.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 09:15:00 | 1085.01 | 1084.81 | 1085.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 1088.72 | 1085.59 | 1086.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 11:15:00 | 1081.06 | 1086.36 | 1086.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 12:45:00 | 1082.25 | 1085.42 | 1086.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 13:30:00 | 1081.41 | 1084.62 | 1085.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 12:15:00 | 1084.52 | 1077.38 | 1077.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2024-06-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 12:15:00 | 1084.52 | 1077.38 | 1077.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 13:15:00 | 1084.77 | 1078.86 | 1077.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 09:15:00 | 1079.38 | 1080.36 | 1078.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 1079.38 | 1080.36 | 1078.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 1079.38 | 1080.36 | 1078.92 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2024-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 11:15:00 | 1067.68 | 1076.44 | 1077.56 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 14:15:00 | 1091.48 | 1078.54 | 1078.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 15:15:00 | 1095.88 | 1082.01 | 1079.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 14:15:00 | 1121.02 | 1126.45 | 1114.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 15:00:00 | 1121.02 | 1126.45 | 1114.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 13:15:00 | 1123.93 | 1129.27 | 1124.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 14:00:00 | 1123.93 | 1129.27 | 1124.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 14:15:00 | 1133.91 | 1130.20 | 1125.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 15:15:00 | 1135.88 | 1130.20 | 1125.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 11:45:00 | 1135.88 | 1134.16 | 1129.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 10:15:00 | 1123.49 | 1130.41 | 1129.51 | SL hit (close<static) qty=1.00 sl=1123.53 alert=retest2 |

### Cycle 97 — SELL (started 2024-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 11:15:00 | 1126.00 | 1129.91 | 1130.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-11 14:15:00 | 1120.08 | 1126.63 | 1128.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-12 10:15:00 | 1125.56 | 1124.76 | 1126.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-12 11:00:00 | 1125.56 | 1124.76 | 1126.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 1129.07 | 1125.62 | 1127.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:00:00 | 1129.07 | 1125.62 | 1127.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 1133.86 | 1127.27 | 1127.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:30:00 | 1135.39 | 1127.27 | 1127.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2024-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 13:15:00 | 1135.88 | 1128.99 | 1128.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 09:15:00 | 1143.83 | 1135.09 | 1132.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 13:15:00 | 1170.60 | 1174.15 | 1165.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-19 14:00:00 | 1170.60 | 1174.15 | 1165.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 1181.76 | 1175.08 | 1167.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 10:15:00 | 1186.21 | 1175.08 | 1167.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 10:45:00 | 1183.93 | 1176.72 | 1169.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-29 09:15:00 | 1194.15 | 1208.36 | 1208.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 09:15:00 | 1194.15 | 1208.36 | 1208.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-29 10:15:00 | 1186.55 | 1203.99 | 1206.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-29 14:15:00 | 1200.20 | 1197.83 | 1202.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-29 15:00:00 | 1200.20 | 1197.83 | 1202.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 15:15:00 | 1200.05 | 1198.27 | 1202.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 09:15:00 | 1202.00 | 1198.27 | 1202.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 1194.95 | 1197.61 | 1201.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 09:15:00 | 1175.65 | 1196.12 | 1198.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 11:15:00 | 1213.75 | 1196.30 | 1194.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 11:15:00 | 1213.75 | 1196.30 | 1194.59 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-08-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 15:15:00 | 1193.00 | 1198.74 | 1198.82 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-05 09:15:00 | 1204.50 | 1199.89 | 1199.34 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 12:15:00 | 1195.75 | 1199.00 | 1199.07 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-05 13:15:00 | 1202.85 | 1199.77 | 1199.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-06 09:15:00 | 1205.95 | 1201.05 | 1200.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-06 12:15:00 | 1200.55 | 1202.86 | 1201.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 12:15:00 | 1200.55 | 1202.86 | 1201.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 12:15:00 | 1200.55 | 1202.86 | 1201.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-06 13:00:00 | 1200.55 | 1202.86 | 1201.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 13:15:00 | 1190.35 | 1200.36 | 1200.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-06 13:45:00 | 1193.60 | 1200.36 | 1200.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2024-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 14:15:00 | 1184.35 | 1197.16 | 1198.86 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 15:15:00 | 1200.00 | 1198.69 | 1198.69 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 09:15:00 | 1176.75 | 1194.31 | 1196.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 1166.95 | 1181.20 | 1185.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 13:15:00 | 1179.20 | 1179.17 | 1182.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-12 13:30:00 | 1180.35 | 1179.17 | 1182.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 1179.95 | 1176.69 | 1179.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:00:00 | 1179.95 | 1176.69 | 1179.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 1180.35 | 1177.43 | 1179.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 13:00:00 | 1180.35 | 1177.43 | 1179.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 1179.50 | 1177.84 | 1179.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 14:45:00 | 1174.65 | 1177.87 | 1179.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 09:15:00 | 1171.45 | 1178.47 | 1179.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 11:30:00 | 1174.95 | 1171.95 | 1173.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 12:15:00 | 1187.85 | 1175.13 | 1175.22 | SL hit (close>static) qty=1.00 sl=1182.75 alert=retest2 |

### Cycle 108 — BUY (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 13:15:00 | 1188.90 | 1177.88 | 1176.46 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 10:15:00 | 1171.75 | 1176.96 | 1177.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 11:15:00 | 1167.55 | 1175.08 | 1176.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 12:15:00 | 1174.20 | 1172.15 | 1173.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 12:15:00 | 1174.20 | 1172.15 | 1173.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 12:15:00 | 1174.20 | 1172.15 | 1173.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 13:00:00 | 1174.20 | 1172.15 | 1173.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 13:15:00 | 1178.50 | 1173.42 | 1174.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 14:00:00 | 1178.50 | 1173.42 | 1174.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 14:15:00 | 1176.65 | 1174.06 | 1174.44 | EMA400 retest candle locked (from downside) |

### Cycle 110 — BUY (started 2024-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 15:15:00 | 1180.00 | 1175.25 | 1174.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 09:15:00 | 1204.85 | 1181.17 | 1177.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 13:15:00 | 1202.15 | 1202.18 | 1194.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 13:30:00 | 1203.55 | 1202.18 | 1194.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 1196.95 | 1201.14 | 1195.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 15:00:00 | 1196.95 | 1201.14 | 1195.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 1196.00 | 1200.11 | 1195.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 1197.65 | 1200.11 | 1195.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 1203.00 | 1200.69 | 1195.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 10:45:00 | 1208.25 | 1202.09 | 1196.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 11:15:00 | 1207.70 | 1210.52 | 1208.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 14:15:00 | 1201.95 | 1207.13 | 1207.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 14:15:00 | 1201.95 | 1207.13 | 1207.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 15:15:00 | 1197.30 | 1205.17 | 1206.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 09:15:00 | 1206.65 | 1205.46 | 1206.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 09:15:00 | 1206.65 | 1205.46 | 1206.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 1206.65 | 1205.46 | 1206.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:00:00 | 1206.65 | 1205.46 | 1206.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 1199.00 | 1204.17 | 1205.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:30:00 | 1205.25 | 1204.17 | 1205.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 1199.00 | 1198.99 | 1202.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 15:00:00 | 1199.00 | 1198.99 | 1202.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 1201.60 | 1198.87 | 1201.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:30:00 | 1201.80 | 1198.87 | 1201.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 1199.00 | 1198.90 | 1201.46 | EMA400 retest candle locked (from downside) |

### Cycle 112 — BUY (started 2024-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 10:15:00 | 1208.60 | 1203.08 | 1202.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 10:15:00 | 1215.60 | 1205.88 | 1203.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 13:15:00 | 1204.90 | 1206.50 | 1204.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 13:15:00 | 1204.90 | 1206.50 | 1204.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 13:15:00 | 1204.90 | 1206.50 | 1204.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 13:45:00 | 1203.85 | 1206.50 | 1204.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 1198.00 | 1204.80 | 1204.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 1198.00 | 1204.80 | 1204.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2024-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 15:15:00 | 1198.00 | 1203.44 | 1203.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 09:15:00 | 1187.25 | 1200.20 | 1202.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 13:15:00 | 1194.20 | 1193.05 | 1197.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-04 14:00:00 | 1194.20 | 1193.05 | 1197.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 14:15:00 | 1195.50 | 1193.54 | 1197.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 14:45:00 | 1196.85 | 1193.54 | 1197.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 1197.85 | 1194.39 | 1197.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:30:00 | 1202.65 | 1194.39 | 1197.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 1196.80 | 1194.87 | 1197.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 11:45:00 | 1194.15 | 1194.43 | 1196.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 14:15:00 | 1194.40 | 1187.51 | 1187.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2024-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 14:15:00 | 1194.40 | 1187.51 | 1187.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 09:15:00 | 1201.85 | 1190.88 | 1189.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 15:15:00 | 1201.55 | 1202.19 | 1196.68 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:15:00 | 1214.55 | 1202.19 | 1196.68 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 1204.05 | 1206.59 | 1202.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 1204.05 | 1206.59 | 1202.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 1204.30 | 1206.39 | 1203.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:45:00 | 1204.05 | 1206.39 | 1203.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 1202.30 | 1205.57 | 1203.27 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-12 12:15:00 | 1202.30 | 1205.57 | 1203.27 | SL hit (close<ema400) qty=1.00 sl=1203.27 alert=retest1 |

### Cycle 115 — SELL (started 2024-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 11:15:00 | 1205.70 | 1213.55 | 1214.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 1202.70 | 1211.38 | 1213.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 1220.45 | 1208.74 | 1210.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 1220.45 | 1208.74 | 1210.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 1220.45 | 1208.74 | 1210.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:00:00 | 1220.45 | 1208.74 | 1210.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 1221.60 | 1211.31 | 1211.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:45:00 | 1223.40 | 1211.31 | 1211.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 11:15:00 | 1214.65 | 1211.98 | 1211.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 09:15:00 | 1224.20 | 1216.08 | 1214.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 14:15:00 | 1215.40 | 1219.16 | 1216.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 14:15:00 | 1215.40 | 1219.16 | 1216.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 1215.40 | 1219.16 | 1216.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 15:00:00 | 1215.40 | 1219.16 | 1216.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 15:15:00 | 1220.60 | 1219.45 | 1216.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 09:30:00 | 1226.40 | 1221.54 | 1218.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 14:15:00 | 1210.95 | 1220.45 | 1219.21 | SL hit (close<static) qty=1.00 sl=1215.60 alert=retest2 |

### Cycle 117 — SELL (started 2024-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 10:15:00 | 1215.85 | 1218.23 | 1218.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 11:15:00 | 1214.95 | 1217.57 | 1218.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 09:15:00 | 1201.85 | 1195.09 | 1202.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 09:15:00 | 1201.85 | 1195.09 | 1202.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 1201.85 | 1195.09 | 1202.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:45:00 | 1205.55 | 1195.09 | 1202.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 1204.95 | 1197.06 | 1202.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:30:00 | 1205.95 | 1197.06 | 1202.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 1204.90 | 1198.63 | 1202.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 11:45:00 | 1204.95 | 1198.63 | 1202.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 1213.25 | 1201.75 | 1203.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 1213.25 | 1201.75 | 1203.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 1210.95 | 1203.59 | 1203.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 09:15:00 | 1203.50 | 1203.59 | 1203.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 09:15:00 | 1206.50 | 1204.17 | 1204.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 09:15:00 | 1206.50 | 1204.17 | 1204.12 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 11:15:00 | 1198.80 | 1204.06 | 1204.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 14:15:00 | 1196.25 | 1201.23 | 1203.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 1203.60 | 1201.03 | 1202.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 1203.60 | 1201.03 | 1202.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 1203.60 | 1201.03 | 1202.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:45:00 | 1204.00 | 1201.03 | 1202.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 1194.10 | 1199.64 | 1201.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:30:00 | 1203.40 | 1199.64 | 1201.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1112.95 | 1117.31 | 1126.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 10:30:00 | 1110.70 | 1114.46 | 1118.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 13:15:00 | 1109.75 | 1112.84 | 1117.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 12:15:00 | 1110.80 | 1113.87 | 1115.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 12:45:00 | 1110.65 | 1112.86 | 1115.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 15:15:00 | 1115.00 | 1112.81 | 1114.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 09:15:00 | 1113.90 | 1112.81 | 1114.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 1107.30 | 1111.71 | 1113.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 11:15:00 | 1104.95 | 1111.64 | 1112.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 10:00:00 | 1105.40 | 1110.85 | 1111.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-10-21 09:15:00 | 999.63 | 1072.89 | 1087.30 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 1010.00 | 991.48 | 989.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 10:15:00 | 1017.50 | 996.68 | 991.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 09:15:00 | 1010.20 | 1013.32 | 1004.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 09:30:00 | 1008.95 | 1013.32 | 1004.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 1002.55 | 1010.34 | 1004.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:45:00 | 1000.80 | 1010.34 | 1004.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 1000.95 | 1008.46 | 1003.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 13:00:00 | 1000.95 | 1008.46 | 1003.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 997.60 | 1006.29 | 1003.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 14:00:00 | 997.60 | 1006.29 | 1003.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 1003.50 | 1005.73 | 1003.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:00:00 | 1008.65 | 1005.78 | 1003.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 983.40 | 1001.05 | 1001.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 983.40 | 1001.05 | 1001.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 978.60 | 996.56 | 999.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-04 14:15:00 | 995.00 | 991.97 | 996.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-04 15:00:00 | 995.00 | 991.97 | 996.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 15:15:00 | 994.50 | 992.47 | 995.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:15:00 | 996.80 | 992.47 | 995.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 996.50 | 993.28 | 996.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:15:00 | 1001.55 | 993.28 | 996.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 1003.80 | 995.38 | 996.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 11:00:00 | 1003.80 | 995.38 | 996.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 1005.00 | 997.31 | 997.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 12:00:00 | 1005.00 | 997.31 | 997.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2024-11-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 12:15:00 | 1008.20 | 999.48 | 998.46 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 10:15:00 | 992.80 | 1000.49 | 1000.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 11:15:00 | 989.70 | 998.33 | 999.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 09:15:00 | 992.70 | 991.88 | 995.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-08 10:00:00 | 992.70 | 991.88 | 995.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 992.30 | 991.55 | 994.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 09:45:00 | 990.00 | 991.13 | 993.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 10:15:00 | 995.95 | 992.09 | 993.67 | SL hit (close>static) qty=1.00 sl=995.45 alert=retest2 |

### Cycle 124 — BUY (started 2024-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 12:15:00 | 940.40 | 928.53 | 928.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 13:15:00 | 944.15 | 931.66 | 929.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 09:15:00 | 957.85 | 960.24 | 953.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 11:15:00 | 953.10 | 958.06 | 953.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 953.10 | 958.06 | 953.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 12:00:00 | 953.10 | 958.06 | 953.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 12:15:00 | 962.55 | 958.96 | 954.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 09:15:00 | 968.90 | 959.66 | 955.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 11:15:00 | 952.80 | 958.55 | 956.39 | SL hit (close<static) qty=1.00 sl=952.90 alert=retest2 |

### Cycle 125 — SELL (started 2024-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 13:15:00 | 944.65 | 953.74 | 954.45 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2024-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 11:15:00 | 960.95 | 953.64 | 953.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 12:15:00 | 962.20 | 955.35 | 954.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 09:15:00 | 954.70 | 956.63 | 955.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 09:15:00 | 954.70 | 956.63 | 955.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 954.70 | 956.63 | 955.47 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2024-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 12:15:00 | 950.35 | 954.70 | 954.81 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2024-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 13:15:00 | 955.75 | 954.91 | 954.90 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 10:15:00 | 952.35 | 954.58 | 954.81 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2024-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 10:15:00 | 956.15 | 954.84 | 954.71 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2024-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 11:15:00 | 951.20 | 954.11 | 954.39 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2024-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 13:15:00 | 963.60 | 956.35 | 955.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 13:15:00 | 965.95 | 959.88 | 957.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 09:15:00 | 935.30 | 964.04 | 963.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 09:15:00 | 935.30 | 964.04 | 963.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 935.30 | 964.04 | 963.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:00:00 | 935.30 | 964.04 | 963.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2024-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 10:15:00 | 938.85 | 959.00 | 961.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 12:15:00 | 933.55 | 950.67 | 956.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 09:15:00 | 939.90 | 933.90 | 940.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 09:15:00 | 939.90 | 933.90 | 940.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 939.90 | 933.90 | 940.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:30:00 | 941.40 | 933.90 | 940.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 921.00 | 931.76 | 936.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 10:15:00 | 919.30 | 931.76 | 936.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 11:15:00 | 919.05 | 929.39 | 935.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 12:00:00 | 919.00 | 927.31 | 933.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 12:45:00 | 918.35 | 925.30 | 932.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 921.65 | 920.21 | 926.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:00:00 | 921.65 | 920.21 | 926.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 927.55 | 922.29 | 926.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:00:00 | 927.55 | 922.29 | 926.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 930.25 | 923.88 | 926.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 930.25 | 923.88 | 926.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 929.00 | 924.90 | 926.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 928.55 | 924.90 | 926.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 920.65 | 925.27 | 926.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 11:45:00 | 918.20 | 923.91 | 925.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 13:00:00 | 918.60 | 922.85 | 925.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 15:15:00 | 920.00 | 921.66 | 924.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 11:15:00 | 909.00 | 903.60 | 903.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2024-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 11:15:00 | 909.00 | 903.60 | 903.45 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2024-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 13:15:00 | 901.50 | 904.19 | 904.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 15:15:00 | 900.00 | 902.92 | 903.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 09:15:00 | 904.50 | 903.24 | 903.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 09:15:00 | 904.50 | 903.24 | 903.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 904.50 | 903.24 | 903.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:00:00 | 904.50 | 903.24 | 903.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 907.05 | 904.00 | 904.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:45:00 | 906.25 | 904.00 | 904.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2024-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 11:15:00 | 905.30 | 904.26 | 904.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 12:15:00 | 910.20 | 905.45 | 904.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 09:15:00 | 902.60 | 905.63 | 905.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 09:15:00 | 902.60 | 905.63 | 905.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 902.60 | 905.63 | 905.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:30:00 | 899.80 | 905.63 | 905.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 909.65 | 906.43 | 905.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 11:45:00 | 911.50 | 907.15 | 905.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 14:15:00 | 899.45 | 905.47 | 905.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 14:15:00 | 899.45 | 905.47 | 905.50 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2024-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 13:15:00 | 913.00 | 905.97 | 905.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 14:15:00 | 915.00 | 907.78 | 906.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 915.70 | 917.31 | 913.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 09:15:00 | 915.70 | 917.31 | 913.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 915.70 | 917.31 | 913.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:30:00 | 915.85 | 917.31 | 913.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 941.20 | 935.63 | 929.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 12:30:00 | 945.50 | 938.51 | 931.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-14 15:15:00 | 963.00 | 966.15 | 966.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2025-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 15:15:00 | 963.00 | 966.15 | 966.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 09:15:00 | 946.40 | 962.20 | 964.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-17 09:15:00 | 948.80 | 941.12 | 947.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 948.80 | 941.12 | 947.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 948.80 | 941.12 | 947.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:45:00 | 946.85 | 941.12 | 947.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 949.70 | 942.84 | 947.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:30:00 | 948.75 | 942.84 | 947.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 951.30 | 944.53 | 947.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 12:00:00 | 951.30 | 944.53 | 947.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — BUY (started 2025-01-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 15:15:00 | 952.40 | 950.18 | 949.89 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 09:15:00 | 947.10 | 949.56 | 949.64 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 10:15:00 | 954.90 | 950.63 | 950.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 11:15:00 | 958.20 | 952.14 | 950.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-22 09:15:00 | 961.55 | 967.72 | 962.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 09:15:00 | 961.55 | 967.72 | 962.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 961.55 | 967.72 | 962.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:30:00 | 962.50 | 967.72 | 962.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 966.30 | 967.44 | 963.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 09:30:00 | 976.50 | 970.16 | 966.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 13:15:00 | 967.25 | 979.97 | 981.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2025-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 13:15:00 | 967.25 | 979.97 | 981.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 13:15:00 | 961.85 | 971.12 | 975.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 13:15:00 | 959.10 | 958.80 | 965.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 14:00:00 | 959.10 | 958.80 | 965.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 971.65 | 961.40 | 965.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:30:00 | 969.30 | 961.40 | 965.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 969.20 | 962.96 | 965.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 11:30:00 | 965.00 | 962.01 | 964.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 13:15:00 | 965.00 | 962.82 | 964.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 14:45:00 | 965.05 | 963.67 | 964.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 09:15:00 | 1006.35 | 973.22 | 969.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — BUY (started 2025-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 09:15:00 | 1006.35 | 973.22 | 969.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 12:15:00 | 1023.80 | 995.02 | 981.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 11:15:00 | 1042.75 | 1051.85 | 1031.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-03 12:00:00 | 1042.75 | 1051.85 | 1031.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 15:15:00 | 1037.00 | 1044.00 | 1033.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:15:00 | 1040.50 | 1044.00 | 1033.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 1028.15 | 1040.83 | 1033.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:00:00 | 1028.15 | 1040.83 | 1033.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 1024.80 | 1037.63 | 1032.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 11:30:00 | 1034.40 | 1036.60 | 1032.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 13:30:00 | 1030.05 | 1033.26 | 1031.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 15:00:00 | 1033.40 | 1033.29 | 1031.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-05 09:15:00 | 1016.80 | 1029.87 | 1030.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 09:15:00 | 1016.80 | 1029.87 | 1030.46 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-10 10:15:00 | 1036.25 | 1024.75 | 1023.26 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-02-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 13:15:00 | 1016.00 | 1024.29 | 1024.65 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 12:15:00 | 1038.20 | 1026.18 | 1024.78 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 15:15:00 | 1019.90 | 1025.98 | 1026.06 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-14 09:15:00 | 1034.45 | 1027.68 | 1026.82 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-02-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 13:15:00 | 1020.00 | 1025.42 | 1026.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-17 09:15:00 | 1009.50 | 1020.89 | 1023.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 14:15:00 | 1023.00 | 1019.37 | 1021.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 14:15:00 | 1023.00 | 1019.37 | 1021.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 1023.00 | 1019.37 | 1021.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 1023.00 | 1019.37 | 1021.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 1021.70 | 1019.84 | 1021.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 1022.05 | 1019.84 | 1021.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 1022.10 | 1020.29 | 1021.65 | EMA400 retest candle locked (from downside) |

### Cycle 152 — BUY (started 2025-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 13:15:00 | 1025.00 | 1022.27 | 1022.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 14:15:00 | 1027.45 | 1023.30 | 1022.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-19 15:15:00 | 1025.00 | 1025.93 | 1024.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 15:15:00 | 1025.00 | 1025.93 | 1024.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 15:15:00 | 1025.00 | 1025.93 | 1024.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 09:15:00 | 1018.80 | 1025.93 | 1024.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 1018.10 | 1024.37 | 1024.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 09:45:00 | 1019.95 | 1024.37 | 1024.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — SELL (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 10:15:00 | 1014.65 | 1022.42 | 1023.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-20 11:15:00 | 1006.30 | 1019.20 | 1021.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-21 14:15:00 | 1005.45 | 1005.22 | 1011.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-21 15:00:00 | 1005.45 | 1005.22 | 1011.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 1005.45 | 1005.01 | 1009.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:30:00 | 1004.05 | 1005.01 | 1009.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 1009.85 | 1006.49 | 1009.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 12:30:00 | 1010.05 | 1006.49 | 1009.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 1008.50 | 1006.89 | 1009.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 14:00:00 | 1008.50 | 1006.89 | 1009.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 14:15:00 | 1002.80 | 1006.07 | 1008.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 15:15:00 | 1002.00 | 1006.07 | 1008.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 09:45:00 | 1001.05 | 1004.72 | 1006.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 12:30:00 | 1002.00 | 1004.27 | 1005.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 13:00:00 | 1001.60 | 1004.27 | 1005.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 1008.80 | 1005.08 | 1005.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 15:00:00 | 1008.80 | 1005.08 | 1005.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 15:15:00 | 1008.00 | 1005.66 | 1006.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 1000.05 | 1005.66 | 1006.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 13:15:00 | 951.90 | 962.66 | 977.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 13:15:00 | 951.00 | 962.66 | 977.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 13:15:00 | 951.90 | 962.66 | 977.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 13:15:00 | 951.52 | 962.66 | 977.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 13:15:00 | 950.05 | 962.66 | 977.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-04 12:15:00 | 955.45 | 955.21 | 966.48 | SL hit (close>ema200) qty=0.50 sl=955.21 alert=retest2 |

### Cycle 154 — BUY (started 2025-03-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 14:15:00 | 962.05 | 959.81 | 959.55 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 956.00 | 960.59 | 960.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 11:15:00 | 949.85 | 956.87 | 958.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 14:15:00 | 959.65 | 956.06 | 957.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 14:15:00 | 959.65 | 956.06 | 957.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 959.65 | 956.06 | 957.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 15:00:00 | 959.65 | 956.06 | 957.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 960.00 | 956.85 | 957.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 09:15:00 | 954.80 | 956.85 | 957.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 09:15:00 | 948.90 | 946.28 | 946.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 948.90 | 946.28 | 946.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 10:15:00 | 955.00 | 948.02 | 947.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 13:15:00 | 946.50 | 949.01 | 947.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-19 13:15:00 | 946.50 | 949.01 | 947.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 13:15:00 | 946.50 | 949.01 | 947.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 14:00:00 | 946.50 | 949.01 | 947.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 14:15:00 | 949.80 | 949.17 | 948.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-19 15:15:00 | 952.00 | 949.17 | 948.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 11:15:00 | 962.00 | 965.35 | 965.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 11:15:00 | 962.00 | 965.35 | 965.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 15:15:00 | 956.95 | 962.30 | 963.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 09:15:00 | 971.40 | 964.12 | 964.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 09:15:00 | 971.40 | 964.12 | 964.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 971.40 | 964.12 | 964.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:00:00 | 971.40 | 964.12 | 964.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — BUY (started 2025-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 10:15:00 | 969.00 | 965.10 | 964.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 15:15:00 | 977.45 | 970.85 | 968.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 11:15:00 | 998.20 | 999.42 | 989.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-01 12:00:00 | 998.20 | 999.42 | 989.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 990.60 | 996.40 | 989.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 1044.85 | 994.92 | 990.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 12:15:00 | 1039.00 | 1059.89 | 1060.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 12:15:00 | 1039.00 | 1059.89 | 1060.00 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 13:15:00 | 1071.40 | 1059.10 | 1058.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 09:15:00 | 1076.10 | 1065.10 | 1061.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 10:15:00 | 1096.40 | 1096.96 | 1088.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-15 11:00:00 | 1096.40 | 1096.96 | 1088.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 09:15:00 | 1102.50 | 1100.50 | 1094.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 09:30:00 | 1091.80 | 1100.50 | 1094.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1101.00 | 1103.49 | 1099.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:15:00 | 1112.10 | 1103.95 | 1099.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 15:15:00 | 1155.00 | 1158.93 | 1159.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 15:15:00 | 1155.00 | 1158.93 | 1159.30 | EMA200 below EMA400 |

### Cycle 162 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 1168.80 | 1160.90 | 1160.16 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 14:15:00 | 1155.90 | 1161.74 | 1161.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 1144.00 | 1156.95 | 1159.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 12:15:00 | 1112.60 | 1111.83 | 1125.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 13:00:00 | 1112.60 | 1111.83 | 1125.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1152.00 | 1120.22 | 1124.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 1152.00 | 1120.22 | 1124.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 1149.40 | 1126.05 | 1127.09 | EMA400 retest candle locked (from downside) |

### Cycle 164 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1146.00 | 1130.04 | 1128.81 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-05-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 13:15:00 | 1122.30 | 1130.80 | 1131.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 14:15:00 | 1116.50 | 1127.94 | 1129.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 10:15:00 | 1120.20 | 1119.47 | 1123.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 10:15:00 | 1120.20 | 1119.47 | 1123.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 1120.20 | 1119.47 | 1123.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 10:45:00 | 1129.40 | 1119.47 | 1123.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 1129.20 | 1121.42 | 1123.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 12:00:00 | 1129.20 | 1121.42 | 1123.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — BUY (started 2025-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 12:15:00 | 1145.90 | 1126.31 | 1125.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 13:15:00 | 1157.00 | 1132.45 | 1128.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 10:15:00 | 1159.50 | 1159.60 | 1149.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 11:00:00 | 1159.50 | 1159.60 | 1149.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 1149.40 | 1156.35 | 1149.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 13:00:00 | 1149.40 | 1156.35 | 1149.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 1147.60 | 1154.60 | 1149.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 1147.60 | 1154.60 | 1149.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 1147.70 | 1153.22 | 1149.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:45:00 | 1147.80 | 1153.22 | 1149.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 1150.00 | 1152.57 | 1149.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 1144.50 | 1152.57 | 1149.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1136.10 | 1149.28 | 1147.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:00:00 | 1136.10 | 1149.28 | 1147.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1141.00 | 1147.62 | 1147.32 | EMA400 retest candle locked (from upside) |

### Cycle 167 — SELL (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 11:15:00 | 1138.00 | 1145.70 | 1146.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 1133.60 | 1142.14 | 1144.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 1143.30 | 1138.95 | 1142.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 1143.30 | 1138.95 | 1142.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1143.30 | 1138.95 | 1142.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 1143.30 | 1138.95 | 1142.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1148.50 | 1140.86 | 1142.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:45:00 | 1152.60 | 1140.86 | 1142.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 1140.40 | 1140.57 | 1142.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:30:00 | 1144.40 | 1140.57 | 1142.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 1140.00 | 1140.45 | 1142.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 1127.00 | 1139.85 | 1141.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 1146.00 | 1132.97 | 1135.58 | SL hit (close>static) qty=1.00 sl=1142.90 alert=retest2 |

### Cycle 168 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 1148.50 | 1138.64 | 1137.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 1149.70 | 1142.05 | 1139.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 10:15:00 | 1141.60 | 1141.96 | 1140.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 10:15:00 | 1141.60 | 1141.96 | 1140.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 1141.60 | 1141.96 | 1140.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:00:00 | 1141.60 | 1141.96 | 1140.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 1144.50 | 1142.47 | 1140.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:45:00 | 1141.60 | 1142.47 | 1140.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1135.50 | 1142.61 | 1141.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 1135.50 | 1142.61 | 1141.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1136.50 | 1141.39 | 1141.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:45:00 | 1131.90 | 1141.39 | 1141.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — SELL (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 11:15:00 | 1137.00 | 1140.51 | 1140.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 1123.00 | 1136.01 | 1138.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 1111.00 | 1110.63 | 1118.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 1111.00 | 1110.63 | 1118.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1111.00 | 1110.63 | 1118.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:30:00 | 1116.20 | 1110.63 | 1118.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1107.50 | 1108.35 | 1113.30 | EMA400 retest candle locked (from downside) |

### Cycle 170 — BUY (started 2025-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 14:15:00 | 1120.90 | 1116.27 | 1115.80 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 1113.60 | 1115.24 | 1115.42 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 09:15:00 | 1118.00 | 1115.51 | 1115.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 10:15:00 | 1122.50 | 1116.91 | 1116.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 09:15:00 | 1120.30 | 1121.53 | 1119.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 09:15:00 | 1120.30 | 1121.53 | 1119.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 1120.30 | 1121.53 | 1119.21 | EMA400 retest candle locked (from upside) |

### Cycle 173 — SELL (started 2025-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 14:15:00 | 1111.60 | 1118.21 | 1118.46 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 14:15:00 | 1121.20 | 1117.02 | 1116.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 15:15:00 | 1122.50 | 1118.12 | 1117.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 13:15:00 | 1119.10 | 1120.00 | 1118.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 13:15:00 | 1119.10 | 1120.00 | 1118.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 1119.10 | 1120.00 | 1118.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 14:00:00 | 1119.10 | 1120.00 | 1118.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 1114.80 | 1118.96 | 1118.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 1114.80 | 1118.96 | 1118.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 1115.00 | 1118.17 | 1118.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:15:00 | 1115.40 | 1118.17 | 1118.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — SELL (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 09:15:00 | 1108.70 | 1116.28 | 1117.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 12:15:00 | 1107.60 | 1112.93 | 1115.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 1080.60 | 1079.72 | 1088.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:15:00 | 1079.40 | 1079.72 | 1088.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1083.00 | 1078.86 | 1084.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:30:00 | 1083.30 | 1078.86 | 1084.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 1084.00 | 1080.63 | 1084.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 1082.60 | 1080.63 | 1084.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 1073.30 | 1079.16 | 1083.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:15:00 | 1072.10 | 1079.16 | 1083.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 09:45:00 | 1071.10 | 1072.89 | 1077.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 11:15:00 | 1087.70 | 1075.33 | 1075.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — BUY (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-19 11:15:00 | 1087.70 | 1075.33 | 1075.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-19 13:15:00 | 1088.20 | 1078.70 | 1076.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 15:15:00 | 1091.50 | 1091.95 | 1086.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 09:15:00 | 1094.50 | 1091.95 | 1086.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1124.60 | 1133.60 | 1129.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 1124.60 | 1133.60 | 1129.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1121.00 | 1131.08 | 1128.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 1117.10 | 1131.08 | 1128.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — SELL (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 09:15:00 | 1102.50 | 1125.37 | 1125.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 10:15:00 | 1098.90 | 1120.07 | 1123.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 12:15:00 | 1094.70 | 1093.06 | 1099.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-02 12:30:00 | 1094.30 | 1093.06 | 1099.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1101.80 | 1095.77 | 1098.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:15:00 | 1091.50 | 1095.29 | 1098.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:45:00 | 1091.80 | 1094.63 | 1097.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 14:30:00 | 1088.80 | 1093.70 | 1096.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 09:30:00 | 1091.00 | 1092.13 | 1095.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1094.20 | 1089.89 | 1092.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 1096.60 | 1089.89 | 1092.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1100.20 | 1091.95 | 1092.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:30:00 | 1098.60 | 1091.95 | 1092.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-07 11:15:00 | 1101.80 | 1093.92 | 1093.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — BUY (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 11:15:00 | 1101.80 | 1093.92 | 1093.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 15:15:00 | 1106.00 | 1099.25 | 1096.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 11:15:00 | 1097.90 | 1100.14 | 1097.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 11:15:00 | 1097.90 | 1100.14 | 1097.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 1097.90 | 1100.14 | 1097.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:45:00 | 1098.40 | 1100.14 | 1097.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 1096.70 | 1099.45 | 1097.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:00:00 | 1096.70 | 1099.45 | 1097.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 1096.80 | 1098.92 | 1097.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:30:00 | 1096.50 | 1098.92 | 1097.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 1100.70 | 1099.28 | 1097.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:45:00 | 1097.30 | 1099.28 | 1097.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 1097.30 | 1098.88 | 1097.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 1102.80 | 1098.88 | 1097.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1099.80 | 1099.07 | 1097.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 11:15:00 | 1105.50 | 1099.39 | 1098.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 12:00:00 | 1105.80 | 1100.67 | 1098.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 15:15:00 | 1095.60 | 1098.83 | 1098.54 | SL hit (close<static) qty=1.00 sl=1096.30 alert=retest2 |

### Cycle 179 — SELL (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 09:15:00 | 1095.00 | 1098.06 | 1098.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 14:15:00 | 1089.30 | 1095.19 | 1096.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 09:15:00 | 1097.10 | 1094.10 | 1095.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 1097.10 | 1094.10 | 1095.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1097.10 | 1094.10 | 1095.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:30:00 | 1097.80 | 1094.10 | 1095.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 1090.30 | 1093.34 | 1095.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 12:15:00 | 1088.00 | 1093.09 | 1095.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 12:45:00 | 1086.10 | 1090.60 | 1093.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 14:15:00 | 1082.00 | 1080.47 | 1080.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — BUY (started 2025-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 14:15:00 | 1082.00 | 1080.47 | 1080.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 10:15:00 | 1092.60 | 1083.96 | 1082.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 11:15:00 | 1097.20 | 1098.45 | 1092.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 11:30:00 | 1096.70 | 1098.45 | 1092.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1082.80 | 1094.53 | 1092.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 1080.10 | 1094.53 | 1092.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 1084.70 | 1092.56 | 1092.00 | EMA400 retest candle locked (from upside) |

### Cycle 181 — SELL (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 11:15:00 | 1086.80 | 1091.41 | 1091.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 09:15:00 | 1082.10 | 1088.85 | 1090.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 1070.60 | 1067.81 | 1074.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 1070.60 | 1067.81 | 1074.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1070.60 | 1067.81 | 1074.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:45:00 | 1065.20 | 1070.97 | 1074.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:30:00 | 1065.20 | 1063.69 | 1066.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 10:00:00 | 1065.20 | 1062.49 | 1064.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 09:15:00 | 1076.20 | 1065.00 | 1064.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 1076.20 | 1065.00 | 1064.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 10:15:00 | 1082.10 | 1068.42 | 1066.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 1068.60 | 1072.50 | 1069.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 1068.60 | 1072.50 | 1069.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1068.60 | 1072.50 | 1069.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 10:15:00 | 1071.00 | 1072.50 | 1069.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:00:00 | 1073.10 | 1072.62 | 1070.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 15:15:00 | 1070.70 | 1073.01 | 1072.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 09:15:00 | 1067.50 | 1071.54 | 1071.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 1067.50 | 1071.54 | 1071.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 10:15:00 | 1063.40 | 1069.91 | 1071.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 1068.60 | 1068.16 | 1070.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 12:15:00 | 1068.60 | 1068.16 | 1070.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 1068.60 | 1068.16 | 1070.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 1068.60 | 1068.16 | 1070.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 1072.50 | 1069.03 | 1070.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:45:00 | 1071.80 | 1069.03 | 1070.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1071.90 | 1069.60 | 1070.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 1071.30 | 1069.60 | 1070.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 1065.90 | 1068.48 | 1069.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:45:00 | 1068.70 | 1068.48 | 1069.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1058.90 | 1064.36 | 1066.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 14:15:00 | 1056.70 | 1061.98 | 1064.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:15:00 | 1056.10 | 1054.15 | 1055.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:30:00 | 1055.70 | 1053.64 | 1054.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 13:00:00 | 1057.40 | 1054.39 | 1054.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 13:15:00 | 1060.30 | 1055.58 | 1055.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — BUY (started 2025-08-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 13:15:00 | 1060.30 | 1055.58 | 1055.09 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 14:15:00 | 1046.90 | 1053.94 | 1054.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 09:15:00 | 1045.20 | 1051.19 | 1053.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 12:15:00 | 1051.90 | 1050.32 | 1052.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 12:15:00 | 1051.90 | 1050.32 | 1052.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 1051.90 | 1050.32 | 1052.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:45:00 | 1053.20 | 1050.32 | 1052.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 1056.40 | 1051.53 | 1052.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:00:00 | 1056.40 | 1051.53 | 1052.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 1056.30 | 1052.49 | 1053.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:45:00 | 1057.90 | 1052.49 | 1053.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — BUY (started 2025-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 15:15:00 | 1057.50 | 1053.49 | 1053.41 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 1051.60 | 1053.47 | 1053.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 15:15:00 | 1045.40 | 1051.59 | 1052.65 | Break + close below crossover candle low |

### Cycle 188 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1081.30 | 1057.53 | 1055.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 15:15:00 | 1091.40 | 1080.40 | 1072.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 1089.80 | 1094.16 | 1085.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 09:45:00 | 1092.40 | 1094.16 | 1085.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 1088.00 | 1091.12 | 1087.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 1085.70 | 1091.12 | 1087.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1085.50 | 1090.00 | 1087.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:00:00 | 1085.50 | 1090.00 | 1087.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1085.50 | 1089.10 | 1087.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:00:00 | 1088.10 | 1087.94 | 1086.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:45:00 | 1089.00 | 1088.41 | 1087.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 10:00:00 | 1087.30 | 1087.01 | 1086.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 10:45:00 | 1087.90 | 1087.31 | 1086.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 1087.30 | 1087.31 | 1086.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:00:00 | 1087.30 | 1087.31 | 1086.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 1086.80 | 1087.21 | 1086.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:30:00 | 1087.40 | 1087.21 | 1086.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-25 13:15:00 | 1084.10 | 1086.59 | 1086.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — SELL (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 13:15:00 | 1084.10 | 1086.59 | 1086.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 1080.30 | 1085.33 | 1086.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 11:15:00 | 1088.90 | 1084.01 | 1084.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 11:15:00 | 1088.90 | 1084.01 | 1084.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 1088.90 | 1084.01 | 1084.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:00:00 | 1088.90 | 1084.01 | 1084.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 1088.90 | 1084.99 | 1085.35 | EMA400 retest candle locked (from downside) |

### Cycle 190 — BUY (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 13:15:00 | 1088.10 | 1085.61 | 1085.60 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 14:15:00 | 1078.90 | 1084.27 | 1084.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 1070.30 | 1081.00 | 1083.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 11:15:00 | 1068.90 | 1066.74 | 1072.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 12:00:00 | 1068.90 | 1066.74 | 1072.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 1075.00 | 1066.83 | 1068.84 | EMA400 retest candle locked (from downside) |

### Cycle 192 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 1090.10 | 1073.95 | 1071.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 14:15:00 | 1099.30 | 1086.70 | 1079.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 1089.00 | 1091.28 | 1083.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 11:00:00 | 1089.00 | 1091.28 | 1083.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1076.70 | 1093.22 | 1089.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 1076.70 | 1093.22 | 1089.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 1078.50 | 1090.28 | 1088.56 | EMA400 retest candle locked (from upside) |

### Cycle 193 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 1075.00 | 1085.43 | 1086.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 1071.10 | 1082.57 | 1085.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1078.20 | 1074.81 | 1078.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 1078.20 | 1074.81 | 1078.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1078.20 | 1074.81 | 1078.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:15:00 | 1076.00 | 1074.81 | 1078.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1076.90 | 1075.23 | 1078.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:00:00 | 1073.20 | 1076.26 | 1077.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 1071.50 | 1076.57 | 1077.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 13:15:00 | 1080.10 | 1076.04 | 1076.84 | SL hit (close>static) qty=1.00 sl=1080.00 alert=retest2 |

### Cycle 194 — BUY (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 15:15:00 | 1082.30 | 1078.34 | 1077.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 1091.00 | 1081.95 | 1079.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 12:15:00 | 1097.90 | 1101.22 | 1096.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 12:15:00 | 1097.90 | 1101.22 | 1096.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 1097.90 | 1101.22 | 1096.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:45:00 | 1097.60 | 1101.22 | 1096.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 1102.00 | 1100.89 | 1097.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 15:15:00 | 1105.00 | 1100.89 | 1097.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 1096.10 | 1100.59 | 1097.72 | SL hit (close<static) qty=1.00 sl=1096.80 alert=retest2 |

### Cycle 195 — SELL (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 13:15:00 | 1092.00 | 1097.50 | 1097.92 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 1120.20 | 1100.99 | 1099.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 13:15:00 | 1132.80 | 1117.48 | 1108.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 12:15:00 | 1127.60 | 1128.93 | 1119.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 13:00:00 | 1127.60 | 1128.93 | 1119.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1119.00 | 1126.90 | 1121.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:30:00 | 1116.80 | 1126.90 | 1121.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 1125.50 | 1126.62 | 1121.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:30:00 | 1122.40 | 1126.62 | 1121.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 1127.00 | 1126.04 | 1122.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 14:30:00 | 1127.50 | 1125.77 | 1122.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 15:15:00 | 1129.10 | 1125.77 | 1122.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 09:30:00 | 1128.70 | 1126.69 | 1123.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 09:15:00 | 1121.00 | 1128.42 | 1126.77 | SL hit (close<static) qty=1.00 sl=1122.00 alert=retest2 |

### Cycle 197 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 11:15:00 | 1126.30 | 1131.59 | 1131.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 12:15:00 | 1122.90 | 1129.85 | 1131.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 1122.00 | 1116.68 | 1122.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:00:00 | 1122.00 | 1116.68 | 1122.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1123.00 | 1117.94 | 1122.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:45:00 | 1129.30 | 1117.94 | 1122.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1125.00 | 1119.35 | 1122.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 1126.00 | 1119.35 | 1122.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1122.20 | 1119.92 | 1122.36 | EMA400 retest candle locked (from downside) |

### Cycle 198 — BUY (started 2025-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 13:15:00 | 1128.10 | 1123.47 | 1123.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 15:15:00 | 1132.50 | 1126.13 | 1124.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 09:15:00 | 1133.70 | 1137.20 | 1132.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 09:45:00 | 1136.60 | 1137.20 | 1132.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 1133.40 | 1136.44 | 1132.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 15:15:00 | 1138.50 | 1136.20 | 1133.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 09:30:00 | 1138.50 | 1136.79 | 1134.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 10:15:00 | 1138.20 | 1136.79 | 1134.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 11:00:00 | 1140.50 | 1137.53 | 1134.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 1134.40 | 1136.91 | 1134.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:45:00 | 1132.70 | 1136.91 | 1134.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 1143.90 | 1138.30 | 1135.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:30:00 | 1140.00 | 1138.30 | 1135.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 1136.60 | 1138.86 | 1136.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:30:00 | 1134.90 | 1138.86 | 1136.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 1132.40 | 1137.57 | 1136.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 1132.40 | 1137.57 | 1136.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 1129.50 | 1135.95 | 1135.87 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-07 11:15:00 | 1129.50 | 1135.95 | 1135.87 | SL hit (close<static) qty=1.00 sl=1130.90 alert=retest2 |

### Cycle 199 — SELL (started 2025-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 12:15:00 | 1129.70 | 1134.70 | 1135.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 13:15:00 | 1127.20 | 1133.20 | 1134.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 12:15:00 | 1125.00 | 1124.65 | 1128.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-08 13:00:00 | 1125.00 | 1124.65 | 1128.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 1121.00 | 1120.02 | 1123.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:00:00 | 1121.00 | 1120.02 | 1123.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 1120.20 | 1120.05 | 1123.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 1120.20 | 1120.05 | 1123.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 1122.70 | 1119.93 | 1122.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:00:00 | 1122.70 | 1119.93 | 1122.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 1125.50 | 1121.04 | 1122.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:30:00 | 1123.20 | 1121.04 | 1122.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 1129.40 | 1122.71 | 1123.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:45:00 | 1129.80 | 1122.71 | 1123.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — BUY (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 14:15:00 | 1125.50 | 1123.72 | 1123.66 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 1121.90 | 1123.42 | 1123.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 12:15:00 | 1115.50 | 1121.84 | 1122.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 1118.10 | 1117.36 | 1119.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 14:15:00 | 1118.10 | 1117.36 | 1119.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 1118.10 | 1117.36 | 1119.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:30:00 | 1123.40 | 1117.36 | 1119.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 1113.10 | 1116.51 | 1118.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 1120.20 | 1116.51 | 1118.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1117.60 | 1116.73 | 1118.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 11:30:00 | 1114.60 | 1116.80 | 1118.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 15:15:00 | 1112.50 | 1117.06 | 1118.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:45:00 | 1111.80 | 1115.40 | 1116.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 11:15:00 | 1122.40 | 1116.80 | 1117.46 | SL hit (close>static) qty=1.00 sl=1122.10 alert=retest2 |

### Cycle 202 — BUY (started 2025-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 12:15:00 | 1133.10 | 1120.06 | 1118.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 13:15:00 | 1136.60 | 1123.37 | 1120.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 12:15:00 | 1173.70 | 1175.63 | 1167.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 13:00:00 | 1173.70 | 1175.63 | 1167.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 1167.40 | 1173.99 | 1167.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:00:00 | 1167.40 | 1173.99 | 1167.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1158.80 | 1170.95 | 1166.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 1158.80 | 1170.95 | 1166.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1159.00 | 1168.56 | 1165.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 1149.20 | 1168.56 | 1165.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 1152.30 | 1162.10 | 1163.06 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 11:15:00 | 1173.70 | 1162.99 | 1161.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 12:15:00 | 1177.70 | 1165.93 | 1163.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 15:15:00 | 1167.10 | 1167.33 | 1164.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 09:15:00 | 1170.70 | 1167.33 | 1164.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1165.10 | 1166.89 | 1164.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 1165.10 | 1166.89 | 1164.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 1167.60 | 1167.03 | 1165.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 15:00:00 | 1169.90 | 1165.94 | 1164.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 1179.30 | 1165.91 | 1165.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:00:00 | 1170.20 | 1174.06 | 1172.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 13:45:00 | 1170.50 | 1172.69 | 1171.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1176.80 | 1174.45 | 1172.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:15:00 | 1171.80 | 1174.45 | 1172.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1172.90 | 1174.14 | 1172.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:30:00 | 1173.20 | 1174.14 | 1172.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 1168.50 | 1173.01 | 1172.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:00:00 | 1168.50 | 1173.01 | 1172.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-31 12:15:00 | 1164.20 | 1171.25 | 1171.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — SELL (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 12:15:00 | 1164.20 | 1171.25 | 1171.72 | EMA200 below EMA400 |

### Cycle 206 — BUY (started 2025-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 13:15:00 | 1197.10 | 1172.94 | 1171.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 14:15:00 | 1199.70 | 1178.29 | 1173.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 09:15:00 | 1167.90 | 1178.97 | 1174.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 09:15:00 | 1167.90 | 1178.97 | 1174.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 1167.90 | 1178.97 | 1174.99 | EMA400 retest candle locked (from upside) |

### Cycle 207 — SELL (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 13:15:00 | 1160.00 | 1176.32 | 1177.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 11:15:00 | 1151.20 | 1164.63 | 1170.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 11:15:00 | 1150.90 | 1149.36 | 1158.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-11 12:00:00 | 1150.90 | 1149.36 | 1158.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 1155.70 | 1151.15 | 1157.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:30:00 | 1156.20 | 1151.15 | 1157.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 1157.00 | 1152.32 | 1157.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 10:00:00 | 1154.00 | 1153.24 | 1156.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 10:15:00 | 1161.80 | 1154.96 | 1157.39 | SL hit (close>static) qty=1.00 sl=1159.00 alert=retest2 |

### Cycle 208 — BUY (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 14:15:00 | 1161.60 | 1158.93 | 1158.75 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 09:15:00 | 1153.80 | 1158.40 | 1158.57 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 11:15:00 | 1161.80 | 1158.63 | 1158.62 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 13:15:00 | 1156.20 | 1158.67 | 1158.67 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 1165.90 | 1158.07 | 1157.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 12:15:00 | 1177.90 | 1164.72 | 1161.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1165.90 | 1170.20 | 1165.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 1165.90 | 1170.20 | 1165.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1165.90 | 1170.20 | 1165.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 1165.90 | 1170.20 | 1165.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1164.00 | 1168.96 | 1165.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:45:00 | 1163.50 | 1168.96 | 1165.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 1166.00 | 1168.37 | 1165.27 | EMA400 retest candle locked (from upside) |

### Cycle 213 — SELL (started 2025-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 14:15:00 | 1153.60 | 1163.74 | 1163.81 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 1175.60 | 1164.02 | 1162.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 13:15:00 | 1181.10 | 1173.90 | 1170.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 12:15:00 | 1177.60 | 1178.93 | 1174.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 12:15:00 | 1177.60 | 1178.93 | 1174.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 1177.60 | 1178.93 | 1174.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 12:45:00 | 1177.00 | 1178.93 | 1174.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1174.90 | 1179.73 | 1176.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:30:00 | 1169.70 | 1179.73 | 1176.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 1168.80 | 1177.54 | 1176.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:00:00 | 1168.80 | 1177.54 | 1176.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 1173.30 | 1176.69 | 1175.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:45:00 | 1167.90 | 1176.69 | 1175.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 1176.80 | 1178.34 | 1176.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 1176.80 | 1178.34 | 1176.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 1174.70 | 1177.61 | 1176.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 1176.80 | 1177.61 | 1176.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1176.90 | 1177.47 | 1176.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 10:30:00 | 1180.50 | 1178.65 | 1177.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 15:15:00 | 1175.50 | 1179.50 | 1179.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 215 — SELL (started 2025-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 15:15:00 | 1175.50 | 1179.50 | 1179.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 10:15:00 | 1174.00 | 1178.26 | 1179.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 13:15:00 | 1145.00 | 1143.65 | 1149.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 13:30:00 | 1144.50 | 1143.65 | 1149.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 1148.20 | 1144.56 | 1149.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 1148.20 | 1144.56 | 1149.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 1145.10 | 1144.67 | 1148.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 1145.10 | 1144.67 | 1148.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1146.50 | 1145.03 | 1148.71 | EMA400 retest candle locked (from downside) |

### Cycle 216 — BUY (started 2025-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 12:15:00 | 1160.90 | 1150.84 | 1150.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 1164.90 | 1155.29 | 1152.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 11:15:00 | 1154.80 | 1157.02 | 1154.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 11:15:00 | 1154.80 | 1157.02 | 1154.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 1154.80 | 1157.02 | 1154.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 1154.80 | 1157.02 | 1154.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 1153.20 | 1156.26 | 1154.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:45:00 | 1153.00 | 1156.26 | 1154.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 1146.90 | 1154.39 | 1153.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:45:00 | 1148.10 | 1154.39 | 1153.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 217 — SELL (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 14:15:00 | 1146.90 | 1152.89 | 1153.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 15:15:00 | 1145.00 | 1151.31 | 1152.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 13:15:00 | 1148.90 | 1146.84 | 1149.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 13:15:00 | 1148.90 | 1146.84 | 1149.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 1148.90 | 1146.84 | 1149.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:00:00 | 1148.90 | 1146.84 | 1149.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 1146.30 | 1146.73 | 1148.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 1143.00 | 1146.50 | 1148.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 1155.00 | 1148.20 | 1149.24 | SL hit (close>static) qty=1.00 sl=1149.10 alert=retest2 |

### Cycle 218 — BUY (started 2025-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 14:15:00 | 1150.70 | 1145.90 | 1145.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 10:15:00 | 1165.80 | 1150.92 | 1148.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 15:15:00 | 1155.00 | 1155.67 | 1152.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:15:00 | 1170.60 | 1155.67 | 1152.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1168.30 | 1158.20 | 1153.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 10:30:00 | 1173.30 | 1161.44 | 1155.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 1173.10 | 1167.13 | 1160.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 09:15:00 | 1176.20 | 1173.92 | 1171.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 1177.20 | 1173.16 | 1171.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 1181.80 | 1174.89 | 1172.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 13:45:00 | 1184.10 | 1178.13 | 1174.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 10:45:00 | 1183.60 | 1180.43 | 1178.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 13:45:00 | 1182.70 | 1181.01 | 1179.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:45:00 | 1184.50 | 1182.68 | 1180.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 1176.70 | 1181.48 | 1180.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:00:00 | 1176.70 | 1181.48 | 1180.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 1174.20 | 1180.03 | 1179.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:30:00 | 1172.90 | 1180.03 | 1179.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-24 13:15:00 | 1177.00 | 1179.02 | 1179.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 219 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 1177.00 | 1179.02 | 1179.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 15:15:00 | 1174.00 | 1176.37 | 1177.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 1179.20 | 1176.93 | 1177.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 1179.20 | 1176.93 | 1177.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1179.20 | 1176.93 | 1177.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 1179.20 | 1176.93 | 1177.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1178.60 | 1177.27 | 1177.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:00:00 | 1178.60 | 1177.27 | 1177.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 220 — BUY (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 11:15:00 | 1187.70 | 1179.35 | 1178.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 12:15:00 | 1188.90 | 1181.26 | 1179.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 09:15:00 | 1184.80 | 1186.67 | 1183.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 09:15:00 | 1184.80 | 1186.67 | 1183.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 1184.80 | 1186.67 | 1183.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:30:00 | 1190.60 | 1186.67 | 1183.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 1179.10 | 1185.16 | 1182.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:45:00 | 1178.30 | 1185.16 | 1182.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 1181.80 | 1184.48 | 1182.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 13:15:00 | 1184.60 | 1183.65 | 1182.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 15:15:00 | 1173.80 | 1180.95 | 1181.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 221 — SELL (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 15:15:00 | 1173.80 | 1180.95 | 1181.45 | EMA200 below EMA400 |

### Cycle 222 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 1189.90 | 1183.29 | 1182.39 | EMA200 above EMA400 |

### Cycle 223 — SELL (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 14:15:00 | 1177.40 | 1182.54 | 1183.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 15:15:00 | 1173.30 | 1180.69 | 1182.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 1181.30 | 1180.81 | 1182.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 09:15:00 | 1181.30 | 1180.81 | 1182.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1181.30 | 1180.81 | 1182.17 | EMA400 retest candle locked (from downside) |

### Cycle 224 — BUY (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 12:15:00 | 1185.80 | 1180.20 | 1179.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 09:15:00 | 1193.90 | 1184.37 | 1182.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 13:15:00 | 1208.10 | 1208.44 | 1199.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 13:30:00 | 1207.90 | 1208.44 | 1199.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1201.90 | 1208.13 | 1201.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:30:00 | 1203.40 | 1208.13 | 1201.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1198.90 | 1206.28 | 1201.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:30:00 | 1202.50 | 1206.28 | 1201.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 1197.60 | 1204.55 | 1201.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 14:45:00 | 1203.00 | 1200.83 | 1200.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:45:00 | 1203.30 | 1200.93 | 1200.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 11:15:00 | 1190.80 | 1198.47 | 1199.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 225 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 11:15:00 | 1190.80 | 1198.47 | 1199.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 1185.90 | 1195.96 | 1198.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 1186.30 | 1185.12 | 1190.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 12:15:00 | 1186.30 | 1185.12 | 1190.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 1186.30 | 1185.12 | 1190.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:30:00 | 1189.70 | 1185.12 | 1190.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 1192.60 | 1186.61 | 1190.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:45:00 | 1193.50 | 1186.61 | 1190.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1192.50 | 1187.79 | 1190.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:30:00 | 1196.00 | 1187.79 | 1190.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1191.90 | 1189.00 | 1190.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:15:00 | 1186.50 | 1189.04 | 1190.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:45:00 | 1185.50 | 1188.17 | 1189.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:45:00 | 1187.30 | 1188.52 | 1189.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 1184.50 | 1188.57 | 1189.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 1181.30 | 1187.12 | 1188.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:15:00 | 1175.70 | 1184.63 | 1187.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 1192.40 | 1180.04 | 1183.39 | SL hit (close>static) qty=1.00 sl=1191.00 alert=retest2 |

### Cycle 226 — BUY (started 2026-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 12:15:00 | 1189.80 | 1185.24 | 1185.23 | EMA200 above EMA400 |

### Cycle 227 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 1180.50 | 1185.11 | 1185.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 15:15:00 | 1174.00 | 1180.96 | 1183.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-20 10:15:00 | 1181.80 | 1180.97 | 1182.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 10:15:00 | 1181.80 | 1180.97 | 1182.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 1181.80 | 1180.97 | 1182.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:30:00 | 1183.60 | 1180.97 | 1182.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 1177.60 | 1180.30 | 1182.32 | EMA400 retest candle locked (from downside) |

### Cycle 228 — BUY (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-20 12:15:00 | 1200.80 | 1184.40 | 1184.00 | EMA200 above EMA400 |

### Cycle 229 — SELL (started 2026-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 11:15:00 | 1179.80 | 1183.63 | 1184.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 13:15:00 | 1160.80 | 1177.45 | 1181.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1183.90 | 1174.95 | 1178.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 1183.90 | 1174.95 | 1178.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1183.90 | 1174.95 | 1178.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 1183.90 | 1174.95 | 1178.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1183.20 | 1176.60 | 1179.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:45:00 | 1185.50 | 1176.60 | 1179.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 1173.80 | 1176.50 | 1178.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 12:30:00 | 1175.70 | 1176.50 | 1178.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 1173.70 | 1175.94 | 1178.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:00:00 | 1171.40 | 1175.81 | 1177.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 13:15:00 | 1171.20 | 1174.27 | 1176.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 13:45:00 | 1162.20 | 1171.90 | 1175.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 13:45:00 | 1169.90 | 1167.39 | 1169.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 1182.50 | 1170.41 | 1171.10 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-27 14:15:00 | 1182.50 | 1170.41 | 1171.10 | SL hit (close>static) qty=1.00 sl=1181.20 alert=retest2 |

### Cycle 230 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 1189.50 | 1174.23 | 1172.78 | EMA200 above EMA400 |

### Cycle 231 — SELL (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 09:15:00 | 1129.00 | 1165.18 | 1168.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 11:15:00 | 1118.10 | 1149.71 | 1160.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 1124.60 | 1115.80 | 1129.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 10:00:00 | 1124.60 | 1115.80 | 1129.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 1133.00 | 1119.24 | 1129.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:00:00 | 1133.00 | 1119.24 | 1129.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 1131.40 | 1121.67 | 1129.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 1103.40 | 1131.11 | 1131.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 13:15:00 | 1124.90 | 1131.07 | 1131.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 14:30:00 | 1120.10 | 1122.95 | 1127.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 13:45:00 | 1123.00 | 1116.35 | 1120.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1124.50 | 1117.98 | 1121.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 1124.50 | 1117.98 | 1121.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1122.00 | 1118.78 | 1121.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 1152.40 | 1118.78 | 1121.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-03 10:15:00 | 1149.00 | 1127.84 | 1125.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 232 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 1149.00 | 1127.84 | 1125.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 1156.00 | 1133.47 | 1127.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 12:15:00 | 1154.80 | 1156.01 | 1145.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 13:00:00 | 1154.80 | 1156.01 | 1145.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 1148.50 | 1153.96 | 1148.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 1148.50 | 1153.96 | 1148.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 1146.70 | 1152.51 | 1148.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:15:00 | 1146.30 | 1152.51 | 1148.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 1147.90 | 1151.58 | 1148.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:45:00 | 1154.60 | 1151.43 | 1148.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 09:45:00 | 1149.80 | 1152.86 | 1149.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 14:15:00 | 1151.90 | 1156.52 | 1156.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 233 — SELL (started 2026-02-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 14:15:00 | 1151.90 | 1156.52 | 1156.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 12:15:00 | 1147.50 | 1151.22 | 1153.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 1140.90 | 1137.54 | 1141.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 14:15:00 | 1140.90 | 1137.54 | 1141.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 1140.90 | 1137.54 | 1141.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 1140.90 | 1137.54 | 1141.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1136.70 | 1137.75 | 1140.91 | EMA400 retest candle locked (from downside) |

### Cycle 234 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 1149.00 | 1143.15 | 1142.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 10:15:00 | 1158.20 | 1148.02 | 1145.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 1158.30 | 1159.19 | 1152.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 10:00:00 | 1158.30 | 1159.19 | 1152.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 1158.10 | 1160.19 | 1155.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:00:00 | 1158.10 | 1160.19 | 1155.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 1159.20 | 1159.99 | 1155.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:45:00 | 1162.10 | 1159.99 | 1155.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 1154.90 | 1158.98 | 1155.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:30:00 | 1161.10 | 1158.78 | 1155.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:15:00 | 1165.10 | 1158.78 | 1155.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 13:45:00 | 1160.60 | 1160.53 | 1157.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 14:30:00 | 1159.80 | 1159.69 | 1157.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 1155.20 | 1158.79 | 1157.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 1166.60 | 1158.79 | 1157.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 11:15:00 | 1159.00 | 1168.61 | 1169.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 235 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 11:15:00 | 1159.00 | 1168.61 | 1169.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 15:15:00 | 1157.00 | 1163.25 | 1166.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 1114.50 | 1108.92 | 1117.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 14:15:00 | 1114.50 | 1108.92 | 1117.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1114.50 | 1108.92 | 1117.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 1121.50 | 1108.92 | 1117.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 1121.10 | 1111.36 | 1117.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 1124.80 | 1111.36 | 1117.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1120.50 | 1113.19 | 1117.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 1094.00 | 1118.26 | 1118.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-16 13:15:00 | 1090.30 | 1080.64 | 1079.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 236 — BUY (started 2026-03-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 13:15:00 | 1090.30 | 1080.64 | 1079.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-16 14:15:00 | 1093.60 | 1083.23 | 1080.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-17 11:15:00 | 1086.80 | 1088.66 | 1084.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 11:15:00 | 1086.80 | 1088.66 | 1084.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 1086.80 | 1088.66 | 1084.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:45:00 | 1084.90 | 1088.66 | 1084.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 1080.00 | 1086.93 | 1084.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:45:00 | 1080.40 | 1086.93 | 1084.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 1085.40 | 1086.62 | 1084.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 13:30:00 | 1079.20 | 1086.62 | 1084.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 1077.30 | 1084.76 | 1083.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 15:00:00 | 1077.30 | 1084.76 | 1083.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 1077.50 | 1083.30 | 1083.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 1078.10 | 1083.30 | 1083.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 237 — SELL (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 09:15:00 | 1077.50 | 1082.14 | 1082.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-18 15:15:00 | 1071.00 | 1077.11 | 1079.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 10:15:00 | 1051.90 | 1051.21 | 1060.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 11:00:00 | 1051.90 | 1051.21 | 1060.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 1027.80 | 1033.53 | 1043.00 | EMA400 retest candle locked (from downside) |

### Cycle 238 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1065.40 | 1048.62 | 1046.48 | EMA200 above EMA400 |

### Cycle 239 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 1042.70 | 1049.16 | 1049.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1040.50 | 1046.02 | 1047.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 11:15:00 | 1031.00 | 1025.99 | 1033.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-01 12:00:00 | 1031.00 | 1025.99 | 1033.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 12:15:00 | 1029.50 | 1024.35 | 1028.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 13:00:00 | 1029.50 | 1024.35 | 1028.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 1028.70 | 1025.22 | 1028.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:15:00 | 1031.70 | 1025.22 | 1028.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 1043.00 | 1028.78 | 1029.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 15:00:00 | 1043.00 | 1028.78 | 1029.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 240 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 1044.90 | 1032.00 | 1031.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 1056.70 | 1036.58 | 1033.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 1075.50 | 1085.45 | 1078.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 1075.50 | 1085.45 | 1078.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1075.50 | 1085.45 | 1078.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 10:00:00 | 1075.50 | 1085.45 | 1078.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 1081.40 | 1084.64 | 1078.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:45:00 | 1085.70 | 1084.89 | 1079.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 10:00:00 | 1090.30 | 1094.71 | 1090.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 12:15:00 | 1194.27 | 1161.10 | 1140.93 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 241 — SELL (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 14:15:00 | 1161.80 | 1167.31 | 1167.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-27 15:15:00 | 1155.00 | 1164.84 | 1166.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 1156.00 | 1153.93 | 1158.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-29 10:15:00 | 1160.70 | 1153.93 | 1158.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 1161.40 | 1155.42 | 1158.90 | EMA400 retest candle locked (from downside) |

### Cycle 242 — BUY (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 14:15:00 | 1168.10 | 1161.84 | 1161.19 | EMA200 above EMA400 |

### Cycle 243 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 1148.80 | 1160.70 | 1160.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 1145.40 | 1157.64 | 1159.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 1163.00 | 1152.23 | 1154.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 1163.00 | 1152.23 | 1154.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1163.00 | 1152.23 | 1154.99 | EMA400 retest candle locked (from downside) |

### Cycle 244 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 1161.40 | 1157.17 | 1156.88 | EMA200 above EMA400 |

### Cycle 245 — SELL (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 09:15:00 | 1152.10 | 1156.61 | 1156.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 12:15:00 | 1141.60 | 1150.57 | 1153.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 14:15:00 | 1153.60 | 1151.07 | 1152.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 14:15:00 | 1153.60 | 1151.07 | 1152.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 1153.60 | 1151.07 | 1152.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:45:00 | 1157.10 | 1151.07 | 1152.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 1149.00 | 1150.65 | 1152.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 1144.90 | 1150.65 | 1152.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 1139.50 | 1148.42 | 1151.41 | EMA400 retest candle locked (from downside) |

### Cycle 246 — BUY (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 09:15:00 | 1167.60 | 1154.52 | 1153.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 11:15:00 | 1181.00 | 1162.31 | 1156.98 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-18 14:00:00 | 768.45 | 2023-05-24 10:15:00 | 763.71 | STOP_HIT | 1.00 | 0.62% |
| BUY | retest2 | 2023-05-31 13:45:00 | 784.89 | 2023-06-05 15:15:00 | 782.77 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2023-06-01 09:30:00 | 786.13 | 2023-06-05 15:15:00 | 782.77 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2023-06-01 14:45:00 | 785.49 | 2023-06-05 15:15:00 | 782.77 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2023-06-02 09:15:00 | 787.71 | 2023-06-05 15:15:00 | 782.77 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2023-06-26 11:15:00 | 845.05 | 2023-06-26 11:15:00 | 843.81 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2023-06-28 11:15:00 | 846.77 | 2023-06-30 09:15:00 | 842.03 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2023-06-28 11:45:00 | 847.02 | 2023-06-30 09:15:00 | 842.03 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2023-06-28 13:00:00 | 847.47 | 2023-06-30 09:15:00 | 842.03 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2023-07-14 10:15:00 | 833.84 | 2023-07-21 15:15:00 | 841.79 | STOP_HIT | 1.00 | 0.95% |
| BUY | retest2 | 2023-07-14 12:00:00 | 836.60 | 2023-07-21 15:15:00 | 841.79 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest2 | 2023-08-16 10:30:00 | 830.92 | 2023-08-17 15:15:00 | 834.53 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2023-08-21 09:15:00 | 834.48 | 2023-08-23 13:15:00 | 833.59 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2023-08-21 10:15:00 | 834.23 | 2023-08-23 13:15:00 | 833.59 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2023-08-21 11:15:00 | 834.13 | 2023-08-23 13:15:00 | 833.59 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2023-08-21 12:15:00 | 835.56 | 2023-08-23 13:15:00 | 833.59 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2023-08-21 15:00:00 | 838.03 | 2023-08-23 13:15:00 | 833.59 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2023-08-22 09:15:00 | 838.58 | 2023-08-23 13:15:00 | 833.59 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2023-08-22 10:15:00 | 838.33 | 2023-08-23 13:15:00 | 833.59 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2023-08-22 10:45:00 | 839.07 | 2023-08-23 13:15:00 | 833.59 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-08-29 09:30:00 | 828.65 | 2023-08-29 10:15:00 | 832.16 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2023-08-29 10:45:00 | 828.90 | 2023-08-30 09:15:00 | 834.87 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2023-08-29 11:30:00 | 828.11 | 2023-08-30 09:15:00 | 834.87 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest1 | 2023-09-04 09:15:00 | 835.86 | 2023-09-04 09:15:00 | 829.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2023-09-13 14:15:00 | 873.15 | 2023-09-20 10:15:00 | 859.47 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2023-09-15 10:00:00 | 871.37 | 2023-09-20 10:15:00 | 859.47 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2023-09-18 09:30:00 | 873.54 | 2023-09-20 10:15:00 | 859.47 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2023-09-18 11:45:00 | 873.30 | 2023-09-20 10:15:00 | 859.47 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2023-10-06 11:30:00 | 861.39 | 2023-10-06 12:15:00 | 861.59 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2023-10-06 12:15:00 | 860.21 | 2023-10-06 12:15:00 | 861.59 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2023-10-13 10:15:00 | 900.16 | 2023-10-18 11:15:00 | 889.15 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2023-10-13 12:00:00 | 898.14 | 2023-10-18 11:15:00 | 889.15 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2023-10-23 12:00:00 | 879.76 | 2023-10-27 13:15:00 | 879.42 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2023-10-23 13:15:00 | 879.22 | 2023-10-27 13:15:00 | 879.42 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2023-10-23 14:30:00 | 878.53 | 2023-10-27 13:15:00 | 879.42 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2023-10-25 10:00:00 | 875.62 | 2023-10-27 13:15:00 | 879.42 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2023-10-26 09:15:00 | 868.50 | 2023-10-27 13:15:00 | 879.42 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2023-11-13 09:15:00 | 896.70 | 2023-11-13 11:15:00 | 906.83 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2023-11-17 09:15:00 | 916.56 | 2023-11-22 12:15:00 | 914.24 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2023-12-01 09:15:00 | 933.94 | 2023-12-08 13:15:00 | 935.27 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2023-12-01 10:30:00 | 930.29 | 2023-12-08 13:15:00 | 935.27 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2023-12-21 14:15:00 | 970.93 | 2023-12-29 10:15:00 | 1068.02 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-22 09:15:00 | 971.92 | 2023-12-29 14:15:00 | 1069.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-17 10:30:00 | 1126.94 | 2024-01-18 09:15:00 | 1112.62 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-01-23 09:15:00 | 1145.81 | 2024-01-23 12:15:00 | 1129.46 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-01-29 13:15:00 | 1124.23 | 2024-02-02 09:15:00 | 1126.00 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2024-01-30 10:30:00 | 1123.88 | 2024-02-02 09:15:00 | 1126.00 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2024-02-06 10:45:00 | 1148.72 | 2024-02-08 09:15:00 | 1126.05 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-02-06 11:15:00 | 1148.72 | 2024-02-08 09:15:00 | 1126.05 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-02-07 11:45:00 | 1145.71 | 2024-02-08 09:15:00 | 1126.05 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-02-07 12:15:00 | 1147.34 | 2024-02-08 09:15:00 | 1126.05 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-02-20 11:30:00 | 1134.40 | 2024-02-22 12:15:00 | 1132.03 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2024-02-22 11:00:00 | 1134.25 | 2024-02-22 12:15:00 | 1132.03 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2024-02-26 11:30:00 | 1147.14 | 2024-02-29 09:15:00 | 1153.81 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2024-03-06 13:30:00 | 1197.86 | 2024-03-12 10:15:00 | 1199.00 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2024-03-06 14:00:00 | 1204.03 | 2024-03-12 10:15:00 | 1199.00 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2024-03-11 11:15:00 | 1200.73 | 2024-03-12 10:15:00 | 1199.00 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2024-03-22 11:15:00 | 1109.95 | 2024-04-02 14:15:00 | 1118.50 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-03-22 12:00:00 | 1109.61 | 2024-04-02 14:15:00 | 1118.50 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-03-22 15:00:00 | 1108.57 | 2024-04-02 14:15:00 | 1118.50 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-04-10 09:30:00 | 1115.63 | 2024-04-15 11:15:00 | 1113.16 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2024-04-10 11:00:00 | 1114.84 | 2024-04-15 11:15:00 | 1113.16 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2024-04-10 12:30:00 | 1117.71 | 2024-04-15 11:15:00 | 1113.16 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-04-15 10:45:00 | 1115.34 | 2024-04-15 11:15:00 | 1113.16 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2024-04-29 11:30:00 | 1081.21 | 2024-04-30 13:15:00 | 1098.20 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-05-06 13:15:00 | 1082.00 | 2024-05-07 09:15:00 | 1100.32 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-05-06 14:45:00 | 1082.35 | 2024-05-07 09:15:00 | 1100.32 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-05-16 13:30:00 | 1059.63 | 2024-05-16 14:15:00 | 1086.50 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2024-05-29 09:15:00 | 1072.47 | 2024-06-04 09:15:00 | 1062.79 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest2 | 2024-06-11 10:15:00 | 1125.81 | 2024-06-12 13:15:00 | 1112.77 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-06-14 11:45:00 | 1103.68 | 2024-06-18 14:15:00 | 1113.21 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-06-18 10:00:00 | 1104.18 | 2024-06-18 14:15:00 | 1113.21 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-06-18 11:30:00 | 1105.76 | 2024-06-18 14:15:00 | 1113.21 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-06-18 12:15:00 | 1106.10 | 2024-06-18 14:15:00 | 1113.21 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-06-25 11:15:00 | 1081.06 | 2024-06-28 12:15:00 | 1084.52 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2024-06-25 12:45:00 | 1082.25 | 2024-06-28 12:15:00 | 1084.52 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-06-25 13:30:00 | 1081.41 | 2024-06-28 12:15:00 | 1084.52 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-07-08 15:15:00 | 1135.88 | 2024-07-10 10:15:00 | 1123.49 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-07-09 11:45:00 | 1135.88 | 2024-07-10 10:15:00 | 1123.49 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-07-10 13:15:00 | 1139.39 | 2024-07-11 11:15:00 | 1126.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-07-10 15:00:00 | 1137.71 | 2024-07-11 11:15:00 | 1126.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-07-22 10:15:00 | 1186.21 | 2024-07-29 09:15:00 | 1194.15 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2024-07-22 10:45:00 | 1183.93 | 2024-07-29 09:15:00 | 1194.15 | STOP_HIT | 1.00 | 0.86% |
| SELL | retest2 | 2024-07-31 09:15:00 | 1175.65 | 2024-08-01 11:15:00 | 1213.75 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2024-08-13 14:45:00 | 1174.65 | 2024-08-16 12:15:00 | 1187.85 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-08-14 09:15:00 | 1171.45 | 2024-08-16 12:15:00 | 1187.85 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-08-16 11:30:00 | 1174.95 | 2024-08-16 12:15:00 | 1187.85 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-08-26 10:45:00 | 1208.25 | 2024-08-28 14:15:00 | 1201.95 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2024-08-28 11:15:00 | 1207.70 | 2024-08-28 14:15:00 | 1201.95 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2024-09-05 11:45:00 | 1194.15 | 2024-09-09 14:15:00 | 1194.40 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest1 | 2024-09-11 09:15:00 | 1214.55 | 2024-09-12 12:15:00 | 1202.30 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-09-12 14:45:00 | 1220.30 | 2024-09-18 11:15:00 | 1205.70 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-09-16 09:45:00 | 1216.05 | 2024-09-18 11:15:00 | 1205.70 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-09-16 12:15:00 | 1216.05 | 2024-09-18 11:15:00 | 1205.70 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-09-16 14:00:00 | 1217.05 | 2024-09-18 11:15:00 | 1205.70 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-09-23 09:30:00 | 1226.40 | 2024-09-23 14:15:00 | 1210.95 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-09-24 09:15:00 | 1224.25 | 2024-09-24 10:15:00 | 1215.85 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-09-27 09:15:00 | 1203.50 | 2024-09-27 09:15:00 | 1206.50 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2024-10-11 10:30:00 | 1110.70 | 2024-10-21 09:15:00 | 999.63 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-11 13:15:00 | 1109.75 | 2024-10-21 09:15:00 | 998.77 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-14 12:15:00 | 1110.80 | 2024-10-21 09:15:00 | 999.72 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-14 12:45:00 | 1110.65 | 2024-10-21 09:15:00 | 999.59 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-16 11:15:00 | 1104.95 | 2024-10-21 09:15:00 | 994.46 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-17 10:00:00 | 1105.40 | 2024-10-21 09:15:00 | 994.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-01 18:00:00 | 1008.65 | 2024-11-04 09:15:00 | 983.40 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2024-11-11 09:45:00 | 990.00 | 2024-11-11 10:15:00 | 995.95 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-11-11 12:30:00 | 988.35 | 2024-11-14 11:15:00 | 938.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 12:30:00 | 988.35 | 2024-11-18 12:15:00 | 933.80 | STOP_HIT | 0.50 | 5.52% |
| BUY | retest2 | 2024-11-28 09:15:00 | 968.90 | 2024-11-28 11:15:00 | 952.80 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-12-12 10:15:00 | 919.30 | 2024-12-24 11:15:00 | 909.00 | STOP_HIT | 1.00 | 1.12% |
| SELL | retest2 | 2024-12-12 11:15:00 | 919.05 | 2024-12-24 11:15:00 | 909.00 | STOP_HIT | 1.00 | 1.09% |
| SELL | retest2 | 2024-12-12 12:00:00 | 919.00 | 2024-12-24 11:15:00 | 909.00 | STOP_HIT | 1.00 | 1.09% |
| SELL | retest2 | 2024-12-12 12:45:00 | 918.35 | 2024-12-24 11:15:00 | 909.00 | STOP_HIT | 1.00 | 1.02% |
| SELL | retest2 | 2024-12-16 11:45:00 | 918.20 | 2024-12-24 11:15:00 | 909.00 | STOP_HIT | 1.00 | 1.00% |
| SELL | retest2 | 2024-12-16 13:00:00 | 918.60 | 2024-12-24 11:15:00 | 909.00 | STOP_HIT | 1.00 | 1.05% |
| SELL | retest2 | 2024-12-16 15:15:00 | 920.00 | 2024-12-24 11:15:00 | 909.00 | STOP_HIT | 1.00 | 1.20% |
| BUY | retest2 | 2024-12-30 11:45:00 | 911.50 | 2024-12-30 14:15:00 | 899.45 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-01-06 12:30:00 | 945.50 | 2025-01-14 15:15:00 | 963.00 | STOP_HIT | 1.00 | 1.85% |
| BUY | retest2 | 2025-01-23 09:30:00 | 976.50 | 2025-01-27 13:15:00 | 967.25 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-01-30 11:30:00 | 965.00 | 2025-01-31 09:15:00 | 1006.35 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest2 | 2025-01-30 13:15:00 | 965.00 | 2025-01-31 09:15:00 | 1006.35 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest2 | 2025-01-30 14:45:00 | 965.05 | 2025-01-31 09:15:00 | 1006.35 | STOP_HIT | 1.00 | -4.28% |
| BUY | retest2 | 2025-02-04 11:30:00 | 1034.40 | 2025-02-05 09:15:00 | 1016.80 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-02-04 13:30:00 | 1030.05 | 2025-02-05 09:15:00 | 1016.80 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-02-04 15:00:00 | 1033.40 | 2025-02-05 09:15:00 | 1016.80 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-02-24 15:15:00 | 1002.00 | 2025-03-03 13:15:00 | 951.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 09:45:00 | 1001.05 | 2025-03-03 13:15:00 | 951.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 12:30:00 | 1002.00 | 2025-03-03 13:15:00 | 951.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 13:00:00 | 1001.60 | 2025-03-03 13:15:00 | 951.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-28 09:15:00 | 1000.05 | 2025-03-03 13:15:00 | 950.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-24 15:15:00 | 1002.00 | 2025-03-04 12:15:00 | 955.45 | STOP_HIT | 0.50 | 4.65% |
| SELL | retest2 | 2025-02-27 09:45:00 | 1001.05 | 2025-03-04 12:15:00 | 955.45 | STOP_HIT | 0.50 | 4.56% |
| SELL | retest2 | 2025-02-27 12:30:00 | 1002.00 | 2025-03-04 12:15:00 | 955.45 | STOP_HIT | 0.50 | 4.65% |
| SELL | retest2 | 2025-02-27 13:00:00 | 1001.60 | 2025-03-04 12:15:00 | 955.45 | STOP_HIT | 0.50 | 4.61% |
| SELL | retest2 | 2025-02-28 09:15:00 | 1000.05 | 2025-03-04 12:15:00 | 955.45 | STOP_HIT | 0.50 | 4.46% |
| SELL | retest2 | 2025-03-12 09:15:00 | 954.80 | 2025-03-19 09:15:00 | 948.90 | STOP_HIT | 1.00 | 0.62% |
| BUY | retest2 | 2025-03-19 15:15:00 | 952.00 | 2025-03-26 11:15:00 | 962.00 | STOP_HIT | 1.00 | 1.05% |
| BUY | retest2 | 2025-04-02 09:15:00 | 1044.85 | 2025-04-07 12:15:00 | 1039.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-04-17 11:15:00 | 1112.10 | 2025-05-02 15:15:00 | 1155.00 | STOP_HIT | 1.00 | 3.86% |
| SELL | retest2 | 2025-05-22 09:15:00 | 1127.00 | 2025-05-23 09:15:00 | 1146.00 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-06-17 10:15:00 | 1072.10 | 2025-06-19 11:15:00 | 1087.70 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-06-18 09:45:00 | 1071.10 | 2025-06-19 11:15:00 | 1087.70 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-07-03 12:15:00 | 1091.50 | 2025-07-07 11:15:00 | 1101.80 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-07-03 12:45:00 | 1091.80 | 2025-07-07 11:15:00 | 1101.80 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-07-03 14:30:00 | 1088.80 | 2025-07-07 11:15:00 | 1101.80 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-07-04 09:30:00 | 1091.00 | 2025-07-07 11:15:00 | 1101.80 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-07-09 11:15:00 | 1105.50 | 2025-07-09 15:15:00 | 1095.60 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-07-09 12:00:00 | 1105.80 | 2025-07-09 15:15:00 | 1095.60 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-07-11 12:15:00 | 1088.00 | 2025-07-16 14:15:00 | 1082.00 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest2 | 2025-07-11 12:45:00 | 1086.10 | 2025-07-16 14:15:00 | 1082.00 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2025-07-25 09:45:00 | 1065.20 | 2025-07-30 09:15:00 | 1076.20 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-07-28 12:30:00 | 1065.20 | 2025-07-30 09:15:00 | 1076.20 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-07-29 10:00:00 | 1065.20 | 2025-07-30 09:15:00 | 1076.20 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-07-31 10:15:00 | 1071.00 | 2025-08-04 09:15:00 | 1067.50 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-07-31 11:00:00 | 1073.10 | 2025-08-04 09:15:00 | 1067.50 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-08-01 15:15:00 | 1070.70 | 2025-08-04 09:15:00 | 1067.50 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-08-06 14:15:00 | 1056.70 | 2025-08-11 13:15:00 | 1060.30 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-08-08 14:15:00 | 1056.10 | 2025-08-11 13:15:00 | 1060.30 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-08-11 11:30:00 | 1055.70 | 2025-08-11 13:15:00 | 1060.30 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-08-11 13:00:00 | 1057.40 | 2025-08-11 13:15:00 | 1060.30 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-08-22 13:00:00 | 1088.10 | 2025-08-25 13:15:00 | 1084.10 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-08-22 13:45:00 | 1089.00 | 2025-08-25 13:15:00 | 1084.10 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-08-25 10:00:00 | 1087.30 | 2025-08-25 13:15:00 | 1084.10 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-08-25 10:45:00 | 1087.90 | 2025-08-25 13:15:00 | 1084.10 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-09-08 15:00:00 | 1073.20 | 2025-09-09 13:15:00 | 1080.10 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-09-09 09:15:00 | 1071.50 | 2025-09-09 13:15:00 | 1080.10 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-09-12 15:15:00 | 1105.00 | 2025-09-15 09:15:00 | 1096.10 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-09-15 11:45:00 | 1104.10 | 2025-09-16 10:15:00 | 1096.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-09-19 14:30:00 | 1127.50 | 2025-09-23 09:15:00 | 1121.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-09-19 15:15:00 | 1129.10 | 2025-09-23 09:15:00 | 1121.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-09-22 09:30:00 | 1128.70 | 2025-09-23 09:15:00 | 1121.00 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-09-23 10:45:00 | 1127.80 | 2025-09-26 11:15:00 | 1126.30 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-09-24 11:15:00 | 1133.00 | 2025-09-26 11:15:00 | 1126.30 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-09-24 12:15:00 | 1133.90 | 2025-09-26 11:15:00 | 1126.30 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-09-25 12:45:00 | 1133.30 | 2025-09-26 11:15:00 | 1126.30 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-09-25 14:30:00 | 1133.00 | 2025-09-26 11:15:00 | 1126.30 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-10-03 15:15:00 | 1138.50 | 2025-10-07 11:15:00 | 1129.50 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-10-06 09:30:00 | 1138.50 | 2025-10-07 11:15:00 | 1129.50 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-10-06 10:15:00 | 1138.20 | 2025-10-07 11:15:00 | 1129.50 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-10-06 11:00:00 | 1140.50 | 2025-10-07 11:15:00 | 1129.50 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-10-15 11:30:00 | 1114.60 | 2025-10-16 11:15:00 | 1122.40 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-10-15 15:15:00 | 1112.50 | 2025-10-16 11:15:00 | 1122.40 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-10-16 10:45:00 | 1111.80 | 2025-10-16 11:15:00 | 1122.40 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-10-28 15:00:00 | 1169.90 | 2025-10-31 12:15:00 | 1164.20 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-10-29 09:15:00 | 1179.30 | 2025-10-31 12:15:00 | 1164.20 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-10-30 12:00:00 | 1170.20 | 2025-10-31 12:15:00 | 1164.20 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-10-30 13:45:00 | 1170.50 | 2025-10-31 12:15:00 | 1164.20 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-11-12 10:00:00 | 1154.00 | 2025-11-12 10:15:00 | 1161.80 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-11-26 10:30:00 | 1180.50 | 2025-11-27 15:15:00 | 1175.50 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-12-10 09:15:00 | 1143.00 | 2025-12-10 09:15:00 | 1155.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-12-10 10:45:00 | 1142.80 | 2025-12-11 10:15:00 | 1149.50 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-12-10 11:15:00 | 1142.50 | 2025-12-11 10:15:00 | 1149.50 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-12-10 11:45:00 | 1142.40 | 2025-12-11 10:15:00 | 1149.50 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-12-16 10:30:00 | 1173.30 | 2025-12-24 13:15:00 | 1177.00 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2025-12-17 09:15:00 | 1173.10 | 2025-12-24 13:15:00 | 1177.00 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2025-12-19 09:15:00 | 1176.20 | 2025-12-24 13:15:00 | 1177.00 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2025-12-19 11:15:00 | 1177.20 | 2025-12-24 13:15:00 | 1177.00 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2025-12-19 13:45:00 | 1184.10 | 2025-12-24 13:15:00 | 1177.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-12-23 10:45:00 | 1183.60 | 2025-12-24 13:15:00 | 1177.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-12-23 13:45:00 | 1182.70 | 2025-12-24 13:15:00 | 1177.00 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-12-24 09:45:00 | 1184.50 | 2025-12-24 13:15:00 | 1177.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-12-30 13:15:00 | 1184.60 | 2025-12-30 15:15:00 | 1173.80 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-01-08 14:45:00 | 1203.00 | 2026-01-09 11:15:00 | 1190.80 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2026-01-09 09:45:00 | 1203.30 | 2026-01-09 11:15:00 | 1190.80 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2026-01-13 13:15:00 | 1186.50 | 2026-01-16 09:15:00 | 1192.40 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2026-01-13 13:45:00 | 1185.50 | 2026-01-16 12:15:00 | 1189.80 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2026-01-13 14:45:00 | 1187.30 | 2026-01-16 12:15:00 | 1189.80 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2026-01-14 09:15:00 | 1184.50 | 2026-01-16 12:15:00 | 1189.80 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2026-01-14 12:15:00 | 1175.70 | 2026-01-16 12:15:00 | 1189.80 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-01-23 11:00:00 | 1171.40 | 2026-01-27 14:15:00 | 1182.50 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-01-23 13:15:00 | 1171.20 | 2026-01-27 14:15:00 | 1182.50 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2026-01-23 13:45:00 | 1162.20 | 2026-01-27 14:15:00 | 1182.50 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2026-01-27 13:45:00 | 1169.90 | 2026-01-27 14:15:00 | 1182.50 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-02-01 12:15:00 | 1103.40 | 2026-02-03 10:15:00 | 1149.00 | STOP_HIT | 1.00 | -4.13% |
| SELL | retest2 | 2026-02-01 13:15:00 | 1124.90 | 2026-02-03 10:15:00 | 1149.00 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2026-02-01 14:30:00 | 1120.10 | 2026-02-03 10:15:00 | 1149.00 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2026-02-02 13:45:00 | 1123.00 | 2026-02-03 10:15:00 | 1149.00 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2026-02-05 13:45:00 | 1154.60 | 2026-02-10 14:15:00 | 1151.90 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2026-02-06 09:45:00 | 1149.80 | 2026-02-10 14:15:00 | 1151.90 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2026-02-20 09:30:00 | 1161.10 | 2026-02-26 11:15:00 | 1159.00 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2026-02-20 10:15:00 | 1165.10 | 2026-02-26 11:15:00 | 1159.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2026-02-20 13:45:00 | 1160.60 | 2026-02-26 11:15:00 | 1159.00 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2026-02-20 14:30:00 | 1159.80 | 2026-02-26 11:15:00 | 1159.00 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2026-02-23 09:15:00 | 1166.60 | 2026-02-26 11:15:00 | 1159.00 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1094.00 | 2026-03-16 13:15:00 | 1090.30 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2026-04-13 11:45:00 | 1085.70 | 2026-04-22 12:15:00 | 1194.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-16 10:00:00 | 1090.30 | 2026-04-27 14:15:00 | 1161.80 | STOP_HIT | 1.00 | 6.56% |
