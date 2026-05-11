# Bharat Forge Ltd. (BHARATFORG)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 1984.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 205 |
| ALERT1 | 144 |
| ALERT2 | 144 |
| ALERT2_SKIP | 70 |
| ALERT3 | 434 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 198 |
| PARTIAL | 14 |
| TARGET_HIT | 16 |
| STOP_HIT | 189 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 219 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 70 / 149
- **Target hits / Stop hits / Partials:** 16 / 189 / 14
- **Avg / median % per leg:** 0.32% / -0.86%
- **Sum % (uncompounded):** 69.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 96 | 23 | 24.0% | 10 | 86 | 0 | 0.16% | 15.8% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.67% | -2.7% |
| BUY @ 3rd Alert (retest2) | 92 | 23 | 25.0% | 10 | 82 | 0 | 0.20% | 18.5% |
| SELL (all) | 123 | 47 | 38.2% | 6 | 103 | 14 | 0.43% | 53.2% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.35% | -7.1% |
| SELL @ 3rd Alert (retest2) | 120 | 47 | 39.2% | 6 | 100 | 14 | 0.50% | 60.3% |
| retest1 (combined) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.39% | -9.7% |
| retest2 (combined) | 212 | 70 | 33.0% | 16 | 182 | 14 | 0.37% | 78.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 14:15:00 | 770.45 | 769.12 | 768.95 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 15:15:00 | 767.65 | 768.83 | 768.83 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 09:15:00 | 773.10 | 769.68 | 769.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-17 14:15:00 | 778.30 | 771.23 | 770.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-18 11:15:00 | 774.30 | 774.53 | 772.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-18 12:00:00 | 774.30 | 774.53 | 772.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 12:15:00 | 771.00 | 773.82 | 772.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-18 12:30:00 | 771.45 | 773.82 | 772.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 13:15:00 | 768.95 | 772.85 | 771.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-18 14:00:00 | 768.95 | 772.85 | 771.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 14:15:00 | 765.40 | 771.36 | 771.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-18 15:00:00 | 765.40 | 771.36 | 771.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2023-05-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 15:15:00 | 764.00 | 769.89 | 770.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-19 09:15:00 | 758.90 | 767.69 | 769.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-22 10:15:00 | 761.55 | 760.60 | 763.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-22 11:00:00 | 761.55 | 760.60 | 763.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 12:15:00 | 761.50 | 760.87 | 763.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 13:15:00 | 758.75 | 760.87 | 763.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 15:15:00 | 760.25 | 760.69 | 762.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-23 10:15:00 | 764.35 | 761.31 | 762.61 | SL hit (close>static) qty=1.00 sl=763.50 alert=retest2 |

### Cycle 5 — BUY (started 2023-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 12:15:00 | 770.00 | 764.44 | 763.89 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2023-05-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 13:15:00 | 763.40 | 764.45 | 764.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 15:15:00 | 761.70 | 763.65 | 764.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-26 09:15:00 | 763.95 | 759.53 | 761.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-26 09:15:00 | 763.95 | 759.53 | 761.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 763.95 | 759.53 | 761.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 10:00:00 | 763.95 | 759.53 | 761.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 10:15:00 | 762.80 | 760.18 | 761.19 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 14:15:00 | 768.00 | 762.91 | 762.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-29 09:15:00 | 774.25 | 765.94 | 763.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 13:15:00 | 782.50 | 783.83 | 777.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-30 14:00:00 | 782.50 | 783.83 | 777.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 787.05 | 783.93 | 779.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 10:15:00 | 788.85 | 783.93 | 779.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 11:00:00 | 788.35 | 792.57 | 789.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 11:30:00 | 788.85 | 791.86 | 789.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-20 09:15:00 | 818.45 | 830.88 | 831.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2023-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 09:15:00 | 818.45 | 830.88 | 831.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-21 10:15:00 | 810.60 | 818.49 | 823.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-22 09:15:00 | 819.80 | 816.62 | 820.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 09:15:00 | 819.80 | 816.62 | 820.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 09:15:00 | 819.80 | 816.62 | 820.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-22 09:30:00 | 818.50 | 816.62 | 820.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 10:15:00 | 825.00 | 818.30 | 820.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-22 11:00:00 | 825.00 | 818.30 | 820.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 11:15:00 | 819.40 | 818.52 | 820.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-22 12:15:00 | 815.00 | 818.52 | 820.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-22 14:00:00 | 816.85 | 817.89 | 819.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-27 09:15:00 | 819.50 | 813.90 | 813.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2023-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 09:15:00 | 819.50 | 813.90 | 813.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 09:15:00 | 832.35 | 820.57 | 817.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 09:15:00 | 837.35 | 838.01 | 832.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-04 09:45:00 | 836.60 | 838.01 | 832.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 14:15:00 | 836.35 | 837.85 | 834.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 14:45:00 | 835.10 | 837.85 | 834.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 837.35 | 837.53 | 835.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-05 10:45:00 | 843.85 | 839.67 | 836.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-07 14:15:00 | 842.30 | 846.57 | 846.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2023-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 14:15:00 | 842.30 | 846.57 | 846.69 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 12:15:00 | 849.00 | 846.67 | 846.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-10 13:15:00 | 853.90 | 848.12 | 847.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-11 13:15:00 | 853.40 | 854.24 | 851.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-11 13:45:00 | 854.75 | 854.24 | 851.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 14:15:00 | 854.05 | 854.20 | 851.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-11 15:00:00 | 854.05 | 854.20 | 851.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 15:15:00 | 854.00 | 854.16 | 851.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-12 09:15:00 | 857.40 | 854.16 | 851.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-12 12:30:00 | 856.40 | 856.10 | 853.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-12 14:15:00 | 854.65 | 855.68 | 853.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-12 15:00:00 | 855.20 | 855.58 | 853.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 15:15:00 | 854.00 | 855.27 | 853.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 09:15:00 | 858.00 | 855.27 | 853.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 11:00:00 | 856.40 | 855.23 | 854.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 11:45:00 | 855.75 | 855.27 | 854.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 12:45:00 | 856.00 | 855.63 | 854.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 13:15:00 | 851.90 | 854.88 | 854.22 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-07-13 13:15:00 | 851.90 | 854.88 | 854.22 | SL hit (close<static) qty=1.00 sl=852.20 alert=retest2 |

### Cycle 12 — SELL (started 2023-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-19 11:15:00 | 857.60 | 864.34 | 864.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-19 12:15:00 | 853.95 | 862.26 | 863.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-21 11:15:00 | 851.60 | 850.30 | 854.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-21 12:00:00 | 851.60 | 850.30 | 854.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 14:15:00 | 852.55 | 851.08 | 853.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-21 14:30:00 | 852.80 | 851.08 | 853.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 14:15:00 | 852.10 | 848.65 | 850.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 15:00:00 | 852.10 | 848.65 | 850.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 15:15:00 | 854.50 | 849.82 | 851.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-25 09:15:00 | 861.30 | 849.82 | 851.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2023-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 10:15:00 | 860.00 | 853.20 | 852.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 11:15:00 | 865.90 | 855.74 | 853.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 09:15:00 | 927.30 | 930.36 | 920.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 10:15:00 | 916.35 | 927.56 | 920.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 916.35 | 927.56 | 920.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:00:00 | 916.35 | 927.56 | 920.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 910.00 | 924.05 | 919.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:45:00 | 910.00 | 924.05 | 919.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2023-08-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 14:15:00 | 906.55 | 914.97 | 915.85 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 13:15:00 | 914.75 | 911.74 | 911.71 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-07 12:15:00 | 909.95 | 911.70 | 911.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-08 10:15:00 | 904.60 | 909.00 | 910.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-09 12:15:00 | 918.05 | 903.67 | 905.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 12:15:00 | 918.05 | 903.67 | 905.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 12:15:00 | 918.05 | 903.67 | 905.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 12:45:00 | 923.60 | 903.67 | 905.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2023-08-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 13:15:00 | 945.55 | 912.04 | 909.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-09 14:15:00 | 965.15 | 922.67 | 914.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-14 09:15:00 | 954.90 | 964.00 | 953.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-14 09:15:00 | 954.90 | 964.00 | 953.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 09:15:00 | 954.90 | 964.00 | 953.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-17 09:15:00 | 970.25 | 957.25 | 954.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-18 11:15:00 | 954.85 | 955.65 | 955.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2023-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 11:15:00 | 954.85 | 955.65 | 955.65 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 13:15:00 | 961.80 | 956.82 | 956.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-18 14:15:00 | 971.25 | 959.71 | 957.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-25 09:15:00 | 1037.60 | 1038.30 | 1025.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-25 10:15:00 | 1032.15 | 1038.30 | 1025.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 12:15:00 | 1026.00 | 1033.64 | 1026.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 13:00:00 | 1026.00 | 1033.64 | 1026.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 13:15:00 | 1026.75 | 1032.26 | 1026.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 14:15:00 | 1024.45 | 1032.26 | 1026.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 14:15:00 | 1022.70 | 1030.35 | 1026.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 15:00:00 | 1022.70 | 1030.35 | 1026.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 15:15:00 | 1021.90 | 1028.66 | 1025.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-28 09:15:00 | 1036.15 | 1028.66 | 1025.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 11:15:00 | 1078.00 | 1099.81 | 1102.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 1078.00 | 1099.81 | 1102.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 12:15:00 | 1076.05 | 1095.06 | 1099.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 14:15:00 | 1080.15 | 1079.07 | 1086.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 14:30:00 | 1077.30 | 1079.07 | 1086.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 1103.15 | 1084.50 | 1087.50 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-09-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 11:15:00 | 1104.80 | 1090.43 | 1089.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 14:15:00 | 1107.95 | 1098.01 | 1093.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 12:15:00 | 1123.45 | 1124.41 | 1115.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-18 13:00:00 | 1123.45 | 1124.41 | 1115.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 1117.55 | 1122.15 | 1116.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-20 10:30:00 | 1129.10 | 1123.52 | 1117.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-20 11:45:00 | 1131.40 | 1125.12 | 1119.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-20 14:00:00 | 1130.25 | 1127.24 | 1121.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-21 09:15:00 | 1136.75 | 1127.64 | 1122.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 11:15:00 | 1123.65 | 1128.85 | 1124.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 12:00:00 | 1123.65 | 1128.85 | 1124.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 12:15:00 | 1098.75 | 1122.83 | 1122.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-09-21 12:15:00 | 1098.75 | 1122.83 | 1122.18 | SL hit (close<static) qty=1.00 sl=1108.65 alert=retest2 |

### Cycle 22 — SELL (started 2023-09-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 13:15:00 | 1098.20 | 1117.90 | 1120.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 09:15:00 | 1093.05 | 1107.72 | 1114.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 12:15:00 | 1100.75 | 1092.68 | 1099.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 12:15:00 | 1100.75 | 1092.68 | 1099.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 12:15:00 | 1100.75 | 1092.68 | 1099.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 13:00:00 | 1100.75 | 1092.68 | 1099.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 13:15:00 | 1095.00 | 1093.15 | 1099.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-26 13:15:00 | 1093.10 | 1100.81 | 1101.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 15:00:00 | 1091.15 | 1088.88 | 1093.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-29 10:00:00 | 1092.25 | 1086.41 | 1088.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-29 13:30:00 | 1093.05 | 1090.04 | 1090.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-29 14:15:00 | 1091.00 | 1090.23 | 1090.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2023-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 14:15:00 | 1091.00 | 1090.23 | 1090.15 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2023-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 09:15:00 | 1073.00 | 1087.15 | 1088.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 09:15:00 | 1072.40 | 1081.28 | 1084.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 1068.80 | 1067.26 | 1074.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 09:15:00 | 1068.80 | 1067.26 | 1074.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 1068.80 | 1067.26 | 1074.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 10:15:00 | 1063.50 | 1067.26 | 1074.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-06 09:15:00 | 1087.95 | 1072.84 | 1073.74 | SL hit (close>static) qty=1.00 sl=1077.05 alert=retest2 |

### Cycle 25 — BUY (started 2023-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 10:15:00 | 1087.50 | 1075.77 | 1074.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 09:15:00 | 1107.35 | 1088.14 | 1083.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 15:15:00 | 1112.55 | 1112.87 | 1105.21 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 09:15:00 | 1116.95 | 1112.87 | 1105.21 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 12:30:00 | 1116.15 | 1116.15 | 1109.43 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 1119.50 | 1117.58 | 1112.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 10:15:00 | 1120.95 | 1117.58 | 1112.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 10:45:00 | 1121.90 | 1118.66 | 1113.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 13:00:00 | 1120.90 | 1118.74 | 1114.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-13 15:15:00 | 1113.75 | 1118.46 | 1115.33 | SL hit (close<ema400) qty=1.00 sl=1115.33 alert=retest1 |

### Cycle 26 — SELL (started 2023-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 11:15:00 | 1113.10 | 1118.73 | 1118.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 14:15:00 | 1106.00 | 1114.11 | 1116.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 1033.75 | 1023.02 | 1034.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 1033.75 | 1023.02 | 1034.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 1033.75 | 1023.02 | 1034.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 10:00:00 | 1033.75 | 1023.02 | 1034.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 1031.70 | 1024.75 | 1034.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-27 11:30:00 | 1028.50 | 1025.87 | 1033.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-27 12:15:00 | 1028.90 | 1025.87 | 1033.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-27 13:30:00 | 1027.70 | 1026.83 | 1032.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-27 14:30:00 | 1028.50 | 1026.09 | 1032.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 11:15:00 | 1030.00 | 1025.74 | 1029.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-30 11:45:00 | 1030.40 | 1025.74 | 1029.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 12:15:00 | 1028.95 | 1026.38 | 1029.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-30 12:30:00 | 1030.10 | 1026.38 | 1029.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 13:15:00 | 1029.15 | 1026.93 | 1029.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-30 13:45:00 | 1029.00 | 1026.93 | 1029.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 14:15:00 | 1028.30 | 1027.21 | 1029.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-30 14:45:00 | 1027.35 | 1027.21 | 1029.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 15:15:00 | 1027.35 | 1027.24 | 1029.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-31 09:15:00 | 1034.40 | 1027.24 | 1029.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 09:15:00 | 1029.50 | 1027.69 | 1029.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-31 11:00:00 | 1026.30 | 1027.41 | 1029.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-01 12:30:00 | 1024.90 | 1023.52 | 1025.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-01 15:00:00 | 1027.15 | 1024.42 | 1025.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-02 10:00:00 | 1027.55 | 1025.05 | 1025.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-02 10:15:00 | 1031.55 | 1026.35 | 1026.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2023-11-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 10:15:00 | 1031.55 | 1026.35 | 1026.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 13:15:00 | 1033.70 | 1027.74 | 1026.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 11:15:00 | 1032.40 | 1033.07 | 1030.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-03 12:00:00 | 1032.40 | 1033.07 | 1030.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 12:15:00 | 1031.00 | 1032.66 | 1030.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-03 12:30:00 | 1030.05 | 1032.66 | 1030.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 13:15:00 | 1031.15 | 1032.36 | 1030.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-03 13:45:00 | 1031.25 | 1032.36 | 1030.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 14:15:00 | 1031.00 | 1032.09 | 1030.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-03 14:45:00 | 1029.60 | 1032.09 | 1030.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 15:15:00 | 1030.00 | 1031.67 | 1030.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 09:15:00 | 1036.55 | 1031.67 | 1030.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 12:15:00 | 1042.60 | 1033.51 | 1031.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-08 12:15:00 | 1032.90 | 1046.48 | 1047.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2023-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 12:15:00 | 1032.90 | 1046.48 | 1047.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-08 13:15:00 | 1028.35 | 1042.85 | 1045.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-09 10:15:00 | 1045.65 | 1040.75 | 1043.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 10:15:00 | 1045.65 | 1040.75 | 1043.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 10:15:00 | 1045.65 | 1040.75 | 1043.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-09 11:00:00 | 1045.65 | 1040.75 | 1043.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 11:15:00 | 1044.00 | 1041.40 | 1043.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-09 11:30:00 | 1044.95 | 1041.40 | 1043.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 12:15:00 | 1035.60 | 1040.24 | 1042.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-09 14:15:00 | 1034.00 | 1039.26 | 1042.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-09 14:45:00 | 1031.65 | 1038.35 | 1041.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-10 09:15:00 | 1025.50 | 1037.54 | 1040.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-12 18:15:00 | 1045.35 | 1039.16 | 1039.44 | SL hit (close>static) qty=1.00 sl=1044.00 alert=retest2 |

### Cycle 29 — BUY (started 2023-11-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 11:15:00 | 1044.25 | 1038.54 | 1038.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 12:15:00 | 1047.20 | 1040.27 | 1039.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 12:15:00 | 1051.75 | 1053.13 | 1047.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-16 13:00:00 | 1051.75 | 1053.13 | 1047.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 15:15:00 | 1090.00 | 1097.02 | 1089.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-24 09:15:00 | 1101.40 | 1097.02 | 1089.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-29 09:15:00 | 1109.10 | 1104.70 | 1103.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-12-18 13:15:00 | 1211.54 | 1200.35 | 1196.81 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 1177.90 | 1205.34 | 1206.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 15:15:00 | 1172.55 | 1198.78 | 1203.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 1198.70 | 1188.02 | 1193.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 09:15:00 | 1198.70 | 1188.02 | 1193.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 1198.70 | 1188.02 | 1193.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 10:00:00 | 1198.70 | 1188.02 | 1193.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 1199.20 | 1190.26 | 1193.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 11:00:00 | 1199.20 | 1190.26 | 1193.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 12:15:00 | 1191.30 | 1190.89 | 1193.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 12:30:00 | 1194.90 | 1190.89 | 1193.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 13:15:00 | 1195.45 | 1191.80 | 1193.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 14:00:00 | 1195.45 | 1191.80 | 1193.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 14:15:00 | 1203.35 | 1194.11 | 1194.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 15:00:00 | 1203.35 | 1194.11 | 1194.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2023-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 15:15:00 | 1201.00 | 1195.49 | 1195.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 09:15:00 | 1211.90 | 1198.77 | 1196.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 11:15:00 | 1218.50 | 1220.11 | 1212.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-27 12:00:00 | 1218.50 | 1220.11 | 1212.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 12:15:00 | 1212.65 | 1218.62 | 1212.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 12:30:00 | 1212.00 | 1218.62 | 1212.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 13:15:00 | 1215.05 | 1217.90 | 1212.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 13:30:00 | 1207.95 | 1217.90 | 1212.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 14:15:00 | 1220.60 | 1218.44 | 1213.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 15:00:00 | 1220.60 | 1218.44 | 1213.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 14:15:00 | 1238.10 | 1236.20 | 1230.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 09:30:00 | 1247.80 | 1239.36 | 1233.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-03 09:15:00 | 1219.65 | 1235.92 | 1237.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 09:15:00 | 1219.65 | 1235.92 | 1237.51 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-01-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 12:15:00 | 1251.15 | 1239.67 | 1238.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 13:15:00 | 1254.70 | 1242.68 | 1239.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 12:15:00 | 1257.60 | 1262.48 | 1256.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-08 12:15:00 | 1257.60 | 1262.48 | 1256.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 12:15:00 | 1257.60 | 1262.48 | 1256.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 13:00:00 | 1257.60 | 1262.48 | 1256.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 13:15:00 | 1255.45 | 1261.08 | 1256.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 13:30:00 | 1255.00 | 1261.08 | 1256.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 14:15:00 | 1258.80 | 1260.62 | 1256.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 14:30:00 | 1252.65 | 1260.62 | 1256.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 15:15:00 | 1249.00 | 1258.30 | 1256.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 09:15:00 | 1279.50 | 1258.30 | 1256.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 09:30:00 | 1263.90 | 1272.73 | 1267.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 11:15:00 | 1260.90 | 1269.89 | 1266.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-12 10:15:00 | 1237.65 | 1266.82 | 1270.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 10:15:00 | 1237.65 | 1266.82 | 1270.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 12:15:00 | 1231.25 | 1246.40 | 1253.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 10:15:00 | 1214.35 | 1214.09 | 1227.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-18 11:00:00 | 1214.35 | 1214.09 | 1227.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 12:15:00 | 1234.20 | 1219.65 | 1227.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 13:00:00 | 1234.20 | 1219.65 | 1227.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 13:15:00 | 1232.15 | 1222.15 | 1227.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-18 14:15:00 | 1224.45 | 1222.15 | 1227.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-18 14:45:00 | 1224.60 | 1222.46 | 1227.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-19 10:15:00 | 1238.00 | 1225.96 | 1227.83 | SL hit (close>static) qty=1.00 sl=1237.50 alert=retest2 |

### Cycle 35 — BUY (started 2024-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 12:15:00 | 1239.50 | 1230.31 | 1229.58 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 15:15:00 | 1226.00 | 1229.94 | 1230.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 11:15:00 | 1206.65 | 1224.33 | 1227.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 12:15:00 | 1200.35 | 1200.30 | 1210.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-24 13:00:00 | 1200.35 | 1200.30 | 1210.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 13:15:00 | 1212.20 | 1202.68 | 1210.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 14:00:00 | 1212.20 | 1202.68 | 1210.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 1214.20 | 1204.99 | 1210.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 15:00:00 | 1214.20 | 1204.99 | 1210.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 15:15:00 | 1220.30 | 1208.05 | 1211.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:15:00 | 1208.50 | 1208.05 | 1211.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 1205.20 | 1207.48 | 1211.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 10:15:00 | 1200.55 | 1207.48 | 1211.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-29 11:15:00 | 1243.65 | 1212.58 | 1210.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 11:15:00 | 1243.65 | 1212.58 | 1210.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-02 10:15:00 | 1248.35 | 1233.94 | 1230.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-07 11:15:00 | 1279.60 | 1282.41 | 1271.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-07 11:30:00 | 1280.45 | 1282.41 | 1271.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 09:15:00 | 1274.70 | 1281.60 | 1275.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 09:45:00 | 1273.25 | 1281.60 | 1275.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 10:15:00 | 1272.30 | 1279.74 | 1275.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 11:00:00 | 1272.30 | 1279.74 | 1275.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 11:15:00 | 1281.70 | 1280.13 | 1275.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-08 12:30:00 | 1293.00 | 1283.05 | 1277.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-08 15:15:00 | 1285.00 | 1283.19 | 1278.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 11:30:00 | 1291.00 | 1290.81 | 1283.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-12 13:15:00 | 1212.00 | 1285.45 | 1288.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 13:15:00 | 1212.00 | 1285.45 | 1288.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 14:15:00 | 1137.20 | 1255.80 | 1275.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 15:15:00 | 1108.00 | 1106.38 | 1142.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-15 09:15:00 | 1109.55 | 1106.38 | 1142.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 10:15:00 | 1129.95 | 1113.90 | 1124.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-16 11:00:00 | 1129.95 | 1113.90 | 1124.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 11:15:00 | 1122.00 | 1115.52 | 1124.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-16 11:30:00 | 1129.85 | 1115.52 | 1124.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 12:15:00 | 1127.70 | 1117.96 | 1124.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-16 13:00:00 | 1127.70 | 1117.96 | 1124.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 13:15:00 | 1126.90 | 1119.74 | 1125.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-16 13:30:00 | 1129.15 | 1119.74 | 1125.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 14:15:00 | 1134.50 | 1122.70 | 1125.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-16 14:45:00 | 1133.50 | 1122.70 | 1125.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 15:15:00 | 1134.70 | 1125.10 | 1126.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-19 09:15:00 | 1125.60 | 1125.10 | 1126.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 10:15:00 | 1124.60 | 1124.05 | 1125.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-19 10:30:00 | 1125.95 | 1124.05 | 1125.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 11:15:00 | 1124.45 | 1124.13 | 1125.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-19 12:00:00 | 1124.45 | 1124.13 | 1125.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 12:15:00 | 1123.15 | 1123.94 | 1125.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-19 12:30:00 | 1123.95 | 1123.94 | 1125.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 13:15:00 | 1130.45 | 1125.24 | 1126.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-19 14:00:00 | 1130.45 | 1125.24 | 1126.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 14:15:00 | 1125.75 | 1125.34 | 1125.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-20 09:15:00 | 1118.65 | 1125.27 | 1125.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-20 15:15:00 | 1118.60 | 1116.68 | 1120.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-21 09:15:00 | 1136.10 | 1120.87 | 1121.37 | SL hit (close>static) qty=1.00 sl=1130.55 alert=retest2 |

### Cycle 39 — BUY (started 2024-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-21 10:15:00 | 1128.60 | 1122.42 | 1122.03 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-02-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 09:15:00 | 1115.15 | 1123.17 | 1123.24 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-02-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 13:15:00 | 1128.60 | 1123.40 | 1123.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 11:15:00 | 1133.75 | 1128.57 | 1126.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-23 14:15:00 | 1133.35 | 1133.79 | 1129.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-23 15:00:00 | 1133.35 | 1133.79 | 1129.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 15:15:00 | 1130.00 | 1133.03 | 1129.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 09:15:00 | 1135.00 | 1133.03 | 1129.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 1132.00 | 1132.83 | 1129.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 09:45:00 | 1128.05 | 1132.83 | 1129.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 10:15:00 | 1140.55 | 1134.37 | 1130.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 10:30:00 | 1130.00 | 1134.37 | 1130.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 10:15:00 | 1163.00 | 1171.61 | 1161.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 10:45:00 | 1161.95 | 1171.61 | 1161.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 11:15:00 | 1157.75 | 1168.84 | 1161.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 12:00:00 | 1157.75 | 1168.84 | 1161.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 12:15:00 | 1142.70 | 1163.61 | 1159.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 13:00:00 | 1142.70 | 1163.61 | 1159.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 13:15:00 | 1153.10 | 1161.51 | 1158.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-28 15:00:00 | 1155.30 | 1160.27 | 1158.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-29 09:15:00 | 1144.75 | 1155.52 | 1156.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-02-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-29 09:15:00 | 1144.75 | 1155.52 | 1156.68 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-03-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 10:15:00 | 1167.00 | 1154.52 | 1154.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 11:15:00 | 1173.60 | 1158.33 | 1155.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 12:15:00 | 1185.35 | 1189.79 | 1182.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-05 13:00:00 | 1185.35 | 1189.79 | 1182.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 13:15:00 | 1185.95 | 1189.02 | 1182.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 13:45:00 | 1182.55 | 1189.02 | 1182.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 1166.75 | 1183.59 | 1181.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:00:00 | 1166.75 | 1183.59 | 1181.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 10:15:00 | 1161.25 | 1179.12 | 1179.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 11:15:00 | 1157.50 | 1174.80 | 1177.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 13:15:00 | 1178.00 | 1174.82 | 1177.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 13:15:00 | 1178.00 | 1174.82 | 1177.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 13:15:00 | 1178.00 | 1174.82 | 1177.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 13:45:00 | 1178.20 | 1174.82 | 1177.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 14:15:00 | 1182.95 | 1176.44 | 1177.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 15:00:00 | 1182.95 | 1176.44 | 1177.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 15:15:00 | 1183.00 | 1177.75 | 1178.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 09:15:00 | 1184.95 | 1177.75 | 1178.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2024-03-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 10:15:00 | 1185.35 | 1179.23 | 1178.74 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-03-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-07 15:15:00 | 1170.90 | 1178.66 | 1178.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 09:15:00 | 1150.00 | 1172.93 | 1176.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-11 11:15:00 | 1173.30 | 1172.61 | 1175.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 11:15:00 | 1173.30 | 1172.61 | 1175.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 11:15:00 | 1173.30 | 1172.61 | 1175.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 13:45:00 | 1168.35 | 1171.81 | 1174.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-14 09:15:00 | 1109.93 | 1130.59 | 1144.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-15 11:15:00 | 1116.95 | 1113.68 | 1125.19 | SL hit (close>ema200) qty=0.50 sl=1113.68 alert=retest2 |

### Cycle 47 — BUY (started 2024-03-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 10:15:00 | 1116.00 | 1107.06 | 1106.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 11:15:00 | 1128.00 | 1111.25 | 1108.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 10:15:00 | 1129.45 | 1130.12 | 1124.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-27 11:00:00 | 1129.45 | 1130.12 | 1124.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 13:15:00 | 1122.65 | 1128.41 | 1124.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 14:00:00 | 1122.65 | 1128.41 | 1124.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 1114.40 | 1125.61 | 1123.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 14:30:00 | 1108.70 | 1125.61 | 1123.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2024-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 09:15:00 | 1109.95 | 1120.36 | 1121.66 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 13:15:00 | 1129.80 | 1122.75 | 1122.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 09:15:00 | 1141.25 | 1128.08 | 1124.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-02 09:15:00 | 1140.30 | 1142.22 | 1135.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-02 09:15:00 | 1140.30 | 1142.22 | 1135.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 09:15:00 | 1140.30 | 1142.22 | 1135.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-04 10:30:00 | 1167.55 | 1146.65 | 1143.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-04 11:15:00 | 1169.60 | 1146.65 | 1143.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-04 14:15:00 | 1162.70 | 1154.26 | 1148.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-05 13:15:00 | 1145.00 | 1146.68 | 1146.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2024-04-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 13:15:00 | 1145.00 | 1146.68 | 1146.79 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 09:15:00 | 1151.50 | 1147.51 | 1147.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 10:15:00 | 1159.55 | 1149.92 | 1148.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-09 10:15:00 | 1161.20 | 1165.94 | 1159.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-09 11:00:00 | 1161.20 | 1165.94 | 1159.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 11:15:00 | 1156.50 | 1164.05 | 1158.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 12:00:00 | 1156.50 | 1164.05 | 1158.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 12:15:00 | 1153.00 | 1161.84 | 1158.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 13:00:00 | 1153.00 | 1161.84 | 1158.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 13:15:00 | 1145.45 | 1158.56 | 1157.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 13:45:00 | 1142.85 | 1158.56 | 1157.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2024-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 15:15:00 | 1151.25 | 1155.73 | 1156.05 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 09:15:00 | 1170.75 | 1158.73 | 1157.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 13:15:00 | 1176.35 | 1166.66 | 1161.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 12:15:00 | 1171.75 | 1175.69 | 1169.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 12:15:00 | 1171.75 | 1175.69 | 1169.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 12:15:00 | 1171.75 | 1175.69 | 1169.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 13:00:00 | 1171.75 | 1175.69 | 1169.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 13:15:00 | 1173.65 | 1175.28 | 1169.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 13:30:00 | 1170.30 | 1175.28 | 1169.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 14:15:00 | 1168.25 | 1173.87 | 1169.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 15:00:00 | 1168.25 | 1173.87 | 1169.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 15:15:00 | 1166.00 | 1172.30 | 1169.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 09:15:00 | 1163.60 | 1172.30 | 1169.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 1159.25 | 1169.69 | 1168.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-16 09:15:00 | 1178.65 | 1168.52 | 1168.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-18 10:15:00 | 1177.60 | 1179.12 | 1175.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 09:30:00 | 1178.05 | 1182.03 | 1179.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 11:30:00 | 1180.00 | 1180.42 | 1178.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 10:15:00 | 1196.25 | 1200.89 | 1195.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 10:45:00 | 1196.90 | 1200.89 | 1195.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 11:15:00 | 1204.65 | 1201.64 | 1195.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 12:15:00 | 1207.95 | 1201.64 | 1195.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-25 09:15:00 | 1296.52 | 1236.56 | 1219.54 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 14:15:00 | 1269.90 | 1283.13 | 1283.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-02 10:15:00 | 1264.65 | 1275.82 | 1279.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-02 13:15:00 | 1275.10 | 1274.88 | 1278.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-02 13:15:00 | 1275.10 | 1274.88 | 1278.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 13:15:00 | 1275.10 | 1274.88 | 1278.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 13:45:00 | 1281.15 | 1274.88 | 1278.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 15:15:00 | 1269.10 | 1273.76 | 1277.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-03 09:15:00 | 1278.00 | 1273.76 | 1277.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 09:15:00 | 1271.50 | 1273.31 | 1276.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 10:15:00 | 1267.25 | 1273.31 | 1276.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-08 13:15:00 | 1346.00 | 1256.71 | 1251.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2024-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 13:15:00 | 1346.00 | 1256.71 | 1251.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 14:15:00 | 1425.90 | 1290.55 | 1267.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-10 11:15:00 | 1400.00 | 1404.96 | 1365.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-10 12:00:00 | 1400.00 | 1404.96 | 1365.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 1385.45 | 1396.90 | 1376.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 09:30:00 | 1375.15 | 1396.90 | 1376.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 10:15:00 | 1379.90 | 1393.50 | 1376.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 10:45:00 | 1373.35 | 1393.50 | 1376.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 11:15:00 | 1385.30 | 1391.86 | 1377.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-13 13:00:00 | 1407.00 | 1394.89 | 1379.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-24 11:15:00 | 1547.70 | 1500.87 | 1490.77 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 1449.40 | 1549.72 | 1563.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 14:15:00 | 1446.30 | 1509.16 | 1539.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 13:15:00 | 1477.10 | 1474.97 | 1505.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 14:00:00 | 1477.10 | 1474.97 | 1505.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 1499.00 | 1479.77 | 1504.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 15:00:00 | 1499.00 | 1479.77 | 1504.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 1511.00 | 1486.02 | 1505.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 1520.10 | 1486.02 | 1505.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1540.05 | 1496.83 | 1508.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 1540.05 | 1496.83 | 1508.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 1569.55 | 1511.37 | 1513.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:00:00 | 1569.55 | 1511.37 | 1513.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 11:15:00 | 1548.10 | 1518.72 | 1516.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 13:15:00 | 1584.90 | 1565.01 | 1546.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 11:15:00 | 1574.45 | 1575.58 | 1559.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 11:45:00 | 1573.60 | 1575.58 | 1559.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 1584.20 | 1575.32 | 1563.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:30:00 | 1564.50 | 1575.32 | 1563.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 1756.70 | 1757.14 | 1743.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 10:30:00 | 1769.10 | 1760.69 | 1746.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 11:15:00 | 1733.25 | 1748.22 | 1746.48 | SL hit (close<static) qty=1.00 sl=1734.00 alert=retest2 |

### Cycle 58 — SELL (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 12:15:00 | 1729.40 | 1744.46 | 1744.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 13:15:00 | 1718.75 | 1739.31 | 1742.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 09:15:00 | 1677.45 | 1677.13 | 1687.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 1677.45 | 1677.13 | 1687.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 1677.45 | 1677.13 | 1687.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:30:00 | 1692.00 | 1677.13 | 1687.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 1689.50 | 1679.61 | 1687.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:30:00 | 1688.15 | 1679.61 | 1687.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 11:15:00 | 1680.50 | 1679.78 | 1686.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 13:00:00 | 1672.75 | 1678.38 | 1685.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 14:15:00 | 1668.45 | 1661.68 | 1661.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2024-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 14:15:00 | 1668.45 | 1661.68 | 1661.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 15:15:00 | 1671.90 | 1663.73 | 1662.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 14:15:00 | 1671.00 | 1671.46 | 1667.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-05 14:30:00 | 1672.65 | 1671.46 | 1667.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 15:15:00 | 1674.00 | 1671.97 | 1668.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 09:15:00 | 1688.75 | 1671.97 | 1668.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 10:00:00 | 1677.85 | 1673.15 | 1669.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 10:15:00 | 1660.60 | 1670.64 | 1668.38 | SL hit (close<static) qty=1.00 sl=1667.45 alert=retest2 |

### Cycle 60 — SELL (started 2024-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 12:15:00 | 1663.60 | 1666.94 | 1666.95 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 15:15:00 | 1675.10 | 1668.30 | 1667.47 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 1637.10 | 1662.85 | 1665.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 10:15:00 | 1621.90 | 1654.66 | 1661.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 11:15:00 | 1655.80 | 1654.89 | 1661.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-10 12:00:00 | 1655.80 | 1654.89 | 1661.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 12:15:00 | 1660.65 | 1656.04 | 1661.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 12:45:00 | 1661.70 | 1656.04 | 1661.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 13:15:00 | 1655.00 | 1655.83 | 1660.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 13:30:00 | 1668.45 | 1655.83 | 1660.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 1661.00 | 1656.87 | 1660.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 15:00:00 | 1661.00 | 1656.87 | 1660.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 1662.00 | 1657.89 | 1660.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 1660.25 | 1657.89 | 1660.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 1657.55 | 1657.83 | 1660.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 14:00:00 | 1653.30 | 1656.43 | 1658.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 10:15:00 | 1655.00 | 1636.09 | 1635.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2024-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 10:15:00 | 1655.00 | 1636.09 | 1635.32 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 1616.55 | 1639.46 | 1639.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 1604.45 | 1622.05 | 1629.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 10:15:00 | 1605.75 | 1596.91 | 1609.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 10:15:00 | 1605.75 | 1596.91 | 1609.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 1605.75 | 1596.91 | 1609.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 1603.00 | 1596.91 | 1609.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 1605.40 | 1598.61 | 1609.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:30:00 | 1612.10 | 1598.61 | 1609.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 1614.40 | 1601.77 | 1609.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:00:00 | 1614.40 | 1601.77 | 1609.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 1626.25 | 1606.66 | 1611.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 1627.15 | 1606.66 | 1611.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 09:15:00 | 1622.25 | 1614.19 | 1613.84 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 12:15:00 | 1590.75 | 1612.34 | 1613.38 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 1661.15 | 1604.86 | 1601.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 10:15:00 | 1681.35 | 1620.16 | 1608.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 15:15:00 | 1736.10 | 1737.04 | 1712.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 09:15:00 | 1733.70 | 1737.04 | 1712.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 1736.00 | 1736.04 | 1725.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:45:00 | 1728.80 | 1736.04 | 1725.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 1731.95 | 1735.23 | 1726.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:45:00 | 1729.00 | 1735.23 | 1726.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 1712.80 | 1730.74 | 1725.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 13:00:00 | 1712.80 | 1730.74 | 1725.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 1715.45 | 1727.68 | 1724.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 15:00:00 | 1720.90 | 1726.33 | 1723.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 09:15:00 | 1678.35 | 1714.73 | 1718.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 1678.35 | 1714.73 | 1718.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 12:15:00 | 1668.50 | 1698.28 | 1709.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 1559.90 | 1544.63 | 1581.25 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-07 11:30:00 | 1541.00 | 1543.99 | 1574.73 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 1544.00 | 1546.59 | 1564.47 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-08 12:15:00 | 1623.00 | 1563.43 | 1567.82 | SL hit (close>ema400) qty=1.00 sl=1567.82 alert=retest1 |

### Cycle 69 — BUY (started 2024-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 14:15:00 | 1606.25 | 1571.90 | 1570.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 09:15:00 | 1640.00 | 1591.11 | 1580.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 1590.70 | 1617.94 | 1603.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 09:15:00 | 1590.70 | 1617.94 | 1603.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 1590.70 | 1617.94 | 1603.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:00:00 | 1590.70 | 1617.94 | 1603.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 1583.70 | 1611.10 | 1601.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:30:00 | 1592.45 | 1611.10 | 1601.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 1591.40 | 1599.39 | 1598.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:30:00 | 1590.70 | 1599.39 | 1598.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2024-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 15:15:00 | 1590.00 | 1597.51 | 1597.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 1583.55 | 1593.77 | 1595.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 1584.00 | 1575.25 | 1581.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 1584.00 | 1575.25 | 1581.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 1584.00 | 1575.25 | 1581.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 13:45:00 | 1579.75 | 1581.17 | 1582.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 10:00:00 | 1578.70 | 1581.78 | 1582.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 12:00:00 | 1576.65 | 1577.65 | 1580.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 15:15:00 | 1593.20 | 1582.17 | 1581.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2024-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 15:15:00 | 1593.20 | 1582.17 | 1581.64 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 09:15:00 | 1565.95 | 1578.93 | 1580.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 10:15:00 | 1562.05 | 1575.55 | 1578.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 13:15:00 | 1575.00 | 1574.59 | 1577.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 13:15:00 | 1575.00 | 1574.59 | 1577.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 13:15:00 | 1575.00 | 1574.59 | 1577.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 14:00:00 | 1575.00 | 1574.59 | 1577.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 1574.60 | 1574.60 | 1577.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 15:00:00 | 1574.60 | 1574.60 | 1577.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 1580.00 | 1575.68 | 1577.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:15:00 | 1582.00 | 1575.68 | 1577.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 09:15:00 | 1603.60 | 1581.26 | 1579.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 09:15:00 | 1611.00 | 1599.37 | 1591.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 15:15:00 | 1617.50 | 1619.46 | 1607.42 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 09:15:00 | 1628.20 | 1619.46 | 1607.42 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 1614.45 | 1618.46 | 1608.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:30:00 | 1606.75 | 1618.46 | 1608.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 1626.05 | 1620.72 | 1614.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:30:00 | 1623.90 | 1620.72 | 1614.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 1637.35 | 1629.07 | 1622.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:45:00 | 1624.65 | 1629.07 | 1622.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 1622.90 | 1627.84 | 1622.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 11:30:00 | 1626.00 | 1627.84 | 1622.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 1618.00 | 1625.87 | 1622.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-27 12:15:00 | 1618.00 | 1625.87 | 1622.08 | SL hit (close<ema400) qty=1.00 sl=1622.08 alert=retest1 |

### Cycle 74 — SELL (started 2024-08-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 15:15:00 | 1607.45 | 1618.06 | 1619.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 09:15:00 | 1587.10 | 1611.87 | 1616.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 14:15:00 | 1577.75 | 1575.43 | 1588.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-29 15:00:00 | 1577.75 | 1575.43 | 1588.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 1575.40 | 1576.14 | 1586.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 09:30:00 | 1574.15 | 1580.49 | 1584.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 09:30:00 | 1569.05 | 1566.95 | 1573.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 10:15:00 | 1570.55 | 1566.95 | 1573.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 12:00:00 | 1573.25 | 1571.15 | 1574.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 12:15:00 | 1573.35 | 1571.59 | 1574.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 13:00:00 | 1573.35 | 1571.59 | 1574.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 1563.35 | 1568.06 | 1571.65 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-04 14:15:00 | 1608.85 | 1579.12 | 1575.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2024-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 14:15:00 | 1608.85 | 1579.12 | 1575.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 15:15:00 | 1613.75 | 1586.04 | 1579.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 1568.95 | 1597.89 | 1591.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 1568.95 | 1597.89 | 1591.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 1568.95 | 1597.89 | 1591.87 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2024-09-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 12:15:00 | 1571.55 | 1588.05 | 1588.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 13:15:00 | 1565.95 | 1583.63 | 1586.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 13:15:00 | 1557.80 | 1556.10 | 1567.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 14:00:00 | 1557.80 | 1556.10 | 1567.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1556.60 | 1555.88 | 1564.66 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2024-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 10:15:00 | 1580.00 | 1568.61 | 1567.59 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 1558.55 | 1566.48 | 1566.96 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 09:15:00 | 1577.35 | 1567.38 | 1567.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 10:15:00 | 1581.55 | 1570.22 | 1568.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 09:15:00 | 1604.25 | 1604.98 | 1593.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 09:30:00 | 1612.70 | 1604.98 | 1593.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 1595.00 | 1602.77 | 1595.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 13:00:00 | 1595.00 | 1602.77 | 1595.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 13:15:00 | 1589.90 | 1600.20 | 1594.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 14:00:00 | 1589.90 | 1600.20 | 1594.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 14:15:00 | 1598.35 | 1599.83 | 1595.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 11:45:00 | 1605.15 | 1598.85 | 1596.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 12:45:00 | 1604.55 | 1600.04 | 1596.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 13:30:00 | 1604.05 | 1599.94 | 1597.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 14:30:00 | 1604.00 | 1598.95 | 1596.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 1593.80 | 1597.92 | 1596.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 09:15:00 | 1597.90 | 1597.92 | 1596.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 12:00:00 | 1596.85 | 1599.90 | 1598.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 12:15:00 | 1574.95 | 1594.91 | 1595.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2024-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 12:15:00 | 1574.95 | 1594.91 | 1595.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 13:15:00 | 1568.30 | 1589.59 | 1593.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 1588.10 | 1570.80 | 1578.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 14:15:00 | 1588.10 | 1570.80 | 1578.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 1588.10 | 1570.80 | 1578.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 15:00:00 | 1588.10 | 1570.80 | 1578.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 1587.50 | 1574.14 | 1579.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 1575.50 | 1574.14 | 1579.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 1597.50 | 1579.97 | 1581.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:00:00 | 1597.50 | 1579.97 | 1581.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 1590.70 | 1582.11 | 1582.18 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2024-09-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 12:15:00 | 1591.50 | 1583.99 | 1583.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 09:15:00 | 1609.50 | 1591.03 | 1586.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 11:15:00 | 1589.25 | 1591.22 | 1587.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-23 12:00:00 | 1589.25 | 1591.22 | 1587.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 13:15:00 | 1577.75 | 1588.31 | 1586.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 14:00:00 | 1577.75 | 1588.31 | 1586.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 14:15:00 | 1580.20 | 1586.69 | 1586.25 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2024-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 09:15:00 | 1568.25 | 1582.59 | 1584.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 15:15:00 | 1557.00 | 1564.52 | 1571.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 09:15:00 | 1529.45 | 1527.40 | 1536.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 09:15:00 | 1529.45 | 1527.40 | 1536.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 1529.45 | 1527.40 | 1536.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 14:15:00 | 1513.55 | 1522.86 | 1531.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 11:00:00 | 1514.00 | 1522.94 | 1525.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 1437.87 | 1468.92 | 1486.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 1438.30 | 1468.92 | 1486.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 10:15:00 | 1462.85 | 1450.12 | 1465.41 | SL hit (close>ema200) qty=0.50 sl=1450.12 alert=retest2 |

### Cycle 83 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 1495.00 | 1472.10 | 1470.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 10:15:00 | 1503.75 | 1478.43 | 1473.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 1501.35 | 1505.38 | 1494.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 13:00:00 | 1501.35 | 1505.38 | 1494.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 1490.30 | 1502.36 | 1494.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 14:00:00 | 1490.30 | 1502.36 | 1494.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 1503.85 | 1502.66 | 1495.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 09:30:00 | 1505.85 | 1501.71 | 1496.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-11 11:15:00 | 1486.85 | 1497.86 | 1495.22 | SL hit (close<static) qty=1.00 sl=1490.10 alert=retest2 |

### Cycle 84 — SELL (started 2024-10-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 15:15:00 | 1490.00 | 1493.85 | 1493.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 09:15:00 | 1478.95 | 1490.87 | 1492.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 13:15:00 | 1490.95 | 1486.45 | 1489.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 13:15:00 | 1490.95 | 1486.45 | 1489.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 1490.95 | 1486.45 | 1489.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 14:00:00 | 1490.95 | 1486.45 | 1489.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 14:15:00 | 1492.50 | 1487.66 | 1489.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 15:15:00 | 1490.00 | 1487.66 | 1489.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 10:00:00 | 1488.80 | 1488.26 | 1489.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 11:00:00 | 1489.95 | 1488.60 | 1489.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 12:30:00 | 1489.35 | 1488.96 | 1489.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 1482.05 | 1484.14 | 1486.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 10:45:00 | 1470.65 | 1481.06 | 1485.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 15:15:00 | 1472.40 | 1474.50 | 1480.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 09:45:00 | 1470.35 | 1469.01 | 1472.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 11:45:00 | 1474.10 | 1472.62 | 1473.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 1469.85 | 1472.07 | 1473.41 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-21 09:15:00 | 1478.95 | 1474.24 | 1474.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 09:15:00 | 1478.95 | 1474.24 | 1474.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-21 10:15:00 | 1487.25 | 1476.84 | 1475.28 | Break + close above crossover candle high |

### Cycle 86 — SELL (started 2024-10-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 12:15:00 | 1459.00 | 1473.45 | 1474.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 13:15:00 | 1445.50 | 1467.86 | 1471.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 12:15:00 | 1424.90 | 1419.14 | 1432.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 12:45:00 | 1425.00 | 1419.14 | 1432.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 14:15:00 | 1428.75 | 1414.97 | 1421.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 15:00:00 | 1428.75 | 1414.97 | 1421.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 15:15:00 | 1430.05 | 1417.98 | 1422.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-25 09:15:00 | 1424.15 | 1417.98 | 1422.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 1375.00 | 1398.29 | 1408.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 11:30:00 | 1373.25 | 1391.10 | 1403.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 12:30:00 | 1371.25 | 1387.08 | 1400.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 13:30:00 | 1370.75 | 1383.76 | 1397.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 11:15:00 | 1411.50 | 1386.01 | 1385.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2024-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 11:15:00 | 1411.50 | 1386.01 | 1385.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 09:15:00 | 1434.00 | 1411.78 | 1399.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 12:15:00 | 1413.70 | 1415.96 | 1405.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 12:45:00 | 1411.95 | 1415.96 | 1405.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 1403.25 | 1413.42 | 1404.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 14:00:00 | 1403.25 | 1413.42 | 1404.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 1416.35 | 1414.00 | 1405.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 14:30:00 | 1403.05 | 1414.00 | 1405.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 15:15:00 | 1411.05 | 1413.41 | 1406.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 1410.05 | 1412.74 | 1406.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 1424.85 | 1415.16 | 1408.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:30:00 | 1418.20 | 1415.16 | 1408.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1410.50 | 1414.59 | 1409.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:30:00 | 1406.70 | 1414.59 | 1409.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 1412.35 | 1414.14 | 1409.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 14:30:00 | 1418.70 | 1415.03 | 1411.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 11:15:00 | 1428.55 | 1446.24 | 1448.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 1428.55 | 1446.24 | 1448.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 13:15:00 | 1421.80 | 1439.57 | 1444.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 12:15:00 | 1333.40 | 1329.17 | 1348.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 13:00:00 | 1333.40 | 1329.17 | 1348.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 1335.45 | 1330.42 | 1347.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 15:00:00 | 1326.55 | 1329.65 | 1345.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 13:15:00 | 1350.00 | 1335.33 | 1334.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2024-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 13:15:00 | 1350.00 | 1335.33 | 1334.31 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 1320.30 | 1332.32 | 1333.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 15:15:00 | 1319.25 | 1329.71 | 1331.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 1310.50 | 1306.29 | 1315.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 10:00:00 | 1310.50 | 1306.29 | 1315.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 1315.50 | 1307.92 | 1314.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 12:00:00 | 1315.50 | 1307.92 | 1314.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 1317.45 | 1309.82 | 1314.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 13:00:00 | 1317.45 | 1309.82 | 1314.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 1325.40 | 1312.94 | 1315.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:00:00 | 1325.40 | 1312.94 | 1315.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 1315.25 | 1313.40 | 1315.77 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 1335.20 | 1318.22 | 1317.57 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 11:15:00 | 1321.00 | 1323.11 | 1323.35 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2024-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 12:15:00 | 1328.50 | 1324.19 | 1323.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 1356.20 | 1331.19 | 1327.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 13:15:00 | 1329.05 | 1334.17 | 1330.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 13:15:00 | 1329.05 | 1334.17 | 1330.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 1329.05 | 1334.17 | 1330.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:00:00 | 1329.05 | 1334.17 | 1330.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 1329.95 | 1333.33 | 1330.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 1329.95 | 1333.33 | 1330.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 1332.00 | 1333.06 | 1330.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:15:00 | 1340.70 | 1333.06 | 1330.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 10:30:00 | 1335.00 | 1333.81 | 1331.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 12:00:00 | 1333.95 | 1333.84 | 1331.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 09:15:00 | 1308.50 | 1327.91 | 1329.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2024-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 09:15:00 | 1308.50 | 1327.91 | 1329.63 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 14:15:00 | 1348.20 | 1330.04 | 1329.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 09:15:00 | 1349.95 | 1336.95 | 1332.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 10:15:00 | 1367.60 | 1369.02 | 1358.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 11:00:00 | 1367.60 | 1369.02 | 1358.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 1366.30 | 1373.01 | 1365.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:45:00 | 1365.60 | 1373.01 | 1365.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 1373.90 | 1373.18 | 1366.40 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2024-12-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 15:15:00 | 1353.00 | 1364.58 | 1364.62 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 10:15:00 | 1372.50 | 1365.12 | 1364.79 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 12:15:00 | 1350.30 | 1363.32 | 1364.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 13:15:00 | 1343.25 | 1359.31 | 1362.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 14:15:00 | 1365.15 | 1360.47 | 1362.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 14:15:00 | 1365.15 | 1360.47 | 1362.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 1365.15 | 1360.47 | 1362.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 15:00:00 | 1365.15 | 1360.47 | 1362.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 1362.50 | 1360.88 | 1362.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:15:00 | 1369.50 | 1360.88 | 1362.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 1366.10 | 1361.92 | 1362.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:45:00 | 1373.65 | 1361.92 | 1362.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2024-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 10:15:00 | 1370.55 | 1363.65 | 1363.51 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 13:15:00 | 1355.15 | 1362.01 | 1362.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 09:15:00 | 1347.00 | 1359.11 | 1361.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 10:15:00 | 1362.00 | 1359.68 | 1361.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 10:15:00 | 1362.00 | 1359.68 | 1361.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 1362.00 | 1359.68 | 1361.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 11:00:00 | 1362.00 | 1359.68 | 1361.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 11:15:00 | 1360.00 | 1359.75 | 1361.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 11:30:00 | 1367.85 | 1359.75 | 1361.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 12:15:00 | 1367.70 | 1361.34 | 1361.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 12:30:00 | 1368.00 | 1361.34 | 1361.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2024-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 13:15:00 | 1377.25 | 1364.52 | 1363.26 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 13:15:00 | 1357.80 | 1363.13 | 1363.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 14:15:00 | 1353.15 | 1361.13 | 1362.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 13:15:00 | 1346.40 | 1346.05 | 1352.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 14:00:00 | 1346.40 | 1346.05 | 1352.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1364.00 | 1350.21 | 1353.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 1367.10 | 1350.21 | 1353.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 1356.85 | 1351.54 | 1353.48 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2024-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 12:15:00 | 1363.05 | 1354.95 | 1354.76 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 12:15:00 | 1346.95 | 1355.83 | 1356.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 1341.55 | 1352.98 | 1354.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 11:15:00 | 1315.05 | 1313.31 | 1326.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 12:00:00 | 1315.05 | 1313.31 | 1326.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 1299.45 | 1295.71 | 1304.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:45:00 | 1300.45 | 1295.71 | 1304.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 1310.15 | 1299.06 | 1302.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:00:00 | 1310.15 | 1299.06 | 1302.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 1315.00 | 1302.25 | 1303.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 11:15:00 | 1320.05 | 1302.25 | 1303.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2024-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 11:15:00 | 1325.95 | 1306.99 | 1305.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 1345.95 | 1321.71 | 1316.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 13:15:00 | 1324.75 | 1325.72 | 1320.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-27 14:00:00 | 1324.75 | 1325.72 | 1320.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 1317.45 | 1324.06 | 1320.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 15:00:00 | 1317.45 | 1324.06 | 1320.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 1314.00 | 1322.05 | 1319.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:15:00 | 1300.15 | 1322.05 | 1319.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 09:15:00 | 1301.25 | 1317.89 | 1317.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 12:15:00 | 1285.75 | 1305.84 | 1311.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 15:15:00 | 1306.20 | 1302.64 | 1308.62 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-31 09:30:00 | 1292.85 | 1298.98 | 1306.41 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 1305.45 | 1298.95 | 1304.42 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-31 12:15:00 | 1305.45 | 1298.95 | 1304.42 | SL hit (close>ema400) qty=1.00 sl=1304.42 alert=retest1 |

### Cycle 107 — BUY (started 2025-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 13:15:00 | 1313.30 | 1303.06 | 1302.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 15:15:00 | 1317.95 | 1307.68 | 1304.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 10:15:00 | 1293.90 | 1305.33 | 1303.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 10:15:00 | 1293.90 | 1305.33 | 1303.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 1293.90 | 1305.33 | 1303.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 11:00:00 | 1293.90 | 1305.33 | 1303.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 1298.95 | 1304.06 | 1303.52 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 13:15:00 | 1295.00 | 1301.63 | 1302.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 1284.10 | 1296.43 | 1299.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 1275.00 | 1272.20 | 1283.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 1275.00 | 1272.20 | 1283.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 1275.00 | 1272.20 | 1283.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 11:00:00 | 1265.55 | 1270.87 | 1281.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 11:45:00 | 1265.40 | 1268.89 | 1279.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 13:45:00 | 1264.70 | 1268.40 | 1277.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 1256.40 | 1268.86 | 1276.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 1202.27 | 1215.25 | 1229.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 1202.13 | 1215.25 | 1229.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 1201.46 | 1215.25 | 1229.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 1193.58 | 1215.25 | 1229.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 10:15:00 | 1200.55 | 1190.80 | 1206.90 | SL hit (close>ema200) qty=0.50 sl=1190.80 alert=retest2 |

### Cycle 109 — BUY (started 2025-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 13:15:00 | 1214.95 | 1201.49 | 1199.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 12:15:00 | 1218.35 | 1209.95 | 1206.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 1209.75 | 1213.24 | 1209.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 1209.75 | 1213.24 | 1209.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 1209.75 | 1213.24 | 1209.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 1209.75 | 1213.24 | 1209.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 1205.85 | 1211.77 | 1209.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:30:00 | 1208.10 | 1211.77 | 1209.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 1211.25 | 1211.66 | 1209.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 1212.40 | 1211.66 | 1209.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 1216.00 | 1212.53 | 1209.87 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 15:15:00 | 1201.35 | 1208.60 | 1208.64 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 14:15:00 | 1227.80 | 1211.21 | 1209.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 09:15:00 | 1235.35 | 1217.92 | 1212.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-23 14:15:00 | 1225.95 | 1226.09 | 1219.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-23 15:00:00 | 1225.95 | 1226.09 | 1219.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 13:15:00 | 1222.00 | 1229.34 | 1224.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:00:00 | 1222.00 | 1229.34 | 1224.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 1215.25 | 1226.52 | 1223.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:30:00 | 1212.40 | 1226.52 | 1223.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 1187.75 | 1216.92 | 1219.80 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 09:15:00 | 1217.05 | 1210.47 | 1210.25 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 12:15:00 | 1197.10 | 1208.65 | 1209.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 09:15:00 | 1189.75 | 1200.79 | 1205.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 09:15:00 | 1197.45 | 1191.73 | 1197.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 09:15:00 | 1197.45 | 1191.73 | 1197.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 1197.45 | 1191.73 | 1197.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:45:00 | 1195.55 | 1191.73 | 1197.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 1214.85 | 1196.36 | 1199.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 10:45:00 | 1215.30 | 1196.36 | 1199.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 11:15:00 | 1217.00 | 1200.49 | 1200.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 12:00:00 | 1217.00 | 1200.49 | 1200.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2025-01-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 12:15:00 | 1216.05 | 1203.60 | 1202.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 14:15:00 | 1224.10 | 1209.55 | 1205.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 1209.80 | 1218.33 | 1211.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 1209.80 | 1218.33 | 1211.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 1209.80 | 1218.33 | 1211.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 1208.45 | 1218.33 | 1211.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 1156.95 | 1206.05 | 1206.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 1137.80 | 1181.45 | 1194.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 1176.65 | 1154.28 | 1169.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 1176.65 | 1154.28 | 1169.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 1176.65 | 1154.28 | 1169.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 11:45:00 | 1157.35 | 1165.15 | 1168.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 12:30:00 | 1161.25 | 1164.23 | 1168.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 1099.48 | 1109.53 | 1123.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 1103.19 | 1109.53 | 1123.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-13 09:15:00 | 1045.12 | 1098.31 | 1110.49 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 117 — BUY (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 10:15:00 | 1078.60 | 1074.45 | 1074.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 14:15:00 | 1085.60 | 1076.60 | 1075.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 1071.60 | 1077.08 | 1075.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 1071.60 | 1077.08 | 1075.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 1071.60 | 1077.08 | 1075.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 1071.60 | 1077.08 | 1075.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 1070.15 | 1075.70 | 1075.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 1070.15 | 1075.70 | 1075.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 1072.35 | 1075.03 | 1074.97 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2025-02-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 12:15:00 | 1071.30 | 1074.28 | 1074.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 13:15:00 | 1064.70 | 1072.37 | 1073.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 12:15:00 | 1046.35 | 1045.56 | 1052.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-25 13:00:00 | 1046.35 | 1045.56 | 1052.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 13:15:00 | 1050.65 | 1046.58 | 1052.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 14:00:00 | 1050.65 | 1046.58 | 1052.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 1010.85 | 1020.17 | 1029.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 11:15:00 | 1008.55 | 1018.44 | 1027.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-03 14:15:00 | 1031.75 | 1022.58 | 1026.74 | SL hit (close>static) qty=1.00 sl=1030.60 alert=retest2 |

### Cycle 119 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 1040.15 | 1027.73 | 1026.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 1048.05 | 1031.80 | 1028.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 10:15:00 | 1084.10 | 1086.33 | 1075.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 11:00:00 | 1084.10 | 1086.33 | 1075.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 1074.35 | 1082.57 | 1076.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:00:00 | 1074.35 | 1082.57 | 1076.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 1069.45 | 1079.94 | 1075.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:45:00 | 1065.45 | 1079.94 | 1075.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 1058.00 | 1073.32 | 1073.47 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 14:15:00 | 1089.35 | 1072.85 | 1071.06 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 11:15:00 | 1050.40 | 1067.22 | 1069.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 13:15:00 | 1038.15 | 1057.64 | 1064.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 1053.25 | 1051.49 | 1059.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 1053.25 | 1051.49 | 1059.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 1053.25 | 1051.49 | 1059.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:45:00 | 1055.65 | 1051.49 | 1059.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 1055.95 | 1052.38 | 1059.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:45:00 | 1055.85 | 1052.38 | 1059.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 1068.45 | 1055.59 | 1059.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 12:00:00 | 1068.45 | 1055.59 | 1059.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 1061.85 | 1056.85 | 1060.08 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2025-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 14:15:00 | 1076.10 | 1063.80 | 1062.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 1085.05 | 1070.40 | 1066.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 11:15:00 | 1173.80 | 1176.45 | 1152.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 12:00:00 | 1173.80 | 1176.45 | 1152.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 1176.15 | 1190.34 | 1181.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 1176.10 | 1190.34 | 1181.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 1180.30 | 1188.33 | 1181.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 13:45:00 | 1184.30 | 1187.72 | 1181.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 15:00:00 | 1187.65 | 1187.71 | 1182.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 1199.20 | 1186.17 | 1182.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 14:00:00 | 1182.90 | 1186.43 | 1184.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 1183.80 | 1185.90 | 1184.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 14:30:00 | 1178.00 | 1185.90 | 1184.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 15:15:00 | 1183.00 | 1185.32 | 1183.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:15:00 | 1143.10 | 1185.32 | 1183.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-27 09:15:00 | 1172.10 | 1182.68 | 1182.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2025-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 09:15:00 | 1172.10 | 1182.68 | 1182.87 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 10:15:00 | 1193.95 | 1184.93 | 1183.88 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 12:15:00 | 1165.80 | 1180.65 | 1182.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 13:15:00 | 1157.65 | 1176.05 | 1179.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 10:15:00 | 1167.55 | 1167.14 | 1173.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-28 11:00:00 | 1167.55 | 1167.14 | 1173.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 11:15:00 | 1171.05 | 1167.92 | 1173.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 12:00:00 | 1171.05 | 1167.92 | 1173.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 1170.70 | 1168.48 | 1173.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 13:00:00 | 1170.70 | 1168.48 | 1173.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 1168.15 | 1168.41 | 1172.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 13:30:00 | 1173.95 | 1168.41 | 1172.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 1172.70 | 1169.27 | 1172.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:45:00 | 1174.20 | 1169.27 | 1172.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 1159.45 | 1167.31 | 1171.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:15:00 | 1172.55 | 1167.31 | 1171.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 1166.05 | 1167.06 | 1171.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:30:00 | 1176.25 | 1167.06 | 1171.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 1150.00 | 1163.64 | 1169.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:15:00 | 1149.50 | 1163.64 | 1169.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 12:45:00 | 1147.75 | 1159.16 | 1166.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 14:15:00 | 1149.00 | 1157.32 | 1164.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 1149.00 | 1154.50 | 1161.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 1152.85 | 1154.17 | 1161.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:30:00 | 1159.20 | 1154.17 | 1161.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 1159.85 | 1155.31 | 1161.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 10:30:00 | 1160.90 | 1155.31 | 1161.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 1160.50 | 1156.35 | 1160.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 1160.50 | 1156.35 | 1160.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 1165.60 | 1158.20 | 1161.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:00:00 | 1165.60 | 1158.20 | 1161.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 1157.65 | 1158.09 | 1161.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 14:30:00 | 1154.10 | 1156.51 | 1160.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-03 09:15:00 | 1092.02 | 1148.41 | 1155.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-03 09:15:00 | 1090.36 | 1148.41 | 1155.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-03 09:15:00 | 1091.55 | 1148.41 | 1155.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-03 09:15:00 | 1091.55 | 1148.41 | 1155.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-03 09:15:00 | 1096.39 | 1148.41 | 1155.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-04 09:15:00 | 1034.55 | 1106.15 | 1129.61 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 127 — BUY (started 2025-04-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 13:15:00 | 995.25 | 982.39 | 982.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 14:15:00 | 1002.75 | 986.46 | 983.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 14:15:00 | 1054.50 | 1057.09 | 1039.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 15:00:00 | 1054.50 | 1057.09 | 1039.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1056.90 | 1056.53 | 1042.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:00:00 | 1058.00 | 1056.82 | 1043.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 1088.80 | 1103.16 | 1103.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 1088.80 | 1103.16 | 1103.38 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 12:15:00 | 1114.20 | 1102.12 | 1101.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 13:15:00 | 1117.20 | 1105.14 | 1102.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 1125.60 | 1126.77 | 1118.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 1125.60 | 1126.77 | 1118.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 1125.60 | 1126.77 | 1118.56 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2025-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 14:15:00 | 1105.90 | 1116.24 | 1116.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 1100.40 | 1113.07 | 1114.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 1116.90 | 1113.84 | 1115.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 1116.90 | 1113.84 | 1115.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 1116.90 | 1113.84 | 1115.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 1116.90 | 1113.84 | 1115.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 1107.50 | 1112.57 | 1114.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:15:00 | 1104.00 | 1112.57 | 1114.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 12:15:00 | 1101.10 | 1111.32 | 1113.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 14:15:00 | 1104.30 | 1110.40 | 1112.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 15:15:00 | 1102.50 | 1110.04 | 1112.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 1109.50 | 1108.73 | 1111.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 1109.50 | 1108.73 | 1111.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1112.30 | 1109.44 | 1111.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:45:00 | 1112.50 | 1109.44 | 1111.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 1116.70 | 1110.89 | 1111.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:00:00 | 1116.70 | 1110.89 | 1111.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 1113.30 | 1111.37 | 1112.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:45:00 | 1121.90 | 1111.37 | 1112.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-05 13:15:00 | 1117.00 | 1112.50 | 1112.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2025-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 13:15:00 | 1117.00 | 1112.50 | 1112.47 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 12:15:00 | 1101.20 | 1111.37 | 1112.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 13:15:00 | 1090.10 | 1107.12 | 1110.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 1121.50 | 1103.34 | 1106.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 10:15:00 | 1121.50 | 1103.34 | 1106.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 1121.50 | 1103.34 | 1106.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:00:00 | 1121.50 | 1103.34 | 1106.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2025-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 11:15:00 | 1134.00 | 1109.47 | 1109.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 13:15:00 | 1138.20 | 1118.87 | 1113.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 1107.60 | 1125.40 | 1121.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 13:15:00 | 1107.60 | 1125.40 | 1121.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 1107.60 | 1125.40 | 1121.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 1107.60 | 1125.40 | 1121.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 1114.20 | 1123.16 | 1120.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:30:00 | 1100.20 | 1123.16 | 1120.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 1115.20 | 1121.57 | 1120.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 1122.00 | 1121.57 | 1120.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 1257.60 | 1261.03 | 1253.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:45:00 | 1254.70 | 1261.03 | 1253.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1258.50 | 1260.19 | 1254.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 1259.00 | 1260.19 | 1254.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1256.80 | 1259.51 | 1254.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:15:00 | 1250.90 | 1259.51 | 1254.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 1246.60 | 1256.93 | 1253.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 1244.10 | 1256.93 | 1253.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 1239.00 | 1253.34 | 1252.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:45:00 | 1235.00 | 1253.34 | 1252.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 1232.80 | 1249.23 | 1250.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 1227.60 | 1244.91 | 1248.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 1245.60 | 1242.66 | 1246.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 1245.60 | 1242.66 | 1246.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1245.60 | 1242.66 | 1246.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 1245.60 | 1242.66 | 1246.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1243.10 | 1242.75 | 1246.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:30:00 | 1244.60 | 1242.75 | 1246.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 1236.20 | 1230.91 | 1236.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:00:00 | 1236.20 | 1230.91 | 1236.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 1237.20 | 1232.16 | 1236.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 14:15:00 | 1233.90 | 1232.16 | 1236.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 15:15:00 | 1240.20 | 1234.87 | 1236.82 | SL hit (close>static) qty=1.00 sl=1240.00 alert=retest2 |

### Cycle 135 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 1247.10 | 1238.29 | 1238.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 1251.40 | 1244.32 | 1241.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 1241.10 | 1246.53 | 1244.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 1241.10 | 1246.53 | 1244.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1241.10 | 1246.53 | 1244.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 1241.10 | 1246.53 | 1244.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1252.40 | 1247.70 | 1245.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:45:00 | 1257.00 | 1250.02 | 1246.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 12:45:00 | 1256.70 | 1250.36 | 1248.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 13:30:00 | 1257.80 | 1251.61 | 1249.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 1266.90 | 1250.62 | 1249.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 1251.70 | 1256.74 | 1253.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:00:00 | 1251.70 | 1256.74 | 1253.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 1256.60 | 1256.71 | 1253.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:30:00 | 1253.20 | 1256.71 | 1253.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 1259.80 | 1257.33 | 1254.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 1253.30 | 1257.33 | 1254.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1241.80 | 1254.22 | 1253.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 1241.80 | 1254.22 | 1253.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1245.60 | 1252.50 | 1252.47 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-30 11:15:00 | 1244.30 | 1250.86 | 1251.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 11:15:00 | 1244.30 | 1250.86 | 1251.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 1240.10 | 1248.63 | 1250.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 1262.00 | 1249.79 | 1250.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 1262.00 | 1249.79 | 1250.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1262.00 | 1249.79 | 1250.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:45:00 | 1260.80 | 1249.79 | 1250.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 10:15:00 | 1261.20 | 1252.07 | 1251.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 10:15:00 | 1266.10 | 1257.77 | 1255.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 1254.00 | 1265.09 | 1261.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 1254.00 | 1265.09 | 1261.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 1254.00 | 1265.09 | 1261.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 1254.40 | 1265.09 | 1261.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 1255.00 | 1263.07 | 1260.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:30:00 | 1252.00 | 1263.07 | 1260.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2025-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 12:15:00 | 1250.80 | 1259.01 | 1259.17 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 14:15:00 | 1268.90 | 1260.31 | 1259.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 13:15:00 | 1283.50 | 1275.53 | 1269.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 15:15:00 | 1335.30 | 1336.17 | 1321.92 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 09:15:00 | 1348.30 | 1336.17 | 1321.92 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 1327.90 | 1335.07 | 1327.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 1330.00 | 1335.07 | 1327.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1334.40 | 1334.93 | 1327.79 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-12 09:15:00 | 1327.50 | 1333.30 | 1328.28 | SL hit (close<ema400) qty=1.00 sl=1328.28 alert=retest1 |

### Cycle 140 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 1310.70 | 1325.83 | 1326.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 14:15:00 | 1305.00 | 1321.66 | 1324.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 1313.30 | 1307.61 | 1313.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 1313.30 | 1307.61 | 1313.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1306.80 | 1307.83 | 1312.91 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 14:15:00 | 1322.60 | 1314.92 | 1314.70 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 11:15:00 | 1304.60 | 1313.99 | 1314.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 12:15:00 | 1301.30 | 1311.46 | 1313.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 1306.80 | 1306.51 | 1310.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 1306.80 | 1306.51 | 1310.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1306.80 | 1306.51 | 1310.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:30:00 | 1301.90 | 1305.01 | 1309.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:45:00 | 1298.50 | 1304.73 | 1308.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:15:00 | 1299.50 | 1304.73 | 1308.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:45:00 | 1299.90 | 1299.69 | 1304.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1285.90 | 1279.88 | 1285.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:30:00 | 1289.80 | 1279.88 | 1285.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 1285.70 | 1281.04 | 1285.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:00:00 | 1285.70 | 1281.04 | 1285.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 1291.80 | 1283.20 | 1286.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:00:00 | 1291.80 | 1283.20 | 1286.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 1290.60 | 1284.68 | 1286.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:30:00 | 1293.20 | 1284.68 | 1286.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-23 14:15:00 | 1295.20 | 1288.16 | 1287.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 14:15:00 | 1295.20 | 1288.16 | 1287.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 15:15:00 | 1305.00 | 1291.53 | 1289.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 10:15:00 | 1293.50 | 1295.04 | 1291.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 11:00:00 | 1293.50 | 1295.04 | 1291.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 1300.00 | 1296.03 | 1292.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:45:00 | 1294.50 | 1296.03 | 1292.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 1293.60 | 1295.76 | 1292.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:30:00 | 1293.40 | 1295.76 | 1292.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 1295.90 | 1295.92 | 1293.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 1303.80 | 1295.92 | 1293.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 11:00:00 | 1296.60 | 1296.52 | 1294.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 11:15:00 | 1283.70 | 1293.96 | 1293.24 | SL hit (close<static) qty=1.00 sl=1292.50 alert=retest2 |

### Cycle 144 — SELL (started 2025-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 12:15:00 | 1286.00 | 1292.36 | 1292.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-25 13:15:00 | 1276.00 | 1289.09 | 1291.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 11:15:00 | 1286.80 | 1283.91 | 1287.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 11:15:00 | 1286.80 | 1283.91 | 1287.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 1286.80 | 1283.91 | 1287.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:00:00 | 1286.80 | 1283.91 | 1287.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 1285.60 | 1284.25 | 1286.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:45:00 | 1287.10 | 1284.25 | 1286.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 1283.80 | 1284.16 | 1286.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 14:15:00 | 1287.20 | 1284.16 | 1286.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 1289.30 | 1285.18 | 1286.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 14:30:00 | 1288.70 | 1285.18 | 1286.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 1285.80 | 1285.31 | 1286.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:15:00 | 1298.40 | 1285.31 | 1286.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 1304.20 | 1289.09 | 1288.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 10:15:00 | 1315.20 | 1294.31 | 1290.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 13:15:00 | 1307.00 | 1310.13 | 1303.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:45:00 | 1308.30 | 1310.13 | 1303.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 1298.30 | 1306.93 | 1303.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:30:00 | 1303.80 | 1306.93 | 1303.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 1280.50 | 1301.64 | 1301.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 12:15:00 | 1272.40 | 1291.53 | 1296.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 11:15:00 | 1289.20 | 1285.43 | 1290.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-02 12:00:00 | 1289.20 | 1285.43 | 1290.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 1315.00 | 1291.34 | 1292.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:00:00 | 1315.00 | 1291.34 | 1292.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 13:15:00 | 1308.40 | 1294.75 | 1294.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 09:15:00 | 1332.50 | 1307.50 | 1300.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 12:15:00 | 1310.20 | 1311.95 | 1304.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-03 13:00:00 | 1310.20 | 1311.95 | 1304.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 1303.10 | 1310.18 | 1304.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:45:00 | 1302.60 | 1310.18 | 1304.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 1297.50 | 1307.65 | 1304.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 15:00:00 | 1297.50 | 1307.65 | 1304.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 1297.80 | 1305.68 | 1303.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 1299.60 | 1305.68 | 1303.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 1309.20 | 1308.47 | 1305.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:45:00 | 1301.00 | 1308.47 | 1305.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 1304.10 | 1307.59 | 1305.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:45:00 | 1300.00 | 1307.59 | 1305.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 1300.90 | 1306.26 | 1304.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:30:00 | 1302.00 | 1306.26 | 1304.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 1316.00 | 1308.20 | 1305.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 14:30:00 | 1312.40 | 1308.20 | 1305.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1301.00 | 1307.59 | 1306.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:45:00 | 1300.50 | 1307.59 | 1306.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1305.10 | 1307.10 | 1306.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 11:30:00 | 1308.40 | 1306.92 | 1306.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 13:00:00 | 1307.50 | 1307.03 | 1306.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 13:45:00 | 1308.00 | 1307.19 | 1306.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 15:00:00 | 1308.90 | 1307.53 | 1306.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 1303.50 | 1306.72 | 1306.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 1303.50 | 1306.72 | 1306.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 1294.30 | 1304.24 | 1305.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 1294.30 | 1304.24 | 1305.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 1286.00 | 1300.59 | 1303.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 1211.90 | 1210.86 | 1222.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 10:15:00 | 1224.60 | 1213.61 | 1222.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 1224.60 | 1213.61 | 1222.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 1224.60 | 1213.61 | 1222.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 1233.10 | 1217.50 | 1223.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:00:00 | 1233.10 | 1217.50 | 1223.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 1236.40 | 1221.28 | 1224.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 1236.40 | 1221.28 | 1224.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2025-07-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 15:15:00 | 1238.10 | 1228.83 | 1227.80 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 14:15:00 | 1223.50 | 1229.11 | 1229.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 11:15:00 | 1219.60 | 1225.98 | 1227.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 1221.90 | 1221.84 | 1224.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 10:00:00 | 1221.90 | 1221.84 | 1224.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 1221.30 | 1221.73 | 1224.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:30:00 | 1221.90 | 1221.73 | 1224.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 1222.40 | 1221.07 | 1223.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:15:00 | 1215.00 | 1221.07 | 1223.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 14:15:00 | 1224.50 | 1214.46 | 1215.72 | SL hit (close>static) qty=1.00 sl=1224.30 alert=retest2 |

### Cycle 151 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 1219.60 | 1216.53 | 1216.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 11:15:00 | 1225.30 | 1218.28 | 1217.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 14:15:00 | 1216.20 | 1219.24 | 1218.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 14:15:00 | 1216.20 | 1219.24 | 1218.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1216.20 | 1219.24 | 1218.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:45:00 | 1215.30 | 1219.24 | 1218.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 1214.20 | 1218.23 | 1217.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 1214.50 | 1218.23 | 1217.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 1203.70 | 1215.32 | 1216.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 1194.10 | 1211.08 | 1214.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 14:15:00 | 1204.70 | 1203.24 | 1208.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-25 15:00:00 | 1204.70 | 1203.24 | 1208.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 1222.00 | 1206.47 | 1209.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:00:00 | 1222.00 | 1206.47 | 1209.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 1212.60 | 1207.70 | 1209.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:00:00 | 1209.00 | 1207.96 | 1209.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:30:00 | 1208.50 | 1204.17 | 1207.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 14:15:00 | 1222.10 | 1203.66 | 1203.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 1222.10 | 1203.66 | 1203.31 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 10:15:00 | 1191.70 | 1201.81 | 1202.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 12:15:00 | 1183.10 | 1195.96 | 1199.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 15:15:00 | 1192.90 | 1192.09 | 1196.75 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-31 09:15:00 | 1157.80 | 1192.09 | 1196.75 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 1166.70 | 1158.00 | 1165.24 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-04 11:15:00 | 1166.70 | 1158.00 | 1165.24 | SL hit (close>ema400) qty=1.00 sl=1165.24 alert=retest1 |

### Cycle 155 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 1178.60 | 1169.03 | 1168.79 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 1157.60 | 1166.74 | 1167.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 12:15:00 | 1155.20 | 1162.84 | 1165.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 14:15:00 | 1161.50 | 1161.08 | 1164.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-05 15:00:00 | 1161.50 | 1161.08 | 1164.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1152.60 | 1159.79 | 1163.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 11:30:00 | 1147.30 | 1156.44 | 1160.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 13:30:00 | 1138.20 | 1154.81 | 1159.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 12:15:00 | 1142.90 | 1137.69 | 1147.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 14:00:00 | 1147.80 | 1140.62 | 1147.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1163.40 | 1145.18 | 1148.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 1163.40 | 1145.18 | 1148.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1158.90 | 1147.92 | 1149.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 1159.60 | 1147.92 | 1149.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 1155.40 | 1149.43 | 1150.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 11:30:00 | 1143.40 | 1148.10 | 1149.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 09:15:00 | 1159.50 | 1145.28 | 1146.95 | SL hit (close>static) qty=1.00 sl=1158.90 alert=retest2 |

### Cycle 157 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 1170.00 | 1150.23 | 1149.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 1180.00 | 1162.90 | 1159.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 1171.00 | 1174.17 | 1168.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 1171.00 | 1174.17 | 1168.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1171.00 | 1174.17 | 1168.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 1171.00 | 1174.17 | 1168.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 1180.70 | 1185.56 | 1180.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:15:00 | 1181.50 | 1185.56 | 1180.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1192.00 | 1186.85 | 1181.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 1177.80 | 1186.85 | 1181.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 1172.90 | 1184.56 | 1183.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 1172.90 | 1184.56 | 1183.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 10:15:00 | 1169.30 | 1181.51 | 1182.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 14:15:00 | 1159.00 | 1172.89 | 1177.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 1170.60 | 1168.86 | 1174.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 10:15:00 | 1170.60 | 1168.86 | 1174.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1170.60 | 1168.86 | 1174.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:45:00 | 1175.60 | 1168.86 | 1174.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 1151.00 | 1139.63 | 1148.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 1151.00 | 1139.63 | 1148.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 1146.10 | 1140.92 | 1148.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 12:15:00 | 1142.20 | 1140.92 | 1148.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 14:45:00 | 1143.30 | 1141.83 | 1147.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 1135.30 | 1119.98 | 1119.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 1135.30 | 1119.98 | 1119.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 1138.90 | 1123.76 | 1121.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 1134.30 | 1134.36 | 1129.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 12:45:00 | 1135.90 | 1134.36 | 1129.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 1132.30 | 1137.20 | 1134.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 1132.30 | 1137.20 | 1134.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 1130.10 | 1135.78 | 1133.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 1130.10 | 1135.78 | 1133.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 1133.00 | 1135.22 | 1133.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 1138.60 | 1135.22 | 1133.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:30:00 | 1136.20 | 1135.36 | 1134.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-16 09:15:00 | 1252.46 | 1234.54 | 1224.40 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 1260.70 | 1262.86 | 1263.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 11:15:00 | 1256.40 | 1261.57 | 1262.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1207.50 | 1196.45 | 1208.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 1207.50 | 1196.45 | 1208.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1207.50 | 1196.45 | 1208.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 1207.50 | 1196.45 | 1208.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 1209.80 | 1199.12 | 1208.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 15:15:00 | 1192.20 | 1210.93 | 1211.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:30:00 | 1196.50 | 1203.28 | 1207.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:15:00 | 1201.10 | 1203.46 | 1206.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 10:15:00 | 1217.90 | 1209.13 | 1208.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 10:15:00 | 1217.90 | 1209.13 | 1208.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 1222.10 | 1214.12 | 1211.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 13:15:00 | 1215.00 | 1220.42 | 1216.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 13:15:00 | 1215.00 | 1220.42 | 1216.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 1215.00 | 1220.42 | 1216.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:45:00 | 1216.60 | 1220.42 | 1216.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 1219.50 | 1220.23 | 1216.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 15:15:00 | 1222.00 | 1220.23 | 1216.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 09:30:00 | 1221.20 | 1219.87 | 1217.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 12:15:00 | 1220.90 | 1218.70 | 1217.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 13:15:00 | 1225.30 | 1218.88 | 1217.38 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 1234.50 | 1227.92 | 1222.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 1205.70 | 1220.83 | 1222.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 1205.70 | 1220.83 | 1222.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 1196.30 | 1215.92 | 1219.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 12:15:00 | 1197.00 | 1195.21 | 1203.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 12:45:00 | 1197.60 | 1195.21 | 1203.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 1203.40 | 1196.85 | 1203.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:00:00 | 1203.40 | 1196.85 | 1203.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 1209.00 | 1199.28 | 1204.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 1209.00 | 1199.28 | 1204.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 1209.30 | 1201.28 | 1204.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 1216.00 | 1201.28 | 1204.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 1215.90 | 1207.44 | 1206.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 14:15:00 | 1226.00 | 1215.38 | 1212.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 09:15:00 | 1214.90 | 1217.06 | 1213.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 1214.90 | 1217.06 | 1213.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 1214.90 | 1217.06 | 1213.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:00:00 | 1214.90 | 1217.06 | 1213.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 1217.00 | 1217.05 | 1214.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 1217.00 | 1217.05 | 1214.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 1205.10 | 1214.66 | 1213.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:00:00 | 1205.10 | 1214.66 | 1213.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 1217.00 | 1215.13 | 1213.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:30:00 | 1208.30 | 1215.13 | 1213.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 1246.80 | 1257.00 | 1248.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 1246.80 | 1257.00 | 1248.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 1240.30 | 1253.66 | 1247.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:45:00 | 1240.40 | 1253.66 | 1247.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 1240.50 | 1251.03 | 1246.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:30:00 | 1239.00 | 1251.03 | 1246.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 1246.80 | 1248.73 | 1246.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 09:15:00 | 1288.90 | 1246.64 | 1246.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 10:15:00 | 1317.20 | 1323.06 | 1323.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 1317.20 | 1323.06 | 1323.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 1302.40 | 1318.93 | 1321.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 14:15:00 | 1316.80 | 1314.59 | 1316.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 14:15:00 | 1316.80 | 1314.59 | 1316.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 1316.80 | 1314.59 | 1316.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 1316.80 | 1314.59 | 1316.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1317.40 | 1315.15 | 1317.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 1315.00 | 1315.15 | 1317.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1315.20 | 1315.16 | 1316.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 10:30:00 | 1309.20 | 1314.53 | 1316.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 13:15:00 | 1329.50 | 1319.71 | 1318.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 1329.50 | 1319.71 | 1318.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 11:15:00 | 1339.10 | 1325.51 | 1321.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 1386.20 | 1387.36 | 1373.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 13:00:00 | 1386.20 | 1387.36 | 1373.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1384.00 | 1386.14 | 1377.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:00:00 | 1398.00 | 1384.75 | 1379.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:30:00 | 1398.00 | 1390.04 | 1382.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 10:15:00 | 1375.30 | 1383.43 | 1383.10 | SL hit (close<static) qty=1.00 sl=1376.80 alert=retest2 |

### Cycle 166 — SELL (started 2025-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 11:15:00 | 1377.70 | 1382.28 | 1382.61 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 13:15:00 | 1396.00 | 1384.10 | 1383.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 09:15:00 | 1421.80 | 1395.89 | 1389.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 14:15:00 | 1435.20 | 1435.54 | 1421.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-20 15:00:00 | 1435.20 | 1435.54 | 1421.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1423.00 | 1433.19 | 1423.11 | EMA400 retest candle locked (from upside) |

### Cycle 168 — SELL (started 2025-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 15:15:00 | 1410.00 | 1419.47 | 1419.91 | EMA200 below EMA400 |

### Cycle 169 — BUY (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 10:15:00 | 1422.00 | 1420.56 | 1420.36 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 14:15:00 | 1409.40 | 1419.78 | 1420.26 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 1431.80 | 1418.00 | 1417.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 15:15:00 | 1434.40 | 1425.96 | 1421.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 13:15:00 | 1433.20 | 1433.29 | 1427.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 13:30:00 | 1432.00 | 1433.29 | 1427.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1414.00 | 1429.33 | 1427.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 1414.00 | 1429.33 | 1427.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 1419.50 | 1427.36 | 1426.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 11:15:00 | 1431.40 | 1427.36 | 1426.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 1410.40 | 1428.65 | 1429.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 1410.40 | 1428.65 | 1429.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 1392.50 | 1415.18 | 1421.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 14:15:00 | 1406.60 | 1405.95 | 1413.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 15:00:00 | 1406.60 | 1405.95 | 1413.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 1412.40 | 1408.00 | 1413.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:30:00 | 1415.40 | 1408.00 | 1413.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1415.40 | 1409.48 | 1413.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 1416.20 | 1409.48 | 1413.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 1407.10 | 1409.00 | 1412.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 12:30:00 | 1398.80 | 1407.38 | 1411.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:30:00 | 1397.60 | 1405.63 | 1410.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 1398.60 | 1402.69 | 1408.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 1396.00 | 1402.18 | 1404.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 1391.00 | 1399.95 | 1403.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 11:30:00 | 1381.40 | 1393.65 | 1399.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 12:45:00 | 1382.80 | 1380.41 | 1387.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:15:00 | 1380.20 | 1381.21 | 1387.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 1383.80 | 1383.07 | 1386.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 1388.60 | 1384.18 | 1387.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:30:00 | 1389.80 | 1384.18 | 1387.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 1379.50 | 1383.24 | 1386.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:30:00 | 1388.10 | 1383.24 | 1386.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 1382.60 | 1382.19 | 1385.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:00:00 | 1382.60 | 1382.19 | 1385.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 1383.70 | 1382.49 | 1384.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 15:00:00 | 1383.70 | 1382.49 | 1384.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 1385.00 | 1382.99 | 1384.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 1386.60 | 1382.99 | 1384.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 1392.10 | 1384.81 | 1385.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:00:00 | 1392.10 | 1384.81 | 1385.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-11 10:15:00 | 1397.50 | 1387.35 | 1386.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — BUY (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 10:15:00 | 1397.50 | 1387.35 | 1386.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 11:15:00 | 1409.80 | 1391.84 | 1388.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1405.70 | 1414.17 | 1406.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 1405.70 | 1414.17 | 1406.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1405.70 | 1414.17 | 1406.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 1403.70 | 1414.17 | 1406.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 1403.80 | 1412.09 | 1405.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:30:00 | 1400.80 | 1412.09 | 1405.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 1400.00 | 1409.67 | 1405.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:30:00 | 1399.50 | 1409.67 | 1405.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 1405.50 | 1407.60 | 1405.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:30:00 | 1406.50 | 1407.60 | 1405.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 1417.30 | 1409.54 | 1406.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 1406.80 | 1407.35 | 1405.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1396.00 | 1405.08 | 1404.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 1396.00 | 1405.08 | 1404.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 1402.40 | 1404.54 | 1404.66 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 12:15:00 | 1413.30 | 1406.29 | 1405.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 13:15:00 | 1419.50 | 1408.94 | 1406.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 15:15:00 | 1410.00 | 1410.89 | 1408.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 1403.30 | 1409.37 | 1407.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 1403.30 | 1409.37 | 1407.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 1403.30 | 1409.37 | 1407.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 1409.60 | 1409.42 | 1407.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:30:00 | 1403.60 | 1409.42 | 1407.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 1411.00 | 1409.73 | 1408.13 | EMA400 retest candle locked (from upside) |

### Cycle 176 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 1388.70 | 1406.45 | 1407.44 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 1421.30 | 1409.46 | 1408.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 1427.30 | 1417.88 | 1413.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 10:15:00 | 1457.20 | 1457.25 | 1448.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 11:00:00 | 1457.20 | 1457.25 | 1448.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 1451.70 | 1456.83 | 1451.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 1451.70 | 1456.83 | 1451.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 1451.00 | 1455.67 | 1451.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 1456.70 | 1455.67 | 1451.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1448.80 | 1454.29 | 1450.92 | EMA400 retest candle locked (from upside) |

### Cycle 178 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 1438.00 | 1448.11 | 1449.18 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 10:15:00 | 1458.20 | 1449.59 | 1448.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 11:15:00 | 1463.70 | 1452.41 | 1449.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 14:15:00 | 1454.70 | 1454.88 | 1451.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-30 15:00:00 | 1454.70 | 1454.88 | 1451.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1456.60 | 1455.22 | 1452.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 09:15:00 | 1478.00 | 1455.22 | 1452.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 10:45:00 | 1459.70 | 1459.12 | 1454.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 11:15:00 | 1462.60 | 1475.25 | 1476.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 1462.60 | 1475.25 | 1476.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 12:15:00 | 1455.20 | 1471.24 | 1474.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 1443.90 | 1443.76 | 1452.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 14:15:00 | 1456.40 | 1446.95 | 1452.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1456.40 | 1446.95 | 1452.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 1457.90 | 1446.95 | 1452.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1453.00 | 1448.16 | 1452.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1460.20 | 1448.16 | 1452.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1459.00 | 1450.33 | 1453.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:15:00 | 1448.60 | 1454.42 | 1454.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 14:15:00 | 1461.40 | 1455.81 | 1455.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — BUY (started 2026-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 14:15:00 | 1461.40 | 1455.81 | 1455.30 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 09:15:00 | 1450.40 | 1454.92 | 1455.00 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 1459.70 | 1455.88 | 1455.42 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 13:15:00 | 1447.50 | 1453.79 | 1454.56 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 1466.80 | 1454.54 | 1454.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 11:15:00 | 1473.50 | 1460.02 | 1457.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 1455.20 | 1459.34 | 1457.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 13:15:00 | 1455.20 | 1459.34 | 1457.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 1455.20 | 1459.34 | 1457.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 1455.20 | 1459.34 | 1457.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 1455.90 | 1458.65 | 1457.21 | EMA400 retest candle locked (from upside) |

### Cycle 186 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 1437.00 | 1453.74 | 1455.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 10:15:00 | 1423.00 | 1447.59 | 1452.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1439.20 | 1396.09 | 1403.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 1439.20 | 1396.09 | 1403.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1439.20 | 1396.09 | 1403.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 1439.20 | 1396.09 | 1403.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1414.30 | 1399.73 | 1404.58 | EMA400 retest candle locked (from downside) |

### Cycle 187 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 1422.10 | 1410.30 | 1408.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 1428.90 | 1414.02 | 1410.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 11:15:00 | 1409.10 | 1419.81 | 1415.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 11:15:00 | 1409.10 | 1419.81 | 1415.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 1409.10 | 1419.81 | 1415.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 12:00:00 | 1409.10 | 1419.81 | 1415.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 1414.80 | 1418.80 | 1415.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 12:30:00 | 1412.30 | 1418.80 | 1415.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 1402.90 | 1415.62 | 1413.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:00:00 | 1402.90 | 1415.62 | 1413.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 1411.20 | 1414.74 | 1413.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:30:00 | 1406.90 | 1414.74 | 1413.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 1410.00 | 1413.79 | 1413.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 1416.50 | 1413.79 | 1413.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 10:15:00 | 1406.10 | 1412.75 | 1412.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 1406.10 | 1412.75 | 1412.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 1399.60 | 1409.90 | 1411.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 1417.60 | 1411.44 | 1412.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 1417.60 | 1411.44 | 1412.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 1417.60 | 1411.44 | 1412.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 1417.60 | 1411.44 | 1412.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 1420.00 | 1413.15 | 1412.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 12:15:00 | 1428.60 | 1419.92 | 1416.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 12:15:00 | 1436.40 | 1438.01 | 1429.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 13:00:00 | 1436.40 | 1438.01 | 1429.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 1430.00 | 1436.53 | 1431.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:15:00 | 1423.00 | 1436.53 | 1431.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1445.50 | 1438.32 | 1432.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:45:00 | 1448.00 | 1438.18 | 1432.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 1447.70 | 1438.63 | 1434.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 1410.90 | 1438.38 | 1436.92 | SL hit (close<static) qty=1.00 sl=1419.20 alert=retest2 |

### Cycle 190 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 1379.40 | 1426.59 | 1431.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 1374.60 | 1416.19 | 1426.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 11:15:00 | 1408.20 | 1407.81 | 1418.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 11:30:00 | 1412.10 | 1407.81 | 1418.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 1422.90 | 1410.82 | 1417.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 14:00:00 | 1422.90 | 1410.82 | 1417.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1443.40 | 1417.34 | 1420.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 1443.40 | 1417.34 | 1420.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1554.50 | 1448.65 | 1434.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 11:15:00 | 1589.60 | 1544.95 | 1504.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 10:15:00 | 1571.70 | 1572.55 | 1538.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 11:00:00 | 1571.70 | 1572.55 | 1538.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1547.20 | 1567.41 | 1551.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:45:00 | 1549.10 | 1567.41 | 1551.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1567.00 | 1567.33 | 1552.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 1589.80 | 1555.91 | 1551.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-12 13:15:00 | 1748.78 | 1667.55 | 1640.83 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 192 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 1858.50 | 1886.97 | 1887.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 1829.10 | 1872.87 | 1880.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 1897.30 | 1857.96 | 1865.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 1897.30 | 1857.96 | 1865.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 1897.30 | 1857.96 | 1865.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 11:15:00 | 1874.40 | 1863.67 | 1867.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:30:00 | 1877.40 | 1867.92 | 1868.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 1895.90 | 1873.52 | 1871.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — BUY (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 14:15:00 | 1895.90 | 1873.52 | 1871.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 11:15:00 | 1920.60 | 1894.00 | 1882.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 1846.80 | 1897.53 | 1890.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 1846.80 | 1897.53 | 1890.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 1846.80 | 1897.53 | 1890.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:45:00 | 1845.50 | 1897.53 | 1890.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 1818.70 | 1881.76 | 1884.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 1794.30 | 1825.84 | 1840.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1694.70 | 1689.02 | 1719.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 1705.00 | 1689.02 | 1719.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1725.90 | 1695.75 | 1717.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 1725.90 | 1695.75 | 1717.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1729.70 | 1702.54 | 1718.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 1729.70 | 1702.54 | 1718.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 1726.70 | 1707.37 | 1719.09 | EMA400 retest candle locked (from downside) |

### Cycle 195 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 1744.90 | 1726.13 | 1725.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 15:15:00 | 1755.00 | 1731.90 | 1728.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1751.40 | 1778.51 | 1761.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1751.40 | 1778.51 | 1761.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1751.40 | 1778.51 | 1761.76 | EMA400 retest candle locked (from upside) |

### Cycle 196 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 1714.30 | 1746.47 | 1750.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 1656.10 | 1711.80 | 1728.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1705.90 | 1671.24 | 1686.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 1705.90 | 1671.24 | 1686.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1705.90 | 1671.24 | 1686.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 1705.90 | 1671.24 | 1686.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1712.30 | 1679.45 | 1689.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 1712.20 | 1679.45 | 1689.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1747.30 | 1701.59 | 1697.55 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 1670.90 | 1707.07 | 1710.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 1624.00 | 1674.97 | 1688.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 09:15:00 | 1657.70 | 1651.65 | 1666.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 1657.70 | 1651.65 | 1666.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1657.70 | 1651.65 | 1666.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:45:00 | 1648.00 | 1651.65 | 1666.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 1665.60 | 1654.62 | 1662.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 15:00:00 | 1665.60 | 1654.62 | 1662.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 15:15:00 | 1672.10 | 1658.12 | 1662.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:15:00 | 1626.80 | 1658.12 | 1662.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 13:45:00 | 1660.90 | 1653.96 | 1657.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 1776.90 | 1683.50 | 1670.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 1776.90 | 1683.50 | 1670.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 1792.00 | 1755.57 | 1737.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 1760.40 | 1779.11 | 1761.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 1760.40 | 1779.11 | 1761.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1760.40 | 1779.11 | 1761.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:30:00 | 1785.50 | 1781.31 | 1765.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 12:15:00 | 1870.00 | 1884.11 | 1884.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — SELL (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 12:15:00 | 1870.00 | 1884.11 | 1884.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 1838.10 | 1866.17 | 1875.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 15:15:00 | 1863.00 | 1860.34 | 1868.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:15:00 | 1880.00 | 1860.34 | 1868.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1868.70 | 1862.01 | 1868.34 | EMA400 retest candle locked (from downside) |

### Cycle 201 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 1913.00 | 1879.44 | 1875.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 1932.00 | 1903.85 | 1894.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 1908.40 | 1917.88 | 1905.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 14:00:00 | 1908.40 | 1917.88 | 1905.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 1907.60 | 1915.82 | 1905.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:45:00 | 1898.60 | 1915.82 | 1905.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 1906.90 | 1914.04 | 1905.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 1885.90 | 1914.04 | 1905.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1872.10 | 1905.65 | 1902.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 1872.10 | 1905.65 | 1902.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 1871.20 | 1898.76 | 1899.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 11:15:00 | 1858.40 | 1890.69 | 1896.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 15:15:00 | 1890.00 | 1880.24 | 1888.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 15:15:00 | 1890.00 | 1880.24 | 1888.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 1890.00 | 1880.24 | 1888.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 1908.00 | 1880.24 | 1888.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1894.30 | 1883.06 | 1888.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:15:00 | 1883.50 | 1886.06 | 1889.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 10:15:00 | 1875.00 | 1871.90 | 1871.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 1875.00 | 1871.90 | 1871.86 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 11:15:00 | 1869.00 | 1871.32 | 1871.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 12:15:00 | 1854.60 | 1867.97 | 1870.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 14:15:00 | 1873.60 | 1867.45 | 1869.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 14:15:00 | 1873.60 | 1867.45 | 1869.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 1873.60 | 1867.45 | 1869.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 15:00:00 | 1873.60 | 1867.45 | 1869.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 1881.90 | 1870.34 | 1870.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 1893.00 | 1870.34 | 1870.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 1869.60 | 1869.08 | 1869.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 10:45:00 | 1871.00 | 1869.08 | 1869.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 11:15:00 | 1878.00 | 1870.87 | 1870.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 12:15:00 | 2015.10 | 1899.71 | 1883.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 15:15:00 | 1984.00 | 1986.05 | 1953.73 | EMA200 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-22 13:15:00 | 758.75 | 2023-05-23 10:15:00 | 764.35 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2023-05-22 15:15:00 | 760.25 | 2023-05-23 10:15:00 | 764.35 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2023-05-31 10:15:00 | 788.85 | 2023-06-20 09:15:00 | 818.45 | STOP_HIT | 1.00 | 3.75% |
| BUY | retest2 | 2023-06-02 11:00:00 | 788.35 | 2023-06-20 09:15:00 | 818.45 | STOP_HIT | 1.00 | 3.82% |
| BUY | retest2 | 2023-06-02 11:30:00 | 788.85 | 2023-06-20 09:15:00 | 818.45 | STOP_HIT | 1.00 | 3.75% |
| SELL | retest2 | 2023-06-22 12:15:00 | 815.00 | 2023-06-27 09:15:00 | 819.50 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2023-06-22 14:00:00 | 816.85 | 2023-06-27 09:15:00 | 819.50 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2023-07-05 10:45:00 | 843.85 | 2023-07-07 14:15:00 | 842.30 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2023-07-12 09:15:00 | 857.40 | 2023-07-13 13:15:00 | 851.90 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2023-07-12 12:30:00 | 856.40 | 2023-07-13 13:15:00 | 851.90 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2023-07-12 14:15:00 | 854.65 | 2023-07-13 13:15:00 | 851.90 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2023-07-12 15:00:00 | 855.20 | 2023-07-13 13:15:00 | 851.90 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2023-07-13 09:15:00 | 858.00 | 2023-07-19 11:15:00 | 857.60 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2023-07-13 11:00:00 | 856.40 | 2023-07-19 11:15:00 | 857.60 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2023-07-13 11:45:00 | 855.75 | 2023-07-19 11:15:00 | 857.60 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2023-07-13 12:45:00 | 856.00 | 2023-07-19 11:15:00 | 857.60 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2023-07-13 15:15:00 | 860.50 | 2023-07-19 11:15:00 | 857.60 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2023-07-19 11:00:00 | 858.85 | 2023-07-19 11:15:00 | 857.60 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2023-08-17 09:15:00 | 970.25 | 2023-08-18 11:15:00 | 954.85 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2023-08-28 09:15:00 | 1036.15 | 2023-09-12 11:15:00 | 1078.00 | STOP_HIT | 1.00 | 4.04% |
| BUY | retest2 | 2023-09-20 10:30:00 | 1129.10 | 2023-09-21 12:15:00 | 1098.75 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2023-09-20 11:45:00 | 1131.40 | 2023-09-21 12:15:00 | 1098.75 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2023-09-20 14:00:00 | 1130.25 | 2023-09-21 12:15:00 | 1098.75 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2023-09-21 09:15:00 | 1136.75 | 2023-09-21 12:15:00 | 1098.75 | STOP_HIT | 1.00 | -3.34% |
| SELL | retest2 | 2023-09-26 13:15:00 | 1093.10 | 2023-09-29 14:15:00 | 1091.00 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2023-09-27 15:00:00 | 1091.15 | 2023-09-29 14:15:00 | 1091.00 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2023-09-29 10:00:00 | 1092.25 | 2023-09-29 14:15:00 | 1091.00 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2023-09-29 13:30:00 | 1093.05 | 2023-09-29 14:15:00 | 1091.00 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2023-10-05 10:15:00 | 1063.50 | 2023-10-06 09:15:00 | 1087.95 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest1 | 2023-10-12 09:15:00 | 1116.95 | 2023-10-13 15:15:00 | 1113.75 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-10-12 12:30:00 | 1116.15 | 2023-10-13 15:15:00 | 1113.75 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2023-10-13 10:15:00 | 1120.95 | 2023-10-18 11:15:00 | 1113.10 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2023-10-13 10:45:00 | 1121.90 | 2023-10-18 11:15:00 | 1113.10 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2023-10-13 13:00:00 | 1120.90 | 2023-10-18 11:15:00 | 1113.10 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2023-10-16 09:45:00 | 1122.70 | 2023-10-18 11:15:00 | 1113.10 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2023-10-27 11:30:00 | 1028.50 | 2023-11-02 10:15:00 | 1031.55 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2023-10-27 12:15:00 | 1028.90 | 2023-11-02 10:15:00 | 1031.55 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2023-10-27 13:30:00 | 1027.70 | 2023-11-02 10:15:00 | 1031.55 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2023-10-27 14:30:00 | 1028.50 | 2023-11-02 10:15:00 | 1031.55 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2023-10-31 11:00:00 | 1026.30 | 2023-11-02 10:15:00 | 1031.55 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2023-11-01 12:30:00 | 1024.90 | 2023-11-02 10:15:00 | 1031.55 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-11-01 15:00:00 | 1027.15 | 2023-11-02 10:15:00 | 1031.55 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2023-11-02 10:00:00 | 1027.55 | 2023-11-02 10:15:00 | 1031.55 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2023-11-06 09:15:00 | 1036.55 | 2023-11-08 12:15:00 | 1032.90 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2023-11-06 12:15:00 | 1042.60 | 2023-11-08 12:15:00 | 1032.90 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2023-11-09 14:15:00 | 1034.00 | 2023-11-12 18:15:00 | 1045.35 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2023-11-09 14:45:00 | 1031.65 | 2023-11-12 18:15:00 | 1045.35 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2023-11-10 09:15:00 | 1025.50 | 2023-11-12 18:15:00 | 1045.35 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2023-11-13 12:45:00 | 1033.10 | 2023-11-15 11:15:00 | 1044.25 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2023-11-24 09:15:00 | 1101.40 | 2023-12-18 13:15:00 | 1211.54 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-29 09:15:00 | 1109.10 | 2023-12-19 12:15:00 | 1220.01 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-01 09:30:00 | 1247.80 | 2024-01-03 09:15:00 | 1219.65 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2024-01-09 09:15:00 | 1279.50 | 2024-01-12 10:15:00 | 1237.65 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2024-01-10 09:30:00 | 1263.90 | 2024-01-12 10:15:00 | 1237.65 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2024-01-10 11:15:00 | 1260.90 | 2024-01-12 10:15:00 | 1237.65 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-01-18 14:15:00 | 1224.45 | 2024-01-19 10:15:00 | 1238.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-01-18 14:45:00 | 1224.60 | 2024-01-19 10:15:00 | 1238.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-01-25 10:15:00 | 1200.55 | 2024-01-29 11:15:00 | 1243.65 | STOP_HIT | 1.00 | -3.59% |
| BUY | retest2 | 2024-02-08 12:30:00 | 1293.00 | 2024-02-12 13:15:00 | 1212.00 | STOP_HIT | 1.00 | -6.26% |
| BUY | retest2 | 2024-02-08 15:15:00 | 1285.00 | 2024-02-12 13:15:00 | 1212.00 | STOP_HIT | 1.00 | -5.68% |
| BUY | retest2 | 2024-02-09 11:30:00 | 1291.00 | 2024-02-12 13:15:00 | 1212.00 | STOP_HIT | 1.00 | -6.12% |
| SELL | retest2 | 2024-02-20 09:15:00 | 1118.65 | 2024-02-21 09:15:00 | 1136.10 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-02-20 15:15:00 | 1118.60 | 2024-02-21 09:15:00 | 1136.10 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-02-28 15:00:00 | 1155.30 | 2024-02-29 09:15:00 | 1144.75 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-03-11 13:45:00 | 1168.35 | 2024-03-14 09:15:00 | 1109.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 13:45:00 | 1168.35 | 2024-03-15 11:15:00 | 1116.95 | STOP_HIT | 0.50 | 4.40% |
| BUY | retest2 | 2024-04-04 10:30:00 | 1167.55 | 2024-04-05 13:15:00 | 1145.00 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-04-04 11:15:00 | 1169.60 | 2024-04-05 13:15:00 | 1145.00 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2024-04-04 14:15:00 | 1162.70 | 2024-04-05 13:15:00 | 1145.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-04-16 09:15:00 | 1178.65 | 2024-04-25 09:15:00 | 1296.52 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-18 10:15:00 | 1177.60 | 2024-04-25 09:15:00 | 1295.36 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-19 09:30:00 | 1178.05 | 2024-04-25 09:15:00 | 1295.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-19 11:30:00 | 1180.00 | 2024-04-25 09:15:00 | 1298.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-23 12:15:00 | 1207.95 | 2024-04-30 14:15:00 | 1269.90 | STOP_HIT | 1.00 | 5.13% |
| SELL | retest2 | 2024-05-03 10:15:00 | 1267.25 | 2024-05-08 13:15:00 | 1346.00 | STOP_HIT | 1.00 | -6.21% |
| BUY | retest2 | 2024-05-13 13:00:00 | 1407.00 | 2024-05-24 11:15:00 | 1547.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-24 10:30:00 | 1769.10 | 2024-06-25 11:15:00 | 1733.25 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-07-01 13:00:00 | 1672.75 | 2024-07-04 14:15:00 | 1668.45 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest2 | 2024-07-08 09:15:00 | 1688.75 | 2024-07-08 10:15:00 | 1660.60 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2024-07-08 10:00:00 | 1677.85 | 2024-07-08 10:15:00 | 1660.60 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-07-11 14:00:00 | 1653.30 | 2024-07-16 10:15:00 | 1655.00 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2024-08-01 15:00:00 | 1720.90 | 2024-08-02 09:15:00 | 1678.35 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest1 | 2024-08-07 11:30:00 | 1541.00 | 2024-08-08 12:15:00 | 1623.00 | STOP_HIT | 1.00 | -5.32% |
| SELL | retest2 | 2024-08-16 13:45:00 | 1579.75 | 2024-08-19 15:15:00 | 1593.20 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-08-19 10:00:00 | 1578.70 | 2024-08-19 15:15:00 | 1593.20 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2024-08-19 12:00:00 | 1576.65 | 2024-08-19 15:15:00 | 1593.20 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest1 | 2024-08-23 09:15:00 | 1628.20 | 2024-08-27 12:15:00 | 1618.00 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-09-02 09:30:00 | 1574.15 | 2024-09-04 14:15:00 | 1608.85 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2024-09-03 09:30:00 | 1569.05 | 2024-09-04 14:15:00 | 1608.85 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2024-09-03 10:15:00 | 1570.55 | 2024-09-04 14:15:00 | 1608.85 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2024-09-03 12:00:00 | 1573.25 | 2024-09-04 14:15:00 | 1608.85 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2024-09-17 11:45:00 | 1605.15 | 2024-09-18 12:15:00 | 1574.95 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2024-09-17 12:45:00 | 1604.55 | 2024-09-18 12:15:00 | 1574.95 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-09-17 13:30:00 | 1604.05 | 2024-09-18 12:15:00 | 1574.95 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-09-17 14:30:00 | 1604.00 | 2024-09-18 12:15:00 | 1574.95 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-09-18 09:15:00 | 1597.90 | 2024-09-18 12:15:00 | 1574.95 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-09-18 12:00:00 | 1596.85 | 2024-09-18 12:15:00 | 1574.95 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-09-30 14:15:00 | 1513.55 | 2024-10-07 10:15:00 | 1437.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 11:00:00 | 1514.00 | 2024-10-07 10:15:00 | 1438.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 14:15:00 | 1513.55 | 2024-10-08 10:15:00 | 1462.85 | STOP_HIT | 0.50 | 3.35% |
| SELL | retest2 | 2024-10-03 11:00:00 | 1514.00 | 2024-10-08 10:15:00 | 1462.85 | STOP_HIT | 0.50 | 3.38% |
| BUY | retest2 | 2024-10-11 09:30:00 | 1505.85 | 2024-10-11 11:15:00 | 1486.85 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-10-14 15:15:00 | 1490.00 | 2024-10-21 09:15:00 | 1478.95 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2024-10-15 10:00:00 | 1488.80 | 2024-10-21 09:15:00 | 1478.95 | STOP_HIT | 1.00 | 0.66% |
| SELL | retest2 | 2024-10-15 11:00:00 | 1489.95 | 2024-10-21 09:15:00 | 1478.95 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2024-10-15 12:30:00 | 1489.35 | 2024-10-21 09:15:00 | 1478.95 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2024-10-16 10:45:00 | 1470.65 | 2024-10-21 09:15:00 | 1478.95 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2024-10-16 15:15:00 | 1472.40 | 2024-10-21 09:15:00 | 1478.95 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2024-10-18 09:45:00 | 1470.35 | 2024-10-21 09:15:00 | 1478.95 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-10-18 11:45:00 | 1474.10 | 2024-10-21 09:15:00 | 1478.95 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2024-10-28 11:30:00 | 1373.25 | 2024-10-30 11:15:00 | 1411.50 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2024-10-28 12:30:00 | 1371.25 | 2024-10-30 11:15:00 | 1411.50 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2024-10-28 13:30:00 | 1370.75 | 2024-10-30 11:15:00 | 1411.50 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2024-11-04 14:30:00 | 1418.70 | 2024-11-08 11:15:00 | 1428.55 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2024-11-14 15:00:00 | 1326.55 | 2024-11-19 13:15:00 | 1350.00 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-11-29 09:15:00 | 1340.70 | 2024-12-02 09:15:00 | 1308.50 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2024-11-29 10:30:00 | 1335.00 | 2024-12-02 09:15:00 | 1308.50 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-11-29 12:00:00 | 1333.95 | 2024-12-02 09:15:00 | 1308.50 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest1 | 2024-12-31 09:30:00 | 1292.85 | 2024-12-31 12:15:00 | 1305.45 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-01-01 09:15:00 | 1296.80 | 2025-01-02 13:15:00 | 1313.30 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-01-01 14:00:00 | 1297.35 | 2025-01-02 13:15:00 | 1313.30 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-01-02 09:15:00 | 1297.95 | 2025-01-02 13:15:00 | 1313.30 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-01-02 09:45:00 | 1295.85 | 2025-01-02 13:15:00 | 1313.30 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-01-07 11:00:00 | 1265.55 | 2025-01-13 09:15:00 | 1202.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-07 11:45:00 | 1265.40 | 2025-01-13 09:15:00 | 1202.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-07 13:45:00 | 1264.70 | 2025-01-13 09:15:00 | 1201.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 1256.40 | 2025-01-13 09:15:00 | 1193.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-07 11:00:00 | 1265.55 | 2025-01-14 10:15:00 | 1200.55 | STOP_HIT | 0.50 | 5.14% |
| SELL | retest2 | 2025-01-07 11:45:00 | 1265.40 | 2025-01-14 10:15:00 | 1200.55 | STOP_HIT | 0.50 | 5.12% |
| SELL | retest2 | 2025-01-07 13:45:00 | 1264.70 | 2025-01-14 10:15:00 | 1200.55 | STOP_HIT | 0.50 | 5.07% |
| SELL | retest2 | 2025-01-08 09:15:00 | 1256.40 | 2025-01-14 10:15:00 | 1200.55 | STOP_HIT | 0.50 | 4.45% |
| SELL | retest2 | 2025-01-15 12:15:00 | 1185.40 | 2025-01-16 09:15:00 | 1216.00 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2025-02-05 11:45:00 | 1157.35 | 2025-02-12 09:15:00 | 1099.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-05 12:30:00 | 1161.25 | 2025-02-12 09:15:00 | 1103.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-05 11:45:00 | 1157.35 | 2025-02-13 09:15:00 | 1045.12 | TARGET_HIT | 0.50 | 9.70% |
| SELL | retest2 | 2025-02-05 12:30:00 | 1161.25 | 2025-02-14 09:15:00 | 1097.70 | STOP_HIT | 0.50 | 5.47% |
| SELL | retest2 | 2025-03-03 11:15:00 | 1008.55 | 2025-03-03 14:15:00 | 1031.75 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-03-25 13:45:00 | 1184.30 | 2025-03-27 09:15:00 | 1172.10 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-03-25 15:00:00 | 1187.65 | 2025-03-27 09:15:00 | 1172.10 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-03-26 09:15:00 | 1199.20 | 2025-03-27 09:15:00 | 1172.10 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-03-26 14:00:00 | 1182.90 | 2025-03-27 09:15:00 | 1172.10 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-04-01 11:15:00 | 1149.50 | 2025-04-03 09:15:00 | 1092.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 12:45:00 | 1147.75 | 2025-04-03 09:15:00 | 1090.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 14:15:00 | 1149.00 | 2025-04-03 09:15:00 | 1091.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-02 09:15:00 | 1149.00 | 2025-04-03 09:15:00 | 1091.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-02 14:30:00 | 1154.10 | 2025-04-03 09:15:00 | 1096.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 11:15:00 | 1149.50 | 2025-04-04 09:15:00 | 1034.55 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-01 12:45:00 | 1147.75 | 2025-04-04 09:15:00 | 1032.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-01 14:15:00 | 1149.00 | 2025-04-04 09:15:00 | 1034.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-02 09:15:00 | 1149.00 | 2025-04-04 09:15:00 | 1034.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-02 14:30:00 | 1154.10 | 2025-04-04 09:15:00 | 1038.69 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-04-17 11:00:00 | 1058.00 | 2025-04-25 10:15:00 | 1088.80 | STOP_HIT | 1.00 | 2.91% |
| SELL | retest2 | 2025-05-02 11:15:00 | 1104.00 | 2025-05-05 13:15:00 | 1117.00 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-05-02 12:15:00 | 1101.10 | 2025-05-05 13:15:00 | 1117.00 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-05-02 14:15:00 | 1104.30 | 2025-05-05 13:15:00 | 1117.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-05-02 15:15:00 | 1102.50 | 2025-05-05 13:15:00 | 1117.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-05-22 14:15:00 | 1233.90 | 2025-05-22 15:15:00 | 1240.20 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-05-27 11:45:00 | 1257.00 | 2025-05-30 11:15:00 | 1244.30 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-05-28 12:45:00 | 1256.70 | 2025-05-30 11:15:00 | 1244.30 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-05-28 13:30:00 | 1257.80 | 2025-05-30 11:15:00 | 1244.30 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-05-29 09:15:00 | 1266.90 | 2025-05-30 11:15:00 | 1244.30 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest1 | 2025-06-11 09:15:00 | 1348.30 | 2025-06-12 09:15:00 | 1327.50 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-06-18 10:30:00 | 1301.90 | 2025-06-23 14:15:00 | 1295.20 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest2 | 2025-06-18 11:45:00 | 1298.50 | 2025-06-23 14:15:00 | 1295.20 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2025-06-18 12:15:00 | 1299.50 | 2025-06-23 14:15:00 | 1295.20 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2025-06-19 09:45:00 | 1299.90 | 2025-06-23 14:15:00 | 1295.20 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2025-06-25 09:15:00 | 1303.80 | 2025-06-25 11:15:00 | 1283.70 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-06-25 11:00:00 | 1296.60 | 2025-06-25 11:15:00 | 1283.70 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-07-07 11:30:00 | 1308.40 | 2025-07-08 09:15:00 | 1294.30 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-07-07 13:00:00 | 1307.50 | 2025-07-08 09:15:00 | 1294.30 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-07-07 13:45:00 | 1308.00 | 2025-07-08 09:15:00 | 1294.30 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-07-07 15:00:00 | 1308.90 | 2025-07-08 09:15:00 | 1294.30 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-07-22 09:15:00 | 1215.00 | 2025-07-23 14:15:00 | 1224.50 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-07-24 09:45:00 | 1218.00 | 2025-07-24 10:15:00 | 1219.60 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-07-28 12:00:00 | 1209.00 | 2025-07-29 14:15:00 | 1222.10 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-07-28 12:30:00 | 1208.50 | 2025-07-29 14:15:00 | 1222.10 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest1 | 2025-07-31 09:15:00 | 1157.80 | 2025-08-04 11:15:00 | 1166.70 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-08-06 11:30:00 | 1147.30 | 2025-08-11 09:15:00 | 1159.50 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-08-06 13:30:00 | 1138.20 | 2025-08-11 10:15:00 | 1170.00 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-08-07 12:15:00 | 1142.90 | 2025-08-11 10:15:00 | 1170.00 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-08-07 14:00:00 | 1147.80 | 2025-08-11 10:15:00 | 1170.00 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-08-08 11:30:00 | 1143.40 | 2025-08-11 10:15:00 | 1170.00 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-08-25 12:15:00 | 1142.20 | 2025-09-02 09:15:00 | 1135.30 | STOP_HIT | 1.00 | 0.60% |
| SELL | retest2 | 2025-08-25 14:45:00 | 1143.30 | 2025-09-02 09:15:00 | 1135.30 | STOP_HIT | 1.00 | 0.70% |
| BUY | retest2 | 2025-09-05 09:15:00 | 1138.60 | 2025-09-16 09:15:00 | 1252.46 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-05 13:30:00 | 1136.20 | 2025-09-16 09:15:00 | 1249.82 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-29 15:15:00 | 1192.20 | 2025-10-01 10:15:00 | 1217.90 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-09-30 11:30:00 | 1196.50 | 2025-10-01 10:15:00 | 1217.90 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-09-30 13:15:00 | 1201.10 | 2025-10-01 10:15:00 | 1217.90 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-10-03 15:15:00 | 1222.00 | 2025-10-08 09:15:00 | 1205.70 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-10-06 09:30:00 | 1221.20 | 2025-10-08 09:15:00 | 1205.70 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-10-06 12:15:00 | 1220.90 | 2025-10-08 09:15:00 | 1205.70 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-10-06 13:15:00 | 1225.30 | 2025-10-08 09:15:00 | 1205.70 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-10-23 09:15:00 | 1288.90 | 2025-11-06 10:15:00 | 1317.20 | STOP_HIT | 1.00 | 2.20% |
| SELL | retest2 | 2025-11-10 10:30:00 | 1309.20 | 2025-11-10 13:15:00 | 1329.50 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-11-14 15:00:00 | 1398.00 | 2025-11-18 10:15:00 | 1375.30 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-11-17 09:30:00 | 1398.00 | 2025-11-18 10:15:00 | 1375.30 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-11-28 11:15:00 | 1431.40 | 2025-12-02 09:15:00 | 1410.40 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-12-04 12:30:00 | 1398.80 | 2025-12-11 10:15:00 | 1397.50 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-12-04 13:30:00 | 1397.60 | 2025-12-11 10:15:00 | 1397.50 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-12-05 09:15:00 | 1398.60 | 2025-12-11 10:15:00 | 1397.50 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2025-12-08 09:15:00 | 1396.00 | 2025-12-11 10:15:00 | 1397.50 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-12-08 11:30:00 | 1381.40 | 2025-12-11 10:15:00 | 1397.50 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-12-09 12:45:00 | 1382.80 | 2025-12-11 10:15:00 | 1397.50 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-12-09 14:15:00 | 1380.20 | 2025-12-11 10:15:00 | 1397.50 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-12-10 09:15:00 | 1383.80 | 2025-12-11 10:15:00 | 1397.50 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-12-31 09:15:00 | 1478.00 | 2026-01-08 11:15:00 | 1462.60 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-12-31 10:45:00 | 1459.70 | 2026-01-08 11:15:00 | 1462.60 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2026-01-13 14:15:00 | 1448.60 | 2026-01-13 14:15:00 | 1461.40 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2026-01-27 09:15:00 | 1416.50 | 2026-01-27 10:15:00 | 1406.10 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2026-01-30 10:45:00 | 1448.00 | 2026-02-01 12:15:00 | 1410.90 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2026-02-01 09:15:00 | 1447.70 | 2026-02-01 12:15:00 | 1410.90 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2026-02-09 09:15:00 | 1589.80 | 2026-02-12 13:15:00 | 1748.78 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-05 11:15:00 | 1874.40 | 2026-03-05 14:15:00 | 1895.90 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2026-03-05 13:30:00 | 1877.40 | 2026-03-05 14:15:00 | 1895.90 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2026-04-07 09:15:00 | 1626.80 | 2026-04-08 09:15:00 | 1776.90 | STOP_HIT | 1.00 | -9.23% |
| SELL | retest2 | 2026-04-07 13:45:00 | 1660.90 | 2026-04-08 09:15:00 | 1776.90 | STOP_HIT | 1.00 | -6.98% |
| BUY | retest2 | 2026-04-13 11:30:00 | 1785.50 | 2026-04-23 12:15:00 | 1870.00 | STOP_HIT | 1.00 | 4.73% |
| SELL | retest2 | 2026-05-04 11:15:00 | 1883.50 | 2026-05-06 10:15:00 | 1875.00 | STOP_HIT | 1.00 | 0.45% |
