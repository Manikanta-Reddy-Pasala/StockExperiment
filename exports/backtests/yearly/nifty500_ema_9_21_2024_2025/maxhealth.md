# Max Healthcare Institute Ltd. (MAXHEALTH)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1013.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 157 |
| ALERT1 | 100 |
| ALERT2 | 100 |
| ALERT2_SKIP | 50 |
| ALERT3 | 258 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 111 |
| PARTIAL | 15 |
| TARGET_HIT | 8 |
| STOP_HIT | 104 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 127 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 67 / 60
- **Target hits / Stop hits / Partials:** 8 / 104 / 15
- **Avg / median % per leg:** 1.25% / 0.26%
- **Sum % (uncompounded):** 159.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 71 | 35 | 49.3% | 7 | 63 | 1 | 0.97% | 68.8% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 0 | 4 | 1 | 1.40% | 7.0% |
| BUY @ 3rd Alert (retest2) | 66 | 31 | 47.0% | 7 | 59 | 0 | 0.94% | 61.8% |
| SELL (all) | 56 | 32 | 57.1% | 1 | 41 | 14 | 1.62% | 90.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 56 | 32 | 57.1% | 1 | 41 | 14 | 1.62% | 90.5% |
| retest1 (combined) | 5 | 4 | 80.0% | 0 | 4 | 1 | 1.40% | 7.0% |
| retest2 (combined) | 122 | 63 | 51.6% | 8 | 100 | 14 | 1.25% | 152.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 809.00 | 803.37 | 802.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 12:15:00 | 822.60 | 808.50 | 805.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 09:15:00 | 835.90 | 837.12 | 829.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 10:00:00 | 835.90 | 837.12 | 829.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 837.25 | 841.49 | 837.08 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2024-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 13:15:00 | 825.05 | 834.21 | 834.73 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 09:15:00 | 853.50 | 837.81 | 836.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 11:15:00 | 865.00 | 845.98 | 840.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 13:15:00 | 845.40 | 847.89 | 842.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-22 14:00:00 | 845.40 | 847.89 | 842.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 833.70 | 845.05 | 841.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 15:00:00 | 833.70 | 845.05 | 841.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 839.00 | 843.84 | 841.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 09:15:00 | 851.85 | 843.84 | 841.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 09:15:00 | 829.45 | 840.96 | 840.26 | SL hit (close<static) qty=1.00 sl=830.20 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 10:15:00 | 822.70 | 837.31 | 838.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 11:15:00 | 816.25 | 833.10 | 836.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 10:15:00 | 800.00 | 797.05 | 807.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-27 11:00:00 | 800.00 | 797.05 | 807.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 13:15:00 | 807.85 | 800.40 | 806.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 13:45:00 | 807.50 | 800.40 | 806.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 14:15:00 | 803.00 | 800.92 | 806.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 15:15:00 | 808.00 | 800.92 | 806.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 15:15:00 | 808.00 | 802.33 | 806.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 09:15:00 | 805.35 | 802.33 | 806.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 798.25 | 801.52 | 805.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 10:15:00 | 797.20 | 801.52 | 805.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 09:15:00 | 757.34 | 778.79 | 785.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 760.00 | 759.98 | 770.70 | SL hit (close>ema200) qty=0.50 sl=759.98 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 13:15:00 | 776.90 | 770.91 | 770.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-04 14:15:00 | 780.15 | 772.76 | 771.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 09:15:00 | 822.40 | 826.41 | 814.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-07 09:15:00 | 822.40 | 826.41 | 814.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 822.40 | 826.41 | 814.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 09:15:00 | 846.00 | 824.04 | 818.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-10 15:15:00 | 810.00 | 818.67 | 818.65 | SL hit (close<static) qty=1.00 sl=813.60 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-11 09:15:00 | 805.50 | 816.03 | 817.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-11 12:15:00 | 803.75 | 810.33 | 814.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-12 09:15:00 | 843.60 | 814.32 | 814.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-12 09:15:00 | 843.60 | 814.32 | 814.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 843.60 | 814.32 | 814.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-12 10:00:00 | 843.60 | 814.32 | 814.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 10:15:00 | 843.00 | 820.05 | 817.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-12 11:15:00 | 857.85 | 827.61 | 820.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-18 09:15:00 | 911.95 | 913.72 | 893.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 09:15:00 | 897.40 | 905.24 | 898.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 897.40 | 905.24 | 898.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 10:45:00 | 919.00 | 906.73 | 899.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 13:30:00 | 903.20 | 904.22 | 900.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 14:00:00 | 904.55 | 904.22 | 900.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 14:45:00 | 903.30 | 902.90 | 900.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 900.00 | 902.32 | 900.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:15:00 | 886.25 | 902.32 | 900.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 895.10 | 900.87 | 899.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:30:00 | 890.25 | 900.87 | 899.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 900.50 | 899.96 | 899.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 12:45:00 | 907.55 | 900.60 | 899.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 12:15:00 | 907.40 | 917.02 | 917.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 12:15:00 | 907.40 | 917.02 | 917.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 14:15:00 | 901.65 | 911.86 | 915.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 14:15:00 | 895.80 | 894.66 | 902.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-25 15:00:00 | 895.80 | 894.66 | 902.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 928.40 | 891.10 | 893.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:45:00 | 928.45 | 891.10 | 893.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 10:15:00 | 924.80 | 897.84 | 896.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 12:15:00 | 933.15 | 909.18 | 902.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 927.00 | 942.25 | 933.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 09:15:00 | 927.00 | 942.25 | 933.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 927.00 | 942.25 | 933.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:00:00 | 927.00 | 942.25 | 933.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 923.45 | 938.49 | 932.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:45:00 | 922.55 | 938.49 | 932.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 13:15:00 | 910.20 | 926.63 | 928.38 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 10:15:00 | 933.65 | 920.12 | 919.56 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 14:15:00 | 917.40 | 921.97 | 922.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 10:15:00 | 905.15 | 917.13 | 920.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-12 10:15:00 | 900.45 | 898.66 | 904.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-12 10:45:00 | 900.40 | 898.66 | 904.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 14:15:00 | 901.40 | 899.48 | 902.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 15:00:00 | 901.40 | 899.48 | 902.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 909.15 | 901.47 | 903.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:00:00 | 909.15 | 901.47 | 903.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 909.25 | 903.02 | 903.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:30:00 | 910.70 | 903.02 | 903.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2024-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 11:15:00 | 915.55 | 905.53 | 904.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 14:15:00 | 923.75 | 911.18 | 907.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 11:15:00 | 911.15 | 913.28 | 910.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 12:00:00 | 911.15 | 913.28 | 910.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 901.00 | 912.90 | 911.31 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2024-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 11:15:00 | 904.30 | 909.33 | 909.85 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 13:15:00 | 926.60 | 913.07 | 911.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-18 14:15:00 | 936.90 | 917.83 | 913.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 918.00 | 919.97 | 915.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 09:15:00 | 918.00 | 919.97 | 915.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 918.00 | 919.97 | 915.59 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2024-07-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 12:15:00 | 898.35 | 911.92 | 912.75 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-07-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-19 14:15:00 | 922.80 | 913.90 | 913.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 09:15:00 | 936.85 | 919.33 | 916.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-22 15:15:00 | 927.00 | 928.92 | 923.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-23 09:15:00 | 931.00 | 928.92 | 923.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 919.90 | 927.12 | 923.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 10:00:00 | 919.90 | 927.12 | 923.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 920.00 | 925.69 | 922.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 10:30:00 | 920.00 | 925.69 | 922.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 935.90 | 926.82 | 923.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 10:45:00 | 942.10 | 933.07 | 928.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-25 12:15:00 | 922.00 | 931.71 | 932.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 12:15:00 | 922.00 | 931.71 | 932.56 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 956.60 | 934.30 | 933.12 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 13:15:00 | 932.65 | 936.90 | 937.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 09:15:00 | 923.45 | 933.48 | 935.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-31 14:15:00 | 922.00 | 920.48 | 924.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 14:15:00 | 922.00 | 920.48 | 924.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 922.00 | 920.48 | 924.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 922.00 | 920.48 | 924.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 931.50 | 923.17 | 925.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 14:00:00 | 916.15 | 924.04 | 925.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 15:15:00 | 920.00 | 923.97 | 925.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 13:15:00 | 874.00 | 896.60 | 908.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 14:15:00 | 870.34 | 890.51 | 904.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-08 09:15:00 | 866.85 | 860.47 | 870.29 | SL hit (close>ema200) qty=0.50 sl=860.47 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 885.00 | 875.01 | 874.26 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 10:15:00 | 866.20 | 875.81 | 876.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 09:15:00 | 858.60 | 866.53 | 869.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 12:15:00 | 873.60 | 866.65 | 868.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 12:15:00 | 873.60 | 866.65 | 868.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 873.60 | 866.65 | 868.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 13:00:00 | 873.60 | 866.65 | 868.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 13:15:00 | 871.50 | 867.62 | 869.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 14:00:00 | 871.50 | 867.62 | 869.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 15:15:00 | 866.50 | 867.91 | 869.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:15:00 | 876.90 | 867.91 | 869.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 09:15:00 | 883.15 | 870.96 | 870.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 11:15:00 | 888.95 | 876.14 | 872.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-16 15:15:00 | 880.00 | 880.81 | 876.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 09:15:00 | 875.00 | 880.81 | 876.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 873.60 | 879.37 | 876.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:00:00 | 873.60 | 879.37 | 876.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 869.30 | 877.35 | 875.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:45:00 | 868.00 | 877.35 | 875.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 13:15:00 | 871.05 | 874.30 | 874.47 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 09:15:00 | 878.45 | 875.32 | 874.91 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 10:15:00 | 871.20 | 874.52 | 874.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-21 12:15:00 | 869.40 | 873.09 | 874.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-22 10:15:00 | 875.80 | 871.22 | 872.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 10:15:00 | 875.80 | 871.22 | 872.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 875.80 | 871.22 | 872.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-22 10:45:00 | 876.10 | 871.22 | 872.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 11:15:00 | 879.00 | 872.78 | 873.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-22 12:00:00 | 879.00 | 872.78 | 873.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 869.05 | 870.09 | 871.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 867.90 | 870.09 | 871.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 868.30 | 869.73 | 871.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 11:15:00 | 859.95 | 868.56 | 870.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 12:30:00 | 861.25 | 858.20 | 858.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 13:15:00 | 860.95 | 858.20 | 858.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 13:15:00 | 863.30 | 859.22 | 858.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 13:15:00 | 863.30 | 859.22 | 858.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-29 15:15:00 | 869.00 | 864.51 | 862.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 09:15:00 | 859.65 | 863.54 | 862.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 859.65 | 863.54 | 862.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 859.65 | 863.54 | 862.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:00:00 | 859.65 | 863.54 | 862.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 863.50 | 863.53 | 862.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:30:00 | 860.35 | 863.53 | 862.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 865.05 | 863.83 | 862.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 12:30:00 | 869.15 | 865.16 | 863.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-30 14:15:00 | 860.65 | 865.67 | 863.85 | SL hit (close<static) qty=1.00 sl=861.15 alert=retest2 |

### Cycle 28 — SELL (started 2024-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 12:15:00 | 880.70 | 887.26 | 887.54 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 894.90 | 887.03 | 886.86 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 14:15:00 | 877.65 | 887.15 | 887.25 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 10:15:00 | 898.10 | 889.07 | 888.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 11:15:00 | 899.60 | 891.18 | 889.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 884.05 | 890.84 | 889.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 13:15:00 | 884.05 | 890.84 | 889.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 884.05 | 890.84 | 889.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 884.05 | 890.84 | 889.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 875.95 | 887.86 | 888.14 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 09:15:00 | 905.15 | 890.77 | 889.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 10:15:00 | 916.50 | 895.91 | 891.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 09:15:00 | 901.75 | 906.81 | 900.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 09:15:00 | 901.75 | 906.81 | 900.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 901.75 | 906.81 | 900.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:30:00 | 899.95 | 906.81 | 900.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 10:15:00 | 906.50 | 906.75 | 901.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 11:15:00 | 909.55 | 906.75 | 901.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 13:00:00 | 915.20 | 908.51 | 902.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 09:15:00 | 929.00 | 908.98 | 904.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-20 09:15:00 | 1000.50 | 975.68 | 959.68 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 10:15:00 | 992.65 | 1028.30 | 1029.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 13:15:00 | 982.75 | 1008.97 | 1019.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 13:15:00 | 996.65 | 995.71 | 1005.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 14:00:00 | 996.65 | 995.71 | 1005.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 995.00 | 991.74 | 997.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 14:45:00 | 995.25 | 991.74 | 997.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 994.00 | 992.19 | 997.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 09:15:00 | 970.30 | 992.19 | 997.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 09:15:00 | 975.50 | 984.68 | 989.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 09:15:00 | 926.72 | 968.01 | 977.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 921.78 | 946.85 | 960.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-07 09:15:00 | 941.50 | 934.41 | 945.90 | SL hit (close>ema200) qty=0.50 sl=934.41 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 13:15:00 | 949.00 | 940.12 | 939.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 14:15:00 | 961.40 | 944.38 | 941.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 15:15:00 | 959.10 | 960.38 | 953.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 09:15:00 | 949.30 | 960.38 | 953.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 950.00 | 958.30 | 953.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 09:45:00 | 947.70 | 958.30 | 953.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 947.30 | 956.10 | 952.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 10:30:00 | 947.10 | 956.10 | 952.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 947.60 | 951.50 | 951.48 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2024-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 10:15:00 | 942.60 | 949.72 | 950.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 11:15:00 | 937.80 | 947.33 | 949.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 09:15:00 | 943.65 | 942.92 | 946.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 09:15:00 | 943.65 | 942.92 | 946.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 943.65 | 942.92 | 946.10 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2024-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 09:15:00 | 950.00 | 946.41 | 946.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 11:15:00 | 969.00 | 951.51 | 948.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 11:15:00 | 963.25 | 965.37 | 958.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-16 11:45:00 | 963.05 | 965.37 | 958.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 12:15:00 | 959.05 | 964.10 | 958.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 12:45:00 | 962.55 | 964.10 | 958.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 13:15:00 | 962.40 | 963.76 | 959.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 13:30:00 | 960.60 | 963.76 | 959.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 964.00 | 963.97 | 960.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:15:00 | 955.30 | 963.97 | 960.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 952.65 | 961.70 | 959.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 952.65 | 961.70 | 959.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 942.05 | 957.77 | 957.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 932.30 | 947.87 | 952.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 09:15:00 | 950.05 | 939.52 | 943.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 09:15:00 | 950.05 | 939.52 | 943.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 950.05 | 939.52 | 943.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 10:15:00 | 953.45 | 939.52 | 943.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 950.70 | 941.76 | 943.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 10:30:00 | 951.50 | 941.76 | 943.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2024-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 13:15:00 | 950.45 | 946.00 | 945.57 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-10-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 09:15:00 | 930.85 | 944.22 | 944.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 10:15:00 | 925.60 | 940.50 | 943.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 12:15:00 | 920.20 | 917.68 | 926.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 13:00:00 | 920.20 | 917.68 | 926.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 924.00 | 911.05 | 915.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-25 09:30:00 | 931.20 | 911.05 | 915.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 923.30 | 913.50 | 916.29 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2024-10-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-25 13:15:00 | 924.95 | 918.90 | 918.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-25 14:15:00 | 936.15 | 922.35 | 919.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 09:15:00 | 983.35 | 983.65 | 966.70 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 15:00:00 | 995.45 | 986.76 | 974.58 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 09:15:00 | 1045.22 | 1012.07 | 998.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-11-05 09:15:00 | 1023.45 | 1024.33 | 1012.45 | SL hit (close<ema200) qty=0.50 sl=1024.33 alert=retest1 |

### Cycle 42 — SELL (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 13:15:00 | 1033.05 | 1060.42 | 1063.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 14:15:00 | 1026.50 | 1053.63 | 1060.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 13:15:00 | 1041.85 | 1041.05 | 1049.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-12 14:00:00 | 1041.85 | 1041.05 | 1049.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 995.30 | 1007.02 | 1017.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 10:15:00 | 984.90 | 997.20 | 1003.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 09:15:00 | 983.05 | 991.80 | 997.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 10:00:00 | 985.50 | 995.30 | 997.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 12:15:00 | 999.85 | 989.95 | 989.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2024-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 12:15:00 | 999.85 | 989.95 | 989.94 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 15:15:00 | 985.30 | 989.54 | 989.86 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 09:15:00 | 992.40 | 990.11 | 990.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 11:15:00 | 999.00 | 992.53 | 991.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 09:15:00 | 992.50 | 999.63 | 995.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 09:15:00 | 992.50 | 999.63 | 995.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 992.50 | 999.63 | 995.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:00:00 | 992.50 | 999.63 | 995.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 989.95 | 997.70 | 995.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:45:00 | 990.15 | 997.70 | 995.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 12:15:00 | 982.00 | 993.23 | 993.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 14:15:00 | 975.60 | 987.84 | 991.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 981.20 | 980.82 | 984.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 09:15:00 | 981.20 | 980.82 | 984.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 981.20 | 980.82 | 984.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:30:00 | 983.25 | 980.82 | 984.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 988.70 | 982.26 | 984.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:30:00 | 990.60 | 982.26 | 984.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 12:15:00 | 993.45 | 984.50 | 985.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 13:00:00 | 993.45 | 984.50 | 985.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2024-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 13:15:00 | 1001.30 | 987.86 | 986.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 14:15:00 | 1022.55 | 994.80 | 990.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 09:15:00 | 1084.50 | 1084.72 | 1064.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-06 09:30:00 | 1084.90 | 1084.72 | 1064.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 1109.25 | 1096.26 | 1080.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 10:15:00 | 1111.50 | 1096.26 | 1080.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 09:30:00 | 1115.70 | 1114.30 | 1106.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-19 14:15:00 | 1181.50 | 1189.72 | 1190.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2024-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 14:15:00 | 1181.50 | 1189.72 | 1190.01 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 11:15:00 | 1194.95 | 1190.15 | 1189.87 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 12:15:00 | 1187.25 | 1189.57 | 1189.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 13:15:00 | 1180.20 | 1187.70 | 1188.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 12:15:00 | 1141.55 | 1139.78 | 1148.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-26 12:45:00 | 1141.50 | 1139.78 | 1148.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 1158.70 | 1144.83 | 1149.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 1158.70 | 1144.83 | 1149.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 1152.85 | 1146.43 | 1149.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 1144.00 | 1146.43 | 1149.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 1130.45 | 1131.73 | 1137.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 13:30:00 | 1132.90 | 1131.73 | 1137.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 1174.15 | 1140.22 | 1140.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 1174.15 | 1140.22 | 1140.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 1140.00 | 1140.17 | 1140.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 10:15:00 | 1137.90 | 1140.26 | 1140.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-31 10:15:00 | 1142.00 | 1140.61 | 1140.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2024-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 10:15:00 | 1142.00 | 1140.61 | 1140.57 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 12:15:00 | 1139.50 | 1140.41 | 1140.49 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 13:15:00 | 1144.00 | 1141.13 | 1140.81 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 14:15:00 | 1129.25 | 1138.75 | 1139.75 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 09:15:00 | 1140.55 | 1139.53 | 1139.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 11:15:00 | 1156.85 | 1143.68 | 1141.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 11:15:00 | 1171.00 | 1171.88 | 1163.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 11:45:00 | 1170.75 | 1171.88 | 1163.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 12:15:00 | 1156.75 | 1168.85 | 1163.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 13:00:00 | 1156.75 | 1168.85 | 1163.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 1152.15 | 1165.51 | 1162.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 13:45:00 | 1152.85 | 1165.51 | 1162.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 1180.65 | 1165.96 | 1162.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 10:15:00 | 1182.60 | 1165.96 | 1162.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 10:00:00 | 1184.75 | 1202.92 | 1198.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 10:30:00 | 1182.65 | 1199.18 | 1197.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 11:00:00 | 1184.20 | 1199.18 | 1197.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-10 11:15:00 | 1169.00 | 1193.14 | 1194.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 11:15:00 | 1169.00 | 1193.14 | 1194.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 12:15:00 | 1158.30 | 1186.17 | 1191.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 11:15:00 | 1036.40 | 1030.02 | 1042.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-20 12:00:00 | 1036.40 | 1030.02 | 1042.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 12:15:00 | 1045.05 | 1033.03 | 1042.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 13:00:00 | 1045.05 | 1033.03 | 1042.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 1054.45 | 1037.31 | 1043.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:00:00 | 1054.45 | 1037.31 | 1043.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 09:15:00 | 1074.75 | 1052.79 | 1049.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-21 11:15:00 | 1081.25 | 1061.89 | 1054.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-22 09:15:00 | 1070.00 | 1072.83 | 1063.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 09:15:00 | 1070.00 | 1072.83 | 1063.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 1070.00 | 1072.83 | 1063.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:45:00 | 1069.70 | 1072.83 | 1063.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 11:15:00 | 1054.85 | 1068.78 | 1063.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:00:00 | 1054.85 | 1068.78 | 1063.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 12:15:00 | 1057.80 | 1066.58 | 1063.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 13:00:00 | 1057.80 | 1066.58 | 1063.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 13:15:00 | 1066.00 | 1066.47 | 1063.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 14:15:00 | 1070.10 | 1066.47 | 1063.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 15:15:00 | 1069.95 | 1066.40 | 1063.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 09:30:00 | 1068.15 | 1067.28 | 1064.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 11:15:00 | 1067.60 | 1067.21 | 1064.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 1069.85 | 1075.82 | 1071.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 1069.85 | 1075.82 | 1071.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 1068.95 | 1074.45 | 1070.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:30:00 | 1070.30 | 1074.45 | 1070.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 11:15:00 | 1076.00 | 1074.76 | 1071.34 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-24 14:15:00 | 1055.70 | 1066.90 | 1068.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 14:15:00 | 1055.70 | 1066.90 | 1068.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 15:15:00 | 1052.10 | 1063.94 | 1066.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 1039.35 | 1031.97 | 1041.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 12:15:00 | 1039.35 | 1031.97 | 1041.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 1039.35 | 1031.97 | 1041.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 1039.35 | 1031.97 | 1041.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 1030.15 | 1025.05 | 1030.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 11:30:00 | 1017.30 | 1023.41 | 1028.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 12:00:00 | 1015.60 | 1023.41 | 1028.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 12:30:00 | 1016.00 | 1025.12 | 1028.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 13:30:00 | 1016.20 | 1021.78 | 1027.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 1040.15 | 1025.45 | 1028.26 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-30 14:15:00 | 1040.15 | 1025.45 | 1028.26 | SL hit (close>static) qty=1.00 sl=1040.10 alert=retest2 |

### Cycle 59 — BUY (started 2025-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 15:15:00 | 1049.40 | 1030.24 | 1030.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 12:15:00 | 1059.55 | 1040.24 | 1035.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 13:15:00 | 1170.15 | 1176.26 | 1154.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-05 14:00:00 | 1170.15 | 1176.26 | 1154.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 1139.90 | 1167.38 | 1155.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:15:00 | 1138.25 | 1167.38 | 1155.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 1143.35 | 1162.58 | 1154.32 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 13:15:00 | 1125.50 | 1147.80 | 1148.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 1120.40 | 1137.20 | 1143.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 14:15:00 | 1144.40 | 1133.37 | 1138.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 14:15:00 | 1144.40 | 1133.37 | 1138.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 1144.40 | 1133.37 | 1138.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 15:00:00 | 1144.40 | 1133.37 | 1138.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 1143.55 | 1135.41 | 1138.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 1107.00 | 1135.41 | 1138.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 1051.65 | 1077.44 | 1101.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-12 09:15:00 | 996.30 | 1028.74 | 1061.36 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 61 — BUY (started 2025-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 09:15:00 | 1033.50 | 1008.18 | 1006.40 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 11:15:00 | 1016.60 | 1023.97 | 1024.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 13:15:00 | 1009.70 | 1020.22 | 1022.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 1015.00 | 998.82 | 1006.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 1015.00 | 998.82 | 1006.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 1015.00 | 998.82 | 1006.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:45:00 | 1014.35 | 998.82 | 1006.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 1015.00 | 1002.06 | 1006.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 10:30:00 | 1015.85 | 1002.06 | 1006.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2025-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 13:15:00 | 1022.95 | 1011.97 | 1010.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-25 15:15:00 | 1024.10 | 1016.18 | 1012.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-27 11:15:00 | 1011.60 | 1016.00 | 1013.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 11:15:00 | 1011.60 | 1016.00 | 1013.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 11:15:00 | 1011.60 | 1016.00 | 1013.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 12:00:00 | 1011.60 | 1016.00 | 1013.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 12:15:00 | 1011.35 | 1015.07 | 1013.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 13:00:00 | 1011.35 | 1015.07 | 1013.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 1017.65 | 1015.01 | 1013.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 14:30:00 | 1008.90 | 1015.01 | 1013.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 995.70 | 1011.61 | 1012.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 11:15:00 | 985.85 | 1003.14 | 1008.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 13:15:00 | 976.10 | 971.20 | 984.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:45:00 | 972.15 | 971.20 | 984.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 973.80 | 971.72 | 983.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 14:45:00 | 987.00 | 971.72 | 983.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 982.70 | 974.12 | 982.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:45:00 | 982.50 | 974.12 | 982.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 985.55 | 976.41 | 982.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:15:00 | 990.45 | 976.41 | 982.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 996.75 | 980.48 | 984.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 12:00:00 | 996.75 | 980.48 | 984.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 12:15:00 | 991.40 | 982.66 | 984.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 12:45:00 | 997.65 | 982.66 | 984.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 14:15:00 | 990.10 | 986.23 | 986.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 09:15:00 | 1004.75 | 990.96 | 988.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 09:15:00 | 1009.00 | 1011.42 | 1002.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-06 09:15:00 | 1009.00 | 1011.42 | 1002.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 1009.00 | 1011.42 | 1002.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 09:45:00 | 1007.75 | 1011.42 | 1002.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 14:15:00 | 997.70 | 1009.23 | 1004.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 15:00:00 | 997.70 | 1009.23 | 1004.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 1001.50 | 1007.68 | 1004.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:15:00 | 1005.45 | 1007.68 | 1004.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 10:15:00 | 1004.40 | 1006.97 | 1004.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 11:00:00 | 1004.40 | 1006.97 | 1004.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 999.60 | 1005.50 | 1004.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:00:00 | 999.60 | 1005.50 | 1004.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-03-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 12:15:00 | 994.80 | 1003.36 | 1003.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 13:15:00 | 988.40 | 1000.37 | 1002.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 09:15:00 | 977.65 | 972.16 | 981.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 09:45:00 | 975.60 | 972.16 | 981.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 984.05 | 975.49 | 981.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 12:00:00 | 984.05 | 975.49 | 981.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 12:15:00 | 982.90 | 976.97 | 981.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 12:30:00 | 982.05 | 976.97 | 981.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 982.65 | 978.11 | 981.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 13:30:00 | 986.95 | 978.11 | 981.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 984.25 | 979.34 | 982.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 15:00:00 | 984.25 | 979.34 | 982.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 982.00 | 979.87 | 982.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 973.10 | 979.87 | 982.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 961.45 | 976.18 | 980.28 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2025-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 15:15:00 | 989.00 | 981.10 | 980.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 09:15:00 | 1002.45 | 985.37 | 982.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 14:15:00 | 992.50 | 994.92 | 989.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 14:15:00 | 992.50 | 994.92 | 989.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 992.50 | 994.92 | 989.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 15:00:00 | 992.50 | 994.92 | 989.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 986.40 | 993.22 | 989.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:15:00 | 994.75 | 993.22 | 989.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 995.15 | 993.60 | 989.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 11:30:00 | 1001.05 | 995.16 | 991.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 13:00:00 | 1000.20 | 996.17 | 991.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 14:30:00 | 1000.80 | 997.44 | 993.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 09:30:00 | 1000.85 | 998.39 | 994.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-20 09:15:00 | 1101.15 | 1075.91 | 1046.41 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-03-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 13:15:00 | 1129.05 | 1146.29 | 1148.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 14:15:00 | 1122.50 | 1141.53 | 1146.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 09:15:00 | 1113.70 | 1107.77 | 1118.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 1113.70 | 1107.77 | 1118.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 1113.70 | 1107.77 | 1118.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:45:00 | 1117.00 | 1107.77 | 1118.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 1096.70 | 1105.55 | 1116.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 12:15:00 | 1092.60 | 1103.44 | 1114.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 13:30:00 | 1093.00 | 1092.01 | 1099.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 14:45:00 | 1092.40 | 1092.34 | 1099.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 12:15:00 | 1110.55 | 1101.98 | 1101.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-04-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 12:15:00 | 1110.55 | 1101.98 | 1101.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 13:15:00 | 1114.30 | 1104.45 | 1102.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1078.75 | 1120.21 | 1116.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 1078.75 | 1120.21 | 1116.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 1078.75 | 1120.21 | 1116.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-07 09:45:00 | 1076.90 | 1120.21 | 1116.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 10:15:00 | 1085.75 | 1113.32 | 1114.05 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 13:15:00 | 1119.00 | 1099.67 | 1097.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 14:15:00 | 1125.95 | 1104.92 | 1099.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 09:15:00 | 1100.00 | 1106.45 | 1101.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 09:15:00 | 1100.00 | 1106.45 | 1101.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 1100.00 | 1106.45 | 1101.51 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2025-04-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-11 13:15:00 | 1087.50 | 1097.50 | 1098.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-15 09:15:00 | 1080.00 | 1091.86 | 1095.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-16 09:15:00 | 1083.20 | 1078.07 | 1084.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-16 09:15:00 | 1083.20 | 1078.07 | 1084.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 09:15:00 | 1083.20 | 1078.07 | 1084.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-16 10:15:00 | 1084.50 | 1078.07 | 1084.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 10:15:00 | 1082.80 | 1079.01 | 1084.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-16 10:45:00 | 1085.00 | 1079.01 | 1084.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1068.70 | 1075.79 | 1080.50 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2025-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 09:15:00 | 1110.90 | 1082.55 | 1080.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 11:15:00 | 1118.10 | 1094.48 | 1086.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 09:15:00 | 1110.00 | 1120.23 | 1111.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 09:15:00 | 1110.00 | 1120.23 | 1111.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 1110.00 | 1120.23 | 1111.40 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 1089.80 | 1106.80 | 1108.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 1070.00 | 1099.44 | 1104.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 1092.30 | 1081.97 | 1091.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 1092.30 | 1081.97 | 1091.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 1092.30 | 1081.97 | 1091.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 1092.30 | 1081.97 | 1091.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 1089.00 | 1083.38 | 1091.41 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 1114.20 | 1097.12 | 1095.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 09:15:00 | 1119.70 | 1105.63 | 1103.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 10:15:00 | 1105.10 | 1105.53 | 1103.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-02 11:00:00 | 1105.10 | 1105.53 | 1103.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 11:15:00 | 1095.50 | 1103.52 | 1102.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 12:15:00 | 1085.50 | 1103.52 | 1102.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2025-05-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 12:15:00 | 1084.30 | 1099.68 | 1101.20 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 1118.50 | 1102.34 | 1101.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 13:15:00 | 1135.00 | 1113.73 | 1107.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 1140.90 | 1149.19 | 1135.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 09:15:00 | 1140.90 | 1149.19 | 1135.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 1140.90 | 1149.19 | 1135.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:15:00 | 1162.50 | 1150.76 | 1137.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 14:00:00 | 1160.10 | 1153.49 | 1142.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 13:15:00 | 1112.90 | 1135.66 | 1138.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 13:15:00 | 1112.90 | 1135.66 | 1138.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 1105.90 | 1126.27 | 1133.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 11:15:00 | 1125.00 | 1121.77 | 1129.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 12:00:00 | 1125.00 | 1121.77 | 1129.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 1127.20 | 1122.80 | 1127.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 15:00:00 | 1127.20 | 1122.80 | 1127.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 1124.00 | 1123.04 | 1127.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 1152.20 | 1123.04 | 1127.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1151.50 | 1128.73 | 1129.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 1151.50 | 1128.73 | 1129.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1150.90 | 1133.17 | 1131.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 1160.80 | 1145.96 | 1138.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 13:15:00 | 1170.50 | 1170.83 | 1162.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 14:00:00 | 1170.50 | 1170.83 | 1162.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 1175.60 | 1184.51 | 1177.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:30:00 | 1180.30 | 1184.51 | 1177.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 1174.20 | 1182.45 | 1177.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 11:00:00 | 1174.20 | 1182.45 | 1177.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 1171.00 | 1180.16 | 1176.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 12:30:00 | 1175.40 | 1178.17 | 1176.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 13:15:00 | 1163.60 | 1175.26 | 1174.89 | SL hit (close<static) qty=1.00 sl=1170.50 alert=retest2 |

### Cycle 80 — SELL (started 2025-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 14:15:00 | 1165.60 | 1173.32 | 1174.04 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 1188.50 | 1176.46 | 1175.26 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 10:15:00 | 1157.00 | 1173.47 | 1175.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 1149.00 | 1162.59 | 1167.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 12:15:00 | 1147.10 | 1145.38 | 1153.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 13:00:00 | 1147.10 | 1145.38 | 1153.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 1153.00 | 1147.61 | 1152.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:15:00 | 1157.80 | 1147.61 | 1152.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 1156.20 | 1149.33 | 1152.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:15:00 | 1160.10 | 1149.33 | 1152.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 1166.90 | 1152.84 | 1154.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 1166.90 | 1152.84 | 1154.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 1165.50 | 1155.37 | 1155.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 13:15:00 | 1169.80 | 1160.14 | 1157.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 14:15:00 | 1173.50 | 1174.02 | 1168.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 15:00:00 | 1173.50 | 1174.02 | 1168.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1167.60 | 1172.48 | 1168.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 1167.60 | 1172.48 | 1168.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1164.30 | 1170.84 | 1168.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:30:00 | 1164.40 | 1170.84 | 1168.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 1155.60 | 1166.52 | 1166.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 12:45:00 | 1156.50 | 1166.52 | 1166.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 1155.90 | 1164.40 | 1165.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 14:15:00 | 1148.10 | 1158.40 | 1161.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 15:15:00 | 1144.50 | 1138.87 | 1147.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 1136.00 | 1138.29 | 1146.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1136.00 | 1138.29 | 1146.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:30:00 | 1142.50 | 1138.29 | 1146.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1150.50 | 1136.15 | 1140.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 1150.50 | 1136.15 | 1140.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 1148.40 | 1138.60 | 1141.52 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 12:15:00 | 1156.30 | 1144.49 | 1143.84 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 1138.70 | 1146.27 | 1146.32 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 1150.00 | 1145.73 | 1145.41 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 12:15:00 | 1138.70 | 1144.32 | 1144.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 09:15:00 | 1132.90 | 1139.33 | 1142.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 10:15:00 | 1148.50 | 1141.17 | 1142.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 10:15:00 | 1148.50 | 1141.17 | 1142.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 1148.50 | 1141.17 | 1142.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:00:00 | 1148.50 | 1141.17 | 1142.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 11:15:00 | 1157.30 | 1144.39 | 1143.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 13:15:00 | 1164.30 | 1150.87 | 1147.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 12:15:00 | 1180.00 | 1183.84 | 1174.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 13:00:00 | 1180.00 | 1183.84 | 1174.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1211.80 | 1192.39 | 1185.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 11:00:00 | 1216.20 | 1197.15 | 1188.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 13:00:00 | 1217.10 | 1203.29 | 1192.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:30:00 | 1219.40 | 1205.50 | 1197.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 10:15:00 | 1203.00 | 1225.03 | 1226.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 10:15:00 | 1203.00 | 1225.03 | 1226.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 11:15:00 | 1190.80 | 1218.19 | 1222.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 1188.00 | 1173.55 | 1187.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 1188.00 | 1173.55 | 1187.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1188.00 | 1173.55 | 1187.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:00:00 | 1188.00 | 1173.55 | 1187.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1193.20 | 1177.48 | 1188.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:45:00 | 1193.60 | 1177.48 | 1188.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1201.80 | 1182.35 | 1189.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:45:00 | 1201.10 | 1182.35 | 1189.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 1213.80 | 1195.48 | 1194.19 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-06-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 15:15:00 | 1190.00 | 1194.36 | 1194.80 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 1198.60 | 1194.96 | 1194.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 1209.20 | 1199.47 | 1197.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 14:15:00 | 1276.90 | 1277.17 | 1266.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 15:00:00 | 1276.90 | 1277.17 | 1266.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1271.50 | 1277.37 | 1273.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:45:00 | 1287.10 | 1279.06 | 1274.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 15:00:00 | 1285.90 | 1280.19 | 1276.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 1290.50 | 1296.38 | 1296.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 09:15:00 | 1290.50 | 1296.38 | 1296.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 10:15:00 | 1266.60 | 1290.42 | 1293.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 1244.90 | 1230.80 | 1243.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 1244.90 | 1230.80 | 1243.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1244.90 | 1230.80 | 1243.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 1247.20 | 1230.80 | 1243.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1253.60 | 1235.36 | 1244.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:45:00 | 1252.50 | 1235.36 | 1244.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 1258.80 | 1240.05 | 1245.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:00:00 | 1258.80 | 1240.05 | 1245.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 1249.50 | 1244.78 | 1246.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:45:00 | 1252.70 | 1244.78 | 1246.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 1252.50 | 1246.33 | 1247.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 1243.90 | 1246.33 | 1247.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 1257.40 | 1248.92 | 1248.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 1262.70 | 1254.73 | 1251.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 14:15:00 | 1265.30 | 1265.58 | 1259.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 14:30:00 | 1264.70 | 1265.58 | 1259.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1252.60 | 1262.73 | 1259.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:00:00 | 1252.60 | 1262.73 | 1259.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 1256.90 | 1261.56 | 1259.17 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 12:15:00 | 1247.40 | 1257.74 | 1257.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 14:15:00 | 1243.30 | 1254.42 | 1256.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 12:15:00 | 1224.90 | 1222.36 | 1231.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 13:00:00 | 1224.90 | 1222.36 | 1231.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1212.60 | 1218.43 | 1226.41 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 10:15:00 | 1249.80 | 1227.00 | 1226.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 14:15:00 | 1256.20 | 1238.80 | 1232.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 1261.20 | 1264.17 | 1253.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 09:30:00 | 1261.10 | 1264.17 | 1253.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 1275.60 | 1266.46 | 1255.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 15:00:00 | 1282.50 | 1270.50 | 1260.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 1249.90 | 1268.59 | 1262.66 | SL hit (close<static) qty=1.00 sl=1255.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 12:15:00 | 1233.50 | 1258.26 | 1258.79 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 09:15:00 | 1266.90 | 1258.23 | 1258.21 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 1249.80 | 1260.13 | 1260.86 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 10:15:00 | 1273.40 | 1253.64 | 1251.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 11:15:00 | 1279.50 | 1258.81 | 1254.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 10:15:00 | 1260.70 | 1265.02 | 1260.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 10:15:00 | 1260.70 | 1265.02 | 1260.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 1260.70 | 1265.02 | 1260.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 1260.70 | 1265.02 | 1260.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 1266.50 | 1265.32 | 1260.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:30:00 | 1258.20 | 1265.32 | 1260.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 1264.40 | 1265.13 | 1261.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:30:00 | 1261.00 | 1265.13 | 1261.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 1258.70 | 1265.46 | 1262.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:30:00 | 1257.70 | 1265.46 | 1262.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 1252.50 | 1262.87 | 1261.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:45:00 | 1255.20 | 1262.87 | 1261.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 11:15:00 | 1251.50 | 1260.59 | 1260.86 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 1263.50 | 1256.23 | 1255.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 1270.20 | 1261.54 | 1258.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 14:15:00 | 1258.40 | 1262.96 | 1260.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 14:15:00 | 1258.40 | 1262.96 | 1260.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 1258.40 | 1262.96 | 1260.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 1258.40 | 1262.96 | 1260.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 1262.50 | 1262.87 | 1260.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 09:15:00 | 1291.50 | 1262.87 | 1260.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 13:00:00 | 1272.90 | 1276.37 | 1269.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 13:45:00 | 1273.30 | 1275.30 | 1269.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 1227.40 | 1263.33 | 1265.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 1227.40 | 1263.33 | 1265.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 12:15:00 | 1212.00 | 1241.43 | 1253.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 11:15:00 | 1218.50 | 1217.99 | 1227.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 11:30:00 | 1215.20 | 1217.99 | 1227.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 1228.60 | 1220.94 | 1227.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:00:00 | 1228.60 | 1220.94 | 1227.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 1228.40 | 1222.44 | 1227.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:30:00 | 1229.60 | 1222.44 | 1227.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 1233.00 | 1224.55 | 1227.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 1221.80 | 1224.55 | 1227.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 10:30:00 | 1226.90 | 1226.15 | 1228.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 14:45:00 | 1226.80 | 1224.37 | 1226.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 09:15:00 | 1242.60 | 1228.60 | 1228.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 1242.60 | 1228.60 | 1228.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 10:15:00 | 1249.00 | 1232.68 | 1229.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 10:15:00 | 1241.00 | 1242.56 | 1237.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 11:00:00 | 1241.00 | 1242.56 | 1237.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 1237.10 | 1240.96 | 1237.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 13:00:00 | 1237.10 | 1240.96 | 1237.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 1236.90 | 1240.15 | 1237.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:15:00 | 1236.90 | 1240.15 | 1237.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 1235.60 | 1239.24 | 1237.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:45:00 | 1234.40 | 1239.24 | 1237.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 1229.90 | 1237.37 | 1236.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 1212.70 | 1237.37 | 1236.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 09:15:00 | 1215.80 | 1233.06 | 1234.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 1181.90 | 1215.04 | 1223.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1168.30 | 1165.93 | 1181.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 1168.30 | 1165.93 | 1181.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1178.90 | 1164.52 | 1173.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 1178.90 | 1164.52 | 1173.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1179.80 | 1167.58 | 1174.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:30:00 | 1182.60 | 1167.58 | 1174.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 1184.50 | 1172.63 | 1175.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 1184.50 | 1172.63 | 1175.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 1180.80 | 1177.17 | 1177.07 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 09:15:00 | 1166.50 | 1175.04 | 1176.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 12:15:00 | 1156.30 | 1167.93 | 1172.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 14:15:00 | 1161.40 | 1158.97 | 1163.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-03 15:00:00 | 1161.40 | 1158.97 | 1163.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 1160.00 | 1159.18 | 1163.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 1167.80 | 1159.18 | 1163.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1170.70 | 1161.48 | 1164.14 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 14:15:00 | 1170.70 | 1165.36 | 1164.69 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 1155.70 | 1163.90 | 1164.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 14:15:00 | 1149.40 | 1161.00 | 1163.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 14:15:00 | 1152.10 | 1150.46 | 1155.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 14:15:00 | 1152.10 | 1150.46 | 1155.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 1152.10 | 1150.46 | 1155.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 1152.10 | 1150.46 | 1155.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 1153.00 | 1150.97 | 1155.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 1168.60 | 1150.97 | 1155.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1171.60 | 1155.10 | 1156.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 1171.60 | 1155.10 | 1156.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 1165.80 | 1157.24 | 1157.52 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 11:15:00 | 1167.00 | 1159.19 | 1158.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 12:15:00 | 1169.30 | 1161.21 | 1159.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 09:15:00 | 1174.50 | 1175.67 | 1170.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 10:00:00 | 1174.50 | 1175.67 | 1170.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1178.00 | 1178.83 | 1174.98 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 09:15:00 | 1158.80 | 1172.47 | 1173.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 11:15:00 | 1150.40 | 1165.80 | 1170.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 15:15:00 | 1162.00 | 1161.59 | 1166.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-17 09:15:00 | 1157.20 | 1161.59 | 1166.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 1152.30 | 1159.73 | 1165.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 10:30:00 | 1147.20 | 1156.27 | 1163.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 13:15:00 | 1170.70 | 1158.07 | 1159.03 | SL hit (close>static) qty=1.00 sl=1166.50 alert=retest2 |

### Cycle 113 — BUY (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 14:15:00 | 1172.30 | 1160.92 | 1160.23 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 12:15:00 | 1155.10 | 1159.64 | 1160.10 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 14:15:00 | 1160.90 | 1160.46 | 1160.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 09:15:00 | 1171.10 | 1162.75 | 1161.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 1159.40 | 1167.30 | 1165.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 1159.40 | 1167.30 | 1165.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 1159.40 | 1167.30 | 1165.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 1155.80 | 1167.30 | 1165.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 1157.20 | 1165.28 | 1164.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:45:00 | 1159.10 | 1165.28 | 1164.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 11:15:00 | 1155.70 | 1163.36 | 1163.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 1139.20 | 1154.66 | 1159.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 1140.80 | 1139.22 | 1147.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 09:30:00 | 1141.00 | 1139.22 | 1147.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 1142.30 | 1139.61 | 1143.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 1113.80 | 1139.61 | 1143.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 09:30:00 | 1131.90 | 1129.32 | 1134.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 10:30:00 | 1132.40 | 1128.92 | 1133.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 10:15:00 | 1075.31 | 1103.42 | 1112.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 10:15:00 | 1075.78 | 1103.42 | 1112.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 1108.90 | 1085.13 | 1096.85 | SL hit (close>ema200) qty=0.50 sl=1085.13 alert=retest2 |

### Cycle 117 — BUY (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 12:15:00 | 1131.10 | 1104.46 | 1103.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 13:15:00 | 1135.10 | 1110.59 | 1106.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 15:15:00 | 1128.00 | 1129.44 | 1121.90 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 09:15:00 | 1135.00 | 1129.44 | 1121.90 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 10:00:00 | 1133.00 | 1130.15 | 1122.91 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1150.30 | 1140.76 | 1132.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 10:15:00 | 1153.80 | 1140.76 | 1132.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 11:00:00 | 1155.00 | 1143.61 | 1134.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 11:00:00 | 1161.90 | 1152.82 | 1144.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 10:15:00 | 1142.30 | 1152.92 | 1149.27 | SL hit (close<ema400) qty=1.00 sl=1149.27 alert=retest1 |

### Cycle 118 — SELL (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 15:15:00 | 1142.60 | 1146.98 | 1147.40 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 11:15:00 | 1152.50 | 1148.39 | 1147.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 13:15:00 | 1157.00 | 1150.45 | 1149.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 14:15:00 | 1155.20 | 1159.07 | 1155.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 14:15:00 | 1155.20 | 1159.07 | 1155.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 1155.20 | 1159.07 | 1155.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 1155.20 | 1159.07 | 1155.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 1158.00 | 1158.86 | 1155.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 1167.40 | 1158.86 | 1155.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 12:15:00 | 1191.80 | 1199.49 | 1200.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 1191.80 | 1199.49 | 1200.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 1185.70 | 1196.73 | 1199.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 14:15:00 | 1187.00 | 1185.70 | 1190.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 15:00:00 | 1187.00 | 1185.70 | 1190.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1183.50 | 1185.47 | 1189.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 1191.70 | 1185.47 | 1189.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1187.90 | 1182.39 | 1185.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 1186.20 | 1182.39 | 1185.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1188.20 | 1183.55 | 1185.95 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 14:15:00 | 1188.60 | 1187.05 | 1186.92 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 1176.50 | 1185.48 | 1186.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 1162.00 | 1176.31 | 1180.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 15:15:00 | 1150.30 | 1146.80 | 1155.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-04 10:00:00 | 1152.60 | 1147.96 | 1155.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 1141.80 | 1139.08 | 1146.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:00:00 | 1141.80 | 1139.08 | 1146.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 1140.90 | 1139.44 | 1145.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:30:00 | 1143.10 | 1139.44 | 1145.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 1122.40 | 1132.14 | 1139.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 10:15:00 | 1121.00 | 1132.14 | 1139.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 12:45:00 | 1120.00 | 1126.09 | 1131.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 12:15:00 | 1114.90 | 1103.38 | 1102.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 12:15:00 | 1114.90 | 1103.38 | 1102.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 09:15:00 | 1127.30 | 1115.17 | 1109.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 14:15:00 | 1118.60 | 1120.19 | 1114.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 14:15:00 | 1118.60 | 1120.19 | 1114.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 1118.60 | 1120.19 | 1114.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 1118.60 | 1120.19 | 1114.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 1139.50 | 1123.26 | 1116.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:15:00 | 1147.00 | 1123.26 | 1116.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 15:15:00 | 1152.00 | 1161.05 | 1162.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2025-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 15:15:00 | 1152.00 | 1161.05 | 1162.16 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 1170.80 | 1164.32 | 1163.49 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 14:15:00 | 1161.80 | 1163.04 | 1163.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 10:15:00 | 1159.10 | 1161.87 | 1162.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 13:15:00 | 1162.00 | 1161.53 | 1162.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 13:15:00 | 1162.00 | 1161.53 | 1162.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1162.00 | 1161.53 | 1162.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:30:00 | 1162.00 | 1161.53 | 1162.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 1161.20 | 1161.46 | 1162.06 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 11:15:00 | 1165.00 | 1162.38 | 1162.31 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 1161.10 | 1162.34 | 1162.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 1155.70 | 1161.01 | 1161.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 10:15:00 | 1085.60 | 1084.47 | 1095.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 10:30:00 | 1086.20 | 1084.47 | 1095.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 1091.50 | 1085.99 | 1094.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:00:00 | 1091.50 | 1085.99 | 1094.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 1096.50 | 1088.10 | 1094.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:45:00 | 1096.40 | 1088.10 | 1094.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 1098.40 | 1090.16 | 1094.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 1098.40 | 1090.16 | 1094.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 1099.30 | 1091.98 | 1095.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 1099.10 | 1091.98 | 1095.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 1076.10 | 1075.02 | 1079.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:30:00 | 1071.30 | 1075.82 | 1079.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 11:15:00 | 1086.40 | 1079.33 | 1079.58 | SL hit (close>static) qty=1.00 sl=1082.00 alert=retest2 |

### Cycle 129 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 1082.30 | 1079.92 | 1079.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 14:15:00 | 1088.00 | 1081.82 | 1080.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 09:15:00 | 1082.90 | 1083.20 | 1081.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 1082.90 | 1083.20 | 1081.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1082.90 | 1083.20 | 1081.62 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 1066.30 | 1078.57 | 1080.11 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 15:15:00 | 1083.50 | 1079.93 | 1079.86 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 1072.50 | 1079.63 | 1080.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 15:15:00 | 1070.00 | 1077.70 | 1079.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 1045.50 | 1043.93 | 1056.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 11:00:00 | 1045.50 | 1043.93 | 1056.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 1056.00 | 1047.09 | 1055.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 1056.00 | 1047.09 | 1055.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 1048.20 | 1047.31 | 1054.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:45:00 | 1052.10 | 1047.31 | 1054.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1072.40 | 1052.62 | 1055.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 1072.40 | 1052.62 | 1055.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 1073.60 | 1059.85 | 1058.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 1078.50 | 1065.38 | 1061.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 1075.10 | 1075.78 | 1070.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:15:00 | 1069.10 | 1075.78 | 1070.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 1071.60 | 1074.94 | 1070.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 1066.00 | 1074.94 | 1070.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 1074.00 | 1074.76 | 1071.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:30:00 | 1069.10 | 1074.76 | 1071.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 1075.00 | 1075.45 | 1072.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 1074.80 | 1075.45 | 1072.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1082.10 | 1076.71 | 1073.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:15:00 | 1082.40 | 1076.71 | 1073.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 11:45:00 | 1082.70 | 1079.32 | 1075.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 13:15:00 | 1072.80 | 1076.27 | 1076.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 13:15:00 | 1072.80 | 1076.27 | 1076.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 1068.00 | 1073.05 | 1074.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 11:15:00 | 1052.20 | 1047.53 | 1054.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 12:00:00 | 1052.20 | 1047.53 | 1054.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 1052.80 | 1046.93 | 1051.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:00:00 | 1052.80 | 1046.93 | 1051.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 1051.70 | 1047.88 | 1051.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:15:00 | 1049.20 | 1047.88 | 1051.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 1055.40 | 1050.28 | 1050.93 | SL hit (close>static) qty=1.00 sl=1054.00 alert=retest2 |

### Cycle 135 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 1059.00 | 1052.02 | 1051.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 12:15:00 | 1061.30 | 1053.88 | 1052.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 1057.70 | 1057.77 | 1055.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 1057.70 | 1057.77 | 1055.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1057.70 | 1057.77 | 1055.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 1049.90 | 1057.77 | 1055.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1055.30 | 1057.28 | 1055.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:00:00 | 1055.30 | 1057.28 | 1055.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 1055.70 | 1056.96 | 1055.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 13:45:00 | 1057.10 | 1055.73 | 1054.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 14:15:00 | 1050.80 | 1054.75 | 1054.55 | SL hit (close<static) qty=1.00 sl=1052.00 alert=retest2 |

### Cycle 136 — SELL (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 15:15:00 | 1050.70 | 1053.94 | 1054.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 1047.60 | 1051.83 | 1053.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 12:15:00 | 1055.70 | 1051.99 | 1052.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 12:15:00 | 1055.70 | 1051.99 | 1052.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 1055.70 | 1051.99 | 1052.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:00:00 | 1055.70 | 1051.99 | 1052.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 1056.60 | 1052.91 | 1053.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 14:15:00 | 1050.30 | 1052.91 | 1053.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 997.78 | 1016.46 | 1024.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 14:15:00 | 1014.10 | 1013.33 | 1019.70 | SL hit (close>ema200) qty=0.50 sl=1013.33 alert=retest2 |

### Cycle 137 — BUY (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 09:15:00 | 1033.20 | 1021.05 | 1019.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 1037.80 | 1027.49 | 1023.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 1028.90 | 1030.44 | 1026.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 1028.90 | 1030.44 | 1026.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1028.90 | 1030.44 | 1026.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 14:30:00 | 1038.40 | 1031.67 | 1028.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 1022.00 | 1030.03 | 1028.18 | SL hit (close<static) qty=1.00 sl=1025.00 alert=retest2 |

### Cycle 138 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 1017.90 | 1026.06 | 1026.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 13:15:00 | 1013.10 | 1021.91 | 1024.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 1002.30 | 996.55 | 1004.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 12:15:00 | 1002.30 | 996.55 | 1004.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 1002.30 | 996.55 | 1004.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 12:45:00 | 1001.20 | 996.55 | 1004.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 1000.30 | 997.30 | 1004.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 997.20 | 1000.23 | 1004.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:45:00 | 997.30 | 999.22 | 1003.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:45:00 | 997.10 | 999.06 | 1002.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:15:00 | 995.40 | 999.06 | 1002.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 997.60 | 997.89 | 1001.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:45:00 | 1001.90 | 997.89 | 1001.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 999.20 | 998.41 | 1000.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:30:00 | 1001.70 | 998.41 | 1000.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 996.90 | 997.79 | 1000.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:30:00 | 998.60 | 997.79 | 1000.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 1003.90 | 999.01 | 1000.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:00:00 | 1003.90 | 999.01 | 1000.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 994.40 | 998.09 | 999.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 15:00:00 | 988.30 | 996.13 | 998.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 09:15:00 | 947.34 | 959.96 | 971.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 09:15:00 | 947.43 | 959.96 | 971.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 09:15:00 | 947.25 | 959.96 | 971.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 09:15:00 | 945.63 | 959.96 | 971.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 962.30 | 955.25 | 962.40 | SL hit (close>ema200) qty=0.50 sl=955.25 alert=retest2 |

### Cycle 139 — BUY (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 11:15:00 | 986.70 | 964.62 | 962.63 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 937.00 | 962.08 | 963.75 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 982.05 | 961.95 | 961.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 1012.50 | 996.03 | 982.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 1011.00 | 1026.51 | 1015.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 1011.00 | 1026.51 | 1015.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1011.00 | 1026.51 | 1015.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 1011.00 | 1026.51 | 1015.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1025.65 | 1026.34 | 1016.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:15:00 | 1028.65 | 1026.34 | 1016.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 15:15:00 | 1010.50 | 1019.25 | 1020.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2026-02-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 15:15:00 | 1010.50 | 1019.25 | 1020.23 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2026-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 15:15:00 | 1021.70 | 1019.86 | 1019.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 09:15:00 | 1043.00 | 1024.49 | 1021.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 1047.65 | 1055.83 | 1047.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 1047.65 | 1055.83 | 1047.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 1047.65 | 1055.83 | 1047.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:00:00 | 1047.65 | 1055.83 | 1047.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 1047.60 | 1054.18 | 1047.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:30:00 | 1054.60 | 1054.35 | 1048.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 13:15:00 | 1052.05 | 1053.82 | 1048.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 15:00:00 | 1052.35 | 1053.17 | 1049.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 1052.45 | 1052.43 | 1049.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 1061.05 | 1054.15 | 1050.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 10:45:00 | 1061.80 | 1055.57 | 1051.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 11:15:00 | 1063.75 | 1055.57 | 1051.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 09:30:00 | 1065.75 | 1070.04 | 1061.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 15:15:00 | 1080.40 | 1082.78 | 1083.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 15:15:00 | 1080.40 | 1082.78 | 1083.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 09:15:00 | 1079.85 | 1082.20 | 1082.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 10:15:00 | 1082.25 | 1082.21 | 1082.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 10:15:00 | 1082.25 | 1082.21 | 1082.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 1082.25 | 1082.21 | 1082.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:30:00 | 1083.70 | 1082.21 | 1082.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 1085.45 | 1082.86 | 1082.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:45:00 | 1085.55 | 1082.86 | 1082.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 1082.30 | 1082.74 | 1082.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 12:30:00 | 1084.15 | 1082.74 | 1082.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — BUY (started 2026-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 13:15:00 | 1085.40 | 1083.28 | 1083.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 1087.80 | 1084.18 | 1083.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 1093.55 | 1099.93 | 1094.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 1093.55 | 1099.93 | 1094.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1093.55 | 1099.93 | 1094.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 1093.55 | 1099.93 | 1094.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 1092.30 | 1098.41 | 1094.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 12:45:00 | 1097.35 | 1097.49 | 1094.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 14:30:00 | 1101.00 | 1096.36 | 1094.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 1079.60 | 1092.11 | 1092.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1079.60 | 1092.11 | 1092.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 1044.20 | 1075.65 | 1083.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 13:15:00 | 1050.30 | 1049.98 | 1060.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 13:45:00 | 1050.90 | 1049.98 | 1060.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1057.40 | 1051.47 | 1060.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 1061.50 | 1051.47 | 1060.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 1053.80 | 1051.93 | 1059.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 1045.40 | 1051.93 | 1059.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:15:00 | 993.13 | 1015.01 | 1024.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-17 09:15:00 | 980.00 | 978.22 | 992.28 | SL hit (close>ema200) qty=0.50 sl=978.22 alert=retest2 |

### Cycle 147 — BUY (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 11:15:00 | 973.40 | 963.12 | 962.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 12:15:00 | 982.00 | 966.89 | 963.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-24 15:15:00 | 966.50 | 968.71 | 965.70 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:15:00 | 976.80 | 968.71 | 965.70 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 976.70 | 970.60 | 967.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:30:00 | 968.30 | 970.60 | 967.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 976.00 | 977.54 | 973.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 954.40 | 972.09 | 972.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest1 |

### Cycle 148 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 954.40 | 972.09 | 972.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 931.85 | 955.84 | 961.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 949.00 | 948.16 | 955.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 13:45:00 | 947.95 | 948.16 | 955.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 928.55 | 943.71 | 951.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 10:15:00 | 925.30 | 943.71 | 951.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:15:00 | 912.00 | 935.41 | 943.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 12:15:00 | 945.20 | 937.49 | 936.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2026-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 12:15:00 | 945.20 | 937.49 | 936.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 09:15:00 | 951.55 | 941.68 | 939.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 09:15:00 | 946.95 | 950.78 | 946.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 09:15:00 | 946.95 | 950.78 | 946.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 946.95 | 950.78 | 946.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:00:00 | 946.95 | 950.78 | 946.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 944.70 | 949.56 | 946.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 11:00:00 | 944.70 | 949.56 | 946.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 950.00 | 949.65 | 946.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 12:45:00 | 955.95 | 950.25 | 947.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 15:15:00 | 957.00 | 950.83 | 947.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 957.15 | 952.64 | 949.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 14:45:00 | 953.95 | 954.59 | 951.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 950.40 | 953.75 | 951.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 976.75 | 953.75 | 951.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 1002.85 | 1008.14 | 1008.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 1002.85 | 1008.14 | 1008.66 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 1010.90 | 1006.51 | 1006.15 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 15:15:00 | 1003.90 | 1006.87 | 1007.18 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 1023.40 | 1010.18 | 1008.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 1028.30 | 1013.80 | 1010.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 1014.80 | 1016.74 | 1012.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 14:00:00 | 1014.80 | 1016.74 | 1012.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 1005.40 | 1014.47 | 1012.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 1005.40 | 1014.47 | 1012.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 1004.00 | 1012.38 | 1011.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 993.50 | 1012.38 | 1011.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 991.10 | 1008.12 | 1009.61 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 1017.00 | 1006.21 | 1005.36 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 11:15:00 | 1002.00 | 1005.16 | 1005.46 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 1007.70 | 1005.35 | 1005.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 11:15:00 | 1016.40 | 1007.56 | 1006.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 09:15:00 | 1010.00 | 1012.46 | 1009.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 10:00:00 | 1010.00 | 1012.46 | 1009.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 1013.55 | 1012.68 | 1009.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 14:30:00 | 1016.25 | 1014.05 | 1011.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 1022.95 | 1013.96 | 1011.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:45:00 | 1018.90 | 1014.26 | 1012.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 10:30:00 | 791.50 | 2024-05-14 09:15:00 | 809.90 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2024-05-23 09:15:00 | 851.85 | 2024-05-23 09:15:00 | 829.45 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2024-05-28 10:15:00 | 797.20 | 2024-05-31 09:15:00 | 757.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-28 10:15:00 | 797.20 | 2024-06-03 09:15:00 | 760.00 | STOP_HIT | 0.50 | 4.67% |
| BUY | retest2 | 2024-06-10 09:15:00 | 846.00 | 2024-06-10 15:15:00 | 810.00 | STOP_HIT | 1.00 | -4.26% |
| BUY | retest2 | 2024-06-19 10:45:00 | 919.00 | 2024-06-24 12:15:00 | 907.40 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-06-19 13:30:00 | 903.20 | 2024-06-24 12:15:00 | 907.40 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest2 | 2024-06-19 14:00:00 | 904.55 | 2024-06-24 12:15:00 | 907.40 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2024-06-19 14:45:00 | 903.30 | 2024-06-24 12:15:00 | 907.40 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest2 | 2024-06-20 12:45:00 | 907.55 | 2024-06-24 12:15:00 | 907.40 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2024-07-24 10:45:00 | 942.10 | 2024-07-25 12:15:00 | 922.00 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-08-01 14:00:00 | 916.15 | 2024-08-05 13:15:00 | 874.00 | PARTIAL | 0.50 | 4.60% |
| SELL | retest2 | 2024-08-01 15:15:00 | 920.00 | 2024-08-05 14:15:00 | 870.34 | PARTIAL | 0.50 | 5.40% |
| SELL | retest2 | 2024-08-01 14:00:00 | 916.15 | 2024-08-08 09:15:00 | 866.85 | STOP_HIT | 0.50 | 5.38% |
| SELL | retest2 | 2024-08-01 15:15:00 | 920.00 | 2024-08-08 09:15:00 | 866.85 | STOP_HIT | 0.50 | 5.78% |
| SELL | retest2 | 2024-08-23 11:15:00 | 859.95 | 2024-08-28 13:15:00 | 863.30 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2024-08-28 12:30:00 | 861.25 | 2024-08-28 13:15:00 | 863.30 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2024-08-28 13:15:00 | 860.95 | 2024-08-28 13:15:00 | 863.30 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2024-08-30 12:30:00 | 869.15 | 2024-08-30 14:15:00 | 860.65 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-09-02 09:15:00 | 880.55 | 2024-09-09 12:15:00 | 880.70 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2024-09-04 09:30:00 | 867.30 | 2024-09-09 12:15:00 | 880.70 | STOP_HIT | 1.00 | 1.55% |
| BUY | retest2 | 2024-09-04 11:30:00 | 868.00 | 2024-09-09 12:15:00 | 880.70 | STOP_HIT | 1.00 | 1.46% |
| BUY | retest2 | 2024-09-05 10:30:00 | 912.50 | 2024-09-09 12:15:00 | 880.70 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest2 | 2024-09-13 11:15:00 | 909.55 | 2024-09-20 09:15:00 | 1000.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-13 13:00:00 | 915.20 | 2024-09-20 09:15:00 | 1006.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-16 09:15:00 | 929.00 | 2024-09-20 12:15:00 | 1021.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-30 09:15:00 | 970.30 | 2024-10-03 09:15:00 | 926.72 | PARTIAL | 0.50 | 4.49% |
| SELL | retest2 | 2024-10-01 09:15:00 | 975.50 | 2024-10-04 09:15:00 | 921.78 | PARTIAL | 0.50 | 5.51% |
| SELL | retest2 | 2024-09-30 09:15:00 | 970.30 | 2024-10-07 09:15:00 | 941.50 | STOP_HIT | 0.50 | 2.97% |
| SELL | retest2 | 2024-10-01 09:15:00 | 975.50 | 2024-10-07 09:15:00 | 941.50 | STOP_HIT | 0.50 | 3.49% |
| BUY | retest1 | 2024-10-30 15:00:00 | 995.45 | 2024-11-04 09:15:00 | 1045.22 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-10-30 15:00:00 | 995.45 | 2024-11-05 09:15:00 | 1023.45 | STOP_HIT | 0.50 | 2.81% |
| SELL | retest2 | 2024-11-21 10:15:00 | 984.90 | 2024-11-26 12:15:00 | 999.85 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-11-22 09:15:00 | 983.05 | 2024-11-26 12:15:00 | 999.85 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-11-25 10:00:00 | 985.50 | 2024-11-26 12:15:00 | 999.85 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-12-09 10:15:00 | 1111.50 | 2024-12-19 14:15:00 | 1181.50 | STOP_HIT | 1.00 | 6.30% |
| BUY | retest2 | 2024-12-11 09:30:00 | 1115.70 | 2024-12-19 14:15:00 | 1181.50 | STOP_HIT | 1.00 | 5.90% |
| SELL | retest2 | 2024-12-31 10:15:00 | 1137.90 | 2024-12-31 10:15:00 | 1142.00 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-01-07 10:15:00 | 1182.60 | 2025-01-10 11:15:00 | 1169.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-01-10 10:00:00 | 1184.75 | 2025-01-10 11:15:00 | 1169.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-01-10 10:30:00 | 1182.65 | 2025-01-10 11:15:00 | 1169.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-01-10 11:00:00 | 1184.20 | 2025-01-10 11:15:00 | 1169.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-01-22 14:15:00 | 1070.10 | 2025-01-24 14:15:00 | 1055.70 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-01-22 15:15:00 | 1069.95 | 2025-01-24 14:15:00 | 1055.70 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-01-23 09:30:00 | 1068.15 | 2025-01-24 14:15:00 | 1055.70 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-01-23 11:15:00 | 1067.60 | 2025-01-24 14:15:00 | 1055.70 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-01-30 11:30:00 | 1017.30 | 2025-01-30 14:15:00 | 1040.15 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-01-30 12:00:00 | 1015.60 | 2025-01-30 14:15:00 | 1040.15 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-01-30 12:30:00 | 1016.00 | 2025-01-30 14:15:00 | 1040.15 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2025-01-30 13:30:00 | 1016.20 | 2025-01-30 14:15:00 | 1040.15 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-02-10 09:15:00 | 1107.00 | 2025-02-11 09:15:00 | 1051.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 1107.00 | 2025-02-12 09:15:00 | 996.30 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-03-17 11:30:00 | 1001.05 | 2025-03-20 09:15:00 | 1101.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-17 13:00:00 | 1000.20 | 2025-03-20 09:15:00 | 1100.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-17 14:30:00 | 1000.80 | 2025-03-20 09:15:00 | 1100.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-18 09:30:00 | 1000.85 | 2025-03-20 09:15:00 | 1100.94 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-01 12:15:00 | 1092.60 | 2025-04-03 12:15:00 | 1110.55 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-04-02 13:30:00 | 1093.00 | 2025-04-03 12:15:00 | 1110.55 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-04-02 14:45:00 | 1092.40 | 2025-04-03 12:15:00 | 1110.55 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-05-07 11:15:00 | 1162.50 | 2025-05-08 13:15:00 | 1112.90 | STOP_HIT | 1.00 | -4.27% |
| BUY | retest2 | 2025-05-07 14:00:00 | 1160.10 | 2025-05-08 13:15:00 | 1112.90 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest2 | 2025-05-16 12:30:00 | 1175.40 | 2025-05-16 13:15:00 | 1163.60 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-06-12 11:00:00 | 1216.20 | 2025-06-18 10:15:00 | 1203.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-06-12 13:00:00 | 1217.10 | 2025-06-18 10:15:00 | 1203.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-06-13 10:30:00 | 1219.40 | 2025-06-18 10:15:00 | 1203.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-07-02 11:45:00 | 1287.10 | 2025-07-09 09:15:00 | 1290.50 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest2 | 2025-07-02 15:00:00 | 1285.90 | 2025-07-09 09:15:00 | 1290.50 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2025-07-25 15:00:00 | 1282.50 | 2025-07-28 10:15:00 | 1249.90 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-08-13 09:15:00 | 1291.50 | 2025-08-14 09:15:00 | 1227.40 | STOP_HIT | 1.00 | -4.96% |
| BUY | retest2 | 2025-08-13 13:00:00 | 1272.90 | 2025-08-14 09:15:00 | 1227.40 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest2 | 2025-08-13 13:45:00 | 1273.30 | 2025-08-14 09:15:00 | 1227.40 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2025-08-20 09:15:00 | 1221.80 | 2025-08-21 09:15:00 | 1242.60 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-08-20 10:30:00 | 1226.90 | 2025-08-21 09:15:00 | 1242.60 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-08-20 14:45:00 | 1226.80 | 2025-08-21 09:15:00 | 1242.60 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-09-17 10:30:00 | 1147.20 | 2025-09-18 13:15:00 | 1170.70 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1113.80 | 2025-10-03 10:15:00 | 1075.31 | PARTIAL | 0.50 | 3.46% |
| SELL | retest2 | 2025-09-29 09:30:00 | 1131.90 | 2025-10-03 10:15:00 | 1075.78 | PARTIAL | 0.50 | 4.96% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1113.80 | 2025-10-06 09:15:00 | 1108.90 | STOP_HIT | 0.50 | 0.44% |
| SELL | retest2 | 2025-09-29 09:30:00 | 1131.90 | 2025-10-06 09:15:00 | 1108.90 | STOP_HIT | 0.50 | 2.03% |
| SELL | retest2 | 2025-09-29 10:30:00 | 1132.40 | 2025-10-06 12:15:00 | 1131.10 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest1 | 2025-10-08 09:15:00 | 1135.00 | 2025-10-13 10:15:00 | 1142.30 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest1 | 2025-10-08 10:00:00 | 1133.00 | 2025-10-13 10:15:00 | 1142.30 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2025-10-09 10:15:00 | 1153.80 | 2025-10-13 15:15:00 | 1142.60 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-10-09 11:00:00 | 1155.00 | 2025-10-13 15:15:00 | 1142.60 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-10-10 11:00:00 | 1161.90 | 2025-10-13 15:15:00 | 1142.60 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-10-16 09:15:00 | 1167.40 | 2025-10-24 12:15:00 | 1191.80 | STOP_HIT | 1.00 | 2.09% |
| SELL | retest2 | 2025-11-07 10:15:00 | 1121.00 | 2025-11-17 12:15:00 | 1114.90 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2025-11-10 12:45:00 | 1120.00 | 2025-11-17 12:15:00 | 1114.90 | STOP_HIT | 1.00 | 0.46% |
| BUY | retest2 | 2025-11-19 10:15:00 | 1147.00 | 2025-11-25 15:15:00 | 1152.00 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2025-12-10 13:30:00 | 1071.30 | 2025-12-11 11:15:00 | 1086.40 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-12-24 10:15:00 | 1082.40 | 2025-12-26 13:15:00 | 1072.80 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-12-24 11:45:00 | 1082.70 | 2025-12-26 13:15:00 | 1072.80 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-01-01 12:15:00 | 1049.20 | 2026-01-02 10:15:00 | 1055.40 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2026-01-02 10:30:00 | 1051.00 | 2026-01-02 11:15:00 | 1059.00 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2026-01-05 13:45:00 | 1057.10 | 2026-01-05 14:15:00 | 1050.80 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2026-01-06 14:15:00 | 1050.30 | 2026-01-12 09:15:00 | 997.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 14:15:00 | 1050.30 | 2026-01-12 14:15:00 | 1014.10 | STOP_HIT | 0.50 | 3.45% |
| BUY | retest2 | 2026-01-16 14:30:00 | 1038.40 | 2026-01-19 09:15:00 | 1022.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2026-01-22 10:15:00 | 997.20 | 2026-01-29 09:15:00 | 947.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 10:45:00 | 997.30 | 2026-01-29 09:15:00 | 947.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 12:45:00 | 997.10 | 2026-01-29 09:15:00 | 947.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 13:15:00 | 995.40 | 2026-01-29 09:15:00 | 945.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 10:15:00 | 997.20 | 2026-01-30 09:15:00 | 962.30 | STOP_HIT | 0.50 | 3.50% |
| SELL | retest2 | 2026-01-22 10:45:00 | 997.30 | 2026-01-30 09:15:00 | 962.30 | STOP_HIT | 0.50 | 3.51% |
| SELL | retest2 | 2026-01-22 12:45:00 | 997.10 | 2026-01-30 09:15:00 | 962.30 | STOP_HIT | 0.50 | 3.49% |
| SELL | retest2 | 2026-01-22 13:15:00 | 995.40 | 2026-01-30 09:15:00 | 962.30 | STOP_HIT | 0.50 | 3.33% |
| SELL | retest2 | 2026-01-23 15:00:00 | 988.30 | 2026-02-01 11:15:00 | 986.70 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest2 | 2026-02-06 11:15:00 | 1028.65 | 2026-02-09 15:15:00 | 1010.50 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-02-13 11:30:00 | 1054.60 | 2026-02-24 15:15:00 | 1080.40 | STOP_HIT | 1.00 | 2.45% |
| BUY | retest2 | 2026-02-13 13:15:00 | 1052.05 | 2026-02-24 15:15:00 | 1080.40 | STOP_HIT | 1.00 | 2.69% |
| BUY | retest2 | 2026-02-13 15:00:00 | 1052.35 | 2026-02-24 15:15:00 | 1080.40 | STOP_HIT | 1.00 | 2.67% |
| BUY | retest2 | 2026-02-16 09:15:00 | 1052.45 | 2026-02-24 15:15:00 | 1080.40 | STOP_HIT | 1.00 | 2.66% |
| BUY | retest2 | 2026-02-16 10:45:00 | 1061.80 | 2026-02-24 15:15:00 | 1080.40 | STOP_HIT | 1.00 | 1.75% |
| BUY | retest2 | 2026-02-16 11:15:00 | 1063.75 | 2026-02-24 15:15:00 | 1080.40 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2026-02-17 09:30:00 | 1065.75 | 2026-02-24 15:15:00 | 1080.40 | STOP_HIT | 1.00 | 1.37% |
| BUY | retest2 | 2026-02-27 12:45:00 | 1097.35 | 2026-03-02 09:15:00 | 1079.60 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2026-02-27 14:30:00 | 1101.00 | 2026-03-02 09:15:00 | 1079.60 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-03-06 09:15:00 | 1045.40 | 2026-03-13 10:15:00 | 993.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 09:15:00 | 1045.40 | 2026-03-17 09:15:00 | 980.00 | STOP_HIT | 0.50 | 6.26% |
| BUY | retest1 | 2026-03-25 09:15:00 | 976.80 | 2026-03-30 09:15:00 | 954.40 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2026-04-06 10:15:00 | 925.30 | 2026-04-08 12:15:00 | 945.20 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2026-04-07 09:15:00 | 912.00 | 2026-04-08 12:15:00 | 945.20 | STOP_HIT | 1.00 | -3.64% |
| BUY | retest2 | 2026-04-10 12:45:00 | 955.95 | 2026-04-23 09:15:00 | 1002.85 | STOP_HIT | 1.00 | 4.91% |
| BUY | retest2 | 2026-04-10 15:15:00 | 957.00 | 2026-04-23 09:15:00 | 1002.85 | STOP_HIT | 1.00 | 4.79% |
| BUY | retest2 | 2026-04-13 10:45:00 | 957.15 | 2026-04-23 09:15:00 | 1002.85 | STOP_HIT | 1.00 | 4.77% |
| BUY | retest2 | 2026-04-13 14:45:00 | 953.95 | 2026-04-23 09:15:00 | 1002.85 | STOP_HIT | 1.00 | 5.13% |
| BUY | retest2 | 2026-04-15 09:15:00 | 976.75 | 2026-04-23 09:15:00 | 1002.85 | STOP_HIT | 1.00 | 2.67% |
