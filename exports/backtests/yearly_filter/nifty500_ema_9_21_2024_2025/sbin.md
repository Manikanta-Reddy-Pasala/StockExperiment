# State Bank of India (SBIN)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3709 bars)
- **Last close:** 1018.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 165 |
| ALERT1 | 103 |
| ALERT2 | 102 |
| ALERT2_SKIP | 52 |
| ALERT3 | 293 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 102 |
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 101 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 107 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 91
- **Target hits / Stop hits / Partials:** 3 / 101 / 3
- **Avg / median % per leg:** -0.19% / -0.67%
- **Sum % (uncompounded):** -20.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 61 | 4 | 6.6% | 1 | 60 | 0 | -0.44% | -26.9% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.76% | -1.8% |
| BUY @ 3rd Alert (retest2) | 60 | 4 | 6.7% | 1 | 59 | 0 | -0.42% | -25.1% |
| SELL (all) | 46 | 12 | 26.1% | 2 | 41 | 3 | 0.14% | 6.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.74% | -1.7% |
| SELL @ 3rd Alert (retest2) | 45 | 12 | 26.7% | 2 | 40 | 3 | 0.19% | 8.3% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.75% | -3.5% |
| retest2 (combined) | 105 | 16 | 15.2% | 3 | 99 | 3 | -0.16% | -16.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 10:15:00 | 802.90 | 813.69 | 814.91 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 13:15:00 | 819.00 | 814.16 | 813.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 10:15:00 | 822.00 | 817.53 | 815.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 10:15:00 | 810.40 | 818.05 | 817.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 10:15:00 | 810.40 | 818.05 | 817.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 810.40 | 818.05 | 817.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:00:00 | 810.40 | 818.05 | 817.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 812.50 | 816.94 | 816.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 12:00:00 | 812.50 | 816.94 | 816.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 12:15:00 | 805.40 | 814.63 | 815.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 13:15:00 | 799.75 | 811.66 | 814.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 14:15:00 | 811.90 | 811.70 | 814.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-16 15:00:00 | 811.90 | 811.70 | 814.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 819.00 | 813.26 | 814.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 10:00:00 | 819.00 | 813.26 | 814.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 10:15:00 | 821.70 | 814.95 | 815.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 11:00:00 | 821.70 | 814.95 | 815.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 11:15:00 | 818.15 | 815.59 | 815.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 09:15:00 | 831.30 | 821.20 | 818.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 09:15:00 | 820.40 | 827.25 | 823.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 09:15:00 | 820.40 | 827.25 | 823.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 820.40 | 827.25 | 823.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 820.90 | 827.25 | 823.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 816.30 | 825.06 | 823.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:00:00 | 816.30 | 825.06 | 823.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2024-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 13:15:00 | 817.45 | 821.34 | 821.80 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 10:15:00 | 831.30 | 823.73 | 822.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 09:15:00 | 836.20 | 830.21 | 826.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 14:15:00 | 828.70 | 832.12 | 829.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 14:15:00 | 828.70 | 832.12 | 829.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 828.70 | 832.12 | 829.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 828.70 | 832.12 | 829.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 828.00 | 831.30 | 829.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 09:15:00 | 831.90 | 831.30 | 829.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 10:15:00 | 829.35 | 830.52 | 829.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 11:30:00 | 829.65 | 831.32 | 829.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 11:45:00 | 829.95 | 832.96 | 831.69 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 832.80 | 832.92 | 831.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:30:00 | 829.85 | 832.92 | 831.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 832.50 | 832.84 | 831.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 13:30:00 | 833.15 | 832.84 | 831.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 832.00 | 832.67 | 831.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 15:00:00 | 832.00 | 832.67 | 831.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 830.90 | 832.32 | 831.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:15:00 | 832.05 | 832.32 | 831.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-29 09:15:00 | 825.60 | 830.97 | 831.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 09:15:00 | 825.60 | 830.97 | 831.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 10:15:00 | 823.45 | 829.47 | 830.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 12:15:00 | 829.35 | 826.48 | 827.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 12:15:00 | 829.35 | 826.48 | 827.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 12:15:00 | 829.35 | 826.48 | 827.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 12:45:00 | 830.20 | 826.48 | 827.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 13:15:00 | 829.45 | 827.07 | 827.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 13:45:00 | 832.00 | 827.07 | 827.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 15:15:00 | 828.90 | 827.15 | 827.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:15:00 | 830.60 | 827.15 | 827.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 822.70 | 826.26 | 827.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 10:45:00 | 822.45 | 825.09 | 826.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 14:15:00 | 830.70 | 827.35 | 827.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2024-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 14:15:00 | 830.70 | 827.35 | 827.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 15:15:00 | 832.30 | 828.34 | 827.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 864.50 | 883.43 | 864.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 864.50 | 883.43 | 864.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 864.50 | 883.43 | 864.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 849.95 | 883.43 | 864.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 815.10 | 869.77 | 859.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 815.10 | 869.77 | 859.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 772.00 | 850.21 | 851.69 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 15:15:00 | 817.50 | 811.61 | 811.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 10:15:00 | 822.15 | 814.58 | 812.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 830.60 | 831.76 | 826.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 09:15:00 | 833.80 | 831.76 | 826.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 841.60 | 840.30 | 837.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 15:15:00 | 844.90 | 841.43 | 838.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 11:00:00 | 844.60 | 843.00 | 839.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 09:30:00 | 846.70 | 842.73 | 841.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 15:15:00 | 845.30 | 845.51 | 843.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 15:15:00 | 845.30 | 845.47 | 843.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 09:15:00 | 846.70 | 845.47 | 843.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 10:45:00 | 847.35 | 846.80 | 844.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 11:30:00 | 849.10 | 847.04 | 844.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 13:45:00 | 847.05 | 847.97 | 845.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 846.95 | 849.37 | 847.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:45:00 | 846.90 | 849.37 | 847.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 847.80 | 849.06 | 847.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 11:30:00 | 850.50 | 848.73 | 847.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 14:15:00 | 844.00 | 847.10 | 846.71 | SL hit (close<static) qty=1.00 sl=846.00 alert=retest2 |

### Cycle 11 — SELL (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 09:15:00 | 843.40 | 846.01 | 846.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 11:15:00 | 837.85 | 843.33 | 844.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 09:15:00 | 838.30 | 835.08 | 837.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 09:15:00 | 838.30 | 835.08 | 837.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 838.30 | 835.08 | 837.55 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2024-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 13:15:00 | 844.60 | 838.71 | 838.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 13:15:00 | 847.10 | 842.30 | 840.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 12:15:00 | 841.85 | 843.94 | 842.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 12:15:00 | 841.85 | 843.94 | 842.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 841.85 | 843.94 | 842.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:00:00 | 841.85 | 843.94 | 842.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 840.95 | 843.34 | 842.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 840.95 | 843.34 | 842.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 844.75 | 843.62 | 842.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:30:00 | 841.05 | 843.62 | 842.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 861.00 | 847.32 | 844.39 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 12:15:00 | 843.60 | 846.51 | 846.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 09:15:00 | 838.20 | 843.43 | 845.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 11:15:00 | 833.70 | 831.99 | 836.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-03 12:00:00 | 833.70 | 831.99 | 836.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 12:15:00 | 836.15 | 832.82 | 836.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 12:30:00 | 834.50 | 832.82 | 836.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 13:15:00 | 842.10 | 834.68 | 836.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 14:00:00 | 842.10 | 834.68 | 836.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 839.30 | 835.60 | 837.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 14:30:00 | 839.05 | 835.60 | 837.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2024-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 11:15:00 | 842.65 | 838.70 | 838.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 10:15:00 | 846.25 | 841.19 | 839.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 14:15:00 | 861.65 | 862.24 | 856.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-09 15:00:00 | 861.65 | 862.24 | 856.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 15:15:00 | 857.90 | 861.37 | 856.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 09:15:00 | 856.40 | 861.37 | 856.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 848.60 | 858.82 | 856.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 848.60 | 858.82 | 856.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 849.70 | 857.00 | 855.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 11:30:00 | 851.45 | 856.11 | 855.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 12:15:00 | 848.25 | 854.54 | 854.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2024-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 12:15:00 | 848.25 | 854.54 | 854.69 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2024-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 13:15:00 | 857.00 | 853.78 | 853.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 09:15:00 | 858.50 | 855.52 | 854.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 887.05 | 888.60 | 882.59 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-19 13:30:00 | 895.95 | 890.04 | 885.19 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 890.00 | 889.85 | 886.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-22 10:15:00 | 880.15 | 887.91 | 885.73 | SL hit (close<ema400) qty=1.00 sl=885.73 alert=retest1 |

### Cycle 17 — SELL (started 2024-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 13:15:00 | 880.10 | 884.10 | 884.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 15:15:00 | 878.00 | 882.10 | 883.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 09:15:00 | 852.80 | 850.12 | 856.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 09:15:00 | 852.80 | 850.12 | 856.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 852.80 | 850.12 | 856.06 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 09:15:00 | 879.80 | 861.75 | 859.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 10:15:00 | 884.75 | 866.35 | 861.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 13:15:00 | 869.50 | 870.41 | 865.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 14:00:00 | 869.50 | 870.41 | 865.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 871.45 | 872.22 | 869.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 877.70 | 872.16 | 870.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 14:15:00 | 864.45 | 873.23 | 872.35 | SL hit (close<static) qty=1.00 sl=868.20 alert=retest2 |

### Cycle 19 — SELL (started 2024-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 15:15:00 | 861.90 | 870.96 | 871.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 849.50 | 866.67 | 869.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 13:15:00 | 806.10 | 805.87 | 815.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 14:00:00 | 806.10 | 805.87 | 815.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 11:15:00 | 816.90 | 809.56 | 813.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 12:00:00 | 816.90 | 809.56 | 813.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 811.40 | 809.93 | 813.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 12:30:00 | 815.70 | 809.93 | 813.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 813.00 | 809.93 | 812.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 11:30:00 | 811.40 | 810.25 | 812.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 13:15:00 | 825.70 | 813.90 | 813.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2024-08-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 13:15:00 | 825.70 | 813.90 | 813.52 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2024-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 12:15:00 | 809.50 | 814.97 | 815.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 14:15:00 | 796.85 | 810.14 | 812.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 807.85 | 805.24 | 807.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 807.85 | 805.24 | 807.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 807.85 | 805.24 | 807.85 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 13:15:00 | 814.50 | 809.14 | 809.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 818.30 | 811.71 | 810.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 14:15:00 | 813.65 | 814.92 | 812.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 14:15:00 | 813.65 | 814.92 | 812.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 14:15:00 | 813.65 | 814.92 | 812.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 14:45:00 | 813.50 | 814.92 | 812.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 819.20 | 815.72 | 813.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 11:00:00 | 820.95 | 816.77 | 814.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 12:30:00 | 820.65 | 818.19 | 815.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-21 13:15:00 | 813.05 | 815.83 | 815.69 | SL hit (close<static) qty=1.00 sl=813.35 alert=retest2 |

### Cycle 23 — SELL (started 2024-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 14:15:00 | 815.25 | 817.48 | 817.68 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 821.60 | 818.04 | 817.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 11:15:00 | 823.00 | 819.38 | 818.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-26 12:15:00 | 819.05 | 819.31 | 818.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-26 13:00:00 | 819.05 | 819.31 | 818.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 817.20 | 818.89 | 818.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 14:00:00 | 817.20 | 818.89 | 818.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2024-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 14:15:00 | 815.15 | 818.14 | 818.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 09:15:00 | 814.55 | 816.98 | 817.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 12:15:00 | 816.90 | 816.56 | 817.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 12:15:00 | 816.90 | 816.56 | 817.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 816.90 | 816.56 | 817.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 13:00:00 | 816.90 | 816.56 | 817.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 13:15:00 | 817.50 | 816.75 | 817.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 13:45:00 | 817.85 | 816.75 | 817.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 14:15:00 | 816.15 | 816.63 | 817.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 14:30:00 | 819.20 | 816.63 | 817.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 814.05 | 815.85 | 816.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 11:15:00 | 812.95 | 815.37 | 816.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 11:45:00 | 812.80 | 815.09 | 816.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 14:30:00 | 812.25 | 814.04 | 815.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 11:30:00 | 812.20 | 813.04 | 814.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 12:15:00 | 814.85 | 813.40 | 814.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 12:30:00 | 815.60 | 813.40 | 814.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 13:15:00 | 812.45 | 813.21 | 814.22 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-30 09:15:00 | 818.80 | 814.96 | 814.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 818.80 | 814.96 | 814.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 13:15:00 | 823.70 | 817.51 | 816.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 10:15:00 | 817.35 | 818.92 | 817.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 10:15:00 | 817.35 | 818.92 | 817.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 10:15:00 | 817.35 | 818.92 | 817.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 11:00:00 | 817.35 | 818.92 | 817.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 11:15:00 | 818.60 | 818.86 | 817.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 12:30:00 | 820.10 | 819.45 | 818.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 14:00:00 | 821.45 | 819.85 | 818.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 09:15:00 | 815.75 | 820.22 | 819.01 | SL hit (close<static) qty=1.00 sl=816.40 alert=retest2 |

### Cycle 27 — SELL (started 2024-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 12:15:00 | 816.40 | 818.23 | 818.29 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2024-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 10:15:00 | 820.00 | 818.53 | 818.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 11:15:00 | 821.50 | 819.13 | 818.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 14:15:00 | 819.05 | 819.49 | 818.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 14:15:00 | 819.05 | 819.49 | 818.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 819.05 | 819.49 | 818.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 15:00:00 | 819.05 | 819.49 | 818.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 15:15:00 | 818.00 | 819.20 | 818.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 09:15:00 | 807.95 | 819.20 | 818.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 797.65 | 814.89 | 816.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 10:15:00 | 789.45 | 809.80 | 814.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 783.65 | 782.78 | 792.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 14:45:00 | 784.85 | 782.78 | 792.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 786.20 | 784.00 | 787.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 10:30:00 | 786.95 | 784.00 | 787.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 780.35 | 777.41 | 780.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 14:00:00 | 780.35 | 777.41 | 780.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 789.35 | 779.80 | 781.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 15:00:00 | 789.35 | 779.80 | 781.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 15:15:00 | 786.50 | 781.14 | 781.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:15:00 | 788.30 | 781.14 | 781.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 788.15 | 782.54 | 782.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 12:15:00 | 794.80 | 786.70 | 784.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 10:15:00 | 788.25 | 789.11 | 786.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 10:45:00 | 788.15 | 789.11 | 786.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 786.50 | 788.58 | 786.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 12:00:00 | 786.50 | 788.58 | 786.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 789.45 | 788.76 | 786.99 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 10:15:00 | 783.15 | 785.79 | 786.10 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2024-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 10:15:00 | 792.90 | 786.40 | 786.00 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2024-09-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-20 14:15:00 | 781.25 | 787.53 | 788.34 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 802.90 | 791.02 | 789.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 12:15:00 | 804.65 | 795.71 | 792.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 14:15:00 | 798.55 | 800.57 | 797.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 14:15:00 | 798.55 | 800.57 | 797.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 798.55 | 800.57 | 797.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 15:00:00 | 798.55 | 800.57 | 797.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 797.90 | 800.03 | 797.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:15:00 | 797.00 | 800.03 | 797.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 792.05 | 798.44 | 797.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 792.05 | 798.44 | 797.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 792.05 | 797.16 | 796.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:30:00 | 791.35 | 797.16 | 796.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2024-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 12:15:00 | 790.60 | 795.16 | 795.77 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2024-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 13:15:00 | 800.50 | 796.40 | 795.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 14:15:00 | 801.25 | 797.37 | 796.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 12:15:00 | 800.95 | 801.24 | 798.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-27 13:00:00 | 800.95 | 801.24 | 798.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 801.55 | 801.30 | 799.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 14:15:00 | 801.80 | 801.30 | 799.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 09:15:00 | 793.20 | 800.05 | 799.21 | SL hit (close<static) qty=1.00 sl=798.45 alert=retest2 |

### Cycle 37 — SELL (started 2024-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 10:15:00 | 790.55 | 798.15 | 798.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 14:15:00 | 788.75 | 794.01 | 796.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 793.75 | 793.00 | 795.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 793.75 | 793.00 | 795.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 793.75 | 793.00 | 795.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:30:00 | 795.75 | 793.00 | 795.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 794.65 | 793.33 | 795.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 11:00:00 | 794.65 | 793.33 | 795.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 11:15:00 | 796.35 | 793.93 | 795.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:00:00 | 796.35 | 793.93 | 795.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 798.25 | 794.80 | 795.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 13:00:00 | 798.25 | 794.80 | 795.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 795.50 | 795.36 | 795.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 794.60 | 795.36 | 795.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 09:15:00 | 800.55 | 796.40 | 796.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-03 09:15:00 | 800.55 | 796.40 | 796.14 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2024-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 12:15:00 | 788.05 | 795.13 | 795.70 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2024-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 10:15:00 | 802.50 | 796.00 | 795.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-04 11:15:00 | 809.05 | 798.61 | 796.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 12:15:00 | 797.50 | 798.39 | 796.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-04 13:00:00 | 797.50 | 798.39 | 796.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 796.75 | 798.06 | 796.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 13:30:00 | 793.45 | 798.06 | 796.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 796.95 | 797.84 | 796.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:45:00 | 795.65 | 797.84 | 796.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 15:15:00 | 797.25 | 797.72 | 796.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-07 09:15:00 | 799.55 | 797.72 | 796.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-07 09:15:00 | 796.50 | 797.48 | 796.95 | SL hit (close<static) qty=1.00 sl=796.60 alert=retest2 |

### Cycle 41 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 784.20 | 794.82 | 795.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 13:15:00 | 773.25 | 787.35 | 791.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 780.65 | 780.59 | 786.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 10:45:00 | 780.50 | 780.59 | 786.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 803.90 | 785.58 | 786.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 803.90 | 785.58 | 786.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 801.00 | 788.66 | 787.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 09:15:00 | 805.80 | 799.83 | 797.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 13:15:00 | 804.15 | 804.18 | 801.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-15 14:00:00 | 804.15 | 804.18 | 801.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 804.30 | 804.20 | 802.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:30:00 | 802.35 | 804.20 | 802.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 807.10 | 804.91 | 802.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 09:15:00 | 815.10 | 805.32 | 803.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 10:00:00 | 811.00 | 806.46 | 804.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 10:00:00 | 812.05 | 811.30 | 808.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-22 11:15:00 | 798.10 | 809.69 | 811.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2024-10-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 11:15:00 | 798.10 | 809.69 | 811.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 12:15:00 | 795.45 | 806.84 | 809.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 09:15:00 | 791.55 | 788.90 | 795.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 10:00:00 | 791.55 | 788.90 | 795.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 791.60 | 789.44 | 794.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:30:00 | 794.75 | 789.44 | 794.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 12:15:00 | 794.05 | 790.62 | 794.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 13:00:00 | 794.05 | 790.62 | 794.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 13:15:00 | 795.85 | 791.67 | 794.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 13:45:00 | 796.15 | 791.67 | 794.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 14:15:00 | 795.00 | 792.33 | 794.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 14:30:00 | 794.45 | 792.33 | 794.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 15:15:00 | 795.35 | 792.94 | 794.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-25 09:15:00 | 797.35 | 792.94 | 794.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 780.65 | 790.48 | 793.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 10:15:00 | 778.70 | 790.48 | 793.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 14:00:00 | 778.35 | 784.27 | 789.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 12:15:00 | 794.50 | 791.06 | 790.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2024-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 12:15:00 | 794.50 | 791.06 | 790.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 09:15:00 | 797.30 | 792.86 | 791.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 14:15:00 | 821.75 | 822.84 | 813.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 15:00:00 | 821.75 | 822.84 | 813.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 817.25 | 821.82 | 816.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:45:00 | 816.70 | 821.82 | 816.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 816.50 | 820.75 | 816.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:45:00 | 815.30 | 820.75 | 816.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 815.45 | 819.69 | 816.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 14:00:00 | 815.45 | 819.69 | 816.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 820.35 | 819.83 | 816.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:00:00 | 824.35 | 820.93 | 817.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:45:00 | 821.95 | 821.14 | 818.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 813.00 | 819.51 | 817.57 | SL hit (close<static) qty=1.00 sl=815.25 alert=retest2 |

### Cycle 45 — SELL (started 2024-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 11:15:00 | 808.50 | 815.85 | 816.16 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2024-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 13:15:00 | 825.85 | 817.68 | 816.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-04 14:15:00 | 828.70 | 819.89 | 818.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 850.00 | 851.98 | 843.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:00:00 | 850.00 | 851.98 | 843.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 852.95 | 856.12 | 850.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 10:45:00 | 850.35 | 856.12 | 850.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 850.35 | 854.96 | 850.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 11:45:00 | 847.65 | 854.96 | 850.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 851.25 | 854.22 | 850.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:45:00 | 848.50 | 854.22 | 850.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 13:15:00 | 839.35 | 851.25 | 849.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 13:45:00 | 843.30 | 851.25 | 849.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 840.75 | 849.15 | 848.90 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2024-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 15:15:00 | 841.10 | 847.54 | 848.19 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 11:15:00 | 850.10 | 848.65 | 848.55 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2024-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 15:15:00 | 845.95 | 848.12 | 848.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 10:15:00 | 843.10 | 846.99 | 847.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 11:15:00 | 808.95 | 807.63 | 814.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-18 11:45:00 | 807.30 | 807.63 | 814.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 814.00 | 809.20 | 814.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 14:00:00 | 814.00 | 809.20 | 814.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 14:15:00 | 815.95 | 810.55 | 814.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 15:00:00 | 815.95 | 810.55 | 814.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 15:15:00 | 814.35 | 811.31 | 814.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:15:00 | 815.50 | 811.31 | 814.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 816.80 | 812.41 | 814.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:45:00 | 815.60 | 812.41 | 814.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 814.65 | 812.86 | 814.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 815.50 | 812.86 | 814.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 809.70 | 812.23 | 814.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:00:00 | 805.60 | 810.34 | 812.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 09:15:00 | 765.32 | 799.54 | 807.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-22 09:15:00 | 797.90 | 787.44 | 795.24 | SL hit (close>ema200) qty=0.50 sl=787.44 alert=retest2 |

### Cycle 50 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 819.05 | 801.62 | 800.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 844.00 | 814.09 | 806.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 11:15:00 | 838.70 | 839.06 | 827.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 12:00:00 | 838.70 | 839.06 | 827.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 833.95 | 837.14 | 833.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 15:00:00 | 833.95 | 837.14 | 833.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 15:15:00 | 834.85 | 836.68 | 833.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:15:00 | 836.35 | 836.68 | 833.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 844.60 | 838.27 | 834.71 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 11:15:00 | 834.75 | 836.46 | 836.52 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 09:15:00 | 851.10 | 839.23 | 837.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 10:15:00 | 855.00 | 842.38 | 839.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 09:15:00 | 860.60 | 862.79 | 858.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-06 09:15:00 | 860.60 | 862.79 | 858.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 860.60 | 862.79 | 858.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:00:00 | 860.60 | 862.79 | 858.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 869.30 | 864.09 | 859.03 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 856.70 | 861.84 | 862.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 10:15:00 | 853.80 | 860.23 | 861.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 12:15:00 | 853.80 | 851.06 | 854.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 12:15:00 | 853.80 | 851.06 | 854.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 853.80 | 851.06 | 854.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:00:00 | 853.80 | 851.06 | 854.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 858.15 | 852.48 | 855.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:45:00 | 858.00 | 852.48 | 855.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 861.95 | 854.37 | 855.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 861.95 | 854.37 | 855.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 09:15:00 | 863.75 | 857.28 | 856.92 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 11:15:00 | 851.35 | 857.70 | 857.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 12:15:00 | 849.90 | 856.14 | 857.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 10:15:00 | 821.10 | 820.77 | 828.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 11:00:00 | 821.10 | 820.77 | 828.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 822.50 | 821.33 | 825.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 11:30:00 | 815.75 | 819.91 | 823.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 09:45:00 | 819.00 | 815.82 | 820.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 15:15:00 | 802.25 | 796.82 | 796.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2025-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 15:15:00 | 802.25 | 796.82 | 796.55 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2025-01-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 15:15:00 | 793.30 | 796.47 | 796.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 788.75 | 794.93 | 796.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 772.10 | 771.44 | 777.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 14:00:00 | 772.10 | 771.44 | 777.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 743.60 | 737.91 | 746.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 09:30:00 | 744.35 | 737.91 | 746.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 752.25 | 740.78 | 746.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 752.25 | 740.78 | 746.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 750.70 | 742.76 | 747.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:15:00 | 749.45 | 742.76 | 747.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 14:30:00 | 748.10 | 746.44 | 748.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-15 09:15:00 | 757.30 | 749.26 | 749.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 09:15:00 | 757.30 | 749.26 | 749.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 768.85 | 756.24 | 753.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 761.70 | 764.06 | 759.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 11:00:00 | 761.70 | 764.06 | 759.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 762.90 | 763.20 | 760.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 13:30:00 | 760.60 | 763.20 | 760.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 764.80 | 773.20 | 769.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 764.80 | 773.20 | 769.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 765.90 | 771.74 | 769.40 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 14:15:00 | 758.90 | 767.43 | 767.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 755.60 | 763.84 | 766.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 755.00 | 753.02 | 759.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 755.00 | 753.02 | 759.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 748.70 | 750.56 | 755.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:45:00 | 751.50 | 750.56 | 755.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 11:15:00 | 752.00 | 748.77 | 752.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 11:45:00 | 751.70 | 748.77 | 752.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 12:15:00 | 748.05 | 748.62 | 751.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:15:00 | 745.45 | 748.13 | 751.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 10:15:00 | 742.50 | 747.02 | 749.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-28 12:15:00 | 755.85 | 751.16 | 750.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2025-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 12:15:00 | 755.85 | 751.16 | 750.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 09:15:00 | 757.85 | 753.21 | 751.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 762.85 | 769.82 | 766.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 762.85 | 769.82 | 766.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 762.85 | 769.82 | 766.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:00:00 | 762.85 | 769.82 | 766.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 755.70 | 766.99 | 765.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 755.70 | 766.99 | 765.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 765.10 | 766.61 | 765.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:15:00 | 765.80 | 766.61 | 765.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:45:00 | 766.00 | 766.45 | 765.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 10:15:00 | 759.10 | 763.72 | 764.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 759.10 | 763.72 | 764.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 11:15:00 | 756.05 | 762.18 | 763.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 13:15:00 | 761.65 | 761.51 | 762.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-03 14:00:00 | 761.65 | 761.51 | 762.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 62 — BUY (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 09:15:00 | 772.25 | 763.57 | 763.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 12:15:00 | 779.20 | 769.80 | 766.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 11:15:00 | 769.20 | 773.31 | 770.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 11:15:00 | 769.20 | 773.31 | 770.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 769.20 | 773.31 | 770.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 12:00:00 | 769.20 | 773.31 | 770.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 12:15:00 | 771.00 | 772.85 | 770.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 13:45:00 | 771.85 | 771.96 | 770.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-05 14:15:00 | 765.85 | 770.74 | 769.81 | SL hit (close<static) qty=1.00 sl=767.05 alert=retest2 |

### Cycle 63 — SELL (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 09:15:00 | 763.85 | 768.52 | 768.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 11:15:00 | 761.05 | 766.09 | 767.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 735.20 | 730.59 | 735.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 11:15:00 | 735.20 | 730.59 | 735.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 735.20 | 730.59 | 735.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 735.20 | 730.59 | 735.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 737.15 | 731.90 | 735.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:45:00 | 736.05 | 731.90 | 735.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 731.95 | 731.91 | 735.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:30:00 | 735.60 | 731.91 | 735.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 732.30 | 732.35 | 734.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 14:00:00 | 728.65 | 732.27 | 733.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 15:00:00 | 727.45 | 731.31 | 733.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:30:00 | 727.10 | 729.98 | 732.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 732.70 | 726.13 | 725.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 732.70 | 726.13 | 725.68 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 725.00 | 727.38 | 727.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 13:15:00 | 722.40 | 725.67 | 726.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 11:15:00 | 690.15 | 689.66 | 697.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 12:00:00 | 690.15 | 689.66 | 697.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 696.15 | 692.31 | 696.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:15:00 | 706.75 | 692.31 | 696.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 714.00 | 696.65 | 697.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 714.00 | 696.65 | 697.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2025-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 10:15:00 | 715.30 | 700.38 | 699.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 13:15:00 | 717.40 | 707.73 | 703.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 12:15:00 | 730.60 | 730.65 | 725.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 13:00:00 | 730.60 | 730.65 | 725.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 729.25 | 731.78 | 729.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:45:00 | 728.65 | 731.78 | 729.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 727.60 | 730.95 | 729.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 727.20 | 730.95 | 729.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 728.55 | 730.47 | 729.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 10:15:00 | 730.30 | 730.47 | 729.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 13:15:00 | 730.65 | 730.52 | 729.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 09:15:00 | 732.85 | 730.39 | 729.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 10:15:00 | 727.35 | 729.45 | 729.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 10:15:00 | 727.35 | 729.45 | 729.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 11:15:00 | 723.15 | 728.19 | 728.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 730.50 | 726.33 | 727.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 09:15:00 | 730.50 | 726.33 | 727.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 730.50 | 726.33 | 727.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:00:00 | 730.50 | 726.33 | 727.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 729.25 | 726.91 | 727.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:30:00 | 729.80 | 726.91 | 727.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 12:15:00 | 729.95 | 727.63 | 727.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 13:00:00 | 729.95 | 727.63 | 727.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2025-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 13:15:00 | 729.30 | 727.97 | 727.95 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2025-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 09:15:00 | 724.40 | 727.24 | 727.62 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 731.15 | 727.52 | 727.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 13:15:00 | 732.40 | 729.03 | 727.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 14:15:00 | 773.55 | 774.55 | 767.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 15:00:00 | 773.55 | 774.55 | 767.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 771.20 | 773.21 | 768.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:00:00 | 771.20 | 773.21 | 768.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 769.75 | 772.39 | 768.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:00:00 | 769.75 | 772.39 | 768.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 12:15:00 | 767.10 | 771.33 | 768.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:00:00 | 767.10 | 771.33 | 768.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 763.95 | 769.85 | 768.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:45:00 | 763.65 | 769.85 | 768.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 764.50 | 768.78 | 767.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 15:15:00 | 762.70 | 768.78 | 767.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 773.30 | 769.01 | 767.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 09:15:00 | 774.75 | 770.00 | 768.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 09:45:00 | 777.90 | 770.97 | 769.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 11:45:00 | 774.40 | 772.35 | 770.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 14:15:00 | 775.20 | 771.80 | 770.38 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 771.20 | 771.76 | 770.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:15:00 | 775.15 | 771.76 | 770.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 778.30 | 773.07 | 771.31 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-02 09:15:00 | 767.40 | 770.42 | 770.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2025-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 09:15:00 | 767.40 | 770.42 | 770.56 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2025-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 10:15:00 | 773.40 | 771.01 | 770.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 776.25 | 772.99 | 771.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 774.05 | 777.34 | 775.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 774.05 | 777.34 | 775.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 774.05 | 777.34 | 775.50 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2025-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 15:15:00 | 768.50 | 774.04 | 774.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 738.40 | 766.91 | 771.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 764.15 | 753.40 | 760.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 764.15 | 753.40 | 760.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 764.15 | 753.40 | 760.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:30:00 | 769.75 | 753.40 | 760.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 767.10 | 756.14 | 760.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 11:00:00 | 767.10 | 756.14 | 760.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 11:15:00 | 768.50 | 758.61 | 761.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:00:00 | 768.50 | 758.61 | 761.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2025-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 14:15:00 | 768.50 | 764.12 | 763.55 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2025-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 10:15:00 | 751.20 | 761.18 | 762.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 13:15:00 | 745.40 | 755.66 | 759.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-11 09:15:00 | 751.25 | 750.94 | 755.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 09:15:00 | 751.25 | 750.94 | 755.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 751.25 | 750.94 | 755.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 10:30:00 | 748.00 | 750.92 | 755.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 769.60 | 756.28 | 756.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 769.60 | 756.28 | 756.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 13:15:00 | 773.95 | 767.32 | 763.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 14:15:00 | 822.15 | 822.62 | 809.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 15:00:00 | 822.15 | 822.62 | 809.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 813.80 | 819.68 | 812.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:00:00 | 813.80 | 819.68 | 812.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 813.20 | 818.39 | 812.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:45:00 | 813.40 | 818.39 | 812.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 13:15:00 | 816.05 | 817.92 | 812.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 12:30:00 | 816.55 | 815.80 | 813.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 13:15:00 | 817.20 | 815.80 | 813.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 14:15:00 | 816.65 | 815.60 | 813.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-24 15:15:00 | 811.20 | 814.35 | 813.55 | SL hit (close<static) qty=1.00 sl=812.85 alert=retest2 |

### Cycle 77 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 801.05 | 811.69 | 812.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 794.15 | 808.18 | 810.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 807.55 | 803.19 | 806.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 807.55 | 803.19 | 806.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 807.55 | 803.19 | 806.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 09:30:00 | 807.15 | 803.19 | 806.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 806.40 | 803.83 | 806.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:30:00 | 809.25 | 803.83 | 806.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 811.35 | 805.34 | 806.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:00:00 | 811.35 | 805.34 | 806.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2025-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 12:15:00 | 821.00 | 808.47 | 808.21 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 09:15:00 | 793.10 | 810.76 | 811.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 10:15:00 | 789.85 | 806.58 | 809.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 795.65 | 795.02 | 801.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-02 09:30:00 | 793.15 | 795.02 | 801.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 798.60 | 795.74 | 801.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:30:00 | 801.80 | 795.74 | 801.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 800.80 | 796.56 | 799.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 14:45:00 | 800.70 | 796.56 | 799.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 15:15:00 | 799.80 | 797.21 | 799.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 09:15:00 | 786.10 | 797.21 | 799.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 794.95 | 780.18 | 778.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 794.95 | 780.18 | 778.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 806.60 | 797.72 | 790.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 799.35 | 799.85 | 793.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 799.35 | 799.85 | 793.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 798.90 | 801.17 | 797.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 797.35 | 801.17 | 797.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 796.80 | 800.13 | 797.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:45:00 | 796.90 | 800.13 | 797.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 796.55 | 799.42 | 797.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:30:00 | 803.00 | 800.78 | 798.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 10:15:00 | 794.70 | 800.43 | 799.52 | SL hit (close<static) qty=1.00 sl=796.00 alert=retest2 |

### Cycle 81 — SELL (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 11:15:00 | 792.05 | 798.76 | 798.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 11:15:00 | 791.25 | 794.05 | 795.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 15:15:00 | 787.60 | 787.20 | 789.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 09:15:00 | 785.55 | 787.20 | 789.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 788.10 | 785.88 | 787.39 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 791.90 | 788.61 | 788.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 10:15:00 | 796.00 | 793.53 | 791.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 13:15:00 | 794.25 | 795.11 | 793.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 14:00:00 | 794.25 | 795.11 | 793.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 794.00 | 794.89 | 793.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 15:00:00 | 794.00 | 794.89 | 793.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 794.00 | 794.71 | 793.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:15:00 | 794.35 | 794.71 | 793.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 795.00 | 794.77 | 793.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 11:00:00 | 798.00 | 795.42 | 793.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 13:15:00 | 792.80 | 794.20 | 794.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 13:15:00 | 792.80 | 794.20 | 794.28 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2025-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 14:15:00 | 797.95 | 794.95 | 794.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 12:15:00 | 802.00 | 796.46 | 795.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 10:15:00 | 811.00 | 812.58 | 808.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 11:00:00 | 811.00 | 812.58 | 808.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 810.95 | 812.25 | 808.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:30:00 | 808.85 | 812.25 | 808.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 807.30 | 811.26 | 808.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:00:00 | 807.30 | 811.26 | 808.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 809.40 | 810.89 | 808.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 14:30:00 | 811.05 | 810.46 | 808.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 810.40 | 810.01 | 808.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 15:15:00 | 805.50 | 807.57 | 807.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 15:15:00 | 805.50 | 807.57 | 807.76 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 811.30 | 808.11 | 807.93 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 803.60 | 806.99 | 807.44 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 813.15 | 808.04 | 807.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 820.20 | 813.02 | 810.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 817.20 | 818.61 | 815.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 10:15:00 | 817.20 | 818.61 | 815.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 817.20 | 818.61 | 815.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:00:00 | 817.20 | 818.61 | 815.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 817.30 | 818.10 | 815.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:30:00 | 816.90 | 818.10 | 815.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 817.20 | 817.78 | 816.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 14:45:00 | 816.35 | 817.78 | 816.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 819.00 | 817.90 | 816.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:30:00 | 813.70 | 817.90 | 816.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 814.10 | 817.97 | 817.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 814.95 | 817.97 | 817.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 815.65 | 817.51 | 816.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:30:00 | 814.10 | 817.51 | 816.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 812.00 | 815.91 | 816.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 803.00 | 811.76 | 814.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 795.05 | 793.16 | 798.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 795.05 | 793.16 | 798.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 795.70 | 793.68 | 797.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 796.05 | 793.68 | 797.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 793.80 | 793.34 | 795.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:30:00 | 794.40 | 793.34 | 795.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 782.10 | 789.51 | 792.30 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2025-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 12:15:00 | 794.80 | 790.93 | 790.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 797.00 | 792.73 | 791.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 790.20 | 792.63 | 791.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 790.20 | 792.63 | 791.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 790.20 | 792.63 | 791.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 790.20 | 792.63 | 791.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 791.50 | 792.40 | 791.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:30:00 | 791.60 | 792.40 | 791.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 791.10 | 792.14 | 791.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:00:00 | 791.10 | 792.14 | 791.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 790.30 | 791.77 | 791.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:00:00 | 790.30 | 791.77 | 791.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 792.30 | 791.88 | 791.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:15:00 | 790.85 | 791.88 | 791.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 790.20 | 791.54 | 791.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:30:00 | 789.80 | 791.54 | 791.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2025-06-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 15:15:00 | 789.95 | 791.22 | 791.35 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 803.00 | 793.58 | 792.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 804.05 | 795.67 | 793.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 794.35 | 797.61 | 795.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 13:15:00 | 794.35 | 797.61 | 795.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 794.35 | 797.61 | 795.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:00:00 | 794.35 | 797.61 | 795.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 794.80 | 797.05 | 795.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 797.05 | 796.79 | 795.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 10:15:00 | 791.85 | 797.71 | 797.08 | SL hit (close<static) qty=1.00 sl=793.65 alert=retest2 |

### Cycle 93 — SELL (started 2025-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 12:15:00 | 794.20 | 796.50 | 796.61 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2025-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 15:15:00 | 797.50 | 796.63 | 796.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 806.40 | 798.59 | 797.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 815.35 | 815.51 | 809.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 10:00:00 | 815.35 | 815.51 | 809.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 815.05 | 817.73 | 814.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:00:00 | 815.05 | 817.73 | 814.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 814.45 | 817.07 | 814.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:45:00 | 814.60 | 817.07 | 814.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 810.50 | 815.76 | 814.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:45:00 | 811.35 | 815.76 | 814.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 813.45 | 815.30 | 814.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:30:00 | 810.50 | 815.30 | 814.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 10:15:00 | 811.25 | 813.55 | 813.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 14:15:00 | 807.20 | 811.21 | 812.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 812.30 | 809.16 | 810.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 14:15:00 | 812.30 | 809.16 | 810.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 812.30 | 809.16 | 810.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 14:30:00 | 812.20 | 809.16 | 810.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 811.30 | 809.59 | 810.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 807.80 | 809.59 | 810.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:00:00 | 810.20 | 809.81 | 810.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 14:15:00 | 813.00 | 809.13 | 809.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2025-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 14:15:00 | 813.00 | 809.13 | 809.13 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 13:15:00 | 808.45 | 809.85 | 809.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 806.65 | 808.57 | 809.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 12:15:00 | 809.30 | 808.41 | 809.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 12:15:00 | 809.30 | 808.41 | 809.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 809.30 | 808.41 | 809.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:30:00 | 809.80 | 808.41 | 809.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 809.10 | 808.55 | 809.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:45:00 | 810.30 | 808.55 | 809.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 808.70 | 808.58 | 809.04 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 09:15:00 | 813.55 | 809.57 | 809.41 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 13:15:00 | 808.65 | 809.39 | 809.41 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 820.50 | 811.45 | 810.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 11:15:00 | 825.60 | 818.65 | 815.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 14:15:00 | 830.15 | 831.23 | 826.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 15:00:00 | 830.15 | 831.23 | 826.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 828.75 | 830.73 | 826.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:15:00 | 827.00 | 830.73 | 826.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 823.25 | 829.24 | 826.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 823.25 | 829.24 | 826.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 821.80 | 827.75 | 825.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:00:00 | 821.80 | 827.75 | 825.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 823.00 | 824.48 | 824.61 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2025-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 12:15:00 | 826.50 | 824.64 | 824.60 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 14:15:00 | 824.30 | 824.57 | 824.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 09:15:00 | 822.30 | 823.87 | 824.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 13:15:00 | 817.45 | 816.95 | 819.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 14:00:00 | 817.45 | 816.95 | 819.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 819.95 | 817.55 | 819.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:45:00 | 821.90 | 817.55 | 819.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 820.05 | 818.05 | 819.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 819.55 | 818.05 | 819.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 817.90 | 815.78 | 817.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:45:00 | 817.10 | 815.78 | 817.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 815.90 | 815.81 | 817.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 10:15:00 | 813.65 | 815.66 | 817.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 11:15:00 | 801.70 | 797.20 | 796.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 11:15:00 | 801.70 | 797.20 | 796.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 11:15:00 | 803.25 | 800.32 | 798.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 09:15:00 | 800.40 | 802.71 | 800.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 800.40 | 802.71 | 800.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 800.40 | 802.71 | 800.83 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2025-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 13:15:00 | 799.10 | 799.91 | 799.93 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 804.95 | 800.92 | 800.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 15:15:00 | 805.60 | 801.86 | 800.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 09:15:00 | 801.85 | 801.86 | 800.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-08 10:00:00 | 801.85 | 801.86 | 800.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 800.95 | 801.67 | 800.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:00:00 | 800.95 | 801.67 | 800.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 799.40 | 801.22 | 800.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:00:00 | 799.40 | 801.22 | 800.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 799.30 | 800.84 | 800.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:00:00 | 799.30 | 800.84 | 800.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 805.25 | 801.72 | 801.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 09:15:00 | 818.00 | 802.41 | 801.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 15:15:00 | 825.40 | 827.98 | 828.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 825.40 | 827.98 | 828.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 09:15:00 | 822.70 | 826.92 | 827.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 12:15:00 | 819.00 | 818.64 | 821.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 13:00:00 | 819.00 | 818.64 | 821.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 804.10 | 803.52 | 806.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:45:00 | 803.00 | 803.68 | 806.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 809.60 | 805.77 | 806.34 | SL hit (close>static) qty=1.00 sl=808.30 alert=retest2 |

### Cycle 108 — BUY (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 12:15:00 | 806.90 | 806.72 | 806.70 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2025-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 13:15:00 | 803.85 | 806.15 | 806.44 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 810.50 | 807.13 | 806.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 14:15:00 | 812.15 | 809.43 | 808.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 11:15:00 | 810.50 | 810.65 | 809.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 12:15:00 | 810.45 | 810.65 | 809.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 809.15 | 810.41 | 809.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 811.10 | 810.30 | 809.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 09:15:00 | 807.50 | 809.74 | 809.31 | SL hit (close<static) qty=1.00 sl=808.00 alert=retest2 |

### Cycle 111 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 806.25 | 808.91 | 809.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 805.00 | 808.13 | 808.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 810.50 | 807.94 | 808.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 810.50 | 807.94 | 808.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 810.50 | 807.94 | 808.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:30:00 | 811.30 | 807.94 | 808.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 809.95 | 808.34 | 808.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:30:00 | 811.10 | 808.34 | 808.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 11:15:00 | 810.30 | 808.73 | 808.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 13:15:00 | 811.40 | 809.60 | 809.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 14:15:00 | 809.25 | 809.53 | 809.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 14:15:00 | 809.25 | 809.53 | 809.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 809.25 | 809.53 | 809.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 15:00:00 | 809.25 | 809.53 | 809.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 808.00 | 809.22 | 808.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:15:00 | 807.25 | 809.22 | 808.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 09:15:00 | 806.50 | 808.68 | 808.75 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 816.00 | 809.87 | 809.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 820.40 | 811.98 | 810.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 11:15:00 | 821.55 | 822.14 | 819.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 11:45:00 | 821.00 | 822.14 | 819.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 822.15 | 823.41 | 821.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 822.30 | 823.41 | 821.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 824.90 | 823.71 | 822.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 11:30:00 | 829.95 | 825.42 | 823.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 15:15:00 | 861.45 | 865.39 | 865.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 861.45 | 865.39 | 865.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 10:15:00 | 858.05 | 863.17 | 864.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 862.90 | 859.77 | 861.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 862.90 | 859.77 | 861.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 862.90 | 859.77 | 861.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 863.95 | 859.77 | 861.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 860.80 | 859.98 | 861.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:15:00 | 860.05 | 859.98 | 861.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 12:15:00 | 867.60 | 861.06 | 861.77 | SL hit (close>static) qty=1.00 sl=864.65 alert=retest2 |

### Cycle 116 — BUY (started 2025-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 14:15:00 | 871.00 | 863.66 | 862.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 09:15:00 | 874.85 | 866.91 | 864.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 09:15:00 | 871.05 | 871.72 | 868.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 09:15:00 | 871.05 | 871.72 | 868.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 871.05 | 871.72 | 868.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:00:00 | 871.05 | 871.72 | 868.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 866.15 | 870.61 | 868.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 866.15 | 870.61 | 868.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 864.90 | 869.47 | 868.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:30:00 | 865.75 | 869.47 | 868.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 14:15:00 | 863.05 | 867.35 | 867.48 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 870.10 | 867.55 | 867.52 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 13:15:00 | 865.30 | 867.29 | 867.46 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2025-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 13:15:00 | 868.90 | 867.74 | 867.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 14:15:00 | 874.05 | 869.00 | 868.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 869.75 | 869.92 | 868.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 09:15:00 | 869.75 | 869.92 | 868.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 869.75 | 869.92 | 868.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:30:00 | 871.40 | 869.92 | 868.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 868.60 | 869.66 | 868.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:15:00 | 869.00 | 869.66 | 868.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 869.35 | 869.59 | 868.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 12:15:00 | 869.70 | 869.59 | 868.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:15:00 | 870.15 | 869.58 | 868.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 14:15:00 | 863.80 | 868.54 | 868.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 14:15:00 | 863.80 | 868.54 | 868.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 859.00 | 865.64 | 867.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 11:15:00 | 861.60 | 860.94 | 863.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 11:15:00 | 861.60 | 860.94 | 863.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 861.60 | 860.94 | 863.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:45:00 | 861.50 | 860.94 | 863.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 860.80 | 860.81 | 862.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:00:00 | 860.80 | 860.81 | 862.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 861.85 | 861.02 | 862.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 861.85 | 861.02 | 862.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 861.40 | 861.09 | 862.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 870.00 | 861.09 | 862.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 879.25 | 864.73 | 864.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 11:15:00 | 882.15 | 870.68 | 867.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 09:15:00 | 879.20 | 881.83 | 877.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 10:00:00 | 879.20 | 881.83 | 877.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 877.10 | 880.88 | 877.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 877.10 | 880.88 | 877.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 875.80 | 879.86 | 877.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 879.15 | 877.27 | 876.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-06 09:15:00 | 967.07 | 952.62 | 947.53 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 14:15:00 | 952.00 | 953.25 | 953.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 15:15:00 | 950.55 | 952.71 | 953.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 13:15:00 | 951.50 | 950.86 | 951.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 13:15:00 | 951.50 | 950.86 | 951.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 951.50 | 950.86 | 951.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:30:00 | 953.10 | 950.86 | 951.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 953.35 | 951.36 | 951.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:30:00 | 953.25 | 951.36 | 951.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 953.50 | 951.79 | 952.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 958.20 | 951.79 | 952.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 957.95 | 953.02 | 952.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 962.45 | 957.46 | 955.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 958.00 | 958.54 | 956.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 12:15:00 | 958.00 | 958.54 | 956.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 958.00 | 958.54 | 956.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:30:00 | 957.90 | 958.54 | 956.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 954.95 | 957.83 | 956.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 954.95 | 957.83 | 956.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 954.00 | 957.06 | 956.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 954.00 | 957.06 | 956.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 954.50 | 956.55 | 956.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 959.00 | 956.55 | 956.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 958.00 | 958.32 | 957.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:00:00 | 958.00 | 958.32 | 957.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 976.50 | 972.78 | 969.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:15:00 | 977.80 | 973.50 | 970.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:45:00 | 978.55 | 974.69 | 971.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 12:15:00 | 977.95 | 978.29 | 977.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 14:15:00 | 972.95 | 976.65 | 976.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 972.95 | 976.65 | 976.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 972.45 | 975.81 | 976.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 977.30 | 976.11 | 976.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 977.30 | 976.11 | 976.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 977.30 | 976.11 | 976.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 977.30 | 976.11 | 976.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 974.00 | 975.69 | 976.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:00:00 | 973.35 | 975.14 | 975.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:30:00 | 973.00 | 974.31 | 975.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 15:00:00 | 971.00 | 974.31 | 975.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 978.55 | 974.42 | 975.21 | SL hit (close>static) qty=1.00 sl=977.60 alert=retest2 |

### Cycle 126 — BUY (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 10:15:00 | 982.05 | 975.95 | 975.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 11:15:00 | 985.40 | 977.84 | 976.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 13:15:00 | 988.00 | 988.14 | 983.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-26 13:45:00 | 988.00 | 988.14 | 983.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 983.65 | 987.24 | 983.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 14:45:00 | 983.50 | 987.24 | 983.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 983.05 | 986.40 | 983.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:30:00 | 980.60 | 984.96 | 983.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 979.35 | 983.84 | 983.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 979.35 | 983.84 | 983.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 11:15:00 | 975.15 | 982.10 | 982.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 13:15:00 | 968.95 | 978.40 | 980.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 10:15:00 | 982.00 | 977.35 | 979.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 10:15:00 | 982.00 | 977.35 | 979.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 982.00 | 977.35 | 979.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:00:00 | 982.00 | 977.35 | 979.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 980.45 | 977.97 | 979.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 13:00:00 | 979.00 | 978.17 | 979.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 15:15:00 | 976.35 | 978.99 | 979.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 985.00 | 979.77 | 979.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 985.00 | 979.77 | 979.71 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 11:15:00 | 975.00 | 979.57 | 979.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 13:15:00 | 974.00 | 977.91 | 978.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 10:15:00 | 976.00 | 975.97 | 977.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 10:15:00 | 976.00 | 975.97 | 977.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 976.00 | 975.97 | 977.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 12:00:00 | 974.25 | 975.63 | 977.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 13:15:00 | 974.95 | 975.50 | 976.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 14:00:00 | 974.00 | 975.20 | 976.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 13:15:00 | 967.35 | 958.41 | 958.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 967.35 | 958.41 | 958.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 971.35 | 961.00 | 959.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 10:15:00 | 962.40 | 963.15 | 961.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 10:15:00 | 962.40 | 963.15 | 961.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 962.40 | 963.15 | 961.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 962.40 | 963.15 | 961.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 958.40 | 962.20 | 960.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 958.40 | 962.20 | 960.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 959.05 | 961.57 | 960.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:30:00 | 957.15 | 961.57 | 960.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 952.85 | 959.83 | 959.99 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 12:15:00 | 961.00 | 959.56 | 959.55 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 15:15:00 | 959.05 | 959.46 | 959.51 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 960.30 | 959.63 | 959.59 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 10:15:00 | 958.85 | 959.47 | 959.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 13:15:00 | 956.80 | 958.64 | 959.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 14:15:00 | 960.65 | 959.04 | 959.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 14:15:00 | 960.65 | 959.04 | 959.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 960.65 | 959.04 | 959.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 15:00:00 | 960.65 | 959.04 | 959.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 958.00 | 958.83 | 959.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 965.00 | 958.83 | 959.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 962.00 | 959.47 | 959.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 11:15:00 | 967.00 | 962.40 | 961.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 963.65 | 964.60 | 963.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 963.65 | 964.60 | 963.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 963.65 | 964.60 | 963.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 963.65 | 964.60 | 963.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 964.90 | 964.66 | 963.35 | EMA400 retest candle locked (from upside) |

### Cycle 137 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 960.00 | 962.50 | 962.70 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 09:15:00 | 976.50 | 965.30 | 963.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 10:15:00 | 977.90 | 967.82 | 965.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 13:15:00 | 976.30 | 976.51 | 972.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-18 14:00:00 | 976.30 | 976.51 | 972.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 973.95 | 978.22 | 976.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 11:00:00 | 973.95 | 978.22 | 976.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 11:15:00 | 974.30 | 977.44 | 976.36 | EMA400 retest candle locked (from upside) |

### Cycle 139 — SELL (started 2025-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 15:15:00 | 974.50 | 975.62 | 975.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 11:15:00 | 973.75 | 975.09 | 975.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 12:15:00 | 975.25 | 975.12 | 975.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 12:15:00 | 975.25 | 975.12 | 975.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 975.25 | 975.12 | 975.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 975.25 | 975.12 | 975.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 972.15 | 974.36 | 975.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 13:15:00 | 970.00 | 973.67 | 974.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 14:15:00 | 970.20 | 973.16 | 974.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 13:45:00 | 970.00 | 969.13 | 971.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 11:45:00 | 970.30 | 966.75 | 967.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 13:15:00 | 973.45 | 968.80 | 968.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 973.45 | 968.80 | 968.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 978.25 | 971.96 | 970.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 09:15:00 | 1008.75 | 1014.40 | 1007.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 10:00:00 | 1008.75 | 1014.40 | 1007.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 1009.15 | 1013.03 | 1008.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:00:00 | 1009.15 | 1013.03 | 1008.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 1005.00 | 1011.42 | 1008.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:30:00 | 1004.90 | 1011.42 | 1008.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 1007.70 | 1010.68 | 1008.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:30:00 | 1009.25 | 1010.16 | 1008.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 10:00:00 | 1011.10 | 1010.16 | 1008.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 10:15:00 | 1004.65 | 1009.06 | 1008.05 | SL hit (close<static) qty=1.00 sl=1004.95 alert=retest2 |

### Cycle 141 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 998.45 | 1006.94 | 1007.18 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2026-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 13:15:00 | 1012.25 | 1004.57 | 1003.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 14:15:00 | 1015.05 | 1006.67 | 1004.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 11:15:00 | 1022.15 | 1023.34 | 1017.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-14 12:00:00 | 1022.15 | 1023.34 | 1017.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 1035.50 | 1038.94 | 1036.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 15:00:00 | 1035.50 | 1038.94 | 1036.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 1033.80 | 1037.91 | 1036.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:15:00 | 1035.80 | 1037.91 | 1036.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 1027.50 | 1035.67 | 1035.72 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 1043.40 | 1036.11 | 1035.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 15:15:00 | 1050.20 | 1043.33 | 1039.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 12:15:00 | 1042.15 | 1044.91 | 1041.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 12:15:00 | 1042.15 | 1044.91 | 1041.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 1042.15 | 1044.91 | 1041.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:00:00 | 1042.15 | 1044.91 | 1041.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 1033.65 | 1042.66 | 1040.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:00:00 | 1033.65 | 1042.66 | 1040.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 1030.65 | 1040.26 | 1039.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:45:00 | 1030.30 | 1040.26 | 1039.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — SELL (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 15:15:00 | 1028.40 | 1037.88 | 1038.90 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2026-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 13:15:00 | 1044.10 | 1040.13 | 1039.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 14:15:00 | 1053.10 | 1042.73 | 1040.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 14:15:00 | 1063.25 | 1065.87 | 1058.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 15:00:00 | 1063.25 | 1065.87 | 1058.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1045.30 | 1069.18 | 1066.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1045.30 | 1069.18 | 1066.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 1044.90 | 1064.32 | 1064.72 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 1058.90 | 1043.56 | 1042.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 1065.00 | 1047.85 | 1044.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 1065.60 | 1070.67 | 1065.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 1065.60 | 1070.67 | 1065.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1065.60 | 1070.67 | 1065.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 1065.60 | 1070.67 | 1065.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1056.80 | 1067.90 | 1064.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:30:00 | 1053.40 | 1067.90 | 1064.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 1058.00 | 1065.92 | 1064.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:15:00 | 1053.30 | 1065.92 | 1064.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 13:15:00 | 1058.90 | 1062.67 | 1063.01 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 1067.40 | 1063.62 | 1063.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 1128.40 | 1076.96 | 1069.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-16 09:15:00 | 1191.70 | 1193.78 | 1183.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-16 09:45:00 | 1190.10 | 1193.78 | 1183.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 1213.10 | 1216.19 | 1211.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:45:00 | 1214.20 | 1216.19 | 1211.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1207.50 | 1214.45 | 1211.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 1207.50 | 1214.45 | 1211.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 1213.10 | 1214.18 | 1211.21 | EMA400 retest candle locked (from upside) |

### Cycle 151 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 1206.30 | 1209.48 | 1209.70 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 14:15:00 | 1216.00 | 1209.89 | 1209.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 1229.60 | 1214.84 | 1211.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 11:15:00 | 1222.90 | 1224.20 | 1219.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 11:15:00 | 1222.90 | 1224.20 | 1219.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 1222.90 | 1224.20 | 1219.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 12:00:00 | 1222.90 | 1224.20 | 1219.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 1224.30 | 1224.44 | 1220.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:30:00 | 1223.90 | 1224.44 | 1220.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 1222.90 | 1224.13 | 1220.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 14:45:00 | 1223.40 | 1224.13 | 1220.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1220.90 | 1223.69 | 1221.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:45:00 | 1216.00 | 1223.69 | 1221.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 1213.10 | 1221.57 | 1220.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:45:00 | 1214.50 | 1221.57 | 1220.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — SELL (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 11:15:00 | 1209.90 | 1219.24 | 1219.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 12:15:00 | 1204.90 | 1216.37 | 1218.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 13:15:00 | 1200.00 | 1199.33 | 1206.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-26 14:00:00 | 1200.00 | 1199.33 | 1206.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 1208.80 | 1201.22 | 1206.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:45:00 | 1209.50 | 1201.22 | 1206.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 1210.50 | 1203.08 | 1207.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 09:30:00 | 1206.20 | 1202.84 | 1206.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 15:00:00 | 1202.40 | 1203.14 | 1205.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 14:15:00 | 1145.89 | 1153.99 | 1163.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 14:15:00 | 1142.28 | 1153.99 | 1163.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 1085.58 | 1135.60 | 1153.35 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 154 — BUY (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 13:15:00 | 1069.00 | 1063.78 | 1063.55 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 1052.00 | 1063.07 | 1063.44 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 1074.10 | 1061.07 | 1060.43 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 1019.70 | 1052.91 | 1057.25 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 1065.00 | 1047.38 | 1045.21 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 1039.50 | 1046.57 | 1047.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 1036.80 | 1044.61 | 1046.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1004.00 | 998.18 | 1014.49 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 10:15:00 | 1000.00 | 998.18 | 1014.49 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 1017.45 | 1002.32 | 1013.56 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-01 11:15:00 | 1017.45 | 1002.32 | 1013.56 | SL hit (close>ema400) qty=1.00 sl=1013.56 alert=retest1 |

### Cycle 160 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 1026.00 | 1012.53 | 1011.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 1034.70 | 1018.51 | 1014.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 1017.35 | 1020.84 | 1016.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 10:15:00 | 1017.35 | 1020.84 | 1016.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 1017.35 | 1020.84 | 1016.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 11:00:00 | 1017.35 | 1020.84 | 1016.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 1020.35 | 1020.74 | 1016.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 11:45:00 | 1021.80 | 1020.74 | 1016.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 12:15:00 | 1041.75 | 1051.91 | 1044.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:00:00 | 1041.75 | 1051.91 | 1044.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 1041.65 | 1049.85 | 1044.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 1056.80 | 1046.62 | 1043.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 1035.85 | 1054.97 | 1051.22 | SL hit (close<static) qty=1.00 sl=1038.30 alert=retest2 |

### Cycle 161 — SELL (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 12:15:00 | 1087.90 | 1098.45 | 1099.41 | EMA200 below EMA400 |

### Cycle 162 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 1108.95 | 1099.78 | 1098.72 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 12:15:00 | 1092.00 | 1102.45 | 1102.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 14:15:00 | 1085.90 | 1095.00 | 1097.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 1079.80 | 1075.81 | 1083.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 1079.80 | 1075.81 | 1083.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1079.80 | 1075.81 | 1083.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:30:00 | 1085.90 | 1075.81 | 1083.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 1073.90 | 1076.45 | 1082.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:30:00 | 1084.70 | 1076.45 | 1082.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 1068.40 | 1064.04 | 1069.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 11:00:00 | 1065.70 | 1064.37 | 1069.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:30:00 | 1064.30 | 1064.90 | 1068.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 14:15:00 | 1093.20 | 1071.62 | 1071.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 1093.20 | 1071.62 | 1071.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 15:15:00 | 1099.00 | 1077.09 | 1073.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 11:15:00 | 1081.20 | 1088.95 | 1084.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 11:15:00 | 1081.20 | 1088.95 | 1084.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 1081.20 | 1088.95 | 1084.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:00:00 | 1081.20 | 1088.95 | 1084.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 1078.30 | 1086.82 | 1083.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:00:00 | 1078.30 | 1086.82 | 1083.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — SELL (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 13:15:00 | 1020.90 | 1073.63 | 1078.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 15:15:00 | 1018.50 | 1054.07 | 1067.98 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-27 09:15:00 | 831.90 | 2024-05-29 09:15:00 | 825.60 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-05-27 10:15:00 | 829.35 | 2024-05-29 09:15:00 | 825.60 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-05-27 11:30:00 | 829.65 | 2024-05-29 09:15:00 | 825.60 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2024-05-28 11:45:00 | 829.95 | 2024-05-29 09:15:00 | 825.60 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-05-31 10:45:00 | 822.45 | 2024-05-31 14:15:00 | 830.70 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-06-13 15:15:00 | 844.90 | 2024-06-20 14:15:00 | 844.00 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2024-06-14 11:00:00 | 844.60 | 2024-06-21 09:15:00 | 843.40 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2024-06-18 09:30:00 | 846.70 | 2024-06-21 09:15:00 | 843.40 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2024-06-18 15:15:00 | 845.30 | 2024-06-21 09:15:00 | 843.40 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2024-06-19 09:15:00 | 846.70 | 2024-06-21 09:15:00 | 843.40 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2024-06-19 10:45:00 | 847.35 | 2024-06-21 09:15:00 | 843.40 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-06-19 11:30:00 | 849.10 | 2024-06-21 09:15:00 | 843.40 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-06-19 13:45:00 | 847.05 | 2024-06-21 09:15:00 | 843.40 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2024-06-20 11:30:00 | 850.50 | 2024-06-21 09:15:00 | 843.40 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-07-10 11:30:00 | 851.45 | 2024-07-10 12:15:00 | 848.25 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-07-19 13:30:00 | 895.95 | 2024-07-22 10:15:00 | 880.15 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-08-01 09:15:00 | 877.70 | 2024-08-01 14:15:00 | 864.45 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-08-09 11:30:00 | 811.40 | 2024-08-09 13:15:00 | 825.70 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-08-20 11:00:00 | 820.95 | 2024-08-21 13:15:00 | 813.05 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-08-20 12:30:00 | 820.65 | 2024-08-21 13:15:00 | 813.05 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-08-22 09:30:00 | 821.05 | 2024-08-23 14:15:00 | 815.25 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-08-22 10:00:00 | 820.35 | 2024-08-23 14:15:00 | 815.25 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-08-28 11:15:00 | 812.95 | 2024-08-30 09:15:00 | 818.80 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-08-28 11:45:00 | 812.80 | 2024-08-30 09:15:00 | 818.80 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-08-28 14:30:00 | 812.25 | 2024-08-30 09:15:00 | 818.80 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2024-08-29 11:30:00 | 812.20 | 2024-08-30 09:15:00 | 818.80 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-09-03 12:30:00 | 820.10 | 2024-09-04 09:15:00 | 815.75 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-09-03 14:00:00 | 821.45 | 2024-09-04 09:15:00 | 815.75 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-09-27 14:15:00 | 801.80 | 2024-09-30 09:15:00 | 793.20 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-10-03 09:15:00 | 794.60 | 2024-10-03 09:15:00 | 800.55 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-10-07 09:15:00 | 799.55 | 2024-10-07 09:15:00 | 796.50 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-10-07 09:45:00 | 798.70 | 2024-10-07 10:15:00 | 784.20 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2024-10-17 09:15:00 | 815.10 | 2024-10-22 11:15:00 | 798.10 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2024-10-17 10:00:00 | 811.00 | 2024-10-22 11:15:00 | 798.10 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-10-18 10:00:00 | 812.05 | 2024-10-22 11:15:00 | 798.10 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-10-25 10:15:00 | 778.70 | 2024-10-28 12:15:00 | 794.50 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-10-25 14:00:00 | 778.35 | 2024-10-28 12:15:00 | 794.50 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2024-11-01 18:00:00 | 824.35 | 2024-11-04 09:15:00 | 813.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-11-01 18:45:00 | 821.95 | 2024-11-04 09:15:00 | 813.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-11-19 14:00:00 | 805.60 | 2024-11-21 09:15:00 | 765.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-19 14:00:00 | 805.60 | 2024-11-22 09:15:00 | 797.90 | STOP_HIT | 0.50 | 0.96% |
| SELL | retest2 | 2024-12-24 11:30:00 | 815.75 | 2025-01-02 15:15:00 | 802.25 | STOP_HIT | 1.00 | 1.65% |
| SELL | retest2 | 2024-12-26 09:45:00 | 819.00 | 2025-01-02 15:15:00 | 802.25 | STOP_HIT | 1.00 | 2.05% |
| SELL | retest2 | 2025-01-14 12:15:00 | 749.45 | 2025-01-15 09:15:00 | 757.30 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-01-14 14:30:00 | 748.10 | 2025-01-15 09:15:00 | 757.30 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-01-24 14:15:00 | 745.45 | 2025-01-28 12:15:00 | 755.85 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-01-27 10:15:00 | 742.50 | 2025-01-28 12:15:00 | 755.85 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-02-01 14:15:00 | 765.80 | 2025-02-03 10:15:00 | 759.10 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-02-01 14:45:00 | 766.00 | 2025-02-03 10:15:00 | 759.10 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-02-05 13:45:00 | 771.85 | 2025-02-05 14:15:00 | 765.85 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-02-13 14:00:00 | 728.65 | 2025-02-19 09:15:00 | 732.70 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-02-13 15:00:00 | 727.45 | 2025-02-19 09:15:00 | 732.70 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-02-14 10:30:00 | 727.10 | 2025-02-19 09:15:00 | 732.70 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-03-11 10:15:00 | 730.30 | 2025-03-12 10:15:00 | 727.35 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-03-11 13:15:00 | 730.65 | 2025-03-12 10:15:00 | 727.35 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-03-12 09:15:00 | 732.85 | 2025-03-12 10:15:00 | 727.35 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-03-28 09:15:00 | 774.75 | 2025-04-02 09:15:00 | 767.40 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-03-28 09:45:00 | 777.90 | 2025-04-02 09:15:00 | 767.40 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-03-28 11:45:00 | 774.40 | 2025-04-02 09:15:00 | 767.40 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-03-28 14:15:00 | 775.20 | 2025-04-02 09:15:00 | 767.40 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-04-11 10:30:00 | 748.00 | 2025-04-15 09:15:00 | 769.60 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2025-04-24 12:30:00 | 816.55 | 2025-04-24 15:15:00 | 811.20 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-04-24 13:15:00 | 817.20 | 2025-04-24 15:15:00 | 811.20 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-04-24 14:15:00 | 816.65 | 2025-04-24 15:15:00 | 811.20 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-05-05 09:15:00 | 786.10 | 2025-05-12 09:15:00 | 794.95 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-05-15 13:30:00 | 803.00 | 2025-05-16 10:15:00 | 794.70 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-05-28 11:00:00 | 798.00 | 2025-05-29 13:15:00 | 792.80 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-06-03 14:30:00 | 811.05 | 2025-06-04 15:15:00 | 805.50 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-06-04 09:15:00 | 810.40 | 2025-06-04 15:15:00 | 805.50 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-06-25 09:15:00 | 797.05 | 2025-06-26 10:15:00 | 791.85 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-07-07 09:15:00 | 807.80 | 2025-07-08 14:15:00 | 813.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-07-07 11:00:00 | 810.20 | 2025-07-08 14:15:00 | 813.00 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-07-25 10:15:00 | 813.65 | 2025-08-05 11:15:00 | 801.70 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest2 | 2025-08-11 09:15:00 | 818.00 | 2025-08-21 15:15:00 | 825.40 | STOP_HIT | 1.00 | 0.90% |
| SELL | retest2 | 2025-09-01 11:45:00 | 803.00 | 2025-09-02 09:15:00 | 809.60 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-09-05 09:15:00 | 811.10 | 2025-09-05 09:15:00 | 807.50 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-09-16 11:30:00 | 829.95 | 2025-09-25 15:15:00 | 861.45 | STOP_HIT | 1.00 | 3.80% |
| SELL | retest2 | 2025-09-29 11:15:00 | 860.05 | 2025-09-29 12:15:00 | 867.60 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-10-07 12:15:00 | 869.70 | 2025-10-07 14:15:00 | 863.80 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-10-07 13:15:00 | 870.15 | 2025-10-07 14:15:00 | 863.80 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-10-15 09:15:00 | 879.15 | 2025-11-06 09:15:00 | 967.07 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-19 12:15:00 | 977.80 | 2025-11-21 14:15:00 | 972.95 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-11-19 12:45:00 | 978.55 | 2025-11-21 14:15:00 | 972.95 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-11-21 12:15:00 | 977.95 | 2025-11-21 14:15:00 | 972.95 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-11-24 14:00:00 | 973.35 | 2025-11-25 09:15:00 | 978.55 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-11-24 14:30:00 | 973.00 | 2025-11-25 09:15:00 | 978.55 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-11-24 15:00:00 | 971.00 | 2025-11-25 09:15:00 | 978.55 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-11-28 13:00:00 | 979.00 | 2025-12-01 09:15:00 | 985.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-11-28 15:15:00 | 976.35 | 2025-12-01 09:15:00 | 985.00 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-12-02 12:00:00 | 974.25 | 2025-12-05 13:15:00 | 967.35 | STOP_HIT | 1.00 | 0.71% |
| SELL | retest2 | 2025-12-02 13:15:00 | 974.95 | 2025-12-05 13:15:00 | 967.35 | STOP_HIT | 1.00 | 0.78% |
| SELL | retest2 | 2025-12-02 14:00:00 | 974.00 | 2025-12-05 13:15:00 | 967.35 | STOP_HIT | 1.00 | 0.68% |
| SELL | retest2 | 2025-12-24 13:15:00 | 970.00 | 2025-12-30 13:15:00 | 973.45 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-12-24 14:15:00 | 970.20 | 2025-12-30 13:15:00 | 973.45 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-12-26 13:45:00 | 970.00 | 2025-12-30 13:15:00 | 973.45 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-12-30 11:45:00 | 970.30 | 2025-12-30 13:15:00 | 973.45 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2026-01-08 09:30:00 | 1009.25 | 2026-01-08 10:15:00 | 1004.65 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2026-01-08 10:00:00 | 1011.10 | 2026-01-08 10:15:00 | 1004.65 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2026-02-27 09:30:00 | 1206.20 | 2026-03-06 14:15:00 | 1145.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 15:00:00 | 1202.40 | 2026-03-06 14:15:00 | 1142.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 09:30:00 | 1206.20 | 2026-03-09 09:15:00 | 1085.58 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-27 15:00:00 | 1202.40 | 2026-03-09 09:15:00 | 1082.16 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2026-04-01 10:15:00 | 1000.00 | 2026-04-01 11:15:00 | 1017.45 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-04-10 09:15:00 | 1056.80 | 2026-04-13 09:15:00 | 1035.85 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-04-13 11:45:00 | 1045.60 | 2026-04-23 12:15:00 | 1087.90 | STOP_HIT | 1.00 | 4.05% |
| SELL | retest2 | 2026-05-06 11:00:00 | 1065.70 | 2026-05-06 14:15:00 | 1093.20 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2026-05-06 12:30:00 | 1064.30 | 2026-05-06 14:15:00 | 1093.20 | STOP_HIT | 1.00 | -2.72% |
