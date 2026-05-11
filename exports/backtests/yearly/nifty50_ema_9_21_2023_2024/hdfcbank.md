# HDFCBANK (HDFCBANK)

## Backtest Summary

- **Window:** 2023-03-13 09:15:00 → 2026-05-08 15:15:00 (5443 bars)
- **Last close:** 781.25
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 235 |
| ALERT1 | 161 |
| ALERT2 | 157 |
| ALERT2_SKIP | 81 |
| ALERT3 | 412 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 187 |
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 188 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 194 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 62 / 132
- **Target hits / Stop hits / Partials:** 3 / 188 / 3
- **Avg / median % per leg:** 0.06% / -0.43%
- **Sum % (uncompounded):** 10.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 86 | 22 | 25.6% | 3 | 83 | 0 | 0.43% | 37.0% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 3 | 0 | -0.25% | -0.7% |
| BUY @ 3rd Alert (retest2) | 83 | 21 | 25.3% | 3 | 80 | 0 | 0.45% | 37.7% |
| SELL (all) | 108 | 40 | 37.0% | 0 | 105 | 3 | -0.24% | -26.3% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.53% | 9.1% |
| SELL @ 3rd Alert (retest2) | 106 | 38 | 35.8% | 0 | 104 | 2 | -0.33% | -35.3% |
| retest1 (combined) | 5 | 3 | 60.0% | 0 | 4 | 1 | 1.66% | 8.3% |
| retest2 (combined) | 189 | 59 | 31.2% | 3 | 184 | 2 | 0.01% | 2.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 12:15:00 | 826.98 | 831.87 | 832.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-16 13:15:00 | 825.43 | 830.58 | 831.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-18 09:15:00 | 823.38 | 821.26 | 824.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 09:15:00 | 823.38 | 821.26 | 824.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 09:15:00 | 823.38 | 821.26 | 824.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-19 09:45:00 | 820.25 | 821.64 | 823.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-19 13:15:00 | 820.03 | 820.86 | 822.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 10:30:00 | 820.63 | 821.87 | 822.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 11:30:00 | 820.58 | 821.59 | 822.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 14:15:00 | 819.45 | 820.74 | 821.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 15:00:00 | 819.45 | 820.74 | 821.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 15:15:00 | 819.23 | 820.44 | 821.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-23 09:15:00 | 821.35 | 820.44 | 821.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 09:15:00 | 820.75 | 820.50 | 821.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-23 13:00:00 | 819.25 | 820.65 | 821.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-23 14:30:00 | 820.25 | 820.22 | 820.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-29 10:15:00 | 819.88 | 811.03 | 809.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-29 10:15:00 | 819.88 | 811.03 | 809.97 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 11:15:00 | 806.95 | 813.13 | 813.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 12:15:00 | 804.05 | 811.32 | 812.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-02 10:15:00 | 804.20 | 804.19 | 806.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-02 11:00:00 | 804.20 | 804.19 | 806.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 13:15:00 | 803.50 | 804.22 | 806.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-02 13:45:00 | 805.53 | 804.22 | 806.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 09:15:00 | 806.60 | 804.34 | 805.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-05 12:15:00 | 805.00 | 805.03 | 805.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-08 09:15:00 | 809.00 | 803.48 | 802.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2023-06-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-08 09:15:00 | 809.00 | 803.48 | 802.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-08 10:15:00 | 811.23 | 805.03 | 803.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 14:15:00 | 803.38 | 806.06 | 804.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 14:15:00 | 803.38 | 806.06 | 804.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 14:15:00 | 803.38 | 806.06 | 804.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 15:00:00 | 803.38 | 806.06 | 804.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 15:15:00 | 804.18 | 805.69 | 804.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-09 09:45:00 | 806.88 | 805.96 | 804.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-12 11:15:00 | 802.50 | 804.72 | 805.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-12 11:15:00 | 802.50 | 804.72 | 805.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-12 12:15:00 | 801.00 | 803.97 | 804.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-13 10:15:00 | 802.30 | 802.30 | 803.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-13 10:30:00 | 802.50 | 802.30 | 803.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 11:15:00 | 801.95 | 802.23 | 803.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 11:30:00 | 803.13 | 802.23 | 803.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 15:15:00 | 803.98 | 802.17 | 802.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-14 09:15:00 | 803.60 | 802.17 | 802.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 09:15:00 | 802.20 | 802.18 | 802.81 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2023-06-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 12:15:00 | 804.78 | 803.44 | 803.30 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-14 14:15:00 | 801.23 | 803.03 | 803.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-15 10:15:00 | 799.70 | 801.73 | 802.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-16 09:15:00 | 797.25 | 795.50 | 798.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-16 10:00:00 | 797.25 | 795.50 | 798.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 10:15:00 | 799.40 | 796.28 | 798.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-16 11:00:00 | 799.40 | 796.28 | 798.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 11:15:00 | 798.28 | 796.68 | 798.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-16 11:30:00 | 799.18 | 796.68 | 798.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 12:15:00 | 799.00 | 797.15 | 798.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-16 13:00:00 | 799.00 | 797.15 | 798.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 13:15:00 | 799.20 | 797.56 | 798.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-16 13:30:00 | 799.75 | 797.56 | 798.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 14:15:00 | 800.45 | 798.14 | 798.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-16 14:45:00 | 804.00 | 798.14 | 798.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 15:15:00 | 801.00 | 798.71 | 798.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-19 09:15:00 | 804.43 | 798.71 | 798.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2023-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-19 09:15:00 | 805.00 | 799.97 | 799.52 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 10:15:00 | 796.10 | 799.87 | 800.11 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-20 14:15:00 | 803.55 | 800.55 | 800.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-20 15:15:00 | 804.70 | 801.38 | 800.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-23 13:15:00 | 821.85 | 823.98 | 819.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-23 14:00:00 | 821.85 | 823.98 | 819.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 819.15 | 822.32 | 819.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-26 10:00:00 | 819.15 | 822.32 | 819.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 10:15:00 | 820.33 | 821.92 | 819.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-27 09:30:00 | 822.35 | 819.88 | 819.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-27 11:45:00 | 822.00 | 820.56 | 819.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-05 11:15:00 | 838.48 | 851.53 | 853.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2023-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-05 11:15:00 | 838.48 | 851.53 | 853.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-06 09:15:00 | 830.93 | 841.16 | 846.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-06 11:15:00 | 842.15 | 840.61 | 845.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-06 11:45:00 | 841.50 | 840.61 | 845.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 833.93 | 832.89 | 837.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-10 09:30:00 | 833.68 | 832.89 | 837.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 10:15:00 | 835.63 | 831.35 | 833.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 11:00:00 | 835.63 | 831.35 | 833.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 11:15:00 | 835.70 | 832.22 | 833.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 11:45:00 | 837.58 | 832.22 | 833.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 14:15:00 | 824.60 | 831.37 | 833.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-12 13:45:00 | 822.33 | 827.51 | 830.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-12 14:45:00 | 821.25 | 824.95 | 828.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-13 14:30:00 | 820.98 | 823.96 | 826.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-14 09:45:00 | 822.48 | 823.25 | 825.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 13:15:00 | 818.63 | 821.71 | 824.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-17 09:15:00 | 817.83 | 822.21 | 823.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-17 11:15:00 | 824.25 | 822.32 | 823.49 | SL hit (close>static) qty=1.00 sl=824.05 alert=retest2 |

### Cycle 12 — BUY (started 2023-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 12:15:00 | 833.63 | 824.58 | 824.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 13:15:00 | 834.03 | 826.47 | 825.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 14:15:00 | 837.73 | 837.79 | 833.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-18 14:45:00 | 838.98 | 837.79 | 833.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 841.75 | 842.77 | 840.43 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 11:15:00 | 836.05 | 839.42 | 839.73 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 09:15:00 | 844.80 | 840.29 | 839.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 12:15:00 | 847.33 | 843.24 | 841.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-26 13:15:00 | 845.70 | 845.88 | 844.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-26 13:45:00 | 846.55 | 845.88 | 844.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 15:15:00 | 844.65 | 845.49 | 844.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-27 09:15:00 | 849.85 | 845.49 | 844.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-27 11:15:00 | 842.10 | 845.16 | 844.46 | SL hit (close<static) qty=1.00 sl=844.08 alert=retest2 |

### Cycle 15 — SELL (started 2023-07-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 13:15:00 | 837.38 | 842.84 | 843.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-28 09:15:00 | 827.03 | 838.16 | 841.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-01 09:15:00 | 828.83 | 825.97 | 829.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-01 09:15:00 | 828.83 | 825.97 | 829.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 09:15:00 | 828.83 | 825.97 | 829.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-01 10:00:00 | 828.83 | 825.97 | 829.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 10:15:00 | 826.03 | 825.98 | 828.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-02 09:15:00 | 824.65 | 828.67 | 829.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-04 15:15:00 | 826.43 | 822.51 | 822.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2023-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 15:15:00 | 826.43 | 822.51 | 822.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 09:15:00 | 828.70 | 823.75 | 822.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 09:15:00 | 822.75 | 824.57 | 823.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 09:15:00 | 822.75 | 824.57 | 823.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 09:15:00 | 822.75 | 824.57 | 823.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 09:45:00 | 823.20 | 824.57 | 823.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 10:15:00 | 824.80 | 824.61 | 823.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-08 12:00:00 | 827.35 | 825.16 | 824.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-08 13:00:00 | 826.00 | 825.33 | 824.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-09 10:15:00 | 817.63 | 823.31 | 823.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2023-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 10:15:00 | 817.63 | 823.31 | 823.80 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-08-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-10 09:15:00 | 826.60 | 823.58 | 823.42 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-08-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 10:15:00 | 820.15 | 822.89 | 823.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-10 13:15:00 | 816.95 | 821.12 | 822.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 14:15:00 | 803.35 | 801.72 | 805.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-16 15:00:00 | 803.35 | 801.72 | 805.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 796.20 | 796.06 | 798.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-21 15:15:00 | 794.00 | 796.04 | 797.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-22 11:00:00 | 794.48 | 795.30 | 796.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-22 12:15:00 | 794.40 | 795.18 | 796.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-22 13:00:00 | 794.13 | 794.97 | 796.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 12:15:00 | 792.15 | 791.85 | 793.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 12:45:00 | 793.03 | 791.85 | 793.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 13:15:00 | 793.50 | 792.18 | 793.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 14:00:00 | 793.50 | 792.18 | 793.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 14:15:00 | 793.38 | 792.42 | 793.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 14:45:00 | 793.05 | 792.42 | 793.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 15:15:00 | 794.25 | 792.79 | 793.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-24 09:15:00 | 795.50 | 792.79 | 793.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 09:15:00 | 796.78 | 793.59 | 794.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-24 09:45:00 | 796.60 | 793.59 | 794.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 10:15:00 | 793.25 | 793.52 | 794.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-24 11:15:00 | 792.13 | 793.52 | 794.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-24 12:00:00 | 792.03 | 793.22 | 793.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-29 09:15:00 | 793.60 | 788.49 | 788.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2023-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 09:15:00 | 793.60 | 788.49 | 788.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 14:15:00 | 795.08 | 792.46 | 790.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 14:15:00 | 789.73 | 794.13 | 792.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 14:15:00 | 789.73 | 794.13 | 792.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 14:15:00 | 789.73 | 794.13 | 792.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-30 15:00:00 | 789.73 | 794.13 | 792.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 15:15:00 | 789.50 | 793.21 | 792.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 09:15:00 | 782.93 | 793.21 | 792.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2023-08-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 09:15:00 | 786.13 | 791.79 | 791.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-01 11:15:00 | 780.95 | 785.29 | 787.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 13:15:00 | 785.90 | 785.17 | 787.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-01 14:00:00 | 785.90 | 785.17 | 787.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 14:15:00 | 787.50 | 785.64 | 787.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 14:30:00 | 788.98 | 785.64 | 787.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 15:15:00 | 786.75 | 785.86 | 787.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 09:15:00 | 791.50 | 785.86 | 787.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 790.00 | 786.69 | 787.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-04 10:30:00 | 788.50 | 787.43 | 787.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-04 11:15:00 | 789.80 | 787.90 | 787.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2023-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 11:15:00 | 789.80 | 787.90 | 787.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 14:15:00 | 792.43 | 789.69 | 788.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 10:15:00 | 788.90 | 789.93 | 789.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-05 10:15:00 | 788.90 | 789.93 | 789.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 10:15:00 | 788.90 | 789.93 | 789.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-05 11:00:00 | 788.90 | 789.93 | 789.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 11:15:00 | 788.08 | 789.56 | 789.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-05 12:00:00 | 788.08 | 789.56 | 789.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 12:15:00 | 787.80 | 789.21 | 788.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-05 13:00:00 | 787.80 | 789.21 | 788.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2023-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-05 13:15:00 | 787.10 | 788.79 | 788.79 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 09:15:00 | 792.05 | 789.09 | 788.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 13:15:00 | 794.88 | 791.73 | 790.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-13 15:15:00 | 820.80 | 821.27 | 817.81 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-14 09:15:00 | 822.98 | 821.27 | 817.81 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 12:15:00 | 820.95 | 821.02 | 818.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-14 12:30:00 | 819.25 | 821.02 | 818.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 822.38 | 826.28 | 823.92 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-09-18 09:15:00 | 822.38 | 826.28 | 823.92 | SL hit (close<ema400) qty=1.00 sl=823.92 alert=retest1 |

### Cycle 25 — SELL (started 2023-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 12:15:00 | 817.28 | 822.57 | 822.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 14:15:00 | 813.80 | 819.84 | 821.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 09:15:00 | 767.25 | 767.05 | 773.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-26 10:00:00 | 767.25 | 767.05 | 773.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 09:15:00 | 765.28 | 764.18 | 766.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-28 13:45:00 | 761.40 | 764.24 | 765.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-28 15:00:00 | 761.30 | 763.66 | 765.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-03 09:15:00 | 755.30 | 763.89 | 764.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-04 10:45:00 | 762.00 | 757.57 | 759.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 11:15:00 | 759.43 | 757.95 | 759.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-04 11:30:00 | 762.10 | 757.95 | 759.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 12:15:00 | 761.38 | 758.63 | 759.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-04 13:00:00 | 761.38 | 758.63 | 759.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 13:15:00 | 763.43 | 759.59 | 760.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-04 14:00:00 | 763.43 | 759.59 | 760.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-10-04 15:15:00 | 765.50 | 761.64 | 761.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2023-10-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-04 15:15:00 | 765.50 | 761.64 | 761.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-05 09:15:00 | 768.15 | 762.94 | 761.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-05 14:15:00 | 767.60 | 767.83 | 765.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-05 15:00:00 | 767.60 | 767.83 | 765.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 14:15:00 | 767.30 | 769.02 | 767.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-06 15:00:00 | 767.30 | 769.02 | 767.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 15:15:00 | 767.50 | 768.72 | 767.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-09 09:15:00 | 759.50 | 768.72 | 767.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 761.75 | 767.32 | 766.73 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 10:15:00 | 758.40 | 765.54 | 765.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 12:15:00 | 757.40 | 762.81 | 764.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 10:15:00 | 760.30 | 760.13 | 762.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-10 10:30:00 | 759.95 | 760.13 | 762.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 11:15:00 | 763.50 | 760.81 | 762.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 12:00:00 | 763.50 | 760.81 | 762.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 12:15:00 | 761.08 | 760.86 | 762.32 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 10:15:00 | 766.50 | 763.12 | 762.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 14:15:00 | 770.13 | 765.87 | 764.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-12 09:15:00 | 766.40 | 766.80 | 765.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-12 10:00:00 | 766.40 | 766.80 | 765.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 10:15:00 | 769.03 | 770.63 | 768.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 10:45:00 | 768.88 | 770.63 | 768.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 13:15:00 | 772.50 | 770.89 | 769.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 13:45:00 | 769.38 | 770.89 | 769.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 14:15:00 | 768.00 | 770.31 | 769.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 15:00:00 | 768.00 | 770.31 | 769.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 15:15:00 | 766.75 | 769.60 | 768.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-16 09:15:00 | 762.53 | 769.60 | 768.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2023-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 10:15:00 | 763.18 | 767.31 | 767.86 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 11:15:00 | 771.73 | 768.29 | 767.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-17 12:15:00 | 772.28 | 769.09 | 768.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 09:15:00 | 767.08 | 769.35 | 768.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 09:15:00 | 767.08 | 769.35 | 768.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 09:15:00 | 767.08 | 769.35 | 768.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 09:45:00 | 765.25 | 769.35 | 768.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2023-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 10:15:00 | 760.78 | 767.64 | 767.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 15:15:00 | 759.55 | 762.75 | 765.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 11:15:00 | 762.65 | 761.08 | 763.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-19 12:00:00 | 762.65 | 761.08 | 763.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 12:15:00 | 761.60 | 759.94 | 761.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 13:00:00 | 761.60 | 759.94 | 761.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 13:15:00 | 761.13 | 760.18 | 761.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 14:30:00 | 760.00 | 760.43 | 761.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 10:15:00 | 759.50 | 760.73 | 761.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 14:15:00 | 760.48 | 761.29 | 761.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-03 09:15:00 | 743.75 | 739.28 | 738.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2023-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 09:15:00 | 743.75 | 739.28 | 738.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 14:15:00 | 747.48 | 743.98 | 742.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 09:15:00 | 741.83 | 744.02 | 742.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 09:15:00 | 741.83 | 744.02 | 742.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 741.83 | 744.02 | 742.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 10:00:00 | 741.83 | 744.02 | 742.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 10:15:00 | 742.70 | 743.75 | 742.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-07 15:15:00 | 744.58 | 742.90 | 742.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 09:45:00 | 744.38 | 743.25 | 742.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 10:30:00 | 744.55 | 743.12 | 742.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 12:00:00 | 745.45 | 743.59 | 742.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 14:15:00 | 745.85 | 744.39 | 743.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 14:30:00 | 743.90 | 744.39 | 743.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 744.08 | 744.57 | 743.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-09 11:45:00 | 745.50 | 744.62 | 743.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-09 13:15:00 | 744.88 | 744.63 | 743.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-10 09:15:00 | 741.45 | 743.43 | 743.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 09:15:00 | 741.45 | 743.43 | 743.59 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 14:15:00 | 745.43 | 743.74 | 743.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-10 15:15:00 | 747.50 | 744.49 | 744.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-13 10:15:00 | 743.25 | 745.30 | 744.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-13 10:15:00 | 743.25 | 745.30 | 744.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 10:15:00 | 743.25 | 745.30 | 744.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-13 11:00:00 | 743.25 | 745.30 | 744.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 11:15:00 | 745.50 | 745.34 | 744.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-13 13:15:00 | 746.50 | 745.40 | 744.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-13 13:45:00 | 746.68 | 745.68 | 744.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-15 09:15:00 | 754.58 | 745.25 | 744.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2023-12-08 09:15:00 | 821.15 | 816.33 | 811.70 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2023-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 15:15:00 | 817.23 | 821.61 | 821.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-13 09:15:00 | 811.18 | 819.53 | 820.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-14 09:15:00 | 827.10 | 818.75 | 819.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 09:15:00 | 827.10 | 818.75 | 819.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 827.10 | 818.75 | 819.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 09:45:00 | 829.15 | 818.75 | 819.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2023-12-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 10:15:00 | 824.48 | 819.90 | 819.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 15:15:00 | 828.03 | 825.18 | 823.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 14:15:00 | 827.35 | 827.84 | 825.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-18 15:00:00 | 827.35 | 827.84 | 825.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 824.65 | 826.95 | 825.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 11:15:00 | 827.90 | 827.03 | 825.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-20 09:15:00 | 832.50 | 826.51 | 826.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 09:15:00 | 831.80 | 828.23 | 827.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-01 15:15:00 | 846.45 | 850.35 | 850.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2024-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-01 15:15:00 | 846.45 | 850.35 | 850.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-03 09:15:00 | 845.70 | 848.53 | 849.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-04 09:15:00 | 843.05 | 840.99 | 844.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-04 10:00:00 | 843.05 | 840.99 | 844.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 10:15:00 | 843.98 | 841.58 | 844.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-04 11:15:00 | 843.53 | 841.58 | 844.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 11:15:00 | 841.65 | 841.60 | 844.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-04 14:00:00 | 840.75 | 841.90 | 843.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-04 15:15:00 | 845.93 | 843.27 | 844.09 | SL hit (close>static) qty=1.00 sl=845.48 alert=retest2 |

### Cycle 38 — BUY (started 2024-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 11:15:00 | 833.83 | 828.49 | 827.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-15 12:15:00 | 835.93 | 829.98 | 828.57 | Break + close above crossover candle high |

### Cycle 39 — SELL (started 2024-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 09:15:00 | 790.60 | 827.48 | 829.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 12:15:00 | 778.58 | 805.17 | 817.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-20 10:15:00 | 744.50 | 742.46 | 756.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-20 11:00:00 | 744.50 | 742.46 | 756.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 15:15:00 | 728.00 | 723.91 | 729.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 09:15:00 | 724.68 | 723.91 | 729.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-29 09:30:00 | 726.60 | 719.87 | 723.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-29 12:15:00 | 729.98 | 723.94 | 724.48 | SL hit (close>static) qty=1.00 sl=729.15 alert=retest2 |

### Cycle 40 — BUY (started 2024-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 14:15:00 | 727.95 | 725.04 | 724.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 15:15:00 | 728.48 | 725.73 | 725.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 09:15:00 | 725.05 | 725.59 | 725.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 09:15:00 | 725.05 | 725.59 | 725.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 09:15:00 | 725.05 | 725.59 | 725.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 09:45:00 | 724.50 | 725.59 | 725.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2024-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 10:15:00 | 722.25 | 724.92 | 724.94 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2024-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 09:15:00 | 733.13 | 725.21 | 724.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 10:15:00 | 734.75 | 727.12 | 725.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-02 12:15:00 | 727.28 | 732.38 | 731.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-02 12:15:00 | 727.28 | 732.38 | 731.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 12:15:00 | 727.28 | 732.38 | 731.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 13:00:00 | 727.28 | 732.38 | 731.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 13:15:00 | 727.50 | 731.40 | 730.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 13:30:00 | 726.15 | 731.40 | 730.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2024-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 14:15:00 | 723.23 | 729.77 | 730.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 09:15:00 | 720.18 | 726.76 | 728.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-05 12:15:00 | 725.30 | 725.10 | 727.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-05 13:00:00 | 725.30 | 725.10 | 727.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 13:15:00 | 722.63 | 721.88 | 724.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 13:45:00 | 723.78 | 721.88 | 724.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 14:15:00 | 721.28 | 721.76 | 723.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 14:30:00 | 723.05 | 721.76 | 723.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 09:15:00 | 720.60 | 721.53 | 723.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 11:15:00 | 718.40 | 721.12 | 723.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-14 09:15:00 | 682.48 | 695.38 | 698.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-14 14:15:00 | 690.65 | 689.58 | 694.25 | SL hit (close>ema200) qty=0.50 sl=689.58 alert=retest2 |

### Cycle 44 — BUY (started 2024-02-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 12:15:00 | 699.60 | 696.80 | 696.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 13:15:00 | 706.08 | 698.65 | 697.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 14:15:00 | 708.48 | 709.37 | 706.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-19 15:00:00 | 708.48 | 709.37 | 706.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 717.43 | 711.02 | 707.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 10:15:00 | 720.25 | 711.02 | 707.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-22 11:15:00 | 708.65 | 715.15 | 715.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2024-02-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 11:15:00 | 708.65 | 715.15 | 715.82 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2024-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-26 13:15:00 | 716.88 | 713.58 | 713.44 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 14:15:00 | 711.20 | 713.11 | 713.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 15:15:00 | 710.98 | 712.68 | 713.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 12:15:00 | 711.53 | 711.11 | 712.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-27 13:00:00 | 711.53 | 711.11 | 712.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 13:15:00 | 713.00 | 711.48 | 712.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 14:00:00 | 713.00 | 711.48 | 712.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 14:15:00 | 709.88 | 711.16 | 711.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 10:45:00 | 706.03 | 710.04 | 711.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 12:15:00 | 708.63 | 705.55 | 706.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 13:00:00 | 708.53 | 706.15 | 706.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-01 13:15:00 | 712.80 | 707.48 | 707.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2024-03-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 13:15:00 | 712.80 | 707.48 | 707.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 14:15:00 | 716.05 | 709.19 | 707.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 11:15:00 | 713.55 | 713.60 | 711.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-04 12:00:00 | 713.55 | 713.60 | 711.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 09:15:00 | 712.88 | 714.58 | 712.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 12:45:00 | 717.40 | 714.84 | 713.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-11 10:15:00 | 717.08 | 720.83 | 719.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-11 11:15:00 | 714.68 | 719.03 | 719.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2024-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 11:15:00 | 714.68 | 719.03 | 719.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 12:15:00 | 713.50 | 717.92 | 718.75 | Break + close below crossover candle low |

### Cycle 50 — BUY (started 2024-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-12 09:15:00 | 730.00 | 718.69 | 718.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-12 15:15:00 | 730.75 | 726.79 | 723.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 11:15:00 | 728.23 | 729.11 | 725.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-13 12:00:00 | 728.23 | 729.11 | 725.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 12:15:00 | 728.55 | 729.00 | 725.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 12:45:00 | 726.90 | 729.00 | 725.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 723.70 | 728.14 | 726.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-14 09:30:00 | 722.55 | 728.14 | 726.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 10:15:00 | 725.10 | 727.53 | 726.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-14 11:00:00 | 725.10 | 727.53 | 726.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 13:15:00 | 726.15 | 727.77 | 726.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-14 14:00:00 | 726.15 | 727.77 | 726.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 14:15:00 | 727.40 | 727.70 | 726.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-14 15:15:00 | 728.50 | 727.70 | 726.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-15 09:15:00 | 723.85 | 727.06 | 726.68 | SL hit (close<static) qty=1.00 sl=724.95 alert=retest2 |

### Cycle 51 — SELL (started 2024-03-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 11:15:00 | 723.95 | 726.15 | 726.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-18 09:15:00 | 720.90 | 725.05 | 725.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-18 11:15:00 | 725.10 | 724.25 | 725.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-18 11:15:00 | 725.10 | 724.25 | 725.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 11:15:00 | 725.10 | 724.25 | 725.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 12:00:00 | 725.10 | 724.25 | 725.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 12:15:00 | 726.00 | 724.60 | 725.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 12:30:00 | 725.23 | 724.60 | 725.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 13:15:00 | 725.73 | 724.83 | 725.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 13:45:00 | 726.15 | 724.83 | 725.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 14:15:00 | 722.50 | 724.36 | 725.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 14:30:00 | 727.40 | 724.36 | 725.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 11:15:00 | 724.00 | 723.65 | 724.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 13:30:00 | 722.53 | 723.10 | 724.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-20 10:00:00 | 720.50 | 723.12 | 723.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-21 11:30:00 | 722.25 | 720.66 | 721.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-21 13:45:00 | 722.65 | 721.25 | 721.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 14:15:00 | 722.60 | 721.52 | 721.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 14:45:00 | 722.98 | 721.52 | 721.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 10:15:00 | 722.50 | 721.71 | 721.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-22 11:00:00 | 722.50 | 721.71 | 721.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-03-22 11:15:00 | 722.10 | 721.79 | 721.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2024-03-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 11:15:00 | 722.10 | 721.79 | 721.76 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-03-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 14:15:00 | 721.15 | 721.69 | 721.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-26 09:15:00 | 716.20 | 720.53 | 721.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 09:15:00 | 717.00 | 716.11 | 718.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-27 10:00:00 | 717.00 | 716.11 | 718.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 10:15:00 | 715.90 | 716.07 | 717.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 10:30:00 | 716.90 | 716.07 | 717.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 720.40 | 716.94 | 717.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 15:00:00 | 720.40 | 716.94 | 717.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 721.98 | 717.95 | 718.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 09:15:00 | 724.20 | 717.95 | 718.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2024-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 09:15:00 | 725.85 | 719.53 | 718.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 10:15:00 | 728.95 | 721.41 | 719.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-28 14:15:00 | 723.00 | 724.24 | 721.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-28 15:00:00 | 723.00 | 724.24 | 721.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 14:15:00 | 740.98 | 742.44 | 738.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 14:30:00 | 740.03 | 742.44 | 738.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 09:15:00 | 771.40 | 772.60 | 770.25 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2024-04-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 09:15:00 | 758.25 | 767.53 | 768.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 10:15:00 | 757.98 | 765.62 | 767.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 11:15:00 | 752.00 | 751.37 | 755.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 12:00:00 | 752.00 | 751.37 | 755.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 14:15:00 | 755.08 | 752.15 | 755.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 15:00:00 | 755.08 | 752.15 | 755.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 15:15:00 | 756.00 | 752.92 | 755.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 09:15:00 | 757.83 | 752.92 | 755.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 753.25 | 752.99 | 755.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 10:15:00 | 752.83 | 752.99 | 755.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 13:15:00 | 750.58 | 754.29 | 755.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 11:45:00 | 752.20 | 751.77 | 753.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-19 12:15:00 | 759.45 | 753.31 | 753.76 | SL hit (close>static) qty=1.00 sl=759.43 alert=retest2 |

### Cycle 56 — BUY (started 2024-04-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-19 13:15:00 | 763.08 | 755.26 | 754.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-19 14:15:00 | 765.88 | 757.39 | 755.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-22 11:15:00 | 758.40 | 759.28 | 757.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-22 12:00:00 | 758.40 | 759.28 | 757.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 12:15:00 | 757.80 | 758.99 | 757.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-22 13:00:00 | 757.80 | 758.99 | 757.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 13:15:00 | 755.93 | 758.37 | 757.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-22 14:00:00 | 755.93 | 758.37 | 757.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 14:15:00 | 756.93 | 758.09 | 757.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 09:15:00 | 758.90 | 757.90 | 757.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-23 11:15:00 | 754.95 | 756.64 | 756.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2024-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 11:15:00 | 754.95 | 756.64 | 756.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-23 12:15:00 | 752.80 | 755.87 | 756.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-24 09:15:00 | 758.48 | 755.52 | 755.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-24 09:15:00 | 758.48 | 755.52 | 755.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 09:15:00 | 758.48 | 755.52 | 755.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 10:00:00 | 758.48 | 755.52 | 755.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 10:15:00 | 758.50 | 756.12 | 756.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 10:30:00 | 759.70 | 756.12 | 756.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2024-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 11:15:00 | 758.98 | 756.69 | 756.43 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 14:15:00 | 755.03 | 756.36 | 756.47 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-04-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 09:15:00 | 759.93 | 757.16 | 756.82 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-04-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 14:15:00 | 753.68 | 756.16 | 756.48 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-29 11:15:00 | 760.20 | 757.22 | 756.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-29 13:15:00 | 760.98 | 758.17 | 757.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-30 14:15:00 | 760.13 | 764.05 | 761.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-30 14:15:00 | 760.13 | 764.05 | 761.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 760.13 | 764.05 | 761.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 15:00:00 | 760.13 | 764.05 | 761.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 757.73 | 762.79 | 761.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 09:15:00 | 764.65 | 762.79 | 761.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 12:00:00 | 761.43 | 763.90 | 763.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-03 12:15:00 | 754.93 | 762.10 | 762.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2024-05-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 12:15:00 | 754.93 | 762.10 | 762.65 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 11:15:00 | 764.40 | 762.44 | 762.38 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 13:15:00 | 761.20 | 762.31 | 762.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 09:15:00 | 756.55 | 760.94 | 761.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-13 10:15:00 | 722.08 | 721.92 | 729.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-13 10:45:00 | 722.60 | 721.92 | 729.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 13:15:00 | 728.15 | 724.04 | 728.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 14:00:00 | 728.15 | 724.04 | 728.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 14:15:00 | 726.50 | 724.53 | 728.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 14:30:00 | 729.23 | 724.53 | 728.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 15:15:00 | 728.48 | 725.32 | 728.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:15:00 | 727.25 | 725.32 | 728.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 729.30 | 726.12 | 728.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:45:00 | 728.53 | 726.12 | 728.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 10:15:00 | 729.68 | 726.83 | 728.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 10:30:00 | 730.30 | 726.83 | 728.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 11:15:00 | 731.10 | 727.68 | 728.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 12:00:00 | 731.10 | 727.68 | 728.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2024-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 14:15:00 | 731.50 | 729.32 | 729.29 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 09:15:00 | 726.00 | 728.89 | 729.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 10:15:00 | 723.58 | 727.83 | 728.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 11:15:00 | 727.75 | 723.32 | 725.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 11:15:00 | 727.75 | 723.32 | 725.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 727.75 | 723.32 | 725.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 12:00:00 | 727.75 | 723.32 | 725.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 722.98 | 723.25 | 724.84 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 15:15:00 | 729.88 | 725.69 | 725.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 10:15:00 | 732.33 | 727.77 | 726.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 728.88 | 731.09 | 729.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 728.88 | 731.09 | 729.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 728.88 | 731.09 | 729.56 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 10:15:00 | 728.13 | 729.28 | 729.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 11:15:00 | 727.20 | 728.86 | 729.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 14:15:00 | 729.85 | 728.20 | 728.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 14:15:00 | 729.85 | 728.20 | 728.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 729.85 | 728.20 | 728.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 15:00:00 | 729.85 | 728.20 | 728.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 731.85 | 728.93 | 728.97 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 09:15:00 | 729.90 | 729.12 | 729.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 10:15:00 | 734.83 | 730.27 | 729.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 13:15:00 | 764.80 | 765.23 | 759.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-28 13:30:00 | 765.23 | 765.23 | 759.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 755.78 | 763.26 | 760.10 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 13:15:00 | 754.88 | 758.05 | 758.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 15:15:00 | 753.68 | 756.69 | 757.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 11:15:00 | 757.83 | 756.43 | 757.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 11:15:00 | 757.83 | 756.43 | 757.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 11:15:00 | 757.83 | 756.43 | 757.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 12:00:00 | 757.83 | 756.43 | 757.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 12:15:00 | 760.35 | 757.21 | 757.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 13:00:00 | 760.35 | 757.21 | 757.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 13:15:00 | 759.65 | 757.70 | 757.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 13:45:00 | 760.53 | 757.70 | 757.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2024-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 15:15:00 | 759.15 | 757.83 | 757.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 09:15:00 | 761.58 | 758.58 | 758.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-31 11:15:00 | 758.48 | 759.16 | 758.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 11:15:00 | 758.48 | 759.16 | 758.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 758.48 | 759.16 | 758.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 12:00:00 | 758.48 | 759.16 | 758.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 762.13 | 759.75 | 758.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 13:15:00 | 763.35 | 759.75 | 758.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 13:45:00 | 764.13 | 760.97 | 759.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 09:45:00 | 767.40 | 777.82 | 771.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 750.23 | 772.30 | 769.67 | SL hit (close<static) qty=1.00 sl=758.30 alert=retest2 |

### Cycle 73 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 741.83 | 766.21 | 767.14 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 14:15:00 | 776.05 | 762.38 | 761.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 14:15:00 | 780.20 | 774.74 | 769.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 09:15:00 | 783.65 | 784.56 | 778.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 09:45:00 | 782.50 | 784.56 | 778.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 781.03 | 783.81 | 780.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 15:00:00 | 781.03 | 783.81 | 780.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 779.78 | 783.00 | 780.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:15:00 | 779.80 | 783.00 | 780.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 781.15 | 782.63 | 780.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 13:00:00 | 782.15 | 781.79 | 780.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 784.95 | 782.23 | 781.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 14:15:00 | 840.53 | 846.81 | 847.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 14:15:00 | 840.53 | 846.81 | 847.03 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 850.23 | 847.34 | 847.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 13:15:00 | 852.40 | 849.00 | 847.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 09:15:00 | 871.75 | 879.37 | 871.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 09:15:00 | 871.75 | 879.37 | 871.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 871.75 | 879.37 | 871.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:30:00 | 872.48 | 879.37 | 871.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 865.88 | 876.67 | 870.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 11:00:00 | 865.88 | 876.67 | 870.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 864.38 | 874.22 | 870.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 12:00:00 | 864.38 | 874.22 | 870.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 09:15:00 | 829.33 | 860.57 | 864.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 10:15:00 | 825.65 | 853.59 | 861.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 09:15:00 | 822.50 | 821.54 | 832.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-09 10:00:00 | 822.50 | 821.54 | 832.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 806.10 | 808.86 | 813.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:30:00 | 812.90 | 808.86 | 813.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 813.50 | 809.79 | 813.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 11:00:00 | 813.50 | 809.79 | 813.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 816.40 | 811.11 | 813.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:00:00 | 816.40 | 811.11 | 813.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 817.50 | 812.39 | 813.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 13:00:00 | 817.50 | 812.39 | 813.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 813.78 | 812.94 | 813.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:15:00 | 813.13 | 812.94 | 813.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 810.85 | 812.52 | 813.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 09:15:00 | 804.75 | 811.17 | 811.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 10:30:00 | 808.15 | 809.63 | 811.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 13:30:00 | 807.60 | 809.29 | 810.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 15:00:00 | 806.90 | 808.81 | 810.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 805.65 | 807.96 | 809.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 10:45:00 | 804.40 | 807.32 | 809.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 12:15:00 | 803.90 | 806.76 | 808.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 15:00:00 | 803.70 | 805.20 | 807.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-22 09:15:00 | 822.48 | 808.24 | 808.37 | SL hit (close>static) qty=1.00 sl=814.90 alert=retest2 |

### Cycle 78 — BUY (started 2024-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 10:15:00 | 821.95 | 810.98 | 809.61 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 14:15:00 | 808.80 | 812.58 | 812.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-24 10:15:00 | 801.68 | 809.32 | 811.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 15:15:00 | 804.18 | 803.82 | 807.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-25 09:15:00 | 805.00 | 803.82 | 807.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 10:15:00 | 804.63 | 804.23 | 806.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 10:45:00 | 805.85 | 804.23 | 806.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 807.15 | 804.82 | 806.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 12:00:00 | 807.15 | 804.82 | 806.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 809.53 | 805.76 | 807.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 13:00:00 | 809.53 | 805.76 | 807.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 811.93 | 806.99 | 807.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 14:00:00 | 811.93 | 806.99 | 807.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 12:15:00 | 807.43 | 805.09 | 806.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 13:00:00 | 807.43 | 805.09 | 806.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 13:15:00 | 808.50 | 805.77 | 806.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 13:45:00 | 809.73 | 805.77 | 806.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2024-07-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 15:15:00 | 808.15 | 806.80 | 806.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 10:15:00 | 813.13 | 808.22 | 807.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 12:15:00 | 803.05 | 807.19 | 807.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 12:15:00 | 803.05 | 807.19 | 807.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 12:15:00 | 803.05 | 807.19 | 807.11 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2024-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 13:15:00 | 801.58 | 806.07 | 806.61 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 12:15:00 | 811.50 | 807.18 | 806.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 13:15:00 | 815.15 | 808.78 | 807.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 14:15:00 | 806.35 | 808.29 | 807.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 14:15:00 | 806.35 | 808.29 | 807.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 806.35 | 808.29 | 807.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 15:00:00 | 806.35 | 808.29 | 807.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 807.10 | 808.05 | 807.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:15:00 | 812.15 | 808.05 | 807.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 11:30:00 | 809.58 | 809.02 | 808.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-31 13:15:00 | 805.70 | 808.10 | 807.79 | SL hit (close<static) qty=1.00 sl=805.98 alert=retest2 |

### Cycle 83 — SELL (started 2024-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 11:15:00 | 806.98 | 818.17 | 818.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 13:15:00 | 799.50 | 807.18 | 811.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 810.58 | 805.86 | 809.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 10:15:00 | 810.58 | 805.86 | 809.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 810.58 | 805.86 | 809.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:45:00 | 813.10 | 805.86 | 809.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 808.88 | 806.47 | 809.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:30:00 | 811.33 | 806.47 | 809.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 811.68 | 807.51 | 809.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:45:00 | 811.78 | 807.51 | 809.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 809.95 | 808.00 | 809.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:15:00 | 812.03 | 808.00 | 809.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 812.00 | 808.80 | 809.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:15:00 | 814.55 | 808.80 | 809.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 814.55 | 809.95 | 810.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 09:15:00 | 811.78 | 809.95 | 810.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 10:15:00 | 819.85 | 812.05 | 811.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 11:15:00 | 825.50 | 814.74 | 812.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 14:15:00 | 824.53 | 825.15 | 820.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 15:00:00 | 824.53 | 825.15 | 820.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 809.38 | 826.92 | 825.35 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2024-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 10:15:00 | 803.70 | 822.27 | 823.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 14:15:00 | 801.35 | 811.72 | 817.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 808.95 | 805.87 | 810.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 808.95 | 805.87 | 810.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 808.95 | 805.87 | 810.00 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2024-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 14:15:00 | 816.58 | 811.66 | 811.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 820.25 | 815.38 | 813.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 14:15:00 | 818.38 | 819.00 | 816.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 15:00:00 | 818.38 | 819.00 | 816.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 814.03 | 817.69 | 816.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:30:00 | 812.80 | 817.69 | 816.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 10:15:00 | 811.45 | 816.44 | 815.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 11:00:00 | 811.45 | 816.44 | 815.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2024-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 12:15:00 | 811.10 | 814.61 | 815.07 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 11:15:00 | 816.30 | 814.97 | 814.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 12:15:00 | 817.38 | 815.45 | 815.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 15:15:00 | 814.48 | 815.55 | 815.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 15:15:00 | 814.48 | 815.55 | 815.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 814.48 | 815.55 | 815.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 814.03 | 815.55 | 815.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 814.18 | 815.28 | 815.21 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 11:15:00 | 814.75 | 815.12 | 815.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 12:15:00 | 814.00 | 814.90 | 815.04 | Break + close below crossover candle low |

### Cycle 90 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 822.68 | 815.45 | 815.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 12:15:00 | 823.75 | 818.75 | 816.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-26 15:15:00 | 819.50 | 819.52 | 817.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-27 09:15:00 | 817.63 | 819.52 | 817.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 819.78 | 819.57 | 817.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:30:00 | 817.03 | 819.57 | 817.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 819.93 | 819.64 | 818.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:30:00 | 818.45 | 819.64 | 818.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 13:15:00 | 819.13 | 819.79 | 818.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 14:15:00 | 819.05 | 819.79 | 818.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 14:15:00 | 819.00 | 819.63 | 818.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 14:45:00 | 819.10 | 819.63 | 818.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 817.23 | 819.15 | 818.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 09:15:00 | 821.53 | 818.29 | 818.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 11:15:00 | 817.45 | 818.10 | 818.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 11:15:00 | 817.45 | 818.10 | 818.18 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 12:15:00 | 819.38 | 818.36 | 818.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-29 15:15:00 | 820.93 | 818.89 | 818.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 14:15:00 | 817.10 | 820.59 | 819.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 14:15:00 | 817.10 | 820.59 | 819.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 817.10 | 820.59 | 819.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 15:00:00 | 817.10 | 820.59 | 819.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — SELL (started 2024-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 15:15:00 | 813.25 | 819.12 | 819.22 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 09:15:00 | 819.55 | 816.95 | 816.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 09:15:00 | 823.80 | 820.41 | 818.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 819.38 | 821.37 | 820.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 819.38 | 821.37 | 820.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 819.38 | 821.37 | 820.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 819.38 | 821.37 | 820.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 820.50 | 821.20 | 820.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 12:30:00 | 821.50 | 820.89 | 820.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 14:00:00 | 821.50 | 821.01 | 820.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 09:15:00 | 819.15 | 819.94 | 820.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 819.15 | 819.94 | 820.03 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 10:15:00 | 821.73 | 820.30 | 820.19 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 12:15:00 | 817.75 | 819.78 | 819.97 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 14:15:00 | 823.58 | 820.63 | 820.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 12:15:00 | 824.90 | 822.46 | 821.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 12:15:00 | 823.60 | 824.16 | 823.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 12:15:00 | 823.60 | 824.16 | 823.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 12:15:00 | 823.60 | 824.16 | 823.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 12:45:00 | 823.03 | 824.16 | 823.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 823.00 | 823.93 | 823.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 823.00 | 823.93 | 823.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 820.78 | 823.30 | 822.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 820.78 | 823.30 | 822.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 825.00 | 823.64 | 823.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:15:00 | 826.43 | 823.64 | 823.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 11:30:00 | 827.15 | 825.19 | 823.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 13:30:00 | 826.48 | 826.23 | 824.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 13:15:00 | 879.38 | 884.00 | 884.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-09-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 13:15:00 | 879.38 | 884.00 | 884.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 14:15:00 | 875.48 | 882.30 | 883.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 15:15:00 | 864.15 | 863.97 | 868.76 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 09:15:00 | 855.28 | 863.97 | 868.76 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 13:15:00 | 812.52 | 824.38 | 836.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-08 09:15:00 | 820.60 | 819.51 | 830.53 | SL hit (close>ema200) qty=0.50 sl=819.51 alert=retest1 |

### Cycle 100 — BUY (started 2024-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 15:15:00 | 830.50 | 825.81 | 825.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 09:15:00 | 838.10 | 827.95 | 826.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 12:15:00 | 839.18 | 839.83 | 835.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-15 13:00:00 | 839.18 | 839.83 | 835.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 839.30 | 846.78 | 843.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 839.30 | 846.78 | 843.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 835.63 | 844.55 | 842.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:00:00 | 835.63 | 844.55 | 842.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2024-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 12:15:00 | 833.65 | 841.18 | 841.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 832.00 | 837.42 | 839.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 12:15:00 | 836.38 | 836.21 | 838.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 12:15:00 | 836.38 | 836.21 | 838.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 836.38 | 836.21 | 838.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:30:00 | 837.55 | 836.21 | 838.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 844.30 | 837.82 | 838.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:00:00 | 844.30 | 837.82 | 838.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 841.00 | 838.46 | 839.03 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 09:15:00 | 869.00 | 845.17 | 842.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-21 10:15:00 | 872.70 | 850.68 | 844.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 11:15:00 | 858.35 | 861.17 | 855.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-22 11:45:00 | 857.78 | 861.17 | 855.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 14:15:00 | 856.33 | 860.33 | 856.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 15:00:00 | 856.33 | 860.33 | 856.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 15:15:00 | 856.55 | 859.57 | 856.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 09:15:00 | 867.75 | 859.57 | 856.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 868.53 | 861.37 | 857.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-23 11:15:00 | 872.90 | 863.31 | 858.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-24 09:15:00 | 880.08 | 866.60 | 862.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-25 15:15:00 | 872.90 | 871.48 | 869.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-28 09:45:00 | 874.33 | 871.75 | 870.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 867.98 | 870.99 | 870.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:00:00 | 867.98 | 870.99 | 870.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 869.08 | 870.61 | 870.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:30:00 | 867.45 | 870.61 | 870.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-28 13:15:00 | 867.15 | 869.40 | 869.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 13:15:00 | 867.15 | 869.40 | 869.53 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 12:15:00 | 873.60 | 869.69 | 869.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 13:15:00 | 881.20 | 872.00 | 870.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 09:15:00 | 871.90 | 872.94 | 871.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 09:15:00 | 871.90 | 872.94 | 871.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 871.90 | 872.94 | 871.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 10:00:00 | 871.90 | 872.94 | 871.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 10:15:00 | 870.95 | 872.54 | 871.38 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2024-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 13:15:00 | 863.68 | 869.79 | 870.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 12:15:00 | 860.08 | 864.68 | 866.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 878.58 | 862.84 | 863.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 878.58 | 862.84 | 863.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 878.58 | 862.84 | 863.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:00:00 | 878.58 | 862.84 | 863.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 14:15:00 | 878.75 | 866.02 | 864.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-11 10:15:00 | 890.00 | 879.65 | 876.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-11 15:15:00 | 881.95 | 882.91 | 879.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-12 09:15:00 | 875.55 | 882.91 | 879.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 874.73 | 881.27 | 879.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:45:00 | 878.00 | 881.27 | 879.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 868.55 | 878.73 | 878.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:45:00 | 867.20 | 878.73 | 878.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2024-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 11:15:00 | 869.98 | 876.98 | 877.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 12:15:00 | 861.30 | 873.84 | 875.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 11:15:00 | 850.33 | 849.59 | 857.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 12:00:00 | 850.33 | 849.59 | 857.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 857.33 | 850.37 | 854.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 09:45:00 | 857.90 | 850.37 | 854.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 10:15:00 | 854.73 | 851.24 | 854.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 14:15:00 | 853.43 | 853.29 | 854.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 863.53 | 855.40 | 855.47 | SL hit (close>static) qty=1.00 sl=858.38 alert=retest2 |

### Cycle 108 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 870.15 | 858.35 | 856.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 11:15:00 | 872.05 | 861.09 | 858.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 15:15:00 | 869.98 | 870.21 | 866.66 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 09:15:00 | 875.08 | 870.21 | 866.66 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 867.08 | 869.93 | 867.17 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-22 10:15:00 | 867.08 | 869.93 | 867.17 | SL hit (close<ema400) qty=1.00 sl=867.17 alert=retest1 |

### Cycle 109 — SELL (started 2024-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 09:15:00 | 890.15 | 895.33 | 895.47 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 14:15:00 | 902.45 | 896.13 | 895.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 09:15:00 | 913.65 | 900.73 | 897.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 09:15:00 | 923.40 | 924.81 | 917.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 10:00:00 | 923.40 | 924.81 | 917.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 924.60 | 929.37 | 923.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:00:00 | 924.60 | 929.37 | 923.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 929.35 | 929.37 | 924.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 09:15:00 | 933.48 | 928.67 | 925.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 09:45:00 | 932.98 | 929.30 | 926.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 10:15:00 | 935.30 | 929.30 | 926.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 14:15:00 | 932.88 | 933.07 | 931.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 934.20 | 933.29 | 931.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 14:30:00 | 931.33 | 933.29 | 931.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 931.80 | 933.08 | 931.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 10:15:00 | 931.13 | 933.08 | 931.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 932.33 | 932.93 | 931.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 11:30:00 | 934.40 | 932.99 | 932.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 14:00:00 | 933.50 | 932.94 | 932.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 09:30:00 | 933.45 | 932.04 | 931.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 10:15:00 | 930.15 | 931.66 | 931.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 10:15:00 | 930.15 | 931.66 | 931.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 12:15:00 | 926.38 | 930.32 | 931.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 11:15:00 | 926.63 | 926.28 | 928.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 11:15:00 | 926.63 | 926.28 | 928.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 926.63 | 926.28 | 928.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:00:00 | 926.63 | 926.28 | 928.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 929.68 | 926.96 | 928.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:00:00 | 929.68 | 926.96 | 928.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 933.05 | 928.18 | 929.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:45:00 | 933.60 | 928.18 | 929.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 935.95 | 929.73 | 929.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 935.95 | 929.73 | 929.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2024-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 15:15:00 | 936.65 | 931.12 | 930.37 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 09:15:00 | 925.15 | 930.18 | 930.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 11:15:00 | 918.20 | 926.87 | 928.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 09:15:00 | 901.83 | 894.09 | 899.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-23 09:15:00 | 901.83 | 894.09 | 899.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 901.83 | 894.09 | 899.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:45:00 | 900.43 | 894.09 | 899.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 901.23 | 895.52 | 899.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:00:00 | 901.23 | 895.52 | 899.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 12:15:00 | 896.80 | 896.48 | 899.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 12:15:00 | 895.45 | 898.58 | 899.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 13:30:00 | 894.10 | 897.49 | 898.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 15:00:00 | 894.63 | 896.92 | 898.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 09:15:00 | 900.88 | 897.60 | 898.22 | SL hit (close>static) qty=1.00 sl=900.00 alert=retest2 |

### Cycle 114 — BUY (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 09:15:00 | 899.25 | 898.41 | 898.31 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 10:15:00 | 896.90 | 898.11 | 898.18 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 11:15:00 | 906.30 | 899.75 | 898.92 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 13:15:00 | 887.35 | 896.79 | 897.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 11:15:00 | 886.48 | 890.64 | 893.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 10:15:00 | 891.75 | 888.57 | 891.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-01 10:15:00 | 891.75 | 888.57 | 891.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 891.75 | 888.57 | 891.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:45:00 | 892.20 | 888.57 | 891.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 11:15:00 | 891.60 | 889.17 | 891.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 12:15:00 | 892.35 | 889.17 | 891.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 12:15:00 | 891.08 | 889.56 | 891.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 12:45:00 | 891.60 | 889.56 | 891.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 13:15:00 | 895.10 | 890.66 | 891.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 14:00:00 | 895.10 | 890.66 | 891.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 14:15:00 | 891.10 | 890.75 | 891.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 09:15:00 | 889.18 | 891.10 | 891.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 10:15:00 | 894.85 | 892.21 | 892.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 10:15:00 | 894.85 | 892.21 | 892.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 13:15:00 | 897.05 | 893.79 | 892.82 | Break + close above crossover candle high |

### Cycle 119 — SELL (started 2025-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 09:15:00 | 883.53 | 892.42 | 892.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 10:15:00 | 880.05 | 889.95 | 891.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 822.48 | 820.69 | 827.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 822.48 | 820.69 | 827.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 822.48 | 820.69 | 827.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 09:45:00 | 824.30 | 820.69 | 827.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 826.90 | 822.95 | 826.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:00:00 | 826.90 | 822.95 | 826.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 823.55 | 823.07 | 826.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 13:15:00 | 819.75 | 824.58 | 826.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 15:00:00 | 820.75 | 823.12 | 825.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 15:15:00 | 827.43 | 825.77 | 825.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2025-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 15:15:00 | 827.43 | 825.77 | 825.55 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2025-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 09:15:00 | 819.28 | 824.47 | 824.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 10:15:00 | 815.33 | 822.64 | 824.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 09:15:00 | 820.48 | 819.69 | 821.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-20 09:45:00 | 820.55 | 819.69 | 821.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 822.83 | 820.32 | 821.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:30:00 | 822.20 | 820.32 | 821.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 11:15:00 | 825.70 | 821.39 | 822.12 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2025-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 12:15:00 | 827.85 | 822.69 | 822.64 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 817.68 | 822.78 | 822.92 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 12:15:00 | 823.40 | 823.05 | 823.03 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 13:15:00 | 822.13 | 822.87 | 822.95 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 09:15:00 | 826.00 | 823.41 | 823.16 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 821.25 | 822.98 | 822.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 12:15:00 | 817.00 | 821.77 | 822.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 13:15:00 | 826.38 | 822.69 | 822.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 13:15:00 | 826.38 | 822.69 | 822.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 13:15:00 | 826.38 | 822.69 | 822.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 14:00:00 | 826.38 | 822.69 | 822.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2025-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 14:15:00 | 832.83 | 824.72 | 823.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-22 15:15:00 | 835.95 | 826.97 | 824.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-23 11:15:00 | 828.03 | 828.96 | 826.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 11:15:00 | 828.03 | 828.96 | 826.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 828.03 | 828.96 | 826.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:30:00 | 826.40 | 828.96 | 826.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 831.45 | 830.96 | 828.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 10:45:00 | 834.00 | 831.66 | 829.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 11:30:00 | 833.78 | 831.92 | 829.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 13:15:00 | 826.58 | 830.18 | 829.03 | SL hit (close<static) qty=1.00 sl=827.18 alert=retest2 |

### Cycle 129 — SELL (started 2025-01-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 15:15:00 | 822.50 | 827.83 | 828.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 816.98 | 825.66 | 827.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 09:15:00 | 835.00 | 821.62 | 823.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 09:15:00 | 835.00 | 821.62 | 823.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 835.00 | 821.62 | 823.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 10:00:00 | 835.00 | 821.62 | 823.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2025-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 10:15:00 | 838.60 | 825.02 | 824.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 10:15:00 | 842.85 | 838.57 | 834.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 843.00 | 847.60 | 844.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 843.00 | 847.60 | 844.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 843.00 | 847.60 | 844.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:00:00 | 843.00 | 847.60 | 844.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 841.10 | 846.30 | 844.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 841.10 | 846.30 | 844.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 844.88 | 846.02 | 844.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:15:00 | 847.95 | 846.02 | 844.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 839.03 | 844.63 | 844.08 | SL hit (close<static) qty=1.00 sl=839.28 alert=retest2 |

### Cycle 131 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 840.63 | 843.54 | 843.66 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 10:15:00 | 847.75 | 843.36 | 843.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 11:15:00 | 849.83 | 844.65 | 843.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 12:15:00 | 863.93 | 865.08 | 860.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 13:00:00 | 863.93 | 865.08 | 860.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 12:15:00 | 865.63 | 869.52 | 865.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 13:00:00 | 865.63 | 869.52 | 865.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 863.48 | 868.31 | 865.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 14:00:00 | 863.48 | 868.31 | 865.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 866.83 | 868.02 | 865.42 | EMA400 retest candle locked (from upside) |

### Cycle 133 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 856.20 | 863.53 | 863.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 849.45 | 857.53 | 860.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 853.63 | 849.62 | 853.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 12:15:00 | 853.63 | 849.62 | 853.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 853.63 | 849.62 | 853.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 853.63 | 849.62 | 853.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 851.43 | 849.98 | 853.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:30:00 | 854.23 | 849.98 | 853.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 853.83 | 850.75 | 853.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 15:00:00 | 853.83 | 850.75 | 853.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 855.33 | 851.67 | 853.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 09:15:00 | 850.40 | 851.67 | 853.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:00:00 | 851.03 | 852.66 | 853.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 13:00:00 | 851.00 | 847.94 | 848.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 859.23 | 851.22 | 850.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 14:15:00 | 859.23 | 851.22 | 850.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 11:15:00 | 859.33 | 854.82 | 852.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-19 15:15:00 | 861.58 | 862.90 | 859.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 09:15:00 | 849.80 | 862.90 | 859.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 845.03 | 859.32 | 858.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 09:45:00 | 847.03 | 859.32 | 858.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — SELL (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 10:15:00 | 845.35 | 856.53 | 857.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-20 12:15:00 | 843.78 | 852.03 | 854.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-21 13:15:00 | 845.88 | 844.62 | 848.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-21 14:00:00 | 845.88 | 844.62 | 848.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 841.50 | 839.05 | 842.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:45:00 | 840.73 | 839.05 | 842.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 838.83 | 839.01 | 841.88 | EMA400 retest candle locked (from downside) |

### Cycle 136 — BUY (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 09:15:00 | 850.23 | 842.54 | 842.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-27 11:15:00 | 851.15 | 845.53 | 843.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-03 10:15:00 | 853.25 | 857.67 | 853.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 10:15:00 | 853.25 | 857.67 | 853.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 10:15:00 | 853.25 | 857.67 | 853.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-03 11:00:00 | 853.25 | 857.67 | 853.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 11:15:00 | 853.00 | 856.74 | 853.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-03 12:30:00 | 854.50 | 855.92 | 853.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-03 13:15:00 | 848.73 | 854.49 | 853.00 | SL hit (close<static) qty=1.00 sl=850.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-04 11:15:00 | 851.03 | 852.27 | 852.30 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 12:15:00 | 855.00 | 852.81 | 852.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 14:15:00 | 855.78 | 853.80 | 853.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 10:15:00 | 850.50 | 853.46 | 853.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-05 10:15:00 | 850.50 | 853.46 | 853.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 850.50 | 853.46 | 853.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:00:00 | 850.50 | 853.46 | 853.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-05 11:15:00 | 850.65 | 852.90 | 852.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-05 13:15:00 | 845.98 | 850.95 | 851.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-06 14:15:00 | 846.00 | 844.24 | 847.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-06 14:45:00 | 845.63 | 844.24 | 847.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 847.93 | 845.29 | 847.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 10:00:00 | 847.93 | 845.29 | 847.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 10:15:00 | 849.25 | 846.08 | 847.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 10:45:00 | 848.53 | 846.08 | 847.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 844.60 | 845.57 | 846.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 15:00:00 | 844.60 | 845.57 | 846.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 844.58 | 845.12 | 846.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 10:45:00 | 842.50 | 844.60 | 845.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 12:15:00 | 848.78 | 845.50 | 846.07 | SL hit (close>static) qty=1.00 sl=846.98 alert=retest2 |

### Cycle 140 — BUY (started 2025-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 09:15:00 | 852.33 | 846.02 | 845.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 12:15:00 | 858.75 | 851.02 | 848.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 12:15:00 | 854.48 | 855.63 | 852.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-13 13:00:00 | 854.48 | 855.63 | 852.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 13:15:00 | 852.40 | 854.98 | 852.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 14:00:00 | 852.40 | 854.98 | 852.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 853.75 | 854.74 | 852.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 14:30:00 | 850.63 | 854.74 | 852.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 852.50 | 854.29 | 852.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 09:15:00 | 856.00 | 854.29 | 852.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 15:00:00 | 855.83 | 855.62 | 854.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 10:15:00 | 887.08 | 904.46 | 906.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 10:15:00 | 887.08 | 904.46 | 906.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 11:15:00 | 886.35 | 900.84 | 904.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 09:15:00 | 896.20 | 892.44 | 898.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 10:00:00 | 896.20 | 892.44 | 898.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 895.60 | 893.08 | 898.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 12:45:00 | 894.40 | 894.18 | 897.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 14:15:00 | 899.35 | 895.48 | 897.68 | SL hit (close>static) qty=1.00 sl=898.13 alert=retest2 |

### Cycle 142 — BUY (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 09:15:00 | 915.45 | 900.70 | 899.25 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 10:15:00 | 876.75 | 897.17 | 899.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 11:15:00 | 873.45 | 892.42 | 897.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 11:15:00 | 884.35 | 882.84 | 888.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 12:00:00 | 884.35 | 882.84 | 888.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 890.93 | 884.46 | 888.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 13:00:00 | 890.93 | 884.46 | 888.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 887.50 | 885.07 | 888.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 883.00 | 884.88 | 888.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:15:00 | 882.33 | 884.63 | 887.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 900.95 | 885.53 | 885.90 | SL hit (close>static) qty=1.00 sl=891.00 alert=retest2 |

### Cycle 144 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 901.10 | 888.64 | 887.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 907.98 | 892.51 | 889.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 968.70 | 974.35 | 965.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 10:00:00 | 968.70 | 974.35 | 965.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 964.50 | 972.38 | 965.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 11:00:00 | 964.50 | 972.38 | 965.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 963.40 | 970.59 | 964.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:00:00 | 963.40 | 970.59 | 964.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 962.70 | 969.01 | 964.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 13:00:00 | 962.70 | 969.01 | 964.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 13:15:00 | 964.10 | 968.03 | 964.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 14:15:00 | 963.40 | 968.03 | 964.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 14:15:00 | 962.90 | 967.00 | 964.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 15:00:00 | 962.90 | 967.00 | 964.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — SELL (started 2025-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 11:15:00 | 960.15 | 962.70 | 963.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 15:15:00 | 957.45 | 960.37 | 961.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-25 13:15:00 | 956.80 | 955.86 | 958.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 13:15:00 | 956.80 | 955.86 | 958.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 13:15:00 | 956.80 | 955.86 | 958.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 13:45:00 | 959.85 | 955.86 | 958.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 956.15 | 955.92 | 958.35 | EMA400 retest candle locked (from downside) |

### Cycle 146 — BUY (started 2025-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 12:15:00 | 962.60 | 959.39 | 959.27 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 09:15:00 | 953.95 | 958.91 | 959.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 11:15:00 | 951.30 | 956.38 | 957.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 09:15:00 | 964.95 | 956.81 | 957.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 964.95 | 956.81 | 957.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 964.95 | 956.81 | 957.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:45:00 | 966.80 | 956.81 | 957.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — BUY (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 10:15:00 | 962.10 | 957.87 | 957.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 09:15:00 | 968.80 | 962.32 | 960.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 14:15:00 | 963.50 | 963.69 | 961.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 14:15:00 | 963.50 | 963.69 | 961.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 963.50 | 963.69 | 961.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 14:30:00 | 962.00 | 963.69 | 961.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 15:15:00 | 961.75 | 963.30 | 961.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 09:15:00 | 970.90 | 963.30 | 961.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 10:45:00 | 963.85 | 966.85 | 965.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 10:00:00 | 964.50 | 968.22 | 967.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 15:00:00 | 966.70 | 968.11 | 967.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 15:15:00 | 960.75 | 966.64 | 966.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 960.75 | 966.64 | 966.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 950.40 | 963.39 | 965.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 975.90 | 955.42 | 958.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 975.90 | 955.42 | 958.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 975.90 | 955.42 | 958.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:45:00 | 976.60 | 955.42 | 958.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 977.50 | 959.84 | 960.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:45:00 | 977.75 | 959.84 | 960.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 975.20 | 962.91 | 961.64 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 15:15:00 | 962.45 | 965.09 | 965.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-14 11:15:00 | 958.85 | 962.98 | 964.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 12:15:00 | 963.95 | 957.38 | 959.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 12:15:00 | 963.95 | 957.38 | 959.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 963.95 | 957.38 | 959.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:00:00 | 963.95 | 957.38 | 959.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 971.35 | 960.18 | 960.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:00:00 | 971.35 | 960.18 | 960.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2025-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 14:15:00 | 966.60 | 961.46 | 961.25 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 10:15:00 | 960.05 | 965.02 | 965.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 957.15 | 962.42 | 963.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 970.80 | 962.54 | 963.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 970.80 | 962.54 | 963.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 970.80 | 962.54 | 963.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:45:00 | 969.85 | 962.54 | 963.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 969.00 | 963.83 | 963.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 11:15:00 | 967.80 | 963.83 | 963.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 10:15:00 | 966.35 | 961.83 | 961.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 966.35 | 961.83 | 961.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 972.15 | 966.15 | 964.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 961.35 | 967.06 | 965.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 961.35 | 967.06 | 965.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 961.35 | 967.06 | 965.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 961.35 | 967.06 | 965.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 965.00 | 966.65 | 965.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:45:00 | 960.10 | 966.65 | 965.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 970.15 | 967.35 | 966.19 | EMA400 retest candle locked (from upside) |

### Cycle 155 — SELL (started 2025-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 14:15:00 | 964.10 | 965.28 | 965.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 10:15:00 | 956.55 | 960.90 | 962.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 14:15:00 | 965.05 | 959.85 | 961.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 14:15:00 | 965.05 | 959.85 | 961.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 965.05 | 959.85 | 961.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 965.05 | 959.85 | 961.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 968.80 | 961.64 | 961.95 | EMA400 retest candle locked (from downside) |

### Cycle 156 — BUY (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 09:15:00 | 967.75 | 962.87 | 962.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 13:15:00 | 969.35 | 965.30 | 963.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 09:15:00 | 959.10 | 965.86 | 964.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 959.10 | 965.86 | 964.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 959.10 | 965.86 | 964.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:30:00 | 957.10 | 965.86 | 964.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 959.25 | 964.53 | 964.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:30:00 | 959.75 | 964.53 | 964.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 964.15 | 964.13 | 964.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:00:00 | 964.15 | 964.13 | 964.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 966.05 | 964.51 | 964.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:30:00 | 964.35 | 964.51 | 964.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 963.15 | 965.93 | 965.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:00:00 | 963.15 | 965.93 | 965.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 964.95 | 965.73 | 965.19 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2025-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 15:15:00 | 961.75 | 964.36 | 964.62 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 967.95 | 965.05 | 964.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 11:15:00 | 970.05 | 966.05 | 965.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 09:15:00 | 973.35 | 973.75 | 971.22 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:15:00 | 981.50 | 973.75 | 971.22 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 997.50 | 978.50 | 973.61 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-10 10:15:00 | 983.95 | 987.74 | 984.97 | SL hit (close<ema400) qty=1.00 sl=984.97 alert=retest1 |

### Cycle 159 — SELL (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 09:15:00 | 979.15 | 983.27 | 983.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 14:15:00 | 975.05 | 979.74 | 981.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 963.95 | 962.65 | 967.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:30:00 | 963.80 | 962.65 | 967.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 967.25 | 963.57 | 967.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 967.25 | 963.57 | 967.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 966.70 | 964.19 | 967.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:45:00 | 966.60 | 964.19 | 967.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 969.15 | 965.19 | 967.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 969.15 | 965.19 | 967.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 967.40 | 965.63 | 967.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:30:00 | 968.50 | 965.63 | 967.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 968.25 | 966.15 | 967.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 968.25 | 966.15 | 967.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 968.45 | 966.61 | 967.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 962.00 | 966.61 | 967.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 960.85 | 965.46 | 967.22 | EMA400 retest candle locked (from downside) |

### Cycle 160 — BUY (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-19 09:15:00 | 971.50 | 965.85 | 965.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 09:15:00 | 975.75 | 969.29 | 967.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 973.60 | 976.72 | 973.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 10:00:00 | 973.60 | 976.72 | 973.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 972.50 | 975.88 | 973.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:00:00 | 972.50 | 975.88 | 973.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 972.50 | 975.20 | 972.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:45:00 | 972.45 | 975.20 | 972.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 972.75 | 974.71 | 972.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 13:15:00 | 975.05 | 974.71 | 972.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 11:15:00 | 996.50 | 1001.20 | 1001.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 11:15:00 | 996.50 | 1001.20 | 1001.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 12:15:00 | 994.70 | 999.90 | 1000.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 998.85 | 996.73 | 998.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 998.85 | 996.73 | 998.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 998.85 | 996.73 | 998.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 998.85 | 996.73 | 998.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 1003.00 | 997.99 | 999.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:00:00 | 1003.00 | 997.99 | 999.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 999.25 | 998.24 | 999.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 13:45:00 | 998.00 | 998.33 | 999.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 14:15:00 | 1001.35 | 996.17 | 995.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — BUY (started 2025-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 14:15:00 | 1001.35 | 996.17 | 995.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 1003.75 | 998.25 | 996.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 1002.50 | 1002.76 | 1000.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 1002.50 | 1002.76 | 1000.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1002.50 | 1002.76 | 1000.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:30:00 | 1001.70 | 1002.76 | 1000.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 1002.60 | 1003.96 | 1001.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 15:00:00 | 1002.60 | 1003.96 | 1001.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 1001.40 | 1003.45 | 1001.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 1001.50 | 1003.45 | 1001.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 997.20 | 1002.20 | 1001.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 997.20 | 1002.20 | 1001.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 995.40 | 1000.84 | 1000.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 11:15:00 | 993.15 | 999.30 | 1000.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 14:15:00 | 991.45 | 990.21 | 993.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 15:00:00 | 991.45 | 990.21 | 993.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 993.65 | 991.10 | 993.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:45:00 | 992.20 | 991.10 | 993.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 998.00 | 992.48 | 993.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 998.00 | 992.48 | 993.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 997.90 | 993.56 | 994.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:30:00 | 999.00 | 993.56 | 994.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 1000.60 | 994.97 | 994.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 1003.50 | 998.38 | 996.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 13:15:00 | 998.80 | 999.40 | 997.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 13:15:00 | 998.80 | 999.40 | 997.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 998.80 | 999.40 | 997.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:00:00 | 998.80 | 999.40 | 997.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 998.40 | 999.20 | 997.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:45:00 | 997.45 | 999.20 | 997.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 998.00 | 998.96 | 997.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 999.15 | 998.96 | 997.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:45:00 | 1000.00 | 998.70 | 997.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 10:45:00 | 999.45 | 998.96 | 998.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 12:15:00 | 995.20 | 998.22 | 997.84 | SL hit (close<static) qty=1.00 sl=996.30 alert=retest2 |

### Cycle 165 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 995.00 | 997.58 | 997.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 15:15:00 | 993.00 | 996.14 | 996.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 997.50 | 987.15 | 990.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 997.50 | 987.15 | 990.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 997.50 | 987.15 | 990.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:45:00 | 998.85 | 987.15 | 990.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 997.40 | 989.20 | 991.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:00:00 | 997.40 | 989.20 | 991.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — BUY (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 13:15:00 | 997.25 | 993.22 | 992.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 14:15:00 | 1000.40 | 994.66 | 993.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 09:15:00 | 1008.50 | 1008.85 | 1004.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 09:30:00 | 1009.65 | 1008.85 | 1004.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 1005.55 | 1008.45 | 1005.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:00:00 | 1005.55 | 1008.45 | 1005.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 1006.50 | 1008.06 | 1005.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:00:00 | 1006.50 | 1008.06 | 1005.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 1005.85 | 1007.50 | 1005.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 1004.80 | 1007.50 | 1005.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 1004.05 | 1006.81 | 1005.63 | EMA400 retest candle locked (from upside) |

### Cycle 167 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 1000.95 | 1004.59 | 1004.76 | EMA200 below EMA400 |

### Cycle 168 — BUY (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 10:15:00 | 1007.50 | 1004.98 | 1004.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 13:15:00 | 1012.10 | 1007.85 | 1006.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 1006.85 | 1010.45 | 1009.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 1006.85 | 1010.45 | 1009.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1006.85 | 1010.45 | 1009.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:15:00 | 1005.55 | 1010.45 | 1009.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 1006.15 | 1009.59 | 1008.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:45:00 | 1005.00 | 1009.59 | 1008.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 1008.90 | 1011.10 | 1009.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 1008.90 | 1011.10 | 1009.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 1009.00 | 1010.68 | 1009.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 1007.40 | 1010.68 | 1009.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1011.40 | 1010.82 | 1010.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:45:00 | 1007.40 | 1010.82 | 1010.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 1008.05 | 1010.27 | 1009.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:00:00 | 1008.05 | 1010.27 | 1009.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 1007.15 | 1009.64 | 1009.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:30:00 | 1006.20 | 1009.64 | 1009.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — SELL (started 2025-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 12:15:00 | 1007.10 | 1009.14 | 1009.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 1005.80 | 1008.27 | 1008.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 15:15:00 | 1008.45 | 1008.31 | 1008.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 15:15:00 | 1008.45 | 1008.31 | 1008.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 1008.45 | 1008.31 | 1008.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 09:30:00 | 1001.65 | 1006.52 | 1008.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 15:15:00 | 999.70 | 994.59 | 994.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 999.70 | 994.59 | 994.46 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 09:15:00 | 990.65 | 993.80 | 994.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 10:15:00 | 989.50 | 992.94 | 993.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 991.40 | 989.48 | 991.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 991.40 | 989.48 | 991.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 991.40 | 989.48 | 991.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:30:00 | 991.20 | 989.48 | 991.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 991.40 | 989.86 | 991.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:45:00 | 992.05 | 989.86 | 991.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 990.50 | 989.99 | 991.20 | EMA400 retest candle locked (from downside) |

### Cycle 172 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 997.85 | 992.52 | 992.12 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 12:15:00 | 990.15 | 991.82 | 992.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 13:15:00 | 987.35 | 990.93 | 991.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 13:15:00 | 991.10 | 988.02 | 989.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 13:15:00 | 991.10 | 988.02 | 989.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 991.10 | 988.02 | 989.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:00:00 | 991.10 | 988.02 | 989.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 990.40 | 988.49 | 989.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:45:00 | 991.10 | 988.49 | 989.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 989.05 | 988.61 | 989.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 993.65 | 988.61 | 989.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 993.50 | 989.58 | 989.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:45:00 | 994.80 | 989.58 | 989.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 10:15:00 | 993.90 | 990.45 | 990.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 12:15:00 | 996.80 | 992.61 | 991.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 999.05 | 1000.19 | 997.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 10:00:00 | 999.05 | 1000.19 | 997.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 994.75 | 999.13 | 997.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:00:00 | 994.75 | 999.13 | 997.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 995.50 | 998.40 | 997.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:45:00 | 995.30 | 998.40 | 997.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — SELL (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 09:15:00 | 993.25 | 996.83 | 996.93 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 13:15:00 | 997.30 | 995.75 | 995.71 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 14:15:00 | 995.15 | 995.63 | 995.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 09:15:00 | 982.65 | 993.00 | 994.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 11:15:00 | 985.50 | 985.01 | 988.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 11:45:00 | 985.25 | 985.01 | 988.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 969.20 | 981.28 | 985.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 957.90 | 975.27 | 980.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:00:00 | 967.90 | 971.22 | 976.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 10:15:00 | 959.55 | 952.75 | 952.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — BUY (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 10:15:00 | 959.55 | 952.75 | 952.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 11:15:00 | 962.10 | 954.62 | 952.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 09:15:00 | 956.60 | 958.28 | 955.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-05 10:00:00 | 956.60 | 958.28 | 955.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 957.10 | 958.04 | 955.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 11:15:00 | 958.80 | 958.04 | 955.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:15:00 | 958.15 | 957.92 | 956.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 11:15:00 | 963.20 | 964.65 | 964.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — SELL (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 11:15:00 | 963.20 | 964.65 | 964.76 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 13:15:00 | 968.10 | 965.47 | 965.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 11:15:00 | 969.95 | 966.32 | 965.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 13:15:00 | 966.40 | 966.63 | 965.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 13:15:00 | 966.40 | 966.63 | 965.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 966.40 | 966.63 | 965.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:00:00 | 966.40 | 966.63 | 965.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 967.95 | 966.89 | 966.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 15:15:00 | 965.60 | 966.89 | 966.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 965.60 | 966.64 | 966.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 966.75 | 966.64 | 966.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 967.75 | 966.86 | 966.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 11:15:00 | 970.00 | 967.07 | 966.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 12:15:00 | 966.05 | 966.83 | 966.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — SELL (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 12:15:00 | 966.05 | 966.83 | 966.85 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 971.45 | 967.59 | 967.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 976.05 | 969.30 | 968.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 12:15:00 | 971.30 | 971.75 | 969.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 12:15:00 | 971.30 | 971.75 | 969.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 971.30 | 971.75 | 969.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 971.30 | 971.75 | 969.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 971.30 | 971.66 | 969.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 14:30:00 | 973.30 | 972.70 | 970.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 09:15:00 | 969.25 | 972.54 | 970.91 | SL hit (close<static) qty=1.00 sl=969.65 alert=retest2 |

### Cycle 183 — SELL (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 11:15:00 | 964.15 | 969.81 | 969.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 962.05 | 965.54 | 967.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 14:15:00 | 952.40 | 951.73 | 956.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 15:00:00 | 952.40 | 951.73 | 956.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 952.00 | 951.76 | 955.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:30:00 | 953.30 | 951.76 | 955.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 12:15:00 | 955.80 | 952.57 | 955.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 13:00:00 | 955.80 | 952.57 | 955.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 954.85 | 953.02 | 955.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 14:15:00 | 953.75 | 953.02 | 955.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 14:30:00 | 953.15 | 950.12 | 950.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 10:15:00 | 953.35 | 950.96 | 950.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — BUY (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 10:15:00 | 953.35 | 950.96 | 950.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 10:15:00 | 960.50 | 953.40 | 952.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 09:15:00 | 957.75 | 960.51 | 957.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 09:15:00 | 957.75 | 960.51 | 957.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 957.75 | 960.51 | 957.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 11:00:00 | 962.90 | 960.99 | 957.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 14:15:00 | 976.45 | 978.10 | 978.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — SELL (started 2025-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 14:15:00 | 976.45 | 978.10 | 978.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 974.25 | 976.97 | 977.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 977.55 | 976.14 | 976.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 14:15:00 | 977.55 | 976.14 | 976.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 977.55 | 976.14 | 976.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 15:00:00 | 977.55 | 976.14 | 976.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 977.00 | 976.31 | 976.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 977.60 | 976.31 | 976.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 978.70 | 976.79 | 977.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:15:00 | 981.35 | 976.79 | 977.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 982.70 | 977.97 | 977.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 10:15:00 | 985.00 | 980.75 | 979.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 13:15:00 | 1001.95 | 1002.88 | 997.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 14:00:00 | 1001.95 | 1002.88 | 997.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1004.95 | 1008.73 | 1005.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:45:00 | 1005.40 | 1008.73 | 1005.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 1002.00 | 1007.38 | 1005.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:45:00 | 1000.75 | 1007.38 | 1005.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2025-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 13:15:00 | 994.25 | 1002.53 | 1003.38 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 11:15:00 | 1007.25 | 1003.20 | 1003.13 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 13:15:00 | 1001.20 | 1003.01 | 1003.07 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 15:15:00 | 1003.50 | 1003.14 | 1003.12 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 09:15:00 | 999.90 | 1002.50 | 1002.83 | EMA200 below EMA400 |

### Cycle 192 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 1005.25 | 1002.84 | 1002.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 11:15:00 | 1008.45 | 1003.96 | 1003.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 10:15:00 | 1004.75 | 1006.75 | 1005.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 10:15:00 | 1004.75 | 1006.75 | 1005.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 1004.75 | 1006.75 | 1005.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:45:00 | 1004.05 | 1006.75 | 1005.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 1002.65 | 1005.93 | 1005.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 12:00:00 | 1002.65 | 1005.93 | 1005.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 14:15:00 | 998.00 | 1003.44 | 1004.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 995.05 | 1001.20 | 1002.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 10:15:00 | 991.60 | 991.04 | 995.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 10:45:00 | 991.35 | 991.04 | 995.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 994.55 | 992.50 | 994.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:00:00 | 994.55 | 992.50 | 994.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 995.00 | 993.00 | 994.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 13:30:00 | 991.10 | 992.19 | 993.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 14:15:00 | 991.45 | 986.43 | 985.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 991.45 | 986.43 | 985.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 993.55 | 988.74 | 987.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 14:15:00 | 989.35 | 991.10 | 989.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 14:15:00 | 989.35 | 991.10 | 989.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 989.35 | 991.10 | 989.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 15:00:00 | 989.35 | 991.10 | 989.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 990.25 | 990.93 | 989.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:15:00 | 985.60 | 990.93 | 989.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 985.40 | 989.82 | 988.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 11:00:00 | 989.15 | 989.69 | 988.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 12:30:00 | 987.95 | 989.46 | 988.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 13:15:00 | 985.00 | 988.57 | 988.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — SELL (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 13:15:00 | 985.00 | 988.57 | 988.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 980.05 | 986.20 | 987.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 12:15:00 | 985.85 | 985.35 | 986.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-14 13:00:00 | 985.85 | 985.35 | 986.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 983.35 | 984.95 | 986.37 | EMA400 retest candle locked (from downside) |

### Cycle 196 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 989.40 | 987.19 | 987.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 12:15:00 | 993.65 | 989.56 | 988.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 12:15:00 | 994.00 | 994.15 | 991.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 13:00:00 | 994.00 | 994.15 | 991.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 991.95 | 993.96 | 992.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 991.95 | 993.96 | 992.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 990.85 | 993.34 | 992.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 986.00 | 993.34 | 992.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 986.90 | 992.05 | 991.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 986.90 | 992.05 | 991.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — SELL (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 10:15:00 | 984.50 | 990.54 | 990.95 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2025-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 15:15:00 | 994.80 | 991.43 | 991.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 09:15:00 | 995.30 | 992.20 | 991.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 999.30 | 1001.70 | 997.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-21 10:00:00 | 999.30 | 1001.70 | 997.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 998.60 | 1001.08 | 997.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:45:00 | 997.65 | 1001.08 | 997.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 1001.25 | 1001.11 | 997.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:30:00 | 998.80 | 1001.11 | 997.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 998.10 | 1000.68 | 998.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 998.10 | 1000.68 | 998.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 998.00 | 1000.15 | 998.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 1002.70 | 1000.15 | 998.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 10:00:00 | 999.10 | 1001.71 | 1000.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 10:15:00 | 993.80 | 1000.13 | 1000.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — SELL (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 10:15:00 | 993.80 | 1000.13 | 1000.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 14:15:00 | 989.75 | 997.04 | 998.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 1000.20 | 996.46 | 998.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 1000.20 | 996.46 | 998.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1000.20 | 996.46 | 998.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 1000.20 | 996.46 | 998.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1000.00 | 997.16 | 998.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:15:00 | 1001.55 | 997.16 | 998.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 1001.65 | 998.06 | 998.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:30:00 | 999.70 | 998.06 | 998.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 1003.95 | 999.24 | 999.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 1005.00 | 1000.39 | 999.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 1005.95 | 1008.37 | 1005.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 1005.95 | 1008.37 | 1005.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1005.95 | 1008.37 | 1005.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:30:00 | 1005.70 | 1008.37 | 1005.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 1011.05 | 1008.91 | 1006.26 | EMA400 retest candle locked (from upside) |

### Cycle 201 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 1002.90 | 1006.43 | 1006.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 990.60 | 1002.04 | 1004.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 13:15:00 | 993.40 | 992.87 | 996.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 13:30:00 | 993.10 | 992.87 | 996.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 1001.30 | 994.56 | 996.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 1001.30 | 994.56 | 996.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 1000.10 | 995.67 | 996.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:15:00 | 1002.00 | 995.67 | 996.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — BUY (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 10:15:00 | 1003.00 | 998.16 | 997.95 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2025-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 12:15:00 | 995.70 | 997.50 | 997.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 13:15:00 | 994.10 | 996.82 | 997.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 14:15:00 | 996.90 | 996.84 | 997.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 14:15:00 | 996.90 | 996.84 | 997.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 996.90 | 996.84 | 997.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 996.90 | 996.84 | 997.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 998.50 | 997.17 | 997.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 994.50 | 997.17 | 997.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 1001.30 | 998.00 | 997.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — BUY (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 09:15:00 | 1001.30 | 998.00 | 997.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 10:15:00 | 1004.20 | 999.24 | 998.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 15:15:00 | 1001.20 | 1001.41 | 999.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 09:15:00 | 999.20 | 1001.41 | 999.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 1003.70 | 1001.87 | 1000.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-08 11:45:00 | 1004.70 | 1002.35 | 1000.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-08 12:15:00 | 1003.90 | 1002.35 | 1000.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-08 12:45:00 | 1005.00 | 1003.02 | 1001.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 13:15:00 | 999.80 | 1000.72 | 1000.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — SELL (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 13:15:00 | 999.80 | 1000.72 | 1000.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 14:15:00 | 997.00 | 999.98 | 1000.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 994.40 | 992.59 | 995.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 10:00:00 | 994.40 | 992.59 | 995.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 995.40 | 993.15 | 995.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:45:00 | 995.00 | 993.15 | 995.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 999.50 | 994.42 | 995.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 999.50 | 994.42 | 995.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 997.60 | 995.06 | 995.89 | EMA400 retest candle locked (from downside) |

### Cycle 206 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 1000.00 | 996.66 | 996.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 15:15:00 | 1002.90 | 1000.60 | 999.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 996.00 | 999.68 | 998.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 996.00 | 999.68 | 998.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 996.00 | 999.68 | 998.76 | EMA400 retest candle locked (from upside) |

### Cycle 207 — SELL (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 13:15:00 | 995.70 | 997.92 | 998.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 994.00 | 996.44 | 997.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 12:15:00 | 995.70 | 995.27 | 996.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 12:15:00 | 995.70 | 995.27 | 996.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 995.70 | 995.27 | 996.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:45:00 | 996.00 | 995.27 | 996.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 996.00 | 995.41 | 996.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:45:00 | 996.50 | 995.41 | 996.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 993.80 | 995.09 | 996.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 989.80 | 994.91 | 996.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 14:15:00 | 988.10 | 986.27 | 986.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — BUY (started 2025-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 14:15:00 | 988.10 | 986.27 | 986.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 994.10 | 988.00 | 986.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 13:15:00 | 994.10 | 995.44 | 992.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 14:00:00 | 994.10 | 995.44 | 992.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 995.00 | 995.83 | 993.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 995.00 | 995.83 | 993.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 991.30 | 994.76 | 993.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 991.30 | 994.76 | 993.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 990.20 | 993.85 | 993.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:45:00 | 990.40 | 993.85 | 993.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 209 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 992.00 | 992.75 | 992.85 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 09:15:00 | 993.70 | 992.94 | 992.93 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 989.40 | 992.23 | 992.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 11:15:00 | 988.30 | 991.45 | 992.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 14:15:00 | 992.20 | 991.18 | 991.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 14:15:00 | 992.20 | 991.18 | 991.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 992.20 | 991.18 | 991.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 988.60 | 991.51 | 991.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 10:00:00 | 989.50 | 991.11 | 991.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 10:30:00 | 989.60 | 990.48 | 991.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 15:15:00 | 995.00 | 989.83 | 990.40 | SL hit (close>static) qty=1.00 sl=993.00 alert=retest2 |

### Cycle 212 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 995.10 | 991.13 | 990.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 13:15:00 | 996.40 | 992.19 | 991.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 14:15:00 | 990.90 | 991.93 | 991.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 14:15:00 | 990.90 | 991.93 | 991.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 990.90 | 991.93 | 991.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 990.90 | 991.93 | 991.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 994.00 | 992.34 | 991.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 995.65 | 992.34 | 991.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 10:30:00 | 994.10 | 992.92 | 991.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 11:15:00 | 990.00 | 992.33 | 991.78 | SL hit (close<static) qty=1.00 sl=990.40 alert=retest2 |

### Cycle 213 — SELL (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 11:15:00 | 983.85 | 992.60 | 993.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 13:15:00 | 978.15 | 988.07 | 991.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 940.35 | 937.80 | 943.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:30:00 | 940.50 | 937.80 | 943.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 942.90 | 938.48 | 942.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:45:00 | 940.65 | 939.83 | 942.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:15:00 | 940.30 | 939.83 | 942.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 15:15:00 | 928.00 | 920.56 | 919.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 214 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 928.00 | 920.56 | 919.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 09:15:00 | 936.25 | 923.70 | 921.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 926.00 | 929.39 | 926.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 926.00 | 929.39 | 926.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 926.00 | 929.39 | 926.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 926.00 | 929.39 | 926.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 930.20 | 929.55 | 926.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:30:00 | 926.30 | 929.55 | 926.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 936.50 | 934.25 | 930.70 | EMA400 retest candle locked (from upside) |

### Cycle 215 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 924.40 | 931.36 | 931.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 923.55 | 929.79 | 930.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 11:15:00 | 926.35 | 925.86 | 928.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 11:45:00 | 925.30 | 925.86 | 928.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 927.60 | 925.52 | 927.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 927.60 | 925.52 | 927.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 927.70 | 925.95 | 927.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 953.25 | 925.95 | 927.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 216 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 947.60 | 930.28 | 929.24 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 939.90 | 945.05 | 945.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-09 12:15:00 | 936.50 | 940.09 | 942.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 09:15:00 | 924.85 | 923.45 | 927.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 10:00:00 | 924.85 | 923.45 | 927.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 919.00 | 922.56 | 926.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 11:45:00 | 916.75 | 921.27 | 925.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 15:15:00 | 925.25 | 922.65 | 922.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 218 — BUY (started 2026-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 15:15:00 | 925.25 | 922.65 | 922.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 15:15:00 | 926.45 | 924.88 | 923.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 923.20 | 924.54 | 923.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 923.20 | 924.54 | 923.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 923.20 | 924.54 | 923.85 | EMA400 retest candle locked (from upside) |

### Cycle 219 — SELL (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 12:15:00 | 922.00 | 923.49 | 923.49 | EMA200 below EMA400 |

### Cycle 220 — BUY (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 14:15:00 | 925.10 | 923.65 | 923.55 | EMA200 above EMA400 |

### Cycle 221 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 922.55 | 923.44 | 923.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 10:15:00 | 920.80 | 922.91 | 923.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 915.60 | 915.16 | 918.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 14:00:00 | 915.60 | 915.16 | 918.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 921.85 | 915.76 | 917.51 | EMA400 retest candle locked (from downside) |

### Cycle 222 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 924.15 | 919.42 | 918.87 | EMA200 above EMA400 |

### Cycle 223 — SELL (started 2026-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 13:15:00 | 916.70 | 918.67 | 918.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 14:15:00 | 910.60 | 917.05 | 918.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 12:15:00 | 878.65 | 877.68 | 884.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 13:00:00 | 878.65 | 877.68 | 884.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 876.40 | 873.72 | 877.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 881.10 | 873.72 | 877.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 875.10 | 874.00 | 877.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 864.80 | 874.00 | 877.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 821.56 | 856.40 | 865.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 847.95 | 843.91 | 853.05 | SL hit (close>ema200) qty=0.50 sl=843.91 alert=retest2 |

### Cycle 224 — BUY (started 2026-03-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 14:15:00 | 842.90 | 830.05 | 829.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 09:15:00 | 845.00 | 834.44 | 831.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 09:15:00 | 833.20 | 839.85 | 836.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 833.20 | 839.85 | 836.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 833.20 | 839.85 | 836.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:00:00 | 833.20 | 839.85 | 836.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 838.35 | 839.55 | 836.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 11:30:00 | 841.00 | 839.77 | 837.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 12:30:00 | 840.75 | 840.06 | 837.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 09:15:00 | 799.70 | 833.21 | 835.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 225 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 799.70 | 833.21 | 835.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 753.40 | 781.21 | 798.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 757.25 | 755.66 | 772.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:15:00 | 758.20 | 755.66 | 772.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 771.90 | 759.08 | 771.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 771.90 | 759.08 | 771.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 767.80 | 760.82 | 770.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 769.90 | 760.82 | 770.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 784.30 | 766.86 | 771.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:45:00 | 785.35 | 766.86 | 771.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 791.30 | 771.75 | 773.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 11:00:00 | 791.30 | 771.75 | 773.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 226 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 792.75 | 775.95 | 774.84 | EMA200 above EMA400 |

### Cycle 227 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 761.55 | 774.32 | 775.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 756.40 | 765.46 | 770.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 745.85 | 743.74 | 753.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 745.85 | 743.74 | 753.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 745.85 | 743.74 | 753.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 741.45 | 743.74 | 753.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 15:00:00 | 740.75 | 744.87 | 750.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 730.80 | 744.87 | 750.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 756.60 | 748.51 | 747.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 228 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 756.60 | 748.51 | 747.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 767.75 | 752.36 | 749.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 762.60 | 762.62 | 756.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 762.60 | 762.62 | 756.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 791.20 | 804.83 | 800.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:30:00 | 789.95 | 804.83 | 800.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 229 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 794.70 | 797.55 | 797.86 | EMA200 below EMA400 |

### Cycle 230 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 813.05 | 799.51 | 798.57 | EMA200 above EMA400 |

### Cycle 231 — SELL (started 2026-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 14:15:00 | 794.30 | 801.53 | 802.23 | EMA200 below EMA400 |

### Cycle 232 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 10:15:00 | 806.75 | 800.23 | 800.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 11:15:00 | 810.60 | 802.30 | 801.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 09:15:00 | 803.60 | 806.91 | 804.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 803.60 | 806.91 | 804.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 803.60 | 806.91 | 804.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:00:00 | 803.60 | 806.91 | 804.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 804.10 | 806.35 | 804.25 | EMA400 retest candle locked (from upside) |

### Cycle 233 — SELL (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 15:15:00 | 799.40 | 802.90 | 803.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 09:15:00 | 794.60 | 801.24 | 802.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 790.65 | 786.41 | 789.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 790.65 | 786.41 | 789.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 790.65 | 786.41 | 789.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 785.50 | 786.88 | 789.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 09:15:00 | 793.25 | 789.74 | 790.05 | SL hit (close>static) qty=1.00 sl=793.00 alert=retest2 |

### Cycle 234 — BUY (started 2026-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 13:15:00 | 781.90 | 776.61 | 776.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 796.00 | 780.49 | 778.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 783.80 | 790.63 | 786.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 783.80 | 790.63 | 786.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 783.80 | 790.63 | 786.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:00:00 | 783.80 | 790.63 | 786.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 783.35 | 789.18 | 786.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 783.35 | 789.18 | 786.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 235 — SELL (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 13:15:00 | 778.25 | 784.24 | 784.47 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-19 09:45:00 | 820.25 | 2023-05-29 10:15:00 | 819.88 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2023-05-19 13:15:00 | 820.03 | 2023-05-29 10:15:00 | 819.88 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2023-05-22 10:30:00 | 820.63 | 2023-05-29 10:15:00 | 819.88 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2023-05-22 11:30:00 | 820.58 | 2023-05-29 10:15:00 | 819.88 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2023-05-23 13:00:00 | 819.25 | 2023-05-29 10:15:00 | 819.88 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2023-05-23 14:30:00 | 820.25 | 2023-05-29 10:15:00 | 819.88 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2023-06-05 12:15:00 | 805.00 | 2023-06-08 09:15:00 | 809.00 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2023-06-09 09:45:00 | 806.88 | 2023-06-12 11:15:00 | 802.50 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2023-06-27 09:30:00 | 822.35 | 2023-07-05 11:15:00 | 838.48 | STOP_HIT | 1.00 | 1.96% |
| BUY | retest2 | 2023-06-27 11:45:00 | 822.00 | 2023-07-05 11:15:00 | 838.48 | STOP_HIT | 1.00 | 2.00% |
| SELL | retest2 | 2023-07-12 13:45:00 | 822.33 | 2023-07-17 11:15:00 | 824.25 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2023-07-12 14:45:00 | 821.25 | 2023-07-17 12:15:00 | 833.63 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2023-07-13 14:30:00 | 820.98 | 2023-07-17 12:15:00 | 833.63 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2023-07-14 09:45:00 | 822.48 | 2023-07-17 12:15:00 | 833.63 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2023-07-17 09:15:00 | 817.83 | 2023-07-17 12:15:00 | 833.63 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2023-07-27 09:15:00 | 849.85 | 2023-07-27 11:15:00 | 842.10 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2023-08-02 09:15:00 | 824.65 | 2023-08-04 15:15:00 | 826.43 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2023-08-08 12:00:00 | 827.35 | 2023-08-09 10:15:00 | 817.63 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2023-08-08 13:00:00 | 826.00 | 2023-08-09 10:15:00 | 817.63 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2023-08-21 15:15:00 | 794.00 | 2023-08-29 09:15:00 | 793.60 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2023-08-22 11:00:00 | 794.48 | 2023-08-29 09:15:00 | 793.60 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2023-08-22 12:15:00 | 794.40 | 2023-08-29 09:15:00 | 793.60 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2023-08-22 13:00:00 | 794.13 | 2023-08-29 09:15:00 | 793.60 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2023-08-24 11:15:00 | 792.13 | 2023-08-29 09:15:00 | 793.60 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2023-08-24 12:00:00 | 792.03 | 2023-08-29 09:15:00 | 793.60 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2023-09-04 10:30:00 | 788.50 | 2023-09-04 11:15:00 | 789.80 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-09-14 09:15:00 | 822.98 | 2023-09-18 09:15:00 | 822.38 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2023-09-28 13:45:00 | 761.40 | 2023-10-04 15:15:00 | 765.50 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2023-09-28 15:00:00 | 761.30 | 2023-10-04 15:15:00 | 765.50 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2023-10-03 09:15:00 | 755.30 | 2023-10-04 15:15:00 | 765.50 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2023-10-04 10:45:00 | 762.00 | 2023-10-04 15:15:00 | 765.50 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2023-10-20 14:30:00 | 760.00 | 2023-11-03 09:15:00 | 743.75 | STOP_HIT | 1.00 | 2.14% |
| SELL | retest2 | 2023-10-23 10:15:00 | 759.50 | 2023-11-03 09:15:00 | 743.75 | STOP_HIT | 1.00 | 2.07% |
| SELL | retest2 | 2023-10-23 14:15:00 | 760.48 | 2023-11-03 09:15:00 | 743.75 | STOP_HIT | 1.00 | 2.20% |
| BUY | retest2 | 2023-11-07 15:15:00 | 744.58 | 2023-11-10 09:15:00 | 741.45 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2023-11-08 09:45:00 | 744.38 | 2023-11-10 09:15:00 | 741.45 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2023-11-08 10:30:00 | 744.55 | 2023-11-10 09:15:00 | 741.45 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2023-11-08 12:00:00 | 745.45 | 2023-11-10 09:15:00 | 741.45 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2023-11-09 11:45:00 | 745.50 | 2023-11-10 09:15:00 | 741.45 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2023-11-09 13:15:00 | 744.88 | 2023-11-10 09:15:00 | 741.45 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2023-11-13 13:15:00 | 746.50 | 2023-12-08 09:15:00 | 821.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-13 13:45:00 | 746.68 | 2023-12-08 09:15:00 | 821.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-15 09:15:00 | 754.58 | 2023-12-11 09:15:00 | 830.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-19 11:15:00 | 827.90 | 2024-01-01 15:15:00 | 846.45 | STOP_HIT | 1.00 | 2.24% |
| BUY | retest2 | 2023-12-20 09:15:00 | 832.50 | 2024-01-01 15:15:00 | 846.45 | STOP_HIT | 1.00 | 1.68% |
| BUY | retest2 | 2023-12-21 09:15:00 | 831.80 | 2024-01-01 15:15:00 | 846.45 | STOP_HIT | 1.00 | 1.76% |
| SELL | retest2 | 2024-01-04 14:00:00 | 840.75 | 2024-01-04 15:15:00 | 845.93 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-01-05 09:30:00 | 840.00 | 2024-01-15 11:15:00 | 833.83 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2024-01-05 10:30:00 | 840.13 | 2024-01-15 11:15:00 | 833.83 | STOP_HIT | 1.00 | 0.75% |
| SELL | retest2 | 2024-01-05 15:00:00 | 840.78 | 2024-01-15 11:15:00 | 833.83 | STOP_HIT | 1.00 | 0.83% |
| SELL | retest2 | 2024-01-08 11:45:00 | 834.80 | 2024-01-15 11:15:00 | 833.83 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2024-01-09 11:15:00 | 836.43 | 2024-01-15 11:15:00 | 833.83 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2024-01-09 12:15:00 | 836.20 | 2024-01-15 11:15:00 | 833.83 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest2 | 2024-01-25 09:15:00 | 724.68 | 2024-01-29 12:15:00 | 729.98 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-01-29 09:30:00 | 726.60 | 2024-01-29 12:15:00 | 729.98 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2024-01-29 14:00:00 | 725.78 | 2024-01-29 14:15:00 | 727.95 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2024-02-07 11:15:00 | 718.40 | 2024-02-14 09:15:00 | 682.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-07 11:15:00 | 718.40 | 2024-02-14 14:15:00 | 690.65 | STOP_HIT | 0.50 | 3.86% |
| BUY | retest2 | 2024-02-20 10:15:00 | 720.25 | 2024-02-22 11:15:00 | 708.65 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-02-28 10:45:00 | 706.03 | 2024-03-01 13:15:00 | 712.80 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-03-01 12:15:00 | 708.63 | 2024-03-01 13:15:00 | 712.80 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2024-03-01 13:00:00 | 708.53 | 2024-03-01 13:15:00 | 712.80 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-03-05 12:45:00 | 717.40 | 2024-03-11 11:15:00 | 714.68 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-03-11 10:15:00 | 717.08 | 2024-03-11 11:15:00 | 714.68 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2024-03-14 15:15:00 | 728.50 | 2024-03-15 09:15:00 | 723.85 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-03-19 13:30:00 | 722.53 | 2024-03-22 11:15:00 | 722.10 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2024-03-20 10:00:00 | 720.50 | 2024-03-22 11:15:00 | 722.10 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-03-21 11:30:00 | 722.25 | 2024-03-22 11:15:00 | 722.10 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2024-03-21 13:45:00 | 722.65 | 2024-03-22 11:15:00 | 722.10 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2024-04-18 10:15:00 | 752.83 | 2024-04-19 12:15:00 | 759.45 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-04-18 13:15:00 | 750.58 | 2024-04-19 12:15:00 | 759.45 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-04-19 11:45:00 | 752.20 | 2024-04-19 12:15:00 | 759.45 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-04-23 09:15:00 | 758.90 | 2024-04-23 11:15:00 | 754.95 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2024-05-02 09:15:00 | 764.65 | 2024-05-03 12:15:00 | 754.93 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-05-03 12:00:00 | 761.43 | 2024-05-03 12:15:00 | 754.93 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-05-31 13:15:00 | 763.35 | 2024-06-04 10:15:00 | 750.23 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-05-31 13:45:00 | 764.13 | 2024-06-04 10:15:00 | 750.23 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2024-06-04 09:45:00 | 767.40 | 2024-06-04 10:15:00 | 750.23 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-06-11 13:00:00 | 782.15 | 2024-06-28 14:15:00 | 840.53 | STOP_HIT | 1.00 | 7.46% |
| BUY | retest2 | 2024-06-12 09:15:00 | 784.95 | 2024-06-28 14:15:00 | 840.53 | STOP_HIT | 1.00 | 7.08% |
| SELL | retest2 | 2024-07-18 09:15:00 | 804.75 | 2024-07-22 09:15:00 | 822.48 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2024-07-18 10:30:00 | 808.15 | 2024-07-22 09:15:00 | 822.48 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2024-07-18 13:30:00 | 807.60 | 2024-07-22 09:15:00 | 822.48 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-07-18 15:00:00 | 806.90 | 2024-07-22 09:15:00 | 822.48 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2024-07-19 10:45:00 | 804.40 | 2024-07-22 09:15:00 | 822.48 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2024-07-19 12:15:00 | 803.90 | 2024-07-22 09:15:00 | 822.48 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2024-07-19 15:00:00 | 803.70 | 2024-07-22 09:15:00 | 822.48 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2024-07-31 09:15:00 | 812.15 | 2024-07-31 13:15:00 | 805.70 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-07-31 11:30:00 | 809.58 | 2024-07-31 13:15:00 | 805.70 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-07-31 15:15:00 | 808.98 | 2024-08-05 10:15:00 | 803.25 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-08-29 09:15:00 | 821.53 | 2024-08-29 11:15:00 | 817.45 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-09-06 12:30:00 | 821.50 | 2024-09-09 09:15:00 | 819.15 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-09-06 14:00:00 | 821.50 | 2024-09-09 09:15:00 | 819.15 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-09-12 09:15:00 | 826.43 | 2024-09-27 13:15:00 | 879.38 | STOP_HIT | 1.00 | 6.41% |
| BUY | retest2 | 2024-09-12 11:30:00 | 827.15 | 2024-09-27 13:15:00 | 879.38 | STOP_HIT | 1.00 | 6.31% |
| BUY | retest2 | 2024-09-12 13:30:00 | 826.48 | 2024-09-27 13:15:00 | 879.38 | STOP_HIT | 1.00 | 6.40% |
| SELL | retest1 | 2024-10-03 09:15:00 | 855.28 | 2024-10-07 13:15:00 | 812.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-10-03 09:15:00 | 855.28 | 2024-10-08 09:15:00 | 820.60 | STOP_HIT | 0.50 | 4.05% |
| SELL | retest2 | 2024-10-09 12:00:00 | 819.28 | 2024-10-10 13:15:00 | 828.25 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-10-10 10:15:00 | 819.45 | 2024-10-10 13:15:00 | 828.25 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-10-23 11:15:00 | 872.90 | 2024-10-28 13:15:00 | 867.15 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-10-24 09:15:00 | 880.08 | 2024-10-28 13:15:00 | 867.15 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-10-25 15:15:00 | 872.90 | 2024-10-28 13:15:00 | 867.15 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-10-28 09:45:00 | 874.33 | 2024-10-28 13:15:00 | 867.15 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-11-18 14:15:00 | 853.43 | 2024-11-19 09:15:00 | 863.53 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest1 | 2024-11-22 09:15:00 | 875.08 | 2024-11-22 10:15:00 | 867.08 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-11-22 14:00:00 | 872.25 | 2024-12-02 09:15:00 | 890.15 | STOP_HIT | 1.00 | 2.05% |
| BUY | retest2 | 2024-11-25 09:15:00 | 886.98 | 2024-12-02 09:15:00 | 890.15 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2024-12-09 09:15:00 | 933.48 | 2024-12-12 10:15:00 | 930.15 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2024-12-09 09:45:00 | 932.98 | 2024-12-12 10:15:00 | 930.15 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2024-12-09 10:15:00 | 935.30 | 2024-12-12 10:15:00 | 930.15 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-12-10 14:15:00 | 932.88 | 2024-12-12 10:15:00 | 930.15 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-12-11 11:30:00 | 934.40 | 2024-12-12 10:15:00 | 930.15 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-12-11 14:00:00 | 933.50 | 2024-12-12 10:15:00 | 930.15 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2024-12-12 09:30:00 | 933.45 | 2024-12-12 10:15:00 | 930.15 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2024-12-26 12:15:00 | 895.45 | 2024-12-27 09:15:00 | 900.88 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2024-12-26 13:30:00 | 894.10 | 2024-12-27 09:15:00 | 900.88 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-12-26 15:00:00 | 894.63 | 2024-12-27 09:15:00 | 900.88 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-12-27 12:00:00 | 895.50 | 2024-12-27 15:15:00 | 900.50 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2024-12-30 09:15:00 | 894.83 | 2024-12-30 09:15:00 | 899.25 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-01-02 09:15:00 | 889.18 | 2025-01-02 10:15:00 | 894.85 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-01-15 13:15:00 | 819.75 | 2025-01-16 15:15:00 | 827.43 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-01-15 15:00:00 | 820.75 | 2025-01-16 15:15:00 | 827.43 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-01-24 10:45:00 | 834.00 | 2025-01-24 13:15:00 | 826.58 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-01-24 11:30:00 | 833.78 | 2025-01-24 13:15:00 | 826.58 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-02-01 14:15:00 | 847.95 | 2025-02-03 09:15:00 | 839.03 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-02-13 09:15:00 | 850.40 | 2025-02-17 14:15:00 | 859.23 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-02-13 13:00:00 | 851.03 | 2025-02-17 14:15:00 | 859.23 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-02-17 13:00:00 | 851.00 | 2025-02-17 14:15:00 | 859.23 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-03-03 12:30:00 | 854.50 | 2025-03-03 13:15:00 | 848.73 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-03-10 10:45:00 | 842.50 | 2025-03-10 12:15:00 | 848.78 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-03-10 14:45:00 | 842.28 | 2025-03-12 09:15:00 | 852.33 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-03-11 09:15:00 | 837.55 | 2025-03-12 09:15:00 | 852.33 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-03-11 12:00:00 | 842.48 | 2025-03-12 09:15:00 | 852.33 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-03-17 09:15:00 | 856.00 | 2025-04-01 10:15:00 | 887.08 | STOP_HIT | 1.00 | 3.63% |
| BUY | retest2 | 2025-03-17 15:00:00 | 855.83 | 2025-04-01 10:15:00 | 887.08 | STOP_HIT | 1.00 | 3.65% |
| SELL | retest2 | 2025-04-02 12:45:00 | 894.40 | 2025-04-02 14:15:00 | 899.35 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-04-09 09:15:00 | 883.00 | 2025-04-11 09:15:00 | 900.95 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-04-09 10:15:00 | 882.33 | 2025-04-11 09:15:00 | 900.95 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-05-05 09:15:00 | 970.90 | 2025-05-08 15:15:00 | 960.75 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-05-06 10:45:00 | 963.85 | 2025-05-08 15:15:00 | 960.75 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-05-08 10:00:00 | 964.50 | 2025-05-08 15:15:00 | 960.75 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-05-08 15:00:00 | 966.70 | 2025-05-08 15:15:00 | 960.75 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-05-21 11:15:00 | 967.80 | 2025-05-23 10:15:00 | 966.35 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest1 | 2025-06-06 10:15:00 | 981.50 | 2025-06-10 10:15:00 | 983.95 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2025-06-23 13:15:00 | 975.05 | 2025-07-02 11:15:00 | 996.50 | STOP_HIT | 1.00 | 2.20% |
| SELL | retest2 | 2025-07-03 13:45:00 | 998.00 | 2025-07-08 14:15:00 | 1001.35 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-07-17 09:15:00 | 999.15 | 2025-07-17 12:15:00 | 995.20 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-07-17 09:45:00 | 1000.00 | 2025-07-17 12:15:00 | 995.20 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-07-17 10:45:00 | 999.45 | 2025-07-17 12:15:00 | 995.20 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-08-04 09:30:00 | 1001.65 | 2025-08-07 15:15:00 | 999.70 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2025-08-28 09:15:00 | 957.90 | 2025-09-04 10:15:00 | 959.55 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-08-28 12:00:00 | 967.90 | 2025-09-04 10:15:00 | 959.55 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2025-09-05 11:15:00 | 958.80 | 2025-09-11 11:15:00 | 963.20 | STOP_HIT | 1.00 | 0.46% |
| BUY | retest2 | 2025-09-05 13:15:00 | 958.15 | 2025-09-11 11:15:00 | 963.20 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2025-09-15 11:15:00 | 970.00 | 2025-09-16 12:15:00 | 966.05 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-09-18 14:30:00 | 973.30 | 2025-09-19 09:15:00 | 969.25 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-09-25 14:15:00 | 953.75 | 2025-09-30 10:15:00 | 953.35 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-09-29 14:30:00 | 953.15 | 2025-09-30 10:15:00 | 953.35 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2025-10-03 11:00:00 | 962.90 | 2025-10-13 14:15:00 | 976.45 | STOP_HIT | 1.00 | 1.41% |
| SELL | retest2 | 2025-11-04 13:30:00 | 991.10 | 2025-11-11 14:15:00 | 991.45 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-11-13 11:00:00 | 989.15 | 2025-11-13 13:15:00 | 985.00 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-11-13 12:30:00 | 987.95 | 2025-11-13 13:15:00 | 985.00 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-11-24 09:15:00 | 1002.70 | 2025-11-25 10:15:00 | 993.80 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-11-25 10:00:00 | 999.10 | 2025-11-25 10:15:00 | 993.80 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-12-05 09:15:00 | 994.50 | 2025-12-05 09:15:00 | 1001.30 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-12-08 11:45:00 | 1004.70 | 2025-12-09 13:15:00 | 999.80 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-12-08 12:15:00 | 1003.90 | 2025-12-09 13:15:00 | 999.80 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-12-08 12:45:00 | 1005.00 | 2025-12-09 13:15:00 | 999.80 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-12-17 09:15:00 | 989.80 | 2025-12-22 14:15:00 | 988.10 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2025-12-30 09:15:00 | 988.60 | 2025-12-30 15:15:00 | 995.00 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-12-30 10:00:00 | 989.50 | 2025-12-30 15:15:00 | 995.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-12-30 10:30:00 | 989.60 | 2025-12-30 15:15:00 | 995.00 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-12-31 09:30:00 | 989.50 | 2025-12-31 12:15:00 | 995.10 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2026-01-01 09:15:00 | 995.65 | 2026-01-01 11:15:00 | 990.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2026-01-01 10:30:00 | 994.10 | 2026-01-01 11:15:00 | 990.00 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2026-01-02 09:15:00 | 994.60 | 2026-01-05 09:15:00 | 990.15 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2026-01-13 11:45:00 | 940.65 | 2026-01-27 15:15:00 | 928.00 | STOP_HIT | 1.00 | 1.34% |
| SELL | retest2 | 2026-01-13 12:15:00 | 940.30 | 2026-01-27 15:15:00 | 928.00 | STOP_HIT | 1.00 | 1.31% |
| SELL | retest2 | 2026-02-13 11:45:00 | 916.75 | 2026-02-16 15:15:00 | 925.25 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-03-06 09:15:00 | 864.80 | 2026-03-09 09:15:00 | 821.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 09:15:00 | 864.80 | 2026-03-10 09:15:00 | 847.95 | STOP_HIT | 0.50 | 1.95% |
| BUY | retest2 | 2026-03-18 11:30:00 | 841.00 | 2026-03-19 09:15:00 | 799.70 | STOP_HIT | 1.00 | -4.91% |
| BUY | retest2 | 2026-03-18 12:30:00 | 840.75 | 2026-03-19 09:15:00 | 799.70 | STOP_HIT | 1.00 | -4.88% |
| SELL | retest2 | 2026-04-01 10:15:00 | 741.45 | 2026-04-06 11:15:00 | 756.60 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2026-04-01 15:00:00 | 740.75 | 2026-04-06 11:15:00 | 756.60 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2026-04-02 09:15:00 | 730.80 | 2026-04-06 11:15:00 | 756.60 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2026-04-27 11:15:00 | 785.50 | 2026-04-28 09:15:00 | 793.25 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2026-04-28 12:15:00 | 785.45 | 2026-05-06 13:15:00 | 781.90 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2026-04-28 12:45:00 | 785.50 | 2026-05-06 13:15:00 | 781.90 | STOP_HIT | 1.00 | 0.46% |
| SELL | retest2 | 2026-04-28 13:45:00 | 785.20 | 2026-05-06 13:15:00 | 781.90 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2026-04-29 14:00:00 | 783.95 | 2026-05-06 13:15:00 | 781.90 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2026-05-04 09:45:00 | 783.55 | 2026-05-06 13:15:00 | 781.90 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2026-05-04 11:15:00 | 784.40 | 2026-05-06 13:15:00 | 781.90 | STOP_HIT | 1.00 | 0.32% |
