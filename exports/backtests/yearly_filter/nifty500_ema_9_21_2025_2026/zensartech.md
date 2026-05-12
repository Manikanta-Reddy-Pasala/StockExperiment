# Zensar Technolgies Ltd. (ZENSARTECH)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 525.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 97 |
| ALERT1 | 54 |
| ALERT2 | 53 |
| ALERT2_SKIP | 27 |
| ALERT3 | 130 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 65 |
| PARTIAL | 8 |
| TARGET_HIT | 1 |
| STOP_HIT | 65 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 74 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 21 / 53
- **Target hits / Stop hits / Partials:** 1 / 65 / 8
- **Avg / median % per leg:** -0.00% / -0.86%
- **Sum % (uncompounded):** -0.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 5 | 14.7% | 0 | 34 | 0 | -1.16% | -39.3% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.95% | -3.0% |
| BUY @ 3rd Alert (retest2) | 33 | 5 | 15.2% | 0 | 33 | 0 | -1.10% | -36.3% |
| SELL (all) | 40 | 16 | 40.0% | 1 | 31 | 8 | 0.97% | 39.0% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| SELL @ 3rd Alert (retest2) | 38 | 14 | 36.8% | 0 | 31 | 7 | 0.63% | 24.0% |
| retest1 (combined) | 3 | 2 | 66.7% | 1 | 1 | 1 | 4.02% | 12.0% |
| retest2 (combined) | 71 | 19 | 26.8% | 0 | 64 | 7 | -0.17% | -12.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 736.00 | 709.47 | 708.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 742.60 | 716.10 | 711.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 14:15:00 | 799.75 | 805.10 | 792.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 15:00:00 | 799.75 | 805.10 | 792.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 15:15:00 | 794.05 | 802.89 | 793.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:15:00 | 792.90 | 802.89 | 793.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 797.30 | 801.77 | 793.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 09:30:00 | 834.80 | 805.20 | 797.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 15:15:00 | 818.00 | 820.28 | 820.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 15:15:00 | 818.00 | 820.28 | 820.50 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 868.25 | 829.88 | 824.84 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 14:15:00 | 829.55 | 834.84 | 835.26 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 10:15:00 | 840.00 | 836.01 | 835.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 09:15:00 | 846.05 | 838.48 | 837.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 837.65 | 843.23 | 840.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 837.65 | 843.23 | 840.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 837.65 | 843.23 | 840.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 837.65 | 843.23 | 840.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 839.00 | 842.38 | 840.69 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 13:15:00 | 834.15 | 839.15 | 839.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 831.15 | 837.55 | 838.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 14:15:00 | 824.95 | 822.03 | 826.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-03 15:00:00 | 824.95 | 822.03 | 826.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 820.70 | 821.77 | 825.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:15:00 | 830.30 | 821.77 | 825.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 828.70 | 823.15 | 826.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 13:15:00 | 820.55 | 826.28 | 827.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 14:15:00 | 820.25 | 826.03 | 826.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 15:00:00 | 822.10 | 825.24 | 826.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 10:15:00 | 834.05 | 827.02 | 826.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 10:15:00 | 834.05 | 827.02 | 826.88 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 11:15:00 | 824.00 | 827.64 | 827.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 14:15:00 | 822.00 | 825.09 | 826.51 | Break + close below crossover candle low |

### Cycle 9 — BUY (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 09:15:00 | 840.75 | 828.10 | 827.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 13:15:00 | 852.90 | 837.94 | 832.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 851.25 | 861.53 | 852.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 851.25 | 861.53 | 852.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 851.25 | 861.53 | 852.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:45:00 | 850.25 | 861.53 | 852.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 852.30 | 859.68 | 852.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:45:00 | 849.70 | 859.68 | 852.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 851.95 | 858.14 | 852.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 12:15:00 | 854.85 | 858.14 | 852.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 14:15:00 | 847.70 | 855.79 | 852.93 | SL hit (close<static) qty=1.00 sl=850.25 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 13:15:00 | 863.70 | 870.75 | 870.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 851.50 | 865.61 | 868.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 852.00 | 849.76 | 856.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 11:00:00 | 852.00 | 849.76 | 856.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 844.55 | 840.79 | 845.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 13:45:00 | 834.00 | 839.34 | 843.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 15:15:00 | 832.70 | 838.77 | 842.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 09:15:00 | 862.45 | 842.53 | 843.66 | SL hit (close>static) qty=1.00 sl=853.30 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 10:15:00 | 856.45 | 845.32 | 844.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 11:15:00 | 866.40 | 849.53 | 846.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 14:15:00 | 852.25 | 853.09 | 849.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 14:45:00 | 854.30 | 853.09 | 849.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 844.70 | 851.13 | 849.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 844.70 | 851.13 | 849.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 847.50 | 850.40 | 848.98 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 13:15:00 | 845.80 | 847.85 | 848.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 12:15:00 | 841.85 | 844.82 | 845.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 13:15:00 | 845.35 | 844.92 | 845.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 13:15:00 | 845.35 | 844.92 | 845.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 845.35 | 844.92 | 845.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:00:00 | 845.35 | 844.92 | 845.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 843.45 | 844.63 | 845.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:30:00 | 845.10 | 844.63 | 845.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 839.00 | 843.23 | 844.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 13:45:00 | 835.70 | 839.48 | 842.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 14:15:00 | 835.90 | 839.48 | 842.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 859.15 | 843.62 | 843.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 09:15:00 | 859.15 | 843.62 | 843.54 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 12:15:00 | 846.00 | 850.41 | 850.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 09:15:00 | 842.05 | 849.13 | 850.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 12:15:00 | 853.05 | 848.44 | 849.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 12:15:00 | 853.05 | 848.44 | 849.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 853.05 | 848.44 | 849.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 13:00:00 | 853.05 | 848.44 | 849.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 13:15:00 | 857.70 | 850.29 | 850.18 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 845.40 | 850.28 | 850.30 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 14:15:00 | 849.05 | 848.05 | 848.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 10:15:00 | 851.05 | 849.04 | 848.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 11:15:00 | 847.25 | 848.68 | 848.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 11:15:00 | 847.25 | 848.68 | 848.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 847.25 | 848.68 | 848.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:00:00 | 847.25 | 848.68 | 848.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 848.30 | 848.61 | 848.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 14:00:00 | 851.70 | 849.23 | 848.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 14:15:00 | 846.30 | 848.64 | 848.48 | SL hit (close<static) qty=1.00 sl=847.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 15:15:00 | 847.00 | 848.31 | 848.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 832.00 | 845.05 | 846.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 10:15:00 | 811.55 | 811.25 | 820.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 10:30:00 | 812.75 | 811.25 | 820.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 822.00 | 813.82 | 817.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:00:00 | 822.00 | 813.82 | 817.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 846.05 | 820.26 | 820.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:45:00 | 851.10 | 820.26 | 820.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 11:15:00 | 842.60 | 824.73 | 822.32 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 10:15:00 | 823.60 | 825.73 | 826.01 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 12:15:00 | 836.50 | 827.84 | 826.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 14:15:00 | 842.80 | 832.43 | 829.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 11:15:00 | 836.00 | 837.55 | 833.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 12:00:00 | 836.00 | 837.55 | 833.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 831.10 | 836.26 | 833.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:00:00 | 831.10 | 836.26 | 833.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 817.60 | 832.53 | 831.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:00:00 | 817.60 | 832.53 | 831.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 14:15:00 | 811.90 | 828.40 | 829.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 795.45 | 819.32 | 825.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 815.30 | 810.57 | 817.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 15:00:00 | 815.30 | 810.57 | 817.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 815.00 | 811.46 | 817.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 814.85 | 811.46 | 817.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 825.20 | 814.21 | 818.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:00:00 | 825.20 | 814.21 | 818.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 824.45 | 816.25 | 818.76 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 13:15:00 | 824.30 | 820.25 | 820.16 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 810.05 | 819.59 | 820.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 802.10 | 816.09 | 818.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 14:15:00 | 800.00 | 791.65 | 799.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 14:15:00 | 800.00 | 791.65 | 799.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 14:15:00 | 800.00 | 791.65 | 799.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 14:45:00 | 800.00 | 791.65 | 799.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 800.10 | 793.34 | 799.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:15:00 | 806.30 | 793.34 | 799.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 805.00 | 795.67 | 800.43 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 810.20 | 802.72 | 802.62 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 11:15:00 | 799.05 | 804.23 | 804.61 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 15:15:00 | 810.00 | 804.82 | 804.63 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 799.20 | 803.70 | 804.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 791.00 | 798.82 | 801.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 13:15:00 | 800.25 | 792.59 | 796.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 13:15:00 | 800.25 | 792.59 | 796.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 800.25 | 792.59 | 796.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:45:00 | 801.30 | 792.59 | 796.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 800.20 | 794.11 | 796.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 803.45 | 794.11 | 796.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 10:15:00 | 802.95 | 797.82 | 797.80 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 790.60 | 798.42 | 798.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 10:15:00 | 784.35 | 795.61 | 797.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 11:15:00 | 787.20 | 787.09 | 790.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 12:00:00 | 787.20 | 787.09 | 790.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 789.80 | 786.01 | 789.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:30:00 | 792.40 | 786.01 | 789.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 791.30 | 787.07 | 789.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 783.70 | 787.07 | 789.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:45:00 | 787.50 | 787.69 | 787.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 12:15:00 | 791.95 | 788.54 | 788.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 791.95 | 788.54 | 788.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 15:15:00 | 793.00 | 789.93 | 789.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 796.55 | 804.05 | 799.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 796.55 | 804.05 | 799.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 796.55 | 804.05 | 799.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:00:00 | 796.55 | 804.05 | 799.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 796.25 | 802.49 | 799.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:00:00 | 796.25 | 802.49 | 799.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 800.00 | 801.21 | 799.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 14:45:00 | 801.20 | 800.97 | 799.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 15:15:00 | 802.10 | 800.97 | 799.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 10:30:00 | 803.60 | 800.76 | 799.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 14:45:00 | 802.60 | 801.08 | 800.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 801.00 | 801.90 | 800.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:45:00 | 801.00 | 801.90 | 800.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 796.20 | 800.76 | 800.39 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-18 11:15:00 | 796.20 | 800.76 | 800.39 | SL hit (close<static) qty=1.00 sl=797.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 12:15:00 | 795.85 | 799.78 | 799.98 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 10:15:00 | 801.65 | 799.91 | 799.88 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 13:15:00 | 797.65 | 799.52 | 799.72 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 14:15:00 | 803.50 | 800.31 | 800.07 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 09:15:00 | 795.30 | 799.90 | 799.96 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 10:15:00 | 804.00 | 800.72 | 800.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 11:15:00 | 814.65 | 803.51 | 801.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 810.50 | 811.70 | 808.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 12:30:00 | 811.40 | 811.70 | 808.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 814.20 | 812.20 | 808.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:30:00 | 809.10 | 812.20 | 808.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 807.00 | 810.94 | 808.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 803.95 | 810.94 | 808.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 806.50 | 810.05 | 808.46 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 800.35 | 806.84 | 807.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 12:15:00 | 798.60 | 805.19 | 806.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 814.65 | 802.69 | 804.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 814.65 | 802.69 | 804.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 814.65 | 802.69 | 804.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:30:00 | 828.55 | 802.69 | 804.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 813.75 | 804.91 | 805.26 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 11:15:00 | 810.50 | 806.02 | 805.73 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 15:15:00 | 796.80 | 805.19 | 806.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 792.60 | 802.67 | 804.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 782.65 | 775.23 | 783.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 782.65 | 775.23 | 783.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 782.65 | 775.23 | 783.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 783.80 | 775.23 | 783.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 781.00 | 776.38 | 783.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:30:00 | 783.95 | 776.38 | 783.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 787.55 | 779.35 | 783.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:30:00 | 786.00 | 779.35 | 783.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 787.85 | 781.05 | 783.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 15:00:00 | 786.30 | 782.10 | 784.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 794.85 | 787.00 | 785.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 794.85 | 787.00 | 785.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 798.05 | 789.21 | 787.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 09:15:00 | 774.70 | 788.60 | 787.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 09:15:00 | 774.70 | 788.60 | 787.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 774.70 | 788.60 | 787.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:45:00 | 775.25 | 788.60 | 787.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 10:15:00 | 777.00 | 786.28 | 786.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 767.25 | 777.20 | 779.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 14:15:00 | 777.20 | 776.11 | 778.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 14:45:00 | 776.90 | 776.11 | 778.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 769.25 | 774.77 | 777.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 10:15:00 | 768.45 | 774.77 | 777.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 12:15:00 | 768.65 | 773.27 | 776.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 13:15:00 | 768.85 | 772.58 | 775.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 14:15:00 | 777.70 | 775.70 | 775.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 14:15:00 | 777.70 | 775.70 | 775.61 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 15:15:00 | 773.25 | 775.21 | 775.40 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 801.95 | 780.56 | 777.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 818.90 | 788.23 | 781.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 09:15:00 | 814.05 | 816.91 | 808.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 09:30:00 | 811.65 | 816.91 | 808.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 807.55 | 815.90 | 812.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:00:00 | 807.55 | 815.90 | 812.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 807.90 | 814.30 | 811.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:45:00 | 807.25 | 814.30 | 811.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 14:15:00 | 806.85 | 810.81 | 810.88 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 813.90 | 811.14 | 811.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 12:15:00 | 823.20 | 814.91 | 812.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 09:15:00 | 847.65 | 849.19 | 840.36 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 15:00:00 | 857.10 | 850.55 | 844.20 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 831.80 | 848.12 | 844.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 831.80 | 848.12 | 844.28 | SL hit (close<ema400) qty=1.00 sl=844.28 alert=retest1 |

### Cycle 48 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 824.00 | 840.04 | 841.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 12:15:00 | 821.35 | 836.30 | 839.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 812.85 | 810.38 | 815.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 812.85 | 810.38 | 815.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 812.85 | 810.38 | 815.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:00:00 | 803.85 | 808.99 | 813.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:15:00 | 763.66 | 774.48 | 787.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 765.00 | 762.25 | 775.31 | SL hit (close>ema200) qty=0.50 sl=762.25 alert=retest2 |

### Cycle 49 — BUY (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 09:15:00 | 770.70 | 762.25 | 762.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 12:15:00 | 782.20 | 772.45 | 768.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 785.10 | 789.80 | 783.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 10:00:00 | 785.10 | 789.80 | 783.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 779.25 | 787.69 | 782.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 779.25 | 787.69 | 782.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 778.80 | 785.91 | 782.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 14:15:00 | 779.50 | 783.33 | 781.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 09:15:00 | 782.80 | 781.31 | 781.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 771.75 | 779.40 | 780.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 771.75 | 779.40 | 780.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 763.25 | 776.17 | 778.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 765.25 | 763.53 | 769.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 765.25 | 763.53 | 769.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 756.10 | 755.18 | 761.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 764.40 | 755.18 | 761.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 761.75 | 756.49 | 761.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:30:00 | 763.50 | 756.49 | 761.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 760.25 | 757.24 | 761.35 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 14:15:00 | 775.40 | 763.94 | 763.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 779.30 | 768.94 | 766.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 09:15:00 | 774.50 | 778.50 | 773.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 774.50 | 778.50 | 773.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 774.50 | 778.50 | 773.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:15:00 | 771.80 | 778.50 | 773.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 770.65 | 776.93 | 773.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:45:00 | 771.00 | 776.93 | 773.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 784.80 | 778.51 | 774.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 12:45:00 | 789.25 | 779.46 | 775.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 09:15:00 | 791.85 | 780.22 | 777.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 10:15:00 | 800.60 | 802.22 | 802.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 10:15:00 | 800.60 | 802.22 | 802.24 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 11:15:00 | 803.00 | 802.37 | 802.31 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 12:15:00 | 801.50 | 802.20 | 802.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 13:15:00 | 800.85 | 801.93 | 802.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 14:15:00 | 803.30 | 802.20 | 802.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 14:15:00 | 803.30 | 802.20 | 802.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 803.30 | 802.20 | 802.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 15:00:00 | 803.30 | 802.20 | 802.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 15:15:00 | 805.50 | 802.86 | 802.52 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 12:15:00 | 801.15 | 802.20 | 802.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 13:15:00 | 799.20 | 801.60 | 802.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 15:15:00 | 703.10 | 702.01 | 717.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 09:15:00 | 698.80 | 702.01 | 717.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 720.25 | 707.67 | 716.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:00:00 | 720.25 | 707.67 | 716.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 721.90 | 710.52 | 717.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:45:00 | 724.30 | 710.52 | 717.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 716.30 | 715.84 | 717.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:45:00 | 717.75 | 715.84 | 717.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 726.20 | 717.64 | 718.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:00:00 | 726.20 | 717.64 | 718.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 13:15:00 | 725.35 | 719.18 | 718.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 14:15:00 | 728.70 | 721.08 | 719.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 733.75 | 736.64 | 730.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 10:00:00 | 733.75 | 736.64 | 730.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 729.45 | 735.20 | 730.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 729.45 | 735.20 | 730.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 729.90 | 734.14 | 730.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:15:00 | 728.25 | 734.14 | 730.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 720.00 | 728.65 | 729.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 10:15:00 | 716.65 | 726.25 | 728.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 14:15:00 | 724.60 | 723.29 | 725.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-14 15:00:00 | 724.60 | 723.29 | 725.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 724.95 | 723.62 | 725.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 729.00 | 723.62 | 725.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 728.00 | 724.50 | 725.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 10:30:00 | 721.05 | 723.88 | 725.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 716.00 | 724.12 | 724.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 11:45:00 | 719.35 | 715.73 | 718.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 13:15:00 | 722.45 | 718.19 | 717.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 722.45 | 718.19 | 717.71 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 710.70 | 717.20 | 717.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 10:15:00 | 704.75 | 710.85 | 713.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 11:15:00 | 716.80 | 712.04 | 713.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 11:15:00 | 716.80 | 712.04 | 713.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 716.80 | 712.04 | 713.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 12:00:00 | 716.80 | 712.04 | 713.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 713.40 | 712.31 | 713.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 12:30:00 | 716.55 | 712.31 | 713.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 715.45 | 712.94 | 713.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 13:45:00 | 716.50 | 712.94 | 713.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 716.15 | 713.58 | 714.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 716.15 | 713.58 | 714.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 716.00 | 714.07 | 714.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:15:00 | 710.55 | 714.07 | 714.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 09:15:00 | 716.25 | 714.50 | 714.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 728.10 | 718.20 | 716.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 743.70 | 747.10 | 741.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 743.70 | 747.10 | 741.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 743.70 | 747.10 | 741.72 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2025-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 11:15:00 | 729.05 | 739.20 | 740.49 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 752.70 | 740.43 | 739.19 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 733.00 | 744.21 | 744.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 731.10 | 741.59 | 743.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 725.85 | 720.69 | 724.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 11:15:00 | 725.85 | 720.69 | 724.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 725.85 | 720.69 | 724.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 725.85 | 720.69 | 724.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 727.15 | 721.98 | 724.62 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 15:15:00 | 731.50 | 726.72 | 726.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 735.95 | 731.34 | 728.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 727.30 | 731.90 | 729.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 727.30 | 731.90 | 729.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 727.30 | 731.90 | 729.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 727.30 | 731.90 | 729.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 728.05 | 731.13 | 729.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 14:30:00 | 730.65 | 730.24 | 729.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 15:00:00 | 730.75 | 730.24 | 729.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 725.00 | 729.27 | 729.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 725.00 | 729.27 | 729.36 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 13:15:00 | 731.25 | 729.25 | 729.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 12:15:00 | 732.30 | 730.93 | 730.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 14:15:00 | 728.25 | 730.41 | 730.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 14:15:00 | 728.25 | 730.41 | 730.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 728.25 | 730.41 | 730.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:30:00 | 727.20 | 730.41 | 730.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 15:15:00 | 726.10 | 729.55 | 729.73 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 14:15:00 | 733.20 | 730.28 | 729.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 10:15:00 | 740.25 | 733.14 | 731.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 14:15:00 | 745.35 | 745.44 | 740.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 15:00:00 | 745.35 | 745.44 | 740.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 743.55 | 745.31 | 741.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 742.65 | 745.31 | 741.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 743.35 | 744.92 | 741.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:30:00 | 740.85 | 744.92 | 741.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 739.30 | 743.52 | 741.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 739.30 | 743.52 | 741.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 737.75 | 742.37 | 741.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:30:00 | 737.00 | 742.37 | 741.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 739.50 | 741.28 | 740.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 736.00 | 741.28 | 740.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 10:15:00 | 736.85 | 739.95 | 740.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 733.20 | 738.30 | 739.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 705.00 | 702.86 | 712.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 704.50 | 701.52 | 705.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 704.50 | 701.52 | 705.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:00:00 | 701.00 | 701.72 | 705.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 14:15:00 | 700.00 | 701.89 | 704.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 15:15:00 | 700.00 | 701.78 | 704.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 10:00:00 | 701.00 | 701.34 | 703.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 701.35 | 701.42 | 703.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:30:00 | 703.00 | 701.42 | 703.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 709.40 | 702.26 | 702.90 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 709.40 | 702.26 | 702.90 | SL hit (close>static) qty=1.00 sl=708.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 11:15:00 | 714.40 | 705.14 | 704.12 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 699.20 | 704.72 | 704.93 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 712.70 | 706.06 | 705.28 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 700.40 | 704.58 | 705.02 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 14:15:00 | 720.60 | 706.81 | 705.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-09 10:15:00 | 733.15 | 715.57 | 710.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 11:15:00 | 711.60 | 714.78 | 710.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-09 12:00:00 | 711.60 | 714.78 | 710.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 709.10 | 713.64 | 710.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:30:00 | 708.25 | 713.64 | 710.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 709.55 | 712.82 | 710.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 13:30:00 | 710.70 | 712.82 | 710.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 710.35 | 712.33 | 710.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:30:00 | 710.70 | 712.33 | 710.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 709.00 | 711.66 | 710.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:15:00 | 700.40 | 711.66 | 710.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 707.90 | 710.91 | 709.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:30:00 | 709.00 | 710.91 | 709.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 714.00 | 711.53 | 710.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-13 09:45:00 | 717.45 | 712.98 | 711.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 10:15:00 | 716.15 | 713.01 | 712.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 10:45:00 | 717.00 | 713.37 | 712.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 11:15:00 | 716.70 | 713.37 | 712.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 711.15 | 713.09 | 712.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:00:00 | 711.15 | 713.09 | 712.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 710.25 | 712.52 | 712.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:30:00 | 710.55 | 712.52 | 712.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-14 15:15:00 | 710.80 | 711.89 | 711.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2026-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 15:15:00 | 710.80 | 711.89 | 711.96 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 720.55 | 713.62 | 712.74 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 708.55 | 714.69 | 715.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 703.15 | 711.78 | 713.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 700.45 | 699.35 | 705.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 13:00:00 | 700.45 | 699.35 | 705.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 719.05 | 703.02 | 705.30 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 721.10 | 708.56 | 707.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 13:15:00 | 722.00 | 712.76 | 709.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 712.70 | 715.54 | 712.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 10:00:00 | 712.70 | 715.54 | 712.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 710.45 | 714.52 | 711.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:00:00 | 710.45 | 714.52 | 711.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 702.80 | 712.17 | 711.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 12:00:00 | 702.80 | 712.17 | 711.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 705.40 | 710.82 | 710.53 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 705.85 | 709.83 | 710.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 703.80 | 708.62 | 709.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 690.50 | 688.39 | 696.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 10:00:00 | 690.50 | 688.39 | 696.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 672.75 | 685.22 | 691.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:15:00 | 670.25 | 685.22 | 691.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 11:45:00 | 671.25 | 666.64 | 670.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 14:00:00 | 670.00 | 669.28 | 671.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 686.20 | 673.01 | 671.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 686.20 | 673.01 | 671.50 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 09:15:00 | 650.95 | 670.60 | 671.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 11:15:00 | 645.00 | 661.84 | 667.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 637.90 | 636.20 | 643.12 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-09 11:30:00 | 630.00 | 634.62 | 641.18 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 631.55 | 629.57 | 635.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:00:00 | 631.55 | 629.57 | 635.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 09:15:00 | 598.50 | 612.17 | 620.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-02-13 09:15:00 | 567.00 | 584.40 | 600.15 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 83 — BUY (started 2026-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 12:15:00 | 556.30 | 552.64 | 552.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 12:15:00 | 561.45 | 556.35 | 554.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 10:15:00 | 562.00 | 566.23 | 562.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 10:15:00 | 562.00 | 566.23 | 562.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 562.00 | 566.23 | 562.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:45:00 | 559.25 | 566.23 | 562.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 556.75 | 564.33 | 562.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 12:00:00 | 556.75 | 564.33 | 562.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 554.80 | 561.69 | 561.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 15:00:00 | 554.80 | 561.69 | 561.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2026-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 15:15:00 | 554.00 | 560.15 | 560.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 10:15:00 | 543.75 | 555.66 | 558.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 09:15:00 | 560.25 | 552.17 | 554.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 09:15:00 | 560.25 | 552.17 | 554.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 560.25 | 552.17 | 554.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 562.15 | 552.17 | 554.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 560.00 | 553.74 | 555.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:00:00 | 560.00 | 553.74 | 555.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 13:15:00 | 560.60 | 556.99 | 556.65 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 541.00 | 554.18 | 555.50 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 560.00 | 553.22 | 552.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 561.20 | 556.94 | 554.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 15:15:00 | 562.00 | 563.43 | 560.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-12 09:15:00 | 561.40 | 563.43 | 560.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 558.85 | 562.51 | 560.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 560.55 | 562.51 | 560.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 564.60 | 562.93 | 560.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:30:00 | 566.30 | 563.58 | 561.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 13:15:00 | 567.85 | 563.83 | 561.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 14:00:00 | 566.05 | 564.27 | 561.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 15:15:00 | 567.50 | 564.57 | 562.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 567.50 | 565.15 | 562.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:15:00 | 562.20 | 565.15 | 562.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 553.85 | 562.89 | 561.86 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 553.85 | 562.89 | 561.86 | SL hit (close<static) qty=1.00 sl=557.25 alert=retest2 |

### Cycle 88 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 557.20 | 560.86 | 561.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 554.35 | 559.56 | 560.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 11:15:00 | 560.00 | 555.70 | 557.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 11:15:00 | 560.00 | 555.70 | 557.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 560.00 | 555.70 | 557.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 12:00:00 | 560.00 | 555.70 | 557.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 540.30 | 552.62 | 556.11 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 567.70 | 557.71 | 556.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 578.40 | 565.58 | 561.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 559.95 | 571.84 | 567.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 559.95 | 571.84 | 567.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 559.95 | 571.84 | 567.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 559.95 | 571.84 | 567.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 559.15 | 569.31 | 566.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 559.15 | 569.31 | 566.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 560.40 | 565.14 | 565.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 559.00 | 563.91 | 564.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 565.30 | 563.40 | 564.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 565.30 | 563.40 | 564.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 565.30 | 563.40 | 564.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 563.45 | 564.17 | 564.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:00:00 | 563.10 | 563.96 | 564.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 547.80 | 564.26 | 564.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 14:15:00 | 535.28 | 543.63 | 547.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 14:15:00 | 534.95 | 543.63 | 547.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 09:15:00 | 520.41 | 537.59 | 543.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 535.00 | 522.95 | 528.01 | SL hit (close>ema200) qty=0.50 sl=522.95 alert=retest2 |

### Cycle 91 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 540.25 | 531.35 | 531.03 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 525.50 | 530.65 | 531.20 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 12:15:00 | 536.85 | 532.23 | 531.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 544.80 | 534.75 | 533.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 12:15:00 | 550.00 | 552.40 | 547.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 13:00:00 | 550.00 | 552.40 | 547.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 13:15:00 | 548.85 | 551.69 | 547.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 14:00:00 | 548.85 | 551.69 | 547.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 547.65 | 550.37 | 547.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 548.85 | 550.37 | 547.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 11:30:00 | 549.90 | 549.89 | 549.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 15:15:00 | 550.00 | 554.55 | 554.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 15:15:00 | 550.00 | 554.55 | 554.57 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 565.10 | 556.66 | 555.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 575.35 | 564.10 | 560.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 590.60 | 595.93 | 586.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 09:15:00 | 590.60 | 595.93 | 586.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 590.60 | 595.93 | 586.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 11:45:00 | 607.25 | 598.70 | 593.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 14:45:00 | 602.15 | 600.28 | 595.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 09:30:00 | 603.15 | 599.98 | 596.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 12:15:00 | 580.00 | 591.42 | 592.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2026-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 12:15:00 | 580.00 | 591.42 | 592.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 11:15:00 | 576.50 | 582.32 | 586.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 10:15:00 | 548.80 | 548.45 | 560.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 10:45:00 | 548.50 | 548.45 | 560.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 525.30 | 515.23 | 516.51 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 522.80 | 518.01 | 517.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 15:15:00 | 525.95 | 521.06 | 519.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 519.90 | 524.97 | 522.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 14:15:00 | 519.90 | 524.97 | 522.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 519.90 | 524.97 | 522.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 15:00:00 | 519.90 | 524.97 | 522.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 519.70 | 523.92 | 522.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 529.80 | 523.92 | 522.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-20 09:30:00 | 834.80 | 2025-05-22 15:15:00 | 818.00 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-06-04 13:15:00 | 820.55 | 2025-06-05 10:15:00 | 834.05 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-06-04 14:15:00 | 820.25 | 2025-06-05 10:15:00 | 834.05 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-06-04 15:00:00 | 822.10 | 2025-06-05 10:15:00 | 834.05 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-06-11 12:15:00 | 854.85 | 2025-06-11 14:15:00 | 847.70 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-06-12 09:15:00 | 857.25 | 2025-06-13 09:15:00 | 848.50 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-06-12 14:15:00 | 857.55 | 2025-06-13 09:15:00 | 848.50 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-06-13 10:30:00 | 859.95 | 2025-06-18 13:15:00 | 863.70 | STOP_HIT | 1.00 | 0.44% |
| BUY | retest2 | 2025-06-16 11:30:00 | 871.20 | 2025-06-18 13:15:00 | 863.70 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-06-18 11:30:00 | 871.50 | 2025-06-18 13:15:00 | 863.70 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-06-24 13:45:00 | 834.00 | 2025-06-25 09:15:00 | 862.45 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2025-06-24 15:15:00 | 832.70 | 2025-06-25 09:15:00 | 862.45 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2025-07-01 13:45:00 | 835.70 | 2025-07-02 09:15:00 | 859.15 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-07-01 14:15:00 | 835.90 | 2025-07-02 09:15:00 | 859.15 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-07-10 14:00:00 | 851.70 | 2025-07-10 14:15:00 | 846.30 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-07-10 14:45:00 | 849.25 | 2025-07-10 15:15:00 | 847.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-08-08 09:15:00 | 783.70 | 2025-08-11 12:15:00 | 791.95 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-08-11 11:45:00 | 787.50 | 2025-08-11 12:15:00 | 791.95 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-08-13 14:45:00 | 801.20 | 2025-08-18 11:15:00 | 796.20 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-08-13 15:15:00 | 802.10 | 2025-08-18 11:15:00 | 796.20 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-08-14 10:30:00 | 803.60 | 2025-08-18 11:15:00 | 796.20 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-08-14 14:45:00 | 802.60 | 2025-08-18 11:15:00 | 796.20 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-09-01 15:00:00 | 786.30 | 2025-09-02 10:15:00 | 794.85 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-09-08 10:15:00 | 768.45 | 2025-09-09 14:15:00 | 777.70 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-09-08 12:15:00 | 768.65 | 2025-09-09 14:15:00 | 777.70 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-09-08 13:15:00 | 768.85 | 2025-09-09 14:15:00 | 777.70 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest1 | 2025-09-19 15:00:00 | 857.10 | 2025-09-22 09:15:00 | 831.80 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2025-09-25 12:00:00 | 803.85 | 2025-09-29 11:15:00 | 763.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 12:00:00 | 803.85 | 2025-09-30 09:15:00 | 765.00 | STOP_HIT | 0.50 | 4.83% |
| BUY | retest2 | 2025-10-13 14:15:00 | 779.50 | 2025-10-14 09:15:00 | 771.75 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-10-14 09:15:00 | 782.80 | 2025-10-14 09:15:00 | 771.75 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-10-20 12:45:00 | 789.25 | 2025-10-30 10:15:00 | 800.60 | STOP_HIT | 1.00 | 1.44% |
| BUY | retest2 | 2025-10-23 09:15:00 | 791.85 | 2025-10-30 10:15:00 | 800.60 | STOP_HIT | 1.00 | 1.11% |
| SELL | retest2 | 2025-11-17 10:30:00 | 721.05 | 2025-11-20 13:15:00 | 722.45 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-11-18 09:15:00 | 716.00 | 2025-11-20 13:15:00 | 722.45 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-11-19 11:45:00 | 719.35 | 2025-11-20 13:15:00 | 722.45 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-12-15 14:30:00 | 730.65 | 2025-12-16 09:15:00 | 725.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-12-15 15:00:00 | 730.75 | 2025-12-16 09:15:00 | 725.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2026-01-01 12:00:00 | 701.00 | 2026-01-05 09:15:00 | 709.40 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-01-01 14:15:00 | 700.00 | 2026-01-05 09:15:00 | 709.40 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-01-01 15:15:00 | 700.00 | 2026-01-05 09:15:00 | 709.40 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-01-02 10:00:00 | 701.00 | 2026-01-05 09:15:00 | 709.40 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-01-05 10:30:00 | 704.00 | 2026-01-05 11:15:00 | 714.40 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2026-01-13 09:45:00 | 717.45 | 2026-01-14 15:15:00 | 710.80 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2026-01-14 10:15:00 | 716.15 | 2026-01-14 15:15:00 | 710.80 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2026-01-14 10:45:00 | 717.00 | 2026-01-14 15:15:00 | 710.80 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-01-14 11:15:00 | 716.70 | 2026-01-14 15:15:00 | 710.80 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-01-29 10:15:00 | 670.25 | 2026-02-03 09:15:00 | 686.20 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2026-02-01 11:45:00 | 671.25 | 2026-02-03 09:15:00 | 686.20 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2026-02-01 14:00:00 | 670.00 | 2026-02-03 09:15:00 | 686.20 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest1 | 2026-02-09 11:30:00 | 630.00 | 2026-02-12 09:15:00 | 598.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-02-09 11:30:00 | 630.00 | 2026-02-13 09:15:00 | 567.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-13 15:15:00 | 586.00 | 2026-02-20 13:15:00 | 559.07 | PARTIAL | 0.50 | 4.59% |
| SELL | retest2 | 2026-02-17 10:00:00 | 588.50 | 2026-02-20 13:15:00 | 559.41 | PARTIAL | 0.50 | 4.94% |
| SELL | retest2 | 2026-02-17 13:30:00 | 588.85 | 2026-02-23 09:15:00 | 556.70 | PARTIAL | 0.50 | 5.46% |
| SELL | retest2 | 2026-02-13 15:15:00 | 586.00 | 2026-02-23 10:15:00 | 567.30 | STOP_HIT | 0.50 | 3.19% |
| SELL | retest2 | 2026-02-17 10:00:00 | 588.50 | 2026-02-23 10:15:00 | 567.30 | STOP_HIT | 0.50 | 3.60% |
| SELL | retest2 | 2026-02-17 13:30:00 | 588.85 | 2026-02-23 10:15:00 | 567.30 | STOP_HIT | 0.50 | 3.66% |
| BUY | retest2 | 2026-03-12 11:30:00 | 566.30 | 2026-03-13 09:15:00 | 553.85 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2026-03-12 13:15:00 | 567.85 | 2026-03-13 09:15:00 | 553.85 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2026-03-12 14:00:00 | 566.05 | 2026-03-13 09:15:00 | 553.85 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2026-03-12 15:15:00 | 567.50 | 2026-03-13 09:15:00 | 553.85 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2026-03-13 11:15:00 | 559.00 | 2026-03-13 11:15:00 | 557.20 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2026-03-20 12:15:00 | 563.45 | 2026-03-25 14:15:00 | 535.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 13:00:00 | 563.10 | 2026-03-25 14:15:00 | 534.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-23 09:15:00 | 547.80 | 2026-03-27 09:15:00 | 520.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 563.45 | 2026-04-01 09:15:00 | 535.00 | STOP_HIT | 0.50 | 5.05% |
| SELL | retest2 | 2026-03-20 13:00:00 | 563.10 | 2026-04-01 09:15:00 | 535.00 | STOP_HIT | 0.50 | 4.99% |
| SELL | retest2 | 2026-03-23 09:15:00 | 547.80 | 2026-04-01 09:15:00 | 535.00 | STOP_HIT | 0.50 | 2.34% |
| BUY | retest2 | 2026-04-08 09:15:00 | 548.85 | 2026-04-13 15:15:00 | 550.00 | STOP_HIT | 1.00 | 0.21% |
| BUY | retest2 | 2026-04-09 11:30:00 | 549.90 | 2026-04-13 15:15:00 | 550.00 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2026-04-21 11:45:00 | 607.25 | 2026-04-22 12:15:00 | 580.00 | STOP_HIT | 1.00 | -4.49% |
| BUY | retest2 | 2026-04-21 14:45:00 | 602.15 | 2026-04-22 12:15:00 | 580.00 | STOP_HIT | 1.00 | -3.68% |
| BUY | retest2 | 2026-04-22 09:30:00 | 603.15 | 2026-04-22 12:15:00 | 580.00 | STOP_HIT | 1.00 | -3.84% |
