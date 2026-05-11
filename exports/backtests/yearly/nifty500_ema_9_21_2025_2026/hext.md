# Hexaware Technologies Ltd. (HEXT)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 486.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 72 |
| ALERT1 | 47 |
| ALERT2 | 46 |
| ALERT2_SKIP | 26 |
| ALERT3 | 129 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 70 |
| PARTIAL | 2 |
| TARGET_HIT | 7 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 72 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 58
- **Target hits / Stop hits / Partials:** 7 / 63 / 2
- **Avg / median % per leg:** 0.13% / -0.95%
- **Sum % (uncompounded):** 9.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 10 | 22.2% | 5 | 40 | 0 | 0.50% | 22.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 45 | 10 | 22.2% | 5 | 40 | 0 | 0.50% | 22.4% |
| SELL (all) | 27 | 4 | 14.8% | 2 | 23 | 2 | -0.48% | -12.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 27 | 4 | 14.8% | 2 | 23 | 2 | -0.48% | -12.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 72 | 14 | 19.4% | 7 | 63 | 2 | 0.13% | 9.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 12:15:00 | 745.85 | 757.66 | 758.37 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 11:15:00 | 766.10 | 758.01 | 757.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 14:15:00 | 790.30 | 767.44 | 762.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 11:15:00 | 795.35 | 802.30 | 793.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-23 12:00:00 | 795.35 | 802.30 | 793.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 799.45 | 801.73 | 794.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:00:00 | 799.45 | 801.73 | 794.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 794.00 | 799.44 | 794.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 15:00:00 | 794.00 | 799.44 | 794.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 796.50 | 798.85 | 794.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 802.55 | 798.85 | 794.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 14:00:00 | 800.95 | 800.43 | 797.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 15:15:00 | 800.10 | 799.76 | 797.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 09:30:00 | 803.40 | 800.43 | 797.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 819.50 | 816.71 | 812.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 09:45:00 | 825.80 | 816.28 | 813.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-30 14:15:00 | 882.81 | 839.97 | 827.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 14:15:00 | 812.55 | 826.66 | 827.28 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 824.35 | 819.15 | 818.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 15:15:00 | 826.00 | 822.46 | 820.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 13:15:00 | 824.60 | 824.83 | 822.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 13:15:00 | 824.60 | 824.83 | 822.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 13:15:00 | 824.60 | 824.83 | 822.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 13:45:00 | 824.55 | 824.83 | 822.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 820.85 | 824.03 | 822.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 820.85 | 824.03 | 822.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 822.05 | 823.63 | 822.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 822.70 | 823.63 | 822.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 824.85 | 823.88 | 822.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 13:30:00 | 831.25 | 826.82 | 824.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 11:15:00 | 825.00 | 835.30 | 835.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 825.00 | 835.30 | 835.92 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 834.75 | 828.44 | 828.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 09:15:00 | 843.70 | 834.81 | 832.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 12:15:00 | 833.00 | 835.14 | 833.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 12:15:00 | 833.00 | 835.14 | 833.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 833.00 | 835.14 | 833.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:45:00 | 832.90 | 835.14 | 833.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 839.25 | 835.96 | 833.80 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 10:15:00 | 818.15 | 831.37 | 832.42 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 11:15:00 | 838.35 | 830.69 | 830.53 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 14:15:00 | 823.65 | 829.27 | 829.95 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 836.50 | 829.32 | 829.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 843.20 | 833.23 | 831.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 837.45 | 838.02 | 834.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:30:00 | 837.95 | 838.02 | 834.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 845.00 | 845.96 | 842.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 13:00:00 | 852.80 | 847.28 | 843.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 15:15:00 | 850.00 | 855.00 | 855.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2025-06-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 15:15:00 | 850.00 | 855.00 | 855.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 09:15:00 | 845.00 | 853.00 | 854.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 13:15:00 | 850.15 | 849.71 | 852.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-01 14:00:00 | 850.15 | 849.71 | 852.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 850.10 | 849.79 | 851.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 14:45:00 | 850.60 | 849.79 | 851.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 841.05 | 848.04 | 850.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 856.05 | 848.04 | 850.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 863.80 | 851.19 | 852.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:45:00 | 863.70 | 851.19 | 852.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 10:15:00 | 862.65 | 853.49 | 853.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 14:15:00 | 875.10 | 862.06 | 857.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 13:15:00 | 869.40 | 871.69 | 865.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 13:15:00 | 869.40 | 871.69 | 865.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 869.40 | 871.69 | 865.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:45:00 | 866.00 | 871.69 | 865.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 870.30 | 871.41 | 865.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 15:00:00 | 870.30 | 871.41 | 865.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 871.50 | 870.80 | 866.47 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 848.95 | 861.64 | 863.16 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 13:15:00 | 867.35 | 863.68 | 863.64 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 14:15:00 | 860.90 | 863.12 | 863.40 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 09:15:00 | 871.10 | 864.38 | 863.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 11:15:00 | 879.45 | 869.04 | 866.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 884.20 | 885.65 | 880.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 11:00:00 | 884.20 | 885.65 | 880.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 878.40 | 884.20 | 880.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:00:00 | 878.40 | 884.20 | 880.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 876.00 | 882.56 | 879.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:45:00 | 876.75 | 882.56 | 879.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 875.85 | 881.22 | 879.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:45:00 | 875.55 | 881.22 | 879.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 873.70 | 877.69 | 878.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 12:15:00 | 862.00 | 871.31 | 874.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 10:15:00 | 855.00 | 850.84 | 857.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 10:15:00 | 855.00 | 850.84 | 857.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 855.00 | 850.84 | 857.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 855.00 | 850.84 | 857.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 853.00 | 850.02 | 854.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 855.15 | 850.02 | 854.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 852.35 | 850.49 | 854.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:45:00 | 855.80 | 850.49 | 854.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 855.95 | 851.58 | 854.24 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 09:15:00 | 864.90 | 856.90 | 856.03 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 11:15:00 | 852.30 | 858.66 | 859.41 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 12:15:00 | 859.00 | 858.80 | 858.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 09:15:00 | 864.10 | 860.23 | 859.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 12:15:00 | 860.00 | 860.57 | 859.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-23 12:30:00 | 860.60 | 860.57 | 859.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 860.00 | 860.46 | 859.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:45:00 | 860.05 | 860.46 | 859.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 860.50 | 860.47 | 859.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:45:00 | 860.25 | 860.47 | 859.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 09:15:00 | 852.10 | 858.85 | 859.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 848.00 | 855.69 | 857.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 14:15:00 | 738.00 | 736.22 | 767.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 15:00:00 | 738.00 | 736.22 | 767.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 705.00 | 698.00 | 703.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:45:00 | 703.70 | 698.00 | 703.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 708.00 | 700.00 | 703.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 708.00 | 700.00 | 703.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 709.55 | 701.91 | 704.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 714.60 | 701.91 | 704.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 714.55 | 705.53 | 705.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:45:00 | 714.55 | 705.53 | 705.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 11:15:00 | 718.00 | 708.02 | 706.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 13:15:00 | 722.00 | 711.82 | 708.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 10:15:00 | 720.35 | 721.62 | 715.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-06 11:00:00 | 720.35 | 721.62 | 715.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 717.15 | 720.72 | 715.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:45:00 | 715.45 | 720.72 | 715.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 720.30 | 720.64 | 715.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 14:45:00 | 722.15 | 720.87 | 716.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 10:00:00 | 722.95 | 721.75 | 717.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 12:15:00 | 712.35 | 721.58 | 720.78 | SL hit (close<static) qty=1.00 sl=715.25 alert=retest2 |

### Cycle 23 — SELL (started 2025-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 13:15:00 | 712.35 | 719.73 | 720.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 10:15:00 | 707.40 | 716.31 | 718.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 708.75 | 708.22 | 712.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-12 09:30:00 | 710.00 | 708.22 | 712.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 720.00 | 709.39 | 710.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:00:00 | 720.00 | 709.39 | 710.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 717.45 | 711.00 | 711.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:00:00 | 717.45 | 711.00 | 711.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 11:15:00 | 720.90 | 712.98 | 712.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 14:15:00 | 723.05 | 717.52 | 714.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 13:15:00 | 722.65 | 724.75 | 720.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 14:00:00 | 722.65 | 724.75 | 720.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 719.75 | 723.75 | 720.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:45:00 | 720.00 | 723.75 | 720.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 716.00 | 722.20 | 719.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 725.90 | 722.20 | 719.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-21 09:15:00 | 798.49 | 766.64 | 751.87 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 09:15:00 | 795.60 | 803.29 | 803.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 775.00 | 787.79 | 795.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 10:15:00 | 755.10 | 754.15 | 763.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-02 10:30:00 | 758.15 | 754.15 | 763.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 745.50 | 751.05 | 757.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 14:00:00 | 742.25 | 751.11 | 753.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 11:15:00 | 745.45 | 732.94 | 732.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 11:15:00 | 745.45 | 732.94 | 732.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 13:15:00 | 750.00 | 738.14 | 735.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 764.95 | 766.60 | 756.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 10:00:00 | 764.95 | 766.60 | 756.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 754.30 | 761.78 | 757.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 754.30 | 761.78 | 757.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 757.80 | 760.99 | 757.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 760.30 | 760.99 | 757.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 764.60 | 761.71 | 758.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 12:15:00 | 765.00 | 762.08 | 759.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 13:30:00 | 765.45 | 763.00 | 760.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 11:30:00 | 771.70 | 763.61 | 761.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 11:00:00 | 766.55 | 764.84 | 763.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 765.85 | 764.78 | 763.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 13:30:00 | 767.00 | 764.25 | 763.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 14:15:00 | 753.90 | 762.18 | 762.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 14:15:00 | 753.90 | 762.18 | 762.37 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 772.25 | 764.14 | 763.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 777.50 | 768.55 | 765.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 14:15:00 | 770.60 | 772.48 | 769.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 14:15:00 | 770.60 | 772.48 | 769.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 770.60 | 772.48 | 769.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:15:00 | 769.45 | 772.48 | 769.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 769.45 | 771.87 | 769.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 762.85 | 771.87 | 769.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 764.85 | 770.47 | 768.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 15:00:00 | 780.40 | 771.14 | 769.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 738.70 | 768.15 | 768.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 738.70 | 768.15 | 768.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 10:15:00 | 728.10 | 760.14 | 765.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 11:15:00 | 702.10 | 702.02 | 718.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 11:45:00 | 701.30 | 702.02 | 718.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 676.70 | 672.55 | 679.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 676.70 | 672.55 | 679.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 668.05 | 669.73 | 673.82 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 11:15:00 | 681.00 | 673.54 | 672.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 12:15:00 | 685.20 | 675.88 | 674.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 11:15:00 | 686.85 | 687.33 | 681.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 12:00:00 | 686.85 | 687.33 | 681.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 708.35 | 715.92 | 711.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:00:00 | 708.35 | 715.92 | 711.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 706.85 | 714.11 | 710.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 11:45:00 | 709.95 | 713.22 | 710.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:45:00 | 709.70 | 712.54 | 710.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 14:00:00 | 709.30 | 711.89 | 710.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 15:00:00 | 709.45 | 711.41 | 710.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 711.50 | 711.95 | 710.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:45:00 | 708.45 | 711.95 | 710.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 714.00 | 712.36 | 711.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 11:30:00 | 716.55 | 713.92 | 711.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 709.20 | 722.67 | 724.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 709.20 | 722.67 | 724.12 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 731.20 | 720.08 | 719.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 14:15:00 | 736.20 | 730.28 | 727.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 730.85 | 731.16 | 728.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 10:00:00 | 730.85 | 731.16 | 728.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 724.80 | 729.89 | 728.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:00:00 | 724.80 | 729.89 | 728.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 719.35 | 727.78 | 727.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:45:00 | 720.60 | 727.78 | 727.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2025-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 12:15:00 | 721.15 | 726.46 | 726.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 14:15:00 | 713.20 | 722.77 | 724.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 13:15:00 | 692.75 | 692.45 | 698.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 14:00:00 | 692.75 | 692.45 | 698.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 691.65 | 692.22 | 696.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:15:00 | 685.60 | 691.44 | 695.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 10:15:00 | 700.50 | 693.09 | 694.44 | SL hit (close>static) qty=1.00 sl=697.45 alert=retest2 |

### Cycle 34 — BUY (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 12:15:00 | 702.65 | 695.79 | 695.48 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 675.00 | 692.70 | 694.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 09:15:00 | 661.70 | 673.40 | 679.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 670.75 | 668.18 | 673.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 670.75 | 668.18 | 673.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 670.75 | 668.18 | 673.21 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 13:15:00 | 691.10 | 677.55 | 676.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 14:15:00 | 696.85 | 681.41 | 678.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 691.00 | 692.52 | 687.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 09:45:00 | 690.65 | 692.52 | 687.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 685.00 | 691.30 | 687.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:45:00 | 685.15 | 691.30 | 687.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 685.00 | 690.04 | 687.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:00:00 | 685.00 | 690.04 | 687.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 686.00 | 688.61 | 687.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 703.20 | 688.61 | 687.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 12:15:00 | 735.85 | 743.30 | 744.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 735.85 | 743.30 | 744.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 12:15:00 | 734.60 | 737.51 | 740.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 10:15:00 | 743.80 | 737.16 | 738.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 10:15:00 | 743.80 | 737.16 | 738.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 743.80 | 737.16 | 738.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:00:00 | 743.80 | 737.16 | 738.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 749.70 | 739.67 | 739.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:00:00 | 749.70 | 739.67 | 739.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 12:15:00 | 750.50 | 741.84 | 740.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 13:15:00 | 757.00 | 744.87 | 742.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 09:15:00 | 773.45 | 777.68 | 765.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 10:15:00 | 770.25 | 776.19 | 765.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 770.25 | 776.19 | 765.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:00:00 | 770.25 | 776.19 | 765.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 765.85 | 774.12 | 765.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:30:00 | 766.55 | 774.12 | 765.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 767.00 | 772.70 | 765.69 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 750.85 | 761.78 | 762.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 745.85 | 758.60 | 760.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 12:15:00 | 759.35 | 756.03 | 759.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 12:15:00 | 759.35 | 756.03 | 759.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 759.35 | 756.03 | 759.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 759.35 | 756.03 | 759.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 758.20 | 756.46 | 758.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 09:15:00 | 749.80 | 755.55 | 758.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 761.50 | 756.59 | 757.22 | SL hit (close>static) qty=1.00 sl=760.30 alert=retest2 |

### Cycle 40 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 761.75 | 758.17 | 757.86 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 10:15:00 | 749.30 | 756.40 | 757.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 11:15:00 | 741.80 | 753.48 | 755.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 12:15:00 | 745.85 | 741.82 | 746.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 12:15:00 | 745.85 | 741.82 | 746.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 745.85 | 741.82 | 746.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 746.60 | 741.82 | 746.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 745.40 | 742.53 | 746.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:30:00 | 746.05 | 742.53 | 746.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 752.50 | 744.53 | 747.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 752.50 | 744.53 | 747.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 748.25 | 745.27 | 747.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 757.25 | 745.27 | 747.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 754.50 | 747.12 | 748.05 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 753.40 | 749.55 | 749.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 760.00 | 752.15 | 750.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 756.85 | 766.01 | 761.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 756.85 | 766.01 | 761.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 756.85 | 766.01 | 761.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:15:00 | 756.35 | 766.01 | 761.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 748.60 | 762.53 | 760.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 748.60 | 762.53 | 760.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 756.05 | 759.59 | 759.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:30:00 | 756.00 | 759.59 | 759.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 758.50 | 759.37 | 759.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:45:00 | 764.25 | 759.47 | 759.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 11:15:00 | 758.55 | 759.17 | 759.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 11:15:00 | 758.55 | 759.17 | 759.20 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 12:15:00 | 763.20 | 759.97 | 759.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 14:15:00 | 765.40 | 761.51 | 760.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 09:15:00 | 757.60 | 761.13 | 760.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 09:15:00 | 757.60 | 761.13 | 760.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 757.60 | 761.13 | 760.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:00:00 | 757.60 | 761.13 | 760.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 755.85 | 760.07 | 759.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:45:00 | 755.65 | 760.07 | 759.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2025-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 11:15:00 | 758.00 | 759.66 | 759.81 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 770.85 | 761.47 | 760.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 11:15:00 | 780.10 | 765.20 | 762.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 09:15:00 | 770.00 | 775.78 | 769.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 770.00 | 775.78 | 769.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 770.00 | 775.78 | 769.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:15:00 | 765.65 | 775.78 | 769.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 759.00 | 772.42 | 768.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 11:00:00 | 759.00 | 772.42 | 768.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 11:15:00 | 755.70 | 769.08 | 767.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 11:45:00 | 754.40 | 769.08 | 767.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 13:15:00 | 753.25 | 763.37 | 764.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-22 14:15:00 | 749.10 | 760.52 | 763.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 10:15:00 | 750.00 | 742.91 | 748.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 10:15:00 | 750.00 | 742.91 | 748.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 750.00 | 742.91 | 748.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:45:00 | 749.65 | 742.91 | 748.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 748.55 | 744.04 | 748.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 742.25 | 746.62 | 748.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 10:30:00 | 742.15 | 744.44 | 747.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 14:00:00 | 746.00 | 744.67 | 746.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 14:15:00 | 751.20 | 745.97 | 746.99 | SL hit (close>static) qty=1.00 sl=751.05 alert=retest2 |

### Cycle 48 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 767.90 | 748.73 | 746.72 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 15:15:00 | 751.20 | 753.88 | 754.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 09:15:00 | 744.95 | 752.09 | 753.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 12:15:00 | 742.45 | 739.51 | 743.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 12:15:00 | 742.45 | 739.51 | 743.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 742.45 | 739.51 | 743.82 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2026-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 14:15:00 | 755.95 | 743.88 | 743.39 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 737.60 | 743.26 | 743.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 737.00 | 741.07 | 742.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 12:15:00 | 736.30 | 731.43 | 736.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 12:15:00 | 736.30 | 731.43 | 736.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 736.30 | 731.43 | 736.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 13:00:00 | 736.30 | 731.43 | 736.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 734.40 | 732.03 | 736.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 15:00:00 | 728.80 | 731.38 | 735.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 10:15:00 | 739.45 | 729.00 | 730.16 | SL hit (close>static) qty=1.00 sl=737.85 alert=retest2 |

### Cycle 52 — BUY (started 2026-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 14:15:00 | 733.50 | 731.02 | 730.91 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 09:15:00 | 715.50 | 728.55 | 729.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 10:15:00 | 710.15 | 724.87 | 728.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 09:15:00 | 728.45 | 718.92 | 722.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 728.45 | 718.92 | 722.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 728.45 | 718.92 | 722.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:00:00 | 728.45 | 718.92 | 722.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 731.15 | 721.36 | 723.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:30:00 | 733.40 | 721.36 | 723.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2026-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 12:15:00 | 744.00 | 727.25 | 725.92 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 14:15:00 | 724.05 | 727.52 | 727.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 720.30 | 725.57 | 726.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 717.90 | 708.74 | 713.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 717.90 | 708.74 | 713.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 717.90 | 708.74 | 713.38 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 722.20 | 716.94 | 716.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 725.65 | 718.68 | 717.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 713.55 | 718.67 | 717.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 713.55 | 718.67 | 717.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 713.55 | 718.67 | 717.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:00:00 | 713.55 | 718.67 | 717.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 712.65 | 717.47 | 717.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:45:00 | 712.50 | 717.47 | 717.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 712.85 | 716.54 | 716.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 09:15:00 | 710.55 | 714.74 | 715.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 15:15:00 | 695.00 | 694.58 | 700.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-29 09:15:00 | 702.75 | 694.58 | 700.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 696.85 | 695.03 | 699.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 13:00:00 | 691.40 | 694.85 | 698.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 14:15:00 | 691.90 | 694.80 | 698.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 690.70 | 694.99 | 697.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 10:15:00 | 692.20 | 694.77 | 697.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 695.60 | 694.94 | 697.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:45:00 | 697.65 | 694.94 | 697.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 698.25 | 695.60 | 697.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:00:00 | 698.25 | 695.60 | 697.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 697.60 | 696.00 | 697.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:30:00 | 698.50 | 696.00 | 697.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 697.60 | 696.32 | 697.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:00:00 | 697.60 | 696.32 | 697.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 699.90 | 697.04 | 697.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 15:00:00 | 699.90 | 697.04 | 697.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 696.00 | 696.83 | 697.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 697.80 | 696.83 | 697.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 700.95 | 697.65 | 697.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-01 10:15:00 | 703.85 | 698.89 | 698.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 703.85 | 698.89 | 698.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 11:15:00 | 709.95 | 701.10 | 699.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 700.80 | 706.34 | 703.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 15:15:00 | 700.80 | 706.34 | 703.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 700.80 | 706.34 | 703.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 694.90 | 706.34 | 703.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 707.10 | 706.50 | 703.41 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 12:15:00 | 694.10 | 700.49 | 701.11 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 719.95 | 704.44 | 702.61 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2026-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 11:15:00 | 690.00 | 706.13 | 707.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 624.00 | 682.28 | 694.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 11:15:00 | 581.00 | 552.36 | 563.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 11:15:00 | 581.00 | 552.36 | 563.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 581.00 | 552.36 | 563.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:00:00 | 581.00 | 552.36 | 563.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 565.00 | 554.89 | 563.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:00:00 | 560.60 | 557.51 | 563.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 547.45 | 558.78 | 563.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 15:15:00 | 575.00 | 563.93 | 563.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2026-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 15:15:00 | 575.00 | 563.93 | 563.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 11:15:00 | 583.45 | 571.22 | 566.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 574.15 | 581.81 | 574.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 574.15 | 581.81 | 574.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 574.15 | 581.81 | 574.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:00:00 | 574.15 | 581.81 | 574.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 572.00 | 579.84 | 574.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:45:00 | 570.50 | 579.84 | 574.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 570.15 | 577.91 | 574.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:30:00 | 568.00 | 577.91 | 574.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 572.00 | 574.05 | 573.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 576.20 | 574.05 | 573.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 10:00:00 | 573.80 | 574.00 | 573.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 10:15:00 | 569.05 | 573.01 | 572.95 | SL hit (close<static) qty=1.00 sl=571.10 alert=retest2 |

### Cycle 63 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 566.00 | 571.61 | 572.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 12:15:00 | 563.65 | 570.02 | 571.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 474.05 | 470.91 | 483.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 474.05 | 470.91 | 483.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 474.05 | 470.91 | 483.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 458.55 | 473.09 | 476.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 14:45:00 | 465.00 | 471.95 | 474.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 12:15:00 | 441.75 | 449.24 | 454.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 13:15:00 | 435.62 | 440.03 | 446.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-12 09:15:00 | 412.69 | 430.45 | 439.83 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 64 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 425.40 | 416.92 | 416.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 436.55 | 420.84 | 417.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 421.70 | 427.32 | 422.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 421.70 | 427.32 | 422.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 421.70 | 427.32 | 422.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 421.90 | 427.32 | 422.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 423.80 | 426.62 | 422.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 11:30:00 | 425.15 | 426.35 | 422.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 15:15:00 | 420.05 | 424.06 | 422.90 | SL hit (close<static) qty=1.00 sl=421.20 alert=retest2 |

### Cycle 65 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 436.80 | 439.40 | 439.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 430.40 | 437.60 | 438.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 10:15:00 | 433.65 | 432.20 | 435.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 10:15:00 | 433.65 | 432.20 | 435.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 433.65 | 432.20 | 435.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:00:00 | 433.65 | 432.20 | 435.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 431.90 | 432.14 | 435.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:00:00 | 431.90 | 432.14 | 435.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 15:15:00 | 434.00 | 432.22 | 434.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 09:15:00 | 432.60 | 432.22 | 434.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 427.15 | 431.20 | 433.46 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 445.25 | 434.73 | 434.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 10:15:00 | 448.65 | 440.66 | 437.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 15:15:00 | 476.35 | 476.93 | 471.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-13 09:15:00 | 466.80 | 476.93 | 471.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 475.35 | 476.61 | 471.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 478.45 | 476.89 | 472.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 482.00 | 475.09 | 473.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 11:15:00 | 481.50 | 489.16 | 489.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2026-04-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 11:15:00 | 481.50 | 489.16 | 489.33 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2026-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 14:15:00 | 491.00 | 488.84 | 488.71 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 467.65 | 484.90 | 486.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 10:15:00 | 462.25 | 480.37 | 484.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 13:15:00 | 465.65 | 462.86 | 469.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-23 13:45:00 | 465.75 | 462.86 | 469.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 453.10 | 452.34 | 455.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:00:00 | 453.10 | 452.34 | 455.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 453.55 | 452.85 | 454.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 453.55 | 452.85 | 454.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 452.70 | 452.82 | 454.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:15:00 | 452.60 | 452.82 | 454.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 453.00 | 452.86 | 454.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 10:45:00 | 450.00 | 452.44 | 453.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 11:30:00 | 450.45 | 452.39 | 453.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 448.80 | 452.97 | 453.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:45:00 | 450.35 | 452.27 | 453.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 452.55 | 451.57 | 452.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:45:00 | 452.50 | 451.57 | 452.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 452.10 | 451.68 | 452.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 13:45:00 | 452.95 | 451.68 | 452.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 447.00 | 450.74 | 452.08 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-04 12:15:00 | 455.70 | 452.40 | 452.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 455.70 | 452.40 | 452.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 13:15:00 | 456.75 | 453.27 | 452.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 450.60 | 454.03 | 453.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 450.60 | 454.03 | 453.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 450.60 | 454.03 | 453.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 11:15:00 | 454.95 | 453.98 | 453.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 12:15:00 | 454.75 | 453.90 | 453.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 14:15:00 | 451.25 | 453.07 | 453.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 14:15:00 | 451.25 | 453.07 | 453.15 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 15:15:00 | 453.70 | 453.20 | 453.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 454.70 | 453.50 | 453.33 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 10:30:00 | 754.75 | 2025-05-19 12:15:00 | 745.85 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-05-14 11:00:00 | 755.00 | 2025-05-19 12:15:00 | 745.85 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-05-14 13:15:00 | 762.00 | 2025-05-19 12:15:00 | 745.85 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-05-15 12:00:00 | 755.45 | 2025-05-19 12:15:00 | 745.85 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-05-26 09:15:00 | 802.55 | 2025-05-30 14:15:00 | 882.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-26 14:00:00 | 800.95 | 2025-05-30 14:15:00 | 881.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-26 15:15:00 | 800.10 | 2025-05-30 14:15:00 | 880.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-27 09:30:00 | 803.40 | 2025-05-30 14:15:00 | 883.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-30 09:45:00 | 825.80 | 2025-06-02 13:15:00 | 808.40 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-06-09 13:30:00 | 831.25 | 2025-06-12 11:15:00 | 825.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-06-26 13:00:00 | 852.80 | 2025-06-30 15:15:00 | 850.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-08-06 14:45:00 | 722.15 | 2025-08-08 12:15:00 | 712.35 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-08-07 10:00:00 | 722.95 | 2025-08-08 12:15:00 | 712.35 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-08-18 09:15:00 | 725.90 | 2025-08-21 09:15:00 | 798.49 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-04 14:00:00 | 742.25 | 2025-09-09 11:15:00 | 745.45 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-09-12 12:15:00 | 765.00 | 2025-09-16 14:15:00 | 753.90 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-09-12 13:30:00 | 765.45 | 2025-09-16 14:15:00 | 753.90 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-09-15 11:30:00 | 771.70 | 2025-09-16 14:15:00 | 753.90 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-09-16 11:00:00 | 766.55 | 2025-09-16 14:15:00 | 753.90 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-09-16 13:30:00 | 767.00 | 2025-09-16 14:15:00 | 753.90 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-09-19 15:00:00 | 780.40 | 2025-09-22 09:15:00 | 738.70 | STOP_HIT | 1.00 | -5.34% |
| BUY | retest2 | 2025-10-13 11:45:00 | 709.95 | 2025-10-17 09:15:00 | 709.20 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-10-13 12:45:00 | 709.70 | 2025-10-17 09:15:00 | 709.20 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-10-13 14:00:00 | 709.30 | 2025-10-17 09:15:00 | 709.20 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2025-10-13 15:00:00 | 709.45 | 2025-10-17 09:15:00 | 709.20 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-10-14 11:30:00 | 716.55 | 2025-10-17 09:15:00 | 709.20 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-11-04 11:15:00 | 685.60 | 2025-11-06 10:15:00 | 700.50 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-11-17 09:15:00 | 703.20 | 2025-12-01 12:15:00 | 735.85 | STOP_HIT | 1.00 | 4.64% |
| SELL | retest2 | 2025-12-09 09:15:00 | 749.80 | 2025-12-09 14:15:00 | 761.50 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-12-17 09:45:00 | 764.25 | 2025-12-17 11:15:00 | 758.55 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-12-26 09:15:00 | 742.25 | 2025-12-26 14:15:00 | 751.20 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-12-26 10:30:00 | 742.15 | 2025-12-26 14:15:00 | 751.20 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-12-26 14:00:00 | 746.00 | 2025-12-26 14:15:00 | 751.20 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-12-29 09:15:00 | 742.85 | 2025-12-30 14:15:00 | 767.90 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2025-12-29 11:15:00 | 743.20 | 2025-12-30 14:15:00 | 767.90 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2025-12-30 10:45:00 | 743.60 | 2025-12-30 14:15:00 | 767.90 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2025-12-30 14:15:00 | 743.90 | 2025-12-30 14:15:00 | 767.90 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2026-01-09 15:00:00 | 728.80 | 2026-01-13 10:15:00 | 739.45 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-01-13 13:00:00 | 728.70 | 2026-01-13 14:15:00 | 733.50 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2026-01-13 14:00:00 | 730.45 | 2026-01-13 14:15:00 | 733.50 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2026-01-29 13:00:00 | 691.40 | 2026-02-01 10:15:00 | 703.85 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2026-01-29 14:15:00 | 691.90 | 2026-02-01 10:15:00 | 703.85 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2026-01-30 09:15:00 | 690.70 | 2026-02-01 10:15:00 | 703.85 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2026-01-30 10:15:00 | 692.20 | 2026-02-01 10:15:00 | 703.85 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-02-13 15:00:00 | 560.60 | 2026-02-16 15:15:00 | 575.00 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2026-02-16 09:15:00 | 547.45 | 2026-02-16 15:15:00 | 575.00 | STOP_HIT | 1.00 | -5.03% |
| BUY | retest2 | 2026-02-19 09:15:00 | 576.20 | 2026-02-19 10:15:00 | 569.05 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2026-02-19 10:00:00 | 573.80 | 2026-02-19 10:15:00 | 569.05 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2026-03-02 09:15:00 | 458.55 | 2026-03-10 12:15:00 | 441.75 | PARTIAL | 0.50 | 3.66% |
| SELL | retest2 | 2026-03-02 14:45:00 | 465.00 | 2026-03-11 13:15:00 | 435.62 | PARTIAL | 0.50 | 6.32% |
| SELL | retest2 | 2026-03-02 09:15:00 | 458.55 | 2026-03-12 09:15:00 | 412.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-02 14:45:00 | 465.00 | 2026-03-12 09:15:00 | 418.50 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-03-19 11:30:00 | 425.15 | 2026-03-19 15:15:00 | 420.05 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-03-20 09:15:00 | 431.45 | 2026-03-30 12:15:00 | 436.80 | STOP_HIT | 1.00 | 1.24% |
| BUY | retest2 | 2026-03-23 09:30:00 | 425.05 | 2026-03-30 12:15:00 | 436.80 | STOP_HIT | 1.00 | 2.76% |
| BUY | retest2 | 2026-03-23 10:15:00 | 425.50 | 2026-03-30 12:15:00 | 436.80 | STOP_HIT | 1.00 | 2.66% |
| BUY | retest2 | 2026-03-23 11:15:00 | 441.90 | 2026-03-30 12:15:00 | 436.80 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2026-03-23 12:00:00 | 440.85 | 2026-03-30 12:15:00 | 436.80 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-03-23 12:30:00 | 440.40 | 2026-03-30 12:15:00 | 436.80 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2026-03-23 13:45:00 | 441.00 | 2026-03-30 12:15:00 | 436.80 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2026-03-24 11:15:00 | 439.60 | 2026-03-30 12:15:00 | 436.80 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2026-03-24 12:45:00 | 440.00 | 2026-03-30 12:15:00 | 436.80 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2026-03-24 14:00:00 | 439.70 | 2026-03-30 12:15:00 | 436.80 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2026-03-25 09:45:00 | 441.00 | 2026-03-30 12:15:00 | 436.80 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2026-04-13 10:45:00 | 478.45 | 2026-04-20 11:15:00 | 481.50 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2026-04-15 09:15:00 | 482.00 | 2026-04-20 11:15:00 | 481.50 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2026-04-29 10:45:00 | 450.00 | 2026-05-04 12:15:00 | 455.70 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-04-29 11:30:00 | 450.45 | 2026-05-04 12:15:00 | 455.70 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2026-04-30 09:15:00 | 448.80 | 2026-05-04 12:15:00 | 455.70 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2026-04-30 09:45:00 | 450.35 | 2026-05-04 12:15:00 | 455.70 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-05-05 11:15:00 | 454.95 | 2026-05-05 14:15:00 | 451.25 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2026-05-05 12:15:00 | 454.75 | 2026-05-05 14:15:00 | 451.25 | STOP_HIT | 1.00 | -0.77% |
