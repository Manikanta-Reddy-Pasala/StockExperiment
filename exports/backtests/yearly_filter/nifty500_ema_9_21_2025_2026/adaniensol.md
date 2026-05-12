# Adani Energy Solutions Ltd. (ADANIENSOL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1351.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 70 |
| ALERT1 | 53 |
| ALERT2 | 52 |
| ALERT2_SKIP | 20 |
| ALERT3 | 135 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 61 |
| PARTIAL | 11 |
| TARGET_HIT | 2 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 72 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 27 / 45
- **Target hits / Stop hits / Partials:** 2 / 59 / 11
- **Avg / median % per leg:** 0.95% / -0.67%
- **Sum % (uncompounded):** 68.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 36 | 5 | 13.9% | 2 | 34 | 0 | -1.03% | -37.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 36 | 5 | 13.9% | 2 | 34 | 0 | -1.03% | -37.2% |
| SELL (all) | 36 | 22 | 61.1% | 0 | 25 | 11 | 2.93% | 105.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 36 | 22 | 61.1% | 0 | 25 | 11 | 2.93% | 105.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 72 | 27 | 37.5% | 2 | 59 | 11 | 0.95% | 68.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 888.25 | 862.19 | 860.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 896.10 | 873.66 | 866.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 890.70 | 895.03 | 884.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 09:45:00 | 890.00 | 895.03 | 884.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 889.80 | 892.84 | 886.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 888.50 | 892.84 | 886.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 887.15 | 891.70 | 886.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:00:00 | 887.15 | 891.70 | 886.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 891.50 | 891.65 | 887.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:45:00 | 896.85 | 892.03 | 888.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:00:00 | 897.30 | 893.53 | 889.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 12:15:00 | 898.00 | 905.96 | 903.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 13:15:00 | 892.00 | 901.33 | 901.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 13:15:00 | 892.00 | 901.33 | 901.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 14:15:00 | 889.55 | 898.98 | 900.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 15:15:00 | 875.05 | 873.09 | 880.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 09:15:00 | 882.50 | 873.09 | 880.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 876.25 | 873.72 | 879.92 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 885.15 | 879.27 | 878.60 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 13:15:00 | 876.00 | 879.77 | 880.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 14:15:00 | 870.80 | 877.98 | 879.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 11:15:00 | 878.70 | 875.63 | 877.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 11:15:00 | 878.70 | 875.63 | 877.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 878.70 | 875.63 | 877.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:00:00 | 878.70 | 875.63 | 877.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 12:15:00 | 894.25 | 879.35 | 878.95 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 870.90 | 878.41 | 878.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 10:15:00 | 866.55 | 876.04 | 877.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 882.00 | 872.23 | 874.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 882.00 | 872.23 | 874.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 882.00 | 872.23 | 874.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 882.00 | 872.23 | 874.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 878.65 | 873.52 | 874.65 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 12:15:00 | 879.50 | 875.45 | 875.37 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 870.65 | 874.66 | 875.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 10:15:00 | 858.80 | 871.49 | 873.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 10:15:00 | 861.95 | 860.64 | 865.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 10:15:00 | 861.95 | 860.64 | 865.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 861.95 | 860.64 | 865.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 861.95 | 860.64 | 865.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 864.10 | 861.54 | 865.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:00:00 | 864.10 | 861.54 | 865.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 868.20 | 862.87 | 865.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:00:00 | 868.20 | 862.87 | 865.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 866.20 | 863.54 | 865.64 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 874.20 | 867.90 | 867.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 884.50 | 874.78 | 871.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 12:15:00 | 905.90 | 910.58 | 903.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 13:00:00 | 905.90 | 910.58 | 903.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 899.80 | 908.42 | 903.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 900.00 | 908.42 | 903.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 905.70 | 907.88 | 903.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 15:15:00 | 909.00 | 907.88 | 903.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 881.95 | 899.17 | 900.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 881.95 | 899.17 | 900.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 15:15:00 | 870.55 | 890.61 | 896.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 866.70 | 864.91 | 874.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 13:00:00 | 866.70 | 864.91 | 874.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 836.50 | 831.95 | 839.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:30:00 | 836.85 | 831.95 | 839.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 835.50 | 832.66 | 839.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 829.45 | 832.66 | 839.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 15:00:00 | 826.10 | 830.32 | 836.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 830.05 | 831.14 | 836.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 11:15:00 | 841.25 | 833.49 | 836.09 | SL hit (close>static) qty=1.00 sl=839.45 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 860.75 | 840.98 | 838.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 891.40 | 864.96 | 857.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 09:15:00 | 877.95 | 879.79 | 870.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 10:00:00 | 877.95 | 879.79 | 870.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 876.65 | 879.42 | 875.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:00:00 | 876.65 | 879.42 | 875.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 875.95 | 878.72 | 875.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:45:00 | 873.05 | 878.72 | 875.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 872.85 | 877.55 | 875.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:30:00 | 872.25 | 877.55 | 875.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 871.75 | 876.39 | 875.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 885.80 | 875.53 | 874.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 870.60 | 874.54 | 874.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 870.60 | 874.54 | 874.55 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 10:15:00 | 875.00 | 874.64 | 874.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 11:15:00 | 879.95 | 875.70 | 875.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 09:15:00 | 874.50 | 876.84 | 876.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 874.50 | 876.84 | 876.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 874.50 | 876.84 | 876.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:45:00 | 875.30 | 876.84 | 876.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 874.40 | 876.35 | 875.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:00:00 | 874.40 | 876.35 | 875.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 11:15:00 | 871.05 | 875.29 | 875.47 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 09:15:00 | 878.05 | 875.57 | 875.43 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 13:15:00 | 871.15 | 875.09 | 875.39 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 09:15:00 | 880.10 | 876.44 | 875.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 12:15:00 | 885.60 | 878.92 | 877.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 13:15:00 | 878.70 | 878.88 | 877.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-07 14:00:00 | 878.70 | 878.88 | 877.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 878.60 | 881.30 | 879.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:00:00 | 878.60 | 881.30 | 879.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 879.35 | 880.91 | 879.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:15:00 | 877.65 | 880.91 | 879.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 878.85 | 880.50 | 879.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:45:00 | 877.90 | 880.50 | 879.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 880.65 | 880.53 | 879.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 888.00 | 880.57 | 879.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 11:15:00 | 881.30 | 886.46 | 886.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 11:15:00 | 881.30 | 886.46 | 886.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 14:15:00 | 875.55 | 882.73 | 884.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 11:15:00 | 881.75 | 879.80 | 882.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 12:00:00 | 881.75 | 879.80 | 882.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 878.15 | 879.47 | 882.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:45:00 | 876.65 | 878.97 | 881.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 10:30:00 | 877.45 | 878.91 | 880.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 11:15:00 | 876.30 | 878.91 | 880.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 12:30:00 | 877.35 | 878.00 | 879.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 880.10 | 877.93 | 879.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 15:00:00 | 880.10 | 877.93 | 879.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 879.90 | 878.32 | 879.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:15:00 | 882.70 | 878.32 | 879.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 884.05 | 879.47 | 880.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-16 09:15:00 | 884.05 | 879.47 | 880.00 | SL hit (close>static) qty=1.00 sl=883.75 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 792.00 | 791.75 | 791.73 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 13:15:00 | 790.35 | 791.47 | 791.60 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 15:15:00 | 793.40 | 791.94 | 791.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 795.90 | 792.73 | 792.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 10:15:00 | 792.00 | 792.58 | 792.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 10:15:00 | 792.00 | 792.58 | 792.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 792.00 | 792.58 | 792.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:00:00 | 792.00 | 792.58 | 792.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 800.20 | 794.11 | 792.89 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 14:15:00 | 785.95 | 791.99 | 792.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 10:15:00 | 783.80 | 787.46 | 789.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 807.10 | 786.13 | 787.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 807.10 | 786.13 | 787.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 807.10 | 786.13 | 787.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 807.10 | 786.13 | 787.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 809.00 | 790.71 | 789.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 11:15:00 | 810.50 | 794.66 | 791.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 13:15:00 | 823.60 | 824.50 | 817.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 13:30:00 | 823.60 | 824.50 | 817.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 818.90 | 822.96 | 819.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 818.90 | 822.96 | 819.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 816.40 | 821.65 | 818.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:15:00 | 817.65 | 821.65 | 818.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 816.15 | 820.55 | 818.72 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 811.95 | 816.99 | 817.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 10:15:00 | 806.00 | 814.79 | 816.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 12:15:00 | 803.70 | 802.05 | 807.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 13:00:00 | 803.70 | 802.05 | 807.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 805.00 | 803.71 | 806.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 792.65 | 803.71 | 806.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 09:15:00 | 753.02 | 760.01 | 763.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-05 13:15:00 | 757.00 | 754.70 | 759.32 | SL hit (close>ema200) qty=0.50 sl=754.70 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 771.85 | 761.30 | 761.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 12:15:00 | 773.55 | 765.65 | 763.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 14:15:00 | 813.60 | 813.75 | 803.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 15:00:00 | 813.60 | 813.75 | 803.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 835.00 | 836.22 | 831.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 10:45:00 | 840.15 | 833.77 | 832.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 868.00 | 835.59 | 833.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-22 10:15:00 | 924.17 | 882.00 | 862.74 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 15:15:00 | 891.00 | 901.87 | 903.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 880.00 | 892.85 | 897.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 14:15:00 | 871.70 | 869.30 | 876.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 15:00:00 | 871.70 | 869.30 | 876.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 870.50 | 869.54 | 875.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 883.90 | 869.54 | 875.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 878.05 | 871.24 | 875.93 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 895.60 | 880.75 | 878.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 901.00 | 887.29 | 882.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 13:15:00 | 925.80 | 926.00 | 916.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 15:15:00 | 918.15 | 923.20 | 917.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 918.15 | 923.20 | 917.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 916.75 | 923.20 | 917.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 916.05 | 921.77 | 916.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 10:30:00 | 924.35 | 920.04 | 916.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 11:15:00 | 912.95 | 918.62 | 916.27 | SL hit (close<static) qty=1.00 sl=913.90 alert=retest2 |

### Cycle 28 — SELL (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 10:15:00 | 922.30 | 927.68 | 927.89 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 933.45 | 929.10 | 928.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 14:15:00 | 940.80 | 932.10 | 930.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 15:15:00 | 945.00 | 945.38 | 939.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 09:15:00 | 944.10 | 945.38 | 939.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 934.85 | 943.27 | 939.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 939.00 | 943.27 | 939.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 940.65 | 942.75 | 939.49 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 14:15:00 | 929.35 | 936.71 | 937.42 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 941.65 | 936.64 | 936.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 13:15:00 | 951.10 | 942.27 | 939.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 09:15:00 | 945.25 | 945.65 | 941.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 10:00:00 | 945.25 | 945.65 | 941.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 948.55 | 946.23 | 942.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:30:00 | 944.85 | 946.23 | 942.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 941.50 | 946.04 | 943.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 941.60 | 946.04 | 943.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 944.30 | 945.69 | 943.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:30:00 | 940.25 | 945.69 | 943.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 945.25 | 945.36 | 943.72 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 09:15:00 | 931.65 | 942.52 | 943.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 10:15:00 | 925.20 | 939.06 | 941.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 933.40 | 929.03 | 934.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 933.40 | 929.03 | 934.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 933.40 | 929.03 | 934.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 933.40 | 929.03 | 934.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 968.90 | 937.00 | 937.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 968.90 | 937.00 | 937.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 959.55 | 941.51 | 939.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 12:15:00 | 990.15 | 971.91 | 962.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 11:15:00 | 993.00 | 995.57 | 986.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 11:15:00 | 993.00 | 995.57 | 986.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 993.00 | 995.57 | 986.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:45:00 | 989.00 | 995.57 | 986.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 987.20 | 993.64 | 987.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:45:00 | 983.70 | 993.64 | 987.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 982.75 | 991.47 | 987.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 981.45 | 989.46 | 986.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 981.85 | 987.94 | 986.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 977.10 | 987.94 | 986.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 968.05 | 983.96 | 984.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 948.40 | 969.40 | 976.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 967.40 | 966.64 | 973.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:00:00 | 967.40 | 966.64 | 973.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 961.35 | 965.59 | 972.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 965.55 | 965.59 | 972.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 968.35 | 965.42 | 970.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:45:00 | 970.45 | 965.42 | 970.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 968.15 | 965.42 | 968.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 13:30:00 | 965.35 | 965.42 | 968.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 956.80 | 963.70 | 967.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:30:00 | 953.00 | 959.96 | 965.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 13:15:00 | 981.40 | 965.09 | 965.95 | SL hit (close>static) qty=1.00 sl=971.35 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 990.40 | 970.15 | 968.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 1011.60 | 981.62 | 973.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 12:15:00 | 1024.50 | 1025.88 | 1013.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 13:00:00 | 1024.50 | 1025.88 | 1013.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 1015.65 | 1022.69 | 1016.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:45:00 | 1014.25 | 1022.69 | 1016.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 1019.15 | 1021.98 | 1016.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 12:30:00 | 1025.85 | 1021.49 | 1016.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 14:15:00 | 1020.90 | 1020.86 | 1017.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 15:15:00 | 1024.00 | 1020.68 | 1017.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 10:15:00 | 1005.35 | 1022.17 | 1021.61 | SL hit (close<static) qty=1.00 sl=1012.80 alert=retest2 |

### Cycle 36 — SELL (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 11:15:00 | 1010.20 | 1019.78 | 1020.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 15:15:00 | 1003.45 | 1012.16 | 1016.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 14:15:00 | 972.70 | 972.58 | 978.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 15:00:00 | 972.70 | 972.58 | 978.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 996.55 | 977.82 | 980.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 996.55 | 977.82 | 980.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 990.50 | 980.36 | 981.20 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 991.75 | 982.64 | 982.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 09:15:00 | 995.50 | 986.94 | 985.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 11:15:00 | 992.80 | 996.79 | 994.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 11:15:00 | 992.80 | 996.79 | 994.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 992.80 | 996.79 | 994.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:00:00 | 992.80 | 996.79 | 994.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 987.45 | 994.92 | 993.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:00:00 | 987.45 | 994.92 | 993.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 13:15:00 | 979.55 | 991.85 | 992.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 15:15:00 | 975.45 | 986.59 | 989.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 12:15:00 | 975.20 | 973.62 | 978.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 13:00:00 | 975.20 | 973.62 | 978.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 978.15 | 974.53 | 978.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:00:00 | 978.15 | 974.53 | 978.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 973.00 | 974.22 | 978.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 965.60 | 973.38 | 977.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 11:45:00 | 965.10 | 971.11 | 973.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 12:45:00 | 966.25 | 966.69 | 968.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 15:15:00 | 975.60 | 970.42 | 970.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 975.60 | 970.42 | 970.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 997.50 | 975.83 | 972.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 14:15:00 | 982.40 | 984.02 | 978.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 15:00:00 | 982.40 | 984.02 | 978.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 981.00 | 983.42 | 978.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 990.95 | 983.42 | 978.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 13:15:00 | 996.70 | 1004.09 | 1004.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 996.70 | 1004.09 | 1004.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 15:15:00 | 992.95 | 1000.47 | 1002.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 984.50 | 983.70 | 990.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:00:00 | 984.50 | 983.70 | 990.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 988.70 | 980.67 | 983.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 988.70 | 980.67 | 983.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 981.90 | 980.92 | 983.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 995.90 | 980.92 | 983.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 996.20 | 983.97 | 984.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:15:00 | 996.50 | 983.97 | 984.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 1001.40 | 987.46 | 986.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 1007.90 | 997.32 | 991.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 11:15:00 | 1000.40 | 1002.83 | 996.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 11:45:00 | 1000.55 | 1002.83 | 996.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 996.65 | 1001.59 | 996.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 996.65 | 1001.59 | 996.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 996.30 | 1000.53 | 996.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:30:00 | 995.55 | 1000.53 | 996.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 995.10 | 999.45 | 996.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 993.50 | 999.45 | 996.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 995.00 | 998.56 | 996.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 1002.70 | 998.56 | 996.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 14:45:00 | 996.40 | 998.16 | 997.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 15:15:00 | 1000.00 | 1003.99 | 1004.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 15:15:00 | 1000.00 | 1003.99 | 1004.32 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 10:15:00 | 1006.00 | 1004.53 | 1004.52 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 11:15:00 | 1002.10 | 1004.04 | 1004.30 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 12:15:00 | 1008.85 | 1005.01 | 1004.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 13:15:00 | 1018.35 | 1007.67 | 1005.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 15:15:00 | 1054.70 | 1055.87 | 1045.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 09:15:00 | 1050.40 | 1055.87 | 1045.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1045.10 | 1053.72 | 1045.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:45:00 | 1049.00 | 1053.72 | 1045.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1052.00 | 1053.37 | 1046.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 12:15:00 | 1054.40 | 1052.98 | 1046.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:45:00 | 1052.90 | 1049.19 | 1046.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 13:15:00 | 1042.40 | 1046.60 | 1046.29 | SL hit (close<static) qty=1.00 sl=1042.50 alert=retest2 |

### Cycle 46 — SELL (started 2026-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 15:15:00 | 1043.40 | 1045.69 | 1045.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 09:15:00 | 1040.20 | 1044.59 | 1045.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 955.00 | 952.52 | 971.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 945.40 | 952.52 | 971.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 918.90 | 911.37 | 917.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 12:00:00 | 908.50 | 911.12 | 916.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 11:15:00 | 920.00 | 908.83 | 907.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 920.00 | 908.83 | 907.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 922.90 | 914.09 | 910.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 908.60 | 914.97 | 911.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 908.60 | 914.97 | 911.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 908.60 | 914.97 | 911.49 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 891.60 | 907.70 | 908.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 866.80 | 899.52 | 904.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 847.90 | 842.87 | 863.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 14:45:00 | 846.50 | 842.87 | 863.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 863.00 | 848.89 | 859.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:00:00 | 863.00 | 848.89 | 859.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 871.10 | 853.34 | 860.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 13:00:00 | 871.10 | 853.34 | 860.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 878.50 | 865.37 | 865.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 12:15:00 | 886.60 | 875.15 | 870.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 12:15:00 | 886.10 | 895.06 | 885.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 13:00:00 | 886.10 | 895.06 | 885.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 893.50 | 894.75 | 886.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 15:00:00 | 896.70 | 895.14 | 887.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 10:00:00 | 896.15 | 895.17 | 888.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 11:15:00 | 876.45 | 892.95 | 889.00 | SL hit (close<static) qty=1.00 sl=883.20 alert=retest2 |

### Cycle 50 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 865.75 | 884.79 | 885.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 843.40 | 876.51 | 881.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 870.30 | 866.98 | 874.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:00:00 | 870.30 | 866.98 | 874.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 872.15 | 868.01 | 873.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:45:00 | 871.50 | 868.01 | 873.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 886.40 | 871.69 | 875.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 886.40 | 871.69 | 875.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 882.90 | 873.93 | 875.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 940.25 | 873.93 | 875.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 944.40 | 888.02 | 881.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 978.00 | 916.87 | 896.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 15:15:00 | 1025.20 | 1025.89 | 1011.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 09:15:00 | 1024.30 | 1025.89 | 1011.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 1023.05 | 1024.77 | 1013.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:30:00 | 1015.65 | 1024.77 | 1013.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 1016.20 | 1022.61 | 1015.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:00:00 | 1016.20 | 1022.61 | 1015.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 1016.05 | 1021.30 | 1015.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:30:00 | 1016.25 | 1021.30 | 1015.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 1014.00 | 1019.84 | 1015.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:15:00 | 1018.40 | 1019.84 | 1015.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1025.50 | 1020.97 | 1016.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:00:00 | 1029.55 | 1022.69 | 1017.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 14:00:00 | 1033.30 | 1027.93 | 1024.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 984.40 | 1017.64 | 1020.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 984.40 | 1017.64 | 1020.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 10:15:00 | 982.85 | 1010.68 | 1017.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 11:15:00 | 998.80 | 993.69 | 1002.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 12:00:00 | 998.80 | 993.69 | 1002.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 999.10 | 994.77 | 1002.17 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 1018.25 | 1006.91 | 1005.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 1029.10 | 1011.34 | 1007.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 1022.45 | 1027.20 | 1021.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 1022.45 | 1027.20 | 1021.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 1022.45 | 1027.20 | 1021.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 1022.45 | 1027.20 | 1021.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 1009.15 | 1023.59 | 1020.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 1009.15 | 1023.59 | 1020.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1010.60 | 1020.99 | 1019.26 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 1006.85 | 1016.51 | 1017.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 1005.00 | 1014.21 | 1016.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 1007.75 | 1000.35 | 1005.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 1007.75 | 1000.35 | 1005.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1007.75 | 1000.35 | 1005.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:00:00 | 1007.75 | 1000.35 | 1005.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1003.35 | 1000.95 | 1005.40 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2026-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 13:15:00 | 1012.55 | 1005.35 | 1005.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 14:15:00 | 1034.55 | 1011.19 | 1007.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 11:15:00 | 1019.95 | 1022.26 | 1015.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 12:00:00 | 1019.95 | 1022.26 | 1015.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 1009.85 | 1019.78 | 1014.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 1009.85 | 1019.78 | 1014.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 1010.65 | 1017.95 | 1014.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 15:15:00 | 1021.70 | 1016.45 | 1013.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 10:30:00 | 1021.50 | 1016.10 | 1014.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 14:15:00 | 1019.50 | 1015.85 | 1014.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 09:15:00 | 1027.35 | 1017.64 | 1015.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1027.10 | 1019.53 | 1016.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 10:15:00 | 1032.05 | 1019.53 | 1016.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 12:45:00 | 1031.05 | 1024.65 | 1020.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 14:15:00 | 1012.25 | 1023.56 | 1020.50 | SL hit (close<static) qty=1.00 sl=1015.05 alert=retest2 |

### Cycle 56 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 971.80 | 1010.88 | 1015.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 958.70 | 988.89 | 1003.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 976.80 | 966.38 | 977.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 976.80 | 966.38 | 977.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 976.80 | 966.38 | 977.85 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 998.30 | 980.71 | 979.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 13:15:00 | 1002.40 | 990.36 | 985.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 971.10 | 987.70 | 985.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 971.10 | 987.70 | 985.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 971.10 | 987.70 | 985.35 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 964.60 | 980.65 | 982.40 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 1001.20 | 984.05 | 982.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 11:15:00 | 1015.20 | 990.28 | 985.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 1004.70 | 1006.77 | 997.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 10:45:00 | 1003.60 | 1006.77 | 997.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 1001.60 | 1005.73 | 998.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 12:00:00 | 1001.60 | 1005.73 | 998.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 1000.00 | 1004.59 | 998.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 12:30:00 | 1000.80 | 1004.59 | 998.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 997.40 | 1003.15 | 998.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:00:00 | 997.40 | 1003.15 | 998.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 990.70 | 1000.66 | 997.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 990.70 | 1000.66 | 997.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 991.00 | 998.73 | 996.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 984.50 | 998.73 | 996.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 991.80 | 997.34 | 996.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:15:00 | 997.50 | 997.34 | 996.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 11:15:00 | 998.60 | 1000.18 | 999.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 12:45:00 | 998.40 | 1000.03 | 999.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 13:15:00 | 997.00 | 999.43 | 999.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 13:15:00 | 997.00 | 999.43 | 999.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 987.90 | 995.30 | 997.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 996.10 | 988.83 | 992.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 996.10 | 988.83 | 992.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 996.10 | 988.83 | 992.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 1001.40 | 988.83 | 992.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 993.00 | 989.66 | 992.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 985.80 | 989.66 | 992.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 985.60 | 988.85 | 991.94 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 1000.00 | 991.36 | 991.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 1017.40 | 996.57 | 993.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1014.00 | 1020.62 | 1010.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1014.00 | 1020.62 | 1010.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1014.00 | 1020.62 | 1010.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 1038.20 | 1012.08 | 1009.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 15:15:00 | 1008.60 | 1013.73 | 1012.50 | SL hit (close<static) qty=1.00 sl=1009.20 alert=retest2 |

### Cycle 62 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 966.20 | 1004.22 | 1008.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 947.80 | 992.94 | 1002.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 966.40 | 955.25 | 970.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 966.10 | 955.25 | 970.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 969.80 | 958.16 | 970.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 970.50 | 958.16 | 970.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 968.00 | 960.13 | 970.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:30:00 | 965.60 | 960.13 | 970.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 968.00 | 961.70 | 970.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 994.70 | 961.70 | 970.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 986.50 | 966.66 | 971.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 990.70 | 966.66 | 971.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 983.40 | 970.01 | 972.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 11:15:00 | 978.90 | 970.01 | 972.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 12:45:00 | 976.60 | 973.54 | 974.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 13:15:00 | 992.00 | 977.23 | 975.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 992.00 | 977.23 | 975.72 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 963.20 | 973.54 | 974.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 955.10 | 967.45 | 971.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 974.05 | 949.23 | 956.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 974.05 | 949.23 | 956.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 974.05 | 949.23 | 956.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 974.05 | 949.23 | 956.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 973.50 | 954.09 | 957.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:45:00 | 980.40 | 954.09 | 957.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 970.20 | 960.78 | 960.46 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 15:15:00 | 956.10 | 960.01 | 960.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 908.95 | 949.80 | 955.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 14:15:00 | 943.95 | 934.74 | 944.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 14:15:00 | 943.95 | 934.74 | 944.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 943.95 | 934.74 | 944.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 15:00:00 | 943.95 | 934.74 | 944.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 945.00 | 936.79 | 944.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:15:00 | 943.65 | 936.79 | 944.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 962.10 | 941.85 | 945.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:00:00 | 962.10 | 941.85 | 945.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 969.00 | 947.28 | 947.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:45:00 | 971.75 | 947.28 | 947.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 961.60 | 950.15 | 949.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 982.05 | 956.53 | 952.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 12:15:00 | 980.75 | 980.92 | 969.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 12:45:00 | 977.05 | 980.92 | 969.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 1320.30 | 1352.46 | 1329.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:30:00 | 1294.70 | 1352.46 | 1329.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 1317.00 | 1345.37 | 1328.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 13:45:00 | 1333.50 | 1338.27 | 1328.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 1351.15 | 1409.71 | 1410.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 1351.15 | 1409.71 | 1410.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 12:15:00 | 1323.30 | 1376.63 | 1393.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 1371.50 | 1359.86 | 1378.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 09:45:00 | 1369.00 | 1359.86 | 1378.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 1401.90 | 1368.27 | 1380.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:00:00 | 1401.90 | 1368.27 | 1380.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 1395.60 | 1373.74 | 1381.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:30:00 | 1412.50 | 1373.74 | 1381.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 1397.70 | 1382.67 | 1384.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 14:15:00 | 1400.30 | 1382.67 | 1384.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-05-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 14:15:00 | 1401.20 | 1386.37 | 1386.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 13:15:00 | 1410.80 | 1396.26 | 1391.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 09:15:00 | 1397.50 | 1400.52 | 1395.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 1397.50 | 1400.52 | 1395.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 1397.50 | 1400.52 | 1395.08 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2026-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 12:15:00 | 1386.30 | 1395.40 | 1396.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 09:15:00 | 1383.70 | 1390.05 | 1393.38 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 10:45:00 | 896.85 | 2025-05-19 13:15:00 | 892.00 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-05-15 14:00:00 | 897.30 | 2025-05-19 13:15:00 | 892.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-05-19 12:15:00 | 898.00 | 2025-05-19 13:15:00 | 892.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-06-11 15:15:00 | 909.00 | 2025-06-12 13:15:00 | 881.95 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2025-06-20 12:15:00 | 829.45 | 2025-06-23 11:15:00 | 841.25 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-06-20 15:00:00 | 826.10 | 2025-06-23 11:15:00 | 841.25 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-06-23 09:15:00 | 830.05 | 2025-06-23 11:15:00 | 841.25 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-07-02 09:15:00 | 885.80 | 2025-07-02 09:15:00 | 870.60 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-07-09 09:15:00 | 888.00 | 2025-07-11 11:15:00 | 881.30 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-07-14 13:45:00 | 876.65 | 2025-07-16 09:15:00 | 884.05 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-07-15 10:30:00 | 877.45 | 2025-07-16 09:15:00 | 884.05 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-07-15 11:15:00 | 876.30 | 2025-07-16 09:15:00 | 884.05 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-07-15 12:30:00 | 877.35 | 2025-07-16 09:15:00 | 884.05 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-07-16 12:15:00 | 878.45 | 2025-07-25 10:15:00 | 834.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-16 15:00:00 | 877.50 | 2025-07-25 10:15:00 | 833.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-17 09:30:00 | 875.60 | 2025-07-25 10:15:00 | 831.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-17 13:00:00 | 876.85 | 2025-07-25 10:15:00 | 833.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 10:15:00 | 871.90 | 2025-07-25 11:15:00 | 828.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 12:30:00 | 872.85 | 2025-07-25 11:15:00 | 829.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 13:45:00 | 872.90 | 2025-07-25 11:15:00 | 829.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 14:45:00 | 872.45 | 2025-07-25 11:15:00 | 828.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 13:00:00 | 869.05 | 2025-07-25 11:15:00 | 825.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 09:45:00 | 868.70 | 2025-07-25 11:15:00 | 825.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-16 12:15:00 | 878.45 | 2025-07-29 12:15:00 | 817.75 | STOP_HIT | 0.50 | 6.91% |
| SELL | retest2 | 2025-07-16 15:00:00 | 877.50 | 2025-07-29 12:15:00 | 817.75 | STOP_HIT | 0.50 | 6.81% |
| SELL | retest2 | 2025-07-17 09:30:00 | 875.60 | 2025-07-29 12:15:00 | 817.75 | STOP_HIT | 0.50 | 6.61% |
| SELL | retest2 | 2025-07-17 13:00:00 | 876.85 | 2025-07-29 12:15:00 | 817.75 | STOP_HIT | 0.50 | 6.74% |
| SELL | retest2 | 2025-07-18 10:15:00 | 871.90 | 2025-07-29 12:15:00 | 817.75 | STOP_HIT | 0.50 | 6.21% |
| SELL | retest2 | 2025-07-18 12:30:00 | 872.85 | 2025-07-29 12:15:00 | 817.75 | STOP_HIT | 0.50 | 6.31% |
| SELL | retest2 | 2025-07-18 13:45:00 | 872.90 | 2025-07-29 12:15:00 | 817.75 | STOP_HIT | 0.50 | 6.32% |
| SELL | retest2 | 2025-07-18 14:45:00 | 872.45 | 2025-07-29 12:15:00 | 817.75 | STOP_HIT | 0.50 | 6.27% |
| SELL | retest2 | 2025-07-22 13:00:00 | 869.05 | 2025-07-29 12:15:00 | 817.75 | STOP_HIT | 0.50 | 5.90% |
| SELL | retest2 | 2025-07-23 09:45:00 | 868.70 | 2025-07-29 12:15:00 | 817.75 | STOP_HIT | 0.50 | 5.87% |
| SELL | retest2 | 2025-08-26 09:15:00 | 792.65 | 2025-09-05 09:15:00 | 753.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 09:15:00 | 792.65 | 2025-09-05 13:15:00 | 757.00 | STOP_HIT | 0.50 | 4.50% |
| BUY | retest2 | 2025-09-18 10:45:00 | 840.15 | 2025-09-22 10:15:00 | 924.17 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-19 09:15:00 | 868.00 | 2025-09-22 13:15:00 | 954.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-08 10:30:00 | 924.35 | 2025-10-08 11:15:00 | 912.95 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-10-09 09:15:00 | 933.00 | 2025-10-15 10:15:00 | 922.30 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-10-10 13:30:00 | 926.90 | 2025-10-15 10:15:00 | 922.30 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-10-14 12:15:00 | 926.35 | 2025-10-15 10:15:00 | 922.30 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-10-14 13:30:00 | 930.40 | 2025-10-15 10:15:00 | 922.30 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-11-11 09:30:00 | 953.00 | 2025-11-11 13:15:00 | 981.40 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2025-11-17 12:30:00 | 1025.85 | 2025-11-19 10:15:00 | 1005.35 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-11-17 14:15:00 | 1020.90 | 2025-11-19 10:15:00 | 1005.35 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-11-17 15:15:00 | 1024.00 | 2025-11-19 10:15:00 | 1005.35 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-12-05 09:15:00 | 965.60 | 2025-12-09 15:15:00 | 975.60 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-12-08 11:45:00 | 965.10 | 2025-12-09 15:15:00 | 975.60 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-12-09 12:45:00 | 966.25 | 2025-12-09 15:15:00 | 975.60 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-12-11 09:15:00 | 990.95 | 2025-12-16 13:15:00 | 996.70 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2025-12-24 09:15:00 | 1002.70 | 2025-12-29 15:15:00 | 1000.00 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-12-24 14:45:00 | 996.40 | 2025-12-29 15:15:00 | 1000.00 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2026-01-05 12:15:00 | 1054.40 | 2026-01-06 13:15:00 | 1042.40 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-01-06 09:45:00 | 1052.90 | 2026-01-06 13:15:00 | 1042.40 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-01-20 12:00:00 | 908.50 | 2026-01-22 11:15:00 | 920.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2026-01-30 15:00:00 | 896.70 | 2026-02-01 11:15:00 | 876.45 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2026-02-01 10:00:00 | 896.15 | 2026-02-01 11:15:00 | 876.45 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2026-02-11 11:00:00 | 1029.55 | 2026-02-13 09:15:00 | 984.40 | STOP_HIT | 1.00 | -4.39% |
| BUY | retest2 | 2026-02-12 14:00:00 | 1033.30 | 2026-02-13 09:15:00 | 984.40 | STOP_HIT | 1.00 | -4.73% |
| BUY | retest2 | 2026-02-25 15:15:00 | 1021.70 | 2026-02-27 14:15:00 | 1012.25 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-02-26 10:30:00 | 1021.50 | 2026-02-27 14:15:00 | 1012.25 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-02-26 14:15:00 | 1019.50 | 2026-03-02 09:15:00 | 971.80 | STOP_HIT | 1.00 | -4.68% |
| BUY | retest2 | 2026-02-27 09:15:00 | 1027.35 | 2026-03-02 09:15:00 | 971.80 | STOP_HIT | 1.00 | -5.41% |
| BUY | retest2 | 2026-02-27 10:15:00 | 1032.05 | 2026-03-02 09:15:00 | 971.80 | STOP_HIT | 1.00 | -5.84% |
| BUY | retest2 | 2026-02-27 12:45:00 | 1031.05 | 2026-03-02 09:15:00 | 971.80 | STOP_HIT | 1.00 | -5.75% |
| BUY | retest2 | 2026-03-12 10:15:00 | 997.50 | 2026-03-13 13:15:00 | 997.00 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2026-03-13 11:15:00 | 998.60 | 2026-03-13 13:15:00 | 997.00 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2026-03-13 12:45:00 | 998.40 | 2026-03-13 13:15:00 | 997.00 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2026-03-20 09:15:00 | 1038.20 | 2026-03-20 15:15:00 | 1008.60 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2026-03-25 11:15:00 | 978.90 | 2026-03-25 13:15:00 | 992.00 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-03-25 12:45:00 | 976.60 | 2026-03-25 13:15:00 | 992.00 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-04-24 13:45:00 | 1333.50 | 2026-04-30 09:15:00 | 1351.15 | STOP_HIT | 1.00 | 1.32% |
