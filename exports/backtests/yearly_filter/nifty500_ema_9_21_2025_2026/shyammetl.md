# Shyam Metalics and Energy Ltd. (SHYAMMETL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 905.65
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 81 |
| ALERT1 | 55 |
| ALERT2 | 55 |
| ALERT2_SKIP | 34 |
| ALERT3 | 147 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 92 |
| PARTIAL | 5 |
| TARGET_HIT | 0 |
| STOP_HIT | 92 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 97 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 66
- **Target hits / Stop hits / Partials:** 0 / 92 / 5
- **Avg / median % per leg:** -0.18% / -0.69%
- **Sum % (uncompounded):** -17.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 35 | 11 | 31.4% | 0 | 35 | 0 | 0.07% | 2.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 35 | 11 | 31.4% | 0 | 35 | 0 | 0.07% | 2.4% |
| SELL (all) | 62 | 20 | 32.3% | 0 | 57 | 5 | -0.32% | -19.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 62 | 20 | 32.3% | 0 | 57 | 5 | -0.32% | -19.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 97 | 31 | 32.0% | 0 | 92 | 5 | -0.18% | -17.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 886.65 | 875.66 | 875.40 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-12 12:15:00 | 870.85 | 874.82 | 875.07 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 880.00 | 875.86 | 875.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 894.85 | 879.66 | 877.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 10:15:00 | 902.00 | 902.40 | 894.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 11:00:00 | 902.00 | 902.40 | 894.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 895.75 | 901.07 | 894.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:30:00 | 893.10 | 901.07 | 894.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 898.55 | 900.56 | 894.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 893.25 | 900.56 | 894.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 897.00 | 928.43 | 924.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 903.80 | 928.43 | 924.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 903.05 | 923.35 | 922.19 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 11:15:00 | 907.85 | 920.25 | 920.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 894.90 | 907.33 | 911.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 905.95 | 905.15 | 909.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 14:30:00 | 906.40 | 905.15 | 909.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 910.00 | 906.12 | 909.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:15:00 | 899.35 | 906.12 | 909.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 905.15 | 905.92 | 908.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 14:15:00 | 897.20 | 903.90 | 906.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 887.45 | 904.34 | 905.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 09:15:00 | 852.34 | 866.54 | 870.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 11:15:00 | 843.08 | 858.05 | 865.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 15:15:00 | 850.65 | 848.46 | 857.89 | SL hit (close>ema200) qty=0.50 sl=848.46 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 13:15:00 | 870.35 | 862.91 | 862.44 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 858.70 | 863.30 | 863.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 15:15:00 | 856.75 | 861.99 | 862.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 13:15:00 | 857.70 | 856.19 | 859.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 13:15:00 | 857.70 | 856.19 | 859.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 857.70 | 856.19 | 859.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 15:15:00 | 848.95 | 855.88 | 858.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 09:15:00 | 861.00 | 855.79 | 858.14 | SL hit (close>static) qty=1.00 sl=859.75 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 13:15:00 | 868.00 | 859.65 | 859.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 09:15:00 | 872.40 | 864.03 | 861.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 14:15:00 | 866.20 | 868.69 | 865.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 15:00:00 | 866.20 | 868.69 | 865.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 872.00 | 869.35 | 865.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:45:00 | 874.60 | 870.75 | 866.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 12:00:00 | 875.45 | 872.69 | 868.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 14:45:00 | 876.90 | 874.34 | 870.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 11:15:00 | 865.05 | 873.79 | 873.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 11:15:00 | 865.05 | 873.79 | 873.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 12:15:00 | 861.35 | 871.31 | 872.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 841.50 | 838.55 | 846.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 10:15:00 | 841.50 | 838.55 | 846.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 841.50 | 838.55 | 846.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:45:00 | 844.70 | 838.55 | 846.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 845.85 | 839.54 | 843.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 846.35 | 839.54 | 843.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 843.70 | 840.37 | 843.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:15:00 | 848.80 | 840.37 | 843.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 841.00 | 840.50 | 843.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:15:00 | 839.60 | 840.50 | 843.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 14:30:00 | 840.10 | 839.79 | 842.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 14:15:00 | 831.95 | 826.56 | 826.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 14:15:00 | 831.95 | 826.56 | 826.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 839.15 | 830.36 | 828.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 833.35 | 834.58 | 831.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:45:00 | 834.55 | 834.58 | 831.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 832.80 | 834.22 | 831.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 832.80 | 834.22 | 831.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 841.45 | 835.39 | 832.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 11:30:00 | 850.10 | 838.35 | 834.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 845.50 | 840.50 | 836.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 12:00:00 | 846.55 | 841.80 | 838.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 09:15:00 | 869.95 | 877.53 | 877.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 869.95 | 877.53 | 877.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 10:15:00 | 866.00 | 875.22 | 876.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 14:15:00 | 872.95 | 870.96 | 873.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 14:15:00 | 872.95 | 870.96 | 873.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 872.95 | 870.96 | 873.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 15:00:00 | 872.95 | 870.96 | 873.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 873.00 | 871.37 | 873.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:30:00 | 866.40 | 870.59 | 873.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 15:15:00 | 867.10 | 867.37 | 870.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:45:00 | 866.50 | 866.90 | 869.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 864.05 | 854.23 | 853.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 864.05 | 854.23 | 853.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 867.90 | 856.96 | 855.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 14:15:00 | 909.45 | 913.14 | 903.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 15:00:00 | 909.45 | 913.14 | 903.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 905.00 | 911.51 | 903.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 09:15:00 | 921.55 | 911.51 | 903.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 15:15:00 | 975.90 | 980.00 | 980.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 15:15:00 | 975.90 | 980.00 | 980.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 971.25 | 978.25 | 979.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 10:15:00 | 980.00 | 978.60 | 979.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 10:15:00 | 980.00 | 978.60 | 979.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 980.00 | 978.60 | 979.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:45:00 | 980.50 | 978.60 | 979.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 983.60 | 979.60 | 980.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 11:45:00 | 984.70 | 979.60 | 980.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 981.00 | 979.88 | 980.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:45:00 | 985.00 | 979.88 | 980.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 980.10 | 979.92 | 980.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:30:00 | 980.40 | 979.92 | 980.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 14:15:00 | 982.10 | 980.36 | 980.30 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 968.65 | 978.48 | 979.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 10:15:00 | 960.35 | 974.85 | 977.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 15:15:00 | 959.00 | 958.66 | 967.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 12:15:00 | 953.60 | 956.49 | 963.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 953.60 | 956.49 | 963.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 12:30:00 | 956.20 | 956.49 | 963.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 965.75 | 958.08 | 962.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 965.75 | 958.08 | 962.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 963.85 | 959.23 | 963.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 953.65 | 959.23 | 963.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 12:15:00 | 960.15 | 960.71 | 962.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 09:30:00 | 958.70 | 955.16 | 958.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:30:00 | 952.20 | 955.19 | 958.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 951.80 | 952.08 | 955.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 09:30:00 | 947.25 | 951.00 | 954.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 13:15:00 | 948.20 | 950.06 | 953.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 14:00:00 | 948.00 | 949.65 | 952.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 15:00:00 | 948.05 | 949.33 | 952.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 966.55 | 952.59 | 953.36 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 966.55 | 952.59 | 953.36 | SL hit (close>static) qty=1.00 sl=956.80 alert=retest2 |

### Cycle 15 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 10:15:00 | 970.25 | 956.12 | 954.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 975.60 | 961.34 | 959.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 14:15:00 | 966.55 | 966.56 | 962.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 14:15:00 | 966.55 | 966.56 | 962.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 966.55 | 966.56 | 962.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 14:30:00 | 965.80 | 966.56 | 962.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 962.00 | 965.22 | 962.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 961.45 | 965.22 | 962.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 957.00 | 963.57 | 962.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 957.00 | 963.57 | 962.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 955.35 | 961.93 | 961.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:45:00 | 953.10 | 961.93 | 961.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 12:15:00 | 952.10 | 959.96 | 960.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 15:15:00 | 947.95 | 955.17 | 958.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 11:15:00 | 954.00 | 953.57 | 956.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 11:45:00 | 954.45 | 953.57 | 956.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 956.15 | 951.59 | 954.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:45:00 | 940.25 | 949.56 | 952.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 09:30:00 | 941.45 | 947.78 | 950.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 10:30:00 | 942.15 | 946.33 | 949.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 935.55 | 945.93 | 946.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 938.60 | 944.46 | 945.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 10:45:00 | 933.85 | 943.97 | 945.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 921.75 | 941.74 | 943.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 10:15:00 | 931.00 | 912.88 | 912.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 931.00 | 912.88 | 912.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 11:15:00 | 931.90 | 916.69 | 914.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 11:15:00 | 926.10 | 927.77 | 922.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 11:45:00 | 926.10 | 927.77 | 922.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 920.30 | 925.81 | 922.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 920.30 | 925.81 | 922.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 923.00 | 925.25 | 922.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 920.70 | 925.25 | 922.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 920.00 | 924.20 | 922.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 11:00:00 | 929.50 | 925.26 | 923.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 15:15:00 | 928.25 | 925.78 | 924.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:30:00 | 944.90 | 930.36 | 926.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 14:45:00 | 930.10 | 934.09 | 933.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 15:15:00 | 927.95 | 932.86 | 932.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 15:15:00 | 927.95 | 932.86 | 932.92 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 936.40 | 933.57 | 933.23 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 11:15:00 | 930.40 | 932.56 | 932.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 12:15:00 | 926.10 | 931.27 | 932.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 14:15:00 | 918.00 | 915.93 | 920.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 15:00:00 | 918.00 | 915.93 | 920.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 916.10 | 915.51 | 919.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:00:00 | 916.10 | 915.51 | 919.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 927.35 | 917.88 | 919.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:00:00 | 927.35 | 917.88 | 919.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 925.00 | 919.30 | 920.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:30:00 | 926.50 | 919.30 | 920.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 928.25 | 921.09 | 921.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 928.75 | 923.55 | 922.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 10:15:00 | 922.30 | 923.30 | 922.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 10:15:00 | 922.30 | 923.30 | 922.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 922.30 | 923.30 | 922.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 922.30 | 923.30 | 922.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 926.80 | 924.00 | 922.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 13:15:00 | 928.45 | 923.90 | 922.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 15:15:00 | 927.80 | 924.73 | 923.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 915.75 | 923.43 | 923.06 | SL hit (close<static) qty=1.00 sl=922.20 alert=retest2 |

### Cycle 22 — SELL (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 10:15:00 | 916.05 | 921.95 | 922.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 13:15:00 | 913.95 | 918.59 | 920.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 14:15:00 | 915.40 | 914.44 | 916.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-18 14:30:00 | 916.60 | 914.44 | 916.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 913.70 | 914.38 | 916.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:00:00 | 901.70 | 910.52 | 913.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:30:00 | 906.45 | 909.95 | 913.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 909.60 | 910.36 | 913.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 922.90 | 912.87 | 914.09 | SL hit (close>static) qty=1.00 sl=916.95 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 11:15:00 | 922.55 | 915.22 | 914.97 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 909.80 | 915.17 | 915.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 13:15:00 | 903.00 | 909.70 | 912.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 916.05 | 908.69 | 911.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 916.05 | 908.69 | 911.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 916.05 | 908.69 | 911.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 916.05 | 908.69 | 911.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 925.85 | 912.12 | 912.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 925.85 | 912.12 | 912.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 11:15:00 | 925.55 | 914.81 | 913.77 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 905.70 | 915.60 | 916.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 15:15:00 | 904.55 | 909.62 | 912.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 918.45 | 911.38 | 913.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 918.45 | 911.38 | 913.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 918.45 | 911.38 | 913.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 928.25 | 911.38 | 913.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 920.00 | 913.11 | 914.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:30:00 | 922.45 | 913.11 | 914.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 915.65 | 913.52 | 914.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:00:00 | 915.65 | 913.52 | 914.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 911.20 | 913.05 | 913.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 908.00 | 911.98 | 913.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:00:00 | 907.40 | 911.06 | 912.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:30:00 | 909.55 | 910.39 | 912.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 12:15:00 | 909.00 | 910.39 | 912.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 907.35 | 909.78 | 911.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:15:00 | 906.15 | 909.78 | 911.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 14:15:00 | 912.40 | 910.34 | 911.54 | SL hit (close>static) qty=1.00 sl=911.70 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 919.60 | 912.44 | 912.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 12:15:00 | 924.45 | 914.85 | 913.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 12:15:00 | 962.20 | 962.57 | 948.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 12:45:00 | 963.20 | 962.57 | 948.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 950.00 | 959.15 | 951.25 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 13:15:00 | 928.70 | 944.79 | 946.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 14:15:00 | 925.80 | 940.99 | 944.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 937.50 | 934.08 | 937.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 937.50 | 934.08 | 937.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 937.50 | 934.08 | 937.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 939.10 | 934.08 | 937.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 948.20 | 936.91 | 938.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 948.20 | 936.91 | 938.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 948.50 | 939.22 | 939.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:45:00 | 949.05 | 939.22 | 939.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 12:15:00 | 944.45 | 940.27 | 939.92 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 10:15:00 | 929.45 | 938.50 | 939.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 923.35 | 931.85 | 935.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 09:15:00 | 930.50 | 927.39 | 930.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 930.50 | 927.39 | 930.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 930.50 | 927.39 | 930.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:30:00 | 921.55 | 925.40 | 929.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 12:00:00 | 914.90 | 923.30 | 928.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 12:45:00 | 920.00 | 923.23 | 927.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 13:15:00 | 922.05 | 923.23 | 927.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 925.05 | 923.02 | 926.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 15:00:00 | 925.05 | 923.02 | 926.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 925.00 | 923.41 | 926.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 922.65 | 923.41 | 926.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 09:45:00 | 921.20 | 917.16 | 920.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 13:45:00 | 922.45 | 919.74 | 920.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 14:15:00 | 925.20 | 919.66 | 919.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 14:15:00 | 925.20 | 919.66 | 919.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 14:15:00 | 927.00 | 922.87 | 920.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 09:15:00 | 913.30 | 920.96 | 920.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 913.30 | 920.96 | 920.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 913.30 | 920.96 | 920.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 913.30 | 920.96 | 920.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 10:15:00 | 909.25 | 918.62 | 919.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 13:15:00 | 908.40 | 914.36 | 916.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 905.30 | 900.03 | 903.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 905.30 | 900.03 | 903.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 905.30 | 900.03 | 903.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:45:00 | 907.35 | 900.03 | 903.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 907.65 | 901.56 | 903.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:45:00 | 908.70 | 901.56 | 903.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 909.10 | 903.06 | 904.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:45:00 | 911.40 | 903.06 | 904.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 903.95 | 903.62 | 904.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 903.95 | 903.62 | 904.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 903.00 | 903.50 | 904.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 903.80 | 903.50 | 904.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 924.40 | 907.68 | 906.08 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 13:15:00 | 905.00 | 908.49 | 908.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 899.45 | 905.71 | 907.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 10:15:00 | 896.00 | 889.23 | 895.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 10:15:00 | 896.00 | 889.23 | 895.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 896.00 | 889.23 | 895.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:00:00 | 896.00 | 889.23 | 895.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 895.00 | 890.38 | 895.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 13:45:00 | 890.00 | 890.48 | 894.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 14:45:00 | 890.00 | 892.10 | 895.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 15:15:00 | 906.00 | 894.88 | 896.00 | SL hit (close>static) qty=1.00 sl=899.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 10:15:00 | 901.00 | 897.09 | 896.87 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 889.70 | 895.61 | 896.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 12:15:00 | 888.25 | 894.14 | 895.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 870.70 | 856.37 | 868.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 10:15:00 | 870.70 | 856.37 | 868.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 870.70 | 856.37 | 868.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:00:00 | 870.70 | 856.37 | 868.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 864.05 | 857.90 | 868.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 14:00:00 | 863.05 | 861.51 | 868.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 15:00:00 | 862.55 | 861.72 | 867.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 854.10 | 862.37 | 867.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 12:00:00 | 859.15 | 853.47 | 855.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 864.40 | 855.66 | 856.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 12:30:00 | 863.45 | 855.66 | 856.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-12 13:15:00 | 865.05 | 857.54 | 857.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 13:15:00 | 865.05 | 857.54 | 857.09 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 14:15:00 | 852.75 | 858.53 | 858.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 849.20 | 855.87 | 857.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 11:15:00 | 856.60 | 855.64 | 856.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 11:15:00 | 856.60 | 855.64 | 856.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 856.60 | 855.64 | 856.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:45:00 | 856.05 | 855.64 | 856.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 853.35 | 855.18 | 856.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:30:00 | 855.35 | 855.18 | 856.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 854.45 | 855.04 | 856.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:45:00 | 854.50 | 855.04 | 856.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 854.85 | 855.00 | 856.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:00:00 | 854.85 | 855.00 | 856.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 856.00 | 855.20 | 856.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 851.05 | 855.20 | 856.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 859.70 | 856.10 | 856.54 | SL hit (close>static) qty=1.00 sl=858.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 828.55 | 824.23 | 823.89 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 818.55 | 823.43 | 823.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 12:15:00 | 815.50 | 820.84 | 822.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 14:15:00 | 821.35 | 820.30 | 821.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-27 15:00:00 | 821.35 | 820.30 | 821.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 825.65 | 817.45 | 818.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:45:00 | 825.45 | 817.45 | 818.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 825.45 | 819.05 | 819.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:15:00 | 821.95 | 819.05 | 819.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 12:15:00 | 820.95 | 819.74 | 819.61 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 14:15:00 | 818.35 | 819.50 | 819.53 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 15:15:00 | 820.00 | 819.60 | 819.57 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 814.20 | 818.52 | 819.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 13:15:00 | 809.75 | 814.44 | 816.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 15:15:00 | 801.90 | 801.76 | 805.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 09:15:00 | 801.80 | 801.76 | 805.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 792.20 | 799.85 | 804.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 11:45:00 | 789.50 | 796.70 | 801.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 09:15:00 | 784.00 | 800.36 | 800.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 12:30:00 | 788.10 | 795.48 | 797.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 819.65 | 802.23 | 800.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 819.65 | 802.23 | 800.50 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 13:15:00 | 796.70 | 801.21 | 801.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 792.70 | 799.51 | 800.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 797.85 | 797.17 | 799.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 797.85 | 797.17 | 799.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 797.85 | 797.17 | 799.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:30:00 | 795.95 | 797.17 | 799.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 790.90 | 795.92 | 798.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:30:00 | 803.80 | 795.92 | 798.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 796.60 | 794.96 | 796.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 797.80 | 794.96 | 796.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 799.50 | 795.87 | 797.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:30:00 | 801.00 | 795.87 | 797.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 799.65 | 796.63 | 797.32 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 801.25 | 797.97 | 797.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 09:15:00 | 809.85 | 801.56 | 799.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 802.90 | 806.17 | 803.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 802.90 | 806.17 | 803.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 802.90 | 806.17 | 803.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 800.35 | 806.17 | 803.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 805.00 | 805.94 | 803.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:30:00 | 801.75 | 805.94 | 803.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 803.50 | 805.45 | 803.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:30:00 | 805.55 | 805.45 | 803.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 808.45 | 806.05 | 804.11 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 10:15:00 | 802.15 | 804.25 | 804.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 13:15:00 | 800.00 | 803.38 | 804.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 801.60 | 801.56 | 802.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 13:15:00 | 801.60 | 801.56 | 802.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 801.60 | 801.56 | 802.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 801.60 | 801.56 | 802.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 820.05 | 805.25 | 804.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 15:15:00 | 821.70 | 808.54 | 805.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 825.00 | 825.38 | 819.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:15:00 | 833.25 | 825.38 | 819.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 829.85 | 826.43 | 822.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:45:00 | 828.00 | 826.43 | 822.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 825.15 | 825.30 | 822.79 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 818.75 | 822.32 | 822.33 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 09:15:00 | 825.60 | 822.98 | 822.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 10:15:00 | 829.85 | 824.35 | 823.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 12:15:00 | 821.40 | 824.42 | 823.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 12:15:00 | 821.40 | 824.42 | 823.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 821.40 | 824.42 | 823.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 13:00:00 | 821.40 | 824.42 | 823.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 823.70 | 824.27 | 823.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 13:30:00 | 821.30 | 824.27 | 823.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 820.90 | 823.60 | 823.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 820.90 | 823.60 | 823.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 15:15:00 | 819.10 | 822.70 | 822.93 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 10:15:00 | 823.80 | 823.10 | 823.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 12:15:00 | 827.25 | 824.39 | 823.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 11:15:00 | 838.50 | 842.27 | 837.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 11:15:00 | 838.50 | 842.27 | 837.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 838.50 | 842.27 | 837.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:00:00 | 838.50 | 842.27 | 837.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 838.40 | 841.50 | 837.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 14:15:00 | 841.20 | 841.31 | 837.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 15:15:00 | 844.00 | 841.15 | 838.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 11:15:00 | 838.85 | 844.07 | 844.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 838.85 | 844.07 | 844.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 14:15:00 | 832.85 | 840.22 | 842.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 806.50 | 803.40 | 809.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 13:00:00 | 806.50 | 803.40 | 809.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 813.95 | 806.16 | 808.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:00:00 | 813.95 | 806.16 | 808.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 814.00 | 807.73 | 809.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:30:00 | 812.95 | 807.73 | 809.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 09:15:00 | 814.05 | 810.62 | 810.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 10:15:00 | 823.95 | 813.29 | 811.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 11:15:00 | 815.65 | 825.80 | 820.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 11:15:00 | 815.65 | 825.80 | 820.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 815.65 | 825.80 | 820.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:00:00 | 815.65 | 825.80 | 820.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 816.50 | 823.94 | 820.53 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 812.90 | 818.35 | 818.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 807.55 | 816.19 | 817.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 13:15:00 | 816.00 | 812.73 | 815.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 13:15:00 | 816.00 | 812.73 | 815.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 816.00 | 812.73 | 815.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 14:00:00 | 816.00 | 812.73 | 815.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 820.15 | 814.21 | 815.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 15:00:00 | 820.15 | 814.21 | 815.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 813.90 | 814.15 | 815.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 807.05 | 814.15 | 815.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 766.70 | 789.63 | 795.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 12:15:00 | 791.25 | 788.80 | 793.42 | SL hit (close>ema200) qty=0.50 sl=788.80 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 812.20 | 796.89 | 795.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 821.50 | 801.81 | 798.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 843.00 | 848.72 | 833.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 10:15:00 | 841.45 | 848.72 | 833.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 846.05 | 850.42 | 841.62 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 830.00 | 841.00 | 841.04 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 847.00 | 839.40 | 839.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 852.60 | 842.04 | 840.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 846.10 | 848.32 | 845.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 11:00:00 | 846.10 | 848.32 | 845.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 847.95 | 848.24 | 845.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:45:00 | 847.50 | 848.24 | 845.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 846.55 | 851.20 | 848.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:30:00 | 843.00 | 851.20 | 848.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 853.45 | 851.65 | 848.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 11:30:00 | 856.20 | 852.86 | 849.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:30:00 | 854.70 | 855.13 | 852.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 13:15:00 | 855.05 | 854.78 | 852.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 881.75 | 893.91 | 894.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 881.75 | 893.91 | 894.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 877.00 | 886.36 | 890.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 13:15:00 | 883.70 | 880.45 | 885.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 13:15:00 | 883.70 | 880.45 | 885.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 883.70 | 880.45 | 885.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:30:00 | 883.30 | 880.45 | 885.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 881.20 | 880.60 | 884.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:45:00 | 884.25 | 880.60 | 884.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 880.10 | 880.50 | 884.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 887.05 | 880.50 | 884.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 891.95 | 882.79 | 884.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 891.95 | 882.79 | 884.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 891.00 | 884.43 | 885.49 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 893.00 | 887.52 | 886.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 14:15:00 | 897.20 | 890.05 | 888.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 12:15:00 | 890.85 | 894.02 | 891.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 12:15:00 | 890.85 | 894.02 | 891.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 12:15:00 | 890.85 | 894.02 | 891.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:00:00 | 890.85 | 894.02 | 891.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 891.75 | 893.57 | 891.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 14:45:00 | 894.00 | 893.58 | 891.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 884.75 | 891.56 | 890.94 | SL hit (close<static) qty=1.00 sl=890.10 alert=retest2 |

### Cycle 62 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 883.00 | 889.85 | 890.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 12:15:00 | 873.75 | 885.38 | 888.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 848.00 | 836.94 | 843.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 848.00 | 836.94 | 843.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 848.00 | 836.94 | 843.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 848.00 | 836.94 | 843.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 854.70 | 840.49 | 844.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:45:00 | 856.45 | 840.49 | 844.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 13:15:00 | 855.50 | 848.10 | 847.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 862.25 | 850.93 | 848.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 10:15:00 | 844.60 | 850.75 | 849.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 10:15:00 | 844.60 | 850.75 | 849.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 844.60 | 850.75 | 849.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 844.60 | 850.75 | 849.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 851.60 | 850.92 | 849.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 13:15:00 | 853.80 | 850.89 | 849.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 10:15:00 | 844.15 | 851.46 | 850.74 | SL hit (close<static) qty=1.00 sl=844.50 alert=retest2 |

### Cycle 64 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 841.90 | 851.05 | 851.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 838.05 | 848.45 | 850.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 781.80 | 775.43 | 787.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 10:15:00 | 786.10 | 777.56 | 786.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 786.10 | 777.56 | 786.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:45:00 | 787.30 | 777.56 | 786.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 786.25 | 779.30 | 786.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 12:00:00 | 786.25 | 779.30 | 786.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 787.80 | 781.00 | 786.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 12:45:00 | 788.00 | 781.00 | 786.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 793.75 | 783.55 | 787.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 13:45:00 | 792.80 | 783.55 | 787.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 790.75 | 784.99 | 787.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:30:00 | 794.50 | 784.99 | 787.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 785.30 | 786.87 | 788.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 12:30:00 | 783.50 | 787.41 | 788.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 13:15:00 | 794.85 | 788.89 | 788.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 13:15:00 | 794.85 | 788.89 | 788.81 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 785.85 | 788.29 | 788.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 776.55 | 786.06 | 787.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 786.10 | 786.07 | 787.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 11:00:00 | 786.10 | 786.07 | 787.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 788.10 | 786.47 | 787.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 788.10 | 786.47 | 787.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 787.40 | 786.66 | 787.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:15:00 | 789.85 | 786.66 | 787.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 788.45 | 787.02 | 787.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:30:00 | 789.70 | 787.02 | 787.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 780.00 | 785.61 | 786.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 15:15:00 | 778.00 | 785.61 | 786.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 09:15:00 | 798.05 | 767.99 | 770.66 | SL hit (close>static) qty=1.00 sl=790.60 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 780.00 | 773.53 | 772.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 788.10 | 778.78 | 775.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 13:15:00 | 783.30 | 785.51 | 782.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 14:00:00 | 783.30 | 785.51 | 782.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 784.90 | 785.39 | 782.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 15:00:00 | 784.90 | 785.39 | 782.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 787.80 | 785.52 | 783.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 787.80 | 785.52 | 783.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 776.60 | 788.65 | 786.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:15:00 | 775.00 | 788.65 | 786.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 776.35 | 786.19 | 785.74 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2026-03-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 11:15:00 | 776.35 | 784.22 | 784.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-24 09:15:00 | 771.95 | 778.92 | 781.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 789.50 | 780.15 | 781.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 789.50 | 780.15 | 781.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 789.50 | 780.15 | 781.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 789.50 | 780.15 | 781.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 789.00 | 781.92 | 782.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:45:00 | 791.75 | 781.92 | 782.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 787.45 | 783.02 | 782.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 809.60 | 788.65 | 785.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 794.20 | 798.60 | 792.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 14:15:00 | 794.20 | 798.60 | 792.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 794.20 | 798.60 | 792.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 15:00:00 | 794.20 | 798.60 | 792.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 791.20 | 797.12 | 792.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 778.90 | 797.12 | 792.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 773.50 | 792.40 | 790.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 773.50 | 792.40 | 790.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 772.45 | 788.41 | 789.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 770.20 | 784.77 | 787.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 09:15:00 | 776.35 | 775.33 | 781.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-30 09:45:00 | 776.20 | 775.33 | 781.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 11:15:00 | 787.85 | 778.08 | 781.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 11:30:00 | 790.60 | 778.08 | 781.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 12:15:00 | 785.60 | 779.59 | 781.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 13:15:00 | 778.60 | 779.59 | 781.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 11:15:00 | 793.25 | 783.69 | 782.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 793.25 | 783.69 | 782.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 798.55 | 786.66 | 783.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 770.00 | 787.23 | 785.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 770.00 | 787.23 | 785.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 770.00 | 787.23 | 785.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 770.00 | 787.23 | 785.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 773.00 | 784.39 | 784.41 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 789.00 | 785.22 | 784.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 793.90 | 786.96 | 785.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 09:15:00 | 778.00 | 785.48 | 785.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 778.00 | 785.48 | 785.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 778.00 | 785.48 | 785.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:00:00 | 778.00 | 785.48 | 785.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 10:15:00 | 781.35 | 784.65 | 784.82 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 789.30 | 785.66 | 785.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 792.70 | 787.07 | 785.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 831.85 | 832.68 | 821.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 831.85 | 832.68 | 821.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 828.00 | 833.28 | 827.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 839.10 | 833.28 | 827.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 823.80 | 834.49 | 831.79 | SL hit (close<static) qty=1.00 sl=826.70 alert=retest2 |

### Cycle 76 — SELL (started 2026-04-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 14:15:00 | 851.65 | 856.94 | 857.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 15:15:00 | 850.00 | 855.55 | 856.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 11:15:00 | 848.15 | 822.05 | 828.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 11:15:00 | 848.15 | 822.05 | 828.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 848.15 | 822.05 | 828.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 12:00:00 | 848.15 | 822.05 | 828.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 831.00 | 823.84 | 828.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 13:15:00 | 829.00 | 823.84 | 828.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 13:00:00 | 823.30 | 821.95 | 825.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 882.70 | 833.51 | 829.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 882.70 | 833.51 | 829.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 900.95 | 885.62 | 872.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 884.00 | 887.83 | 877.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 13:30:00 | 879.80 | 887.83 | 877.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 873.45 | 883.87 | 877.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 11:15:00 | 884.50 | 883.38 | 878.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 12:15:00 | 865.00 | 878.26 | 876.75 | SL hit (close<static) qty=1.00 sl=866.40 alert=retest2 |

### Cycle 78 — SELL (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 15:15:00 | 870.00 | 875.16 | 875.58 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 888.10 | 877.75 | 876.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 10:15:00 | 896.00 | 881.40 | 878.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 11:15:00 | 889.40 | 894.75 | 888.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 11:15:00 | 889.40 | 894.75 | 888.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 889.40 | 894.75 | 888.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:00:00 | 889.40 | 894.75 | 888.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 890.25 | 893.85 | 888.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:00:00 | 890.25 | 893.85 | 888.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 885.40 | 892.16 | 888.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:00:00 | 885.40 | 892.16 | 888.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 890.40 | 891.81 | 888.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:30:00 | 895.65 | 892.61 | 889.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 11:45:00 | 893.75 | 893.72 | 890.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 13:15:00 | 894.50 | 893.68 | 890.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 13:45:00 | 894.10 | 893.78 | 891.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 887.70 | 892.46 | 891.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 09:15:00 | 895.10 | 892.46 | 891.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 10:00:00 | 894.15 | 892.80 | 891.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 10:15:00 | 883.00 | 890.84 | 890.61 | SL hit (close<static) qty=1.00 sl=885.40 alert=retest2 |

### Cycle 80 — SELL (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 11:15:00 | 882.15 | 889.10 | 889.84 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 10:15:00 | 923.65 | 892.35 | 890.30 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-22 14:15:00 | 897.20 | 2025-05-30 09:15:00 | 852.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-26 09:15:00 | 887.45 | 2025-05-30 11:15:00 | 843.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-22 14:15:00 | 897.20 | 2025-05-30 15:15:00 | 850.65 | STOP_HIT | 0.50 | 5.19% |
| SELL | retest2 | 2025-05-26 09:15:00 | 887.45 | 2025-05-30 15:15:00 | 850.65 | STOP_HIT | 0.50 | 4.15% |
| SELL | retest2 | 2025-06-04 15:15:00 | 848.95 | 2025-06-05 09:15:00 | 861.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-06-09 09:45:00 | 874.60 | 2025-06-11 11:15:00 | 865.05 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-06-09 12:00:00 | 875.45 | 2025-06-11 11:15:00 | 865.05 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-06-09 14:45:00 | 876.90 | 2025-06-11 11:15:00 | 865.05 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-06-17 12:15:00 | 839.60 | 2025-06-23 14:15:00 | 831.95 | STOP_HIT | 1.00 | 0.91% |
| SELL | retest2 | 2025-06-17 14:30:00 | 840.10 | 2025-06-23 14:15:00 | 831.95 | STOP_HIT | 1.00 | 0.97% |
| BUY | retest2 | 2025-06-25 11:30:00 | 850.10 | 2025-07-07 09:15:00 | 869.95 | STOP_HIT | 1.00 | 2.34% |
| BUY | retest2 | 2025-06-26 09:15:00 | 845.50 | 2025-07-07 09:15:00 | 869.95 | STOP_HIT | 1.00 | 2.89% |
| BUY | retest2 | 2025-06-26 12:00:00 | 846.55 | 2025-07-07 09:15:00 | 869.95 | STOP_HIT | 1.00 | 2.76% |
| SELL | retest2 | 2025-07-08 09:30:00 | 866.40 | 2025-07-15 10:15:00 | 864.05 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2025-07-08 15:15:00 | 867.10 | 2025-07-15 10:15:00 | 864.05 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2025-07-09 09:45:00 | 866.50 | 2025-07-15 10:15:00 | 864.05 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2025-07-21 09:15:00 | 921.55 | 2025-08-04 15:15:00 | 975.90 | STOP_HIT | 1.00 | 5.90% |
| SELL | retest2 | 2025-08-08 09:15:00 | 953.65 | 2025-08-13 09:15:00 | 966.55 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-08-08 12:15:00 | 960.15 | 2025-08-13 09:15:00 | 966.55 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-08-11 09:30:00 | 958.70 | 2025-08-13 09:15:00 | 966.55 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-08-11 11:30:00 | 952.20 | 2025-08-13 09:15:00 | 966.55 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-08-12 09:30:00 | 947.25 | 2025-08-13 10:15:00 | 970.25 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-08-12 13:15:00 | 948.20 | 2025-08-13 10:15:00 | 970.25 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-08-12 14:00:00 | 948.00 | 2025-08-13 10:15:00 | 970.25 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-08-12 15:00:00 | 948.05 | 2025-08-13 10:15:00 | 970.25 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-08-21 12:45:00 | 940.25 | 2025-09-03 10:15:00 | 931.00 | STOP_HIT | 1.00 | 0.98% |
| SELL | retest2 | 2025-08-22 09:30:00 | 941.45 | 2025-09-03 10:15:00 | 931.00 | STOP_HIT | 1.00 | 1.11% |
| SELL | retest2 | 2025-08-22 10:30:00 | 942.15 | 2025-09-03 10:15:00 | 931.00 | STOP_HIT | 1.00 | 1.18% |
| SELL | retest2 | 2025-08-26 09:15:00 | 935.55 | 2025-09-03 10:15:00 | 931.00 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2025-08-26 10:45:00 | 933.85 | 2025-09-03 10:15:00 | 931.00 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2025-08-28 09:15:00 | 921.75 | 2025-09-03 10:15:00 | 931.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-09-05 11:00:00 | 929.50 | 2025-09-09 15:15:00 | 927.95 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-09-05 15:15:00 | 928.25 | 2025-09-09 15:15:00 | 927.95 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-09-08 09:30:00 | 944.90 | 2025-09-09 15:15:00 | 927.95 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-09-09 14:45:00 | 930.10 | 2025-09-09 15:15:00 | 927.95 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-09-16 13:15:00 | 928.45 | 2025-09-17 09:15:00 | 915.75 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-09-16 15:15:00 | 927.80 | 2025-09-17 09:15:00 | 915.75 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-09-19 14:00:00 | 901.70 | 2025-09-22 09:15:00 | 922.90 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-09-19 14:30:00 | 906.45 | 2025-09-22 09:15:00 | 922.90 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-09-22 09:15:00 | 909.60 | 2025-09-22 09:15:00 | 922.90 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-09-30 09:15:00 | 908.00 | 2025-09-30 14:15:00 | 912.40 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-09-30 10:00:00 | 907.40 | 2025-10-01 11:15:00 | 919.60 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-09-30 11:30:00 | 909.55 | 2025-10-01 11:15:00 | 919.60 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-09-30 12:15:00 | 909.00 | 2025-10-01 11:15:00 | 919.60 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-09-30 13:15:00 | 906.15 | 2025-10-01 11:15:00 | 919.60 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-10-14 10:30:00 | 921.55 | 2025-10-20 14:15:00 | 925.20 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-10-14 12:00:00 | 914.90 | 2025-10-20 14:15:00 | 925.20 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-10-14 12:45:00 | 920.00 | 2025-10-20 14:15:00 | 925.20 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-10-14 13:15:00 | 922.05 | 2025-10-20 14:15:00 | 925.20 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-10-15 09:15:00 | 922.65 | 2025-10-20 14:15:00 | 925.20 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-10-16 09:45:00 | 921.20 | 2025-10-20 14:15:00 | 925.20 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-10-16 13:45:00 | 922.45 | 2025-10-20 14:15:00 | 925.20 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-11-03 13:45:00 | 890.00 | 2025-11-03 15:15:00 | 906.00 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-11-03 14:45:00 | 890.00 | 2025-11-03 15:15:00 | 906.00 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-11-07 14:00:00 | 863.05 | 2025-11-12 13:15:00 | 865.05 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-11-07 15:00:00 | 862.55 | 2025-11-12 13:15:00 | 865.05 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-11-10 09:15:00 | 854.10 | 2025-11-12 13:15:00 | 865.05 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-11-12 12:00:00 | 859.15 | 2025-11-12 13:15:00 | 865.05 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-11-17 09:15:00 | 851.05 | 2025-11-17 09:15:00 | 859.70 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-11-18 09:15:00 | 851.20 | 2025-11-24 14:15:00 | 808.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-24 09:30:00 | 847.15 | 2025-11-24 15:15:00 | 804.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 09:15:00 | 851.20 | 2025-11-25 14:15:00 | 817.95 | STOP_HIT | 0.50 | 3.91% |
| SELL | retest2 | 2025-11-24 09:30:00 | 847.15 | 2025-11-25 14:15:00 | 817.95 | STOP_HIT | 0.50 | 3.45% |
| SELL | retest2 | 2025-12-05 11:45:00 | 789.50 | 2025-12-09 14:15:00 | 819.65 | STOP_HIT | 1.00 | -3.82% |
| SELL | retest2 | 2025-12-09 09:15:00 | 784.00 | 2025-12-09 14:15:00 | 819.65 | STOP_HIT | 1.00 | -4.55% |
| SELL | retest2 | 2025-12-09 12:30:00 | 788.10 | 2025-12-09 14:15:00 | 819.65 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest2 | 2026-01-01 14:15:00 | 841.20 | 2026-01-06 11:15:00 | 838.85 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2026-01-01 15:15:00 | 844.00 | 2026-01-06 11:15:00 | 838.85 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2026-01-20 09:15:00 | 807.05 | 2026-01-27 09:15:00 | 766.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 09:15:00 | 807.05 | 2026-01-27 12:15:00 | 791.25 | STOP_HIT | 0.50 | 1.96% |
| BUY | retest2 | 2026-02-05 11:30:00 | 856.20 | 2026-02-13 10:15:00 | 881.75 | STOP_HIT | 1.00 | 2.98% |
| BUY | retest2 | 2026-02-06 11:30:00 | 854.70 | 2026-02-13 10:15:00 | 881.75 | STOP_HIT | 1.00 | 3.16% |
| BUY | retest2 | 2026-02-06 13:15:00 | 855.05 | 2026-02-13 10:15:00 | 881.75 | STOP_HIT | 1.00 | 3.12% |
| BUY | retest2 | 2026-02-18 14:45:00 | 894.00 | 2026-02-19 09:15:00 | 884.75 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2026-02-26 13:15:00 | 853.80 | 2026-02-27 10:15:00 | 844.15 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2026-02-27 12:15:00 | 857.10 | 2026-03-02 10:15:00 | 843.25 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2026-03-02 09:30:00 | 853.90 | 2026-03-02 10:15:00 | 843.25 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2026-03-11 12:30:00 | 783.50 | 2026-03-11 13:15:00 | 794.85 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-03-12 15:15:00 | 778.00 | 2026-03-17 09:15:00 | 798.05 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2026-03-17 11:00:00 | 779.60 | 2026-03-17 12:15:00 | 780.00 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2026-03-17 11:30:00 | 778.45 | 2026-03-17 12:15:00 | 780.00 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2026-03-30 13:15:00 | 778.60 | 2026-04-01 11:15:00 | 793.25 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2026-04-10 09:15:00 | 839.10 | 2026-04-13 09:15:00 | 823.80 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2026-04-13 10:45:00 | 834.00 | 2026-04-20 14:15:00 | 851.65 | STOP_HIT | 1.00 | 2.12% |
| BUY | retest2 | 2026-04-13 11:15:00 | 833.45 | 2026-04-20 14:15:00 | 851.65 | STOP_HIT | 1.00 | 2.18% |
| BUY | retest2 | 2026-04-13 11:45:00 | 833.25 | 2026-04-20 14:15:00 | 851.65 | STOP_HIT | 1.00 | 2.21% |
| BUY | retest2 | 2026-04-15 09:15:00 | 848.50 | 2026-04-20 14:15:00 | 851.65 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2026-04-23 13:15:00 | 829.00 | 2026-04-27 09:15:00 | 882.70 | STOP_HIT | 1.00 | -6.48% |
| SELL | retest2 | 2026-04-24 13:00:00 | 823.30 | 2026-04-27 09:15:00 | 882.70 | STOP_HIT | 1.00 | -7.21% |
| BUY | retest2 | 2026-04-30 11:15:00 | 884.50 | 2026-04-30 12:15:00 | 865.00 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2026-05-06 09:30:00 | 895.65 | 2026-05-07 10:15:00 | 883.00 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2026-05-06 11:45:00 | 893.75 | 2026-05-07 10:15:00 | 883.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-05-06 13:15:00 | 894.50 | 2026-05-07 10:15:00 | 883.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2026-05-06 13:45:00 | 894.10 | 2026-05-07 10:15:00 | 883.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2026-05-07 09:15:00 | 895.10 | 2026-05-07 10:15:00 | 883.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2026-05-07 10:00:00 | 894.15 | 2026-05-07 10:15:00 | 883.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2026-05-07 10:30:00 | 894.70 | 2026-05-07 11:15:00 | 882.15 | STOP_HIT | 1.00 | -1.40% |
