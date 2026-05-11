# UTI Asset Management Company Ltd. (UTIAMC)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 973.15
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 145 |
| ALERT1 | 97 |
| ALERT2 | 97 |
| ALERT2_SKIP | 46 |
| ALERT3 | 268 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 9 |
| ENTRY2 | 106 |
| PARTIAL | 12 |
| TARGET_HIT | 7 |
| STOP_HIT | 110 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 127 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 44 / 83
- **Target hits / Stop hits / Partials:** 7 / 108 / 12
- **Avg / median % per leg:** 0.25% / -0.88%
- **Sum % (uncompounded):** 32.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 51 | 15 | 29.4% | 5 | 45 | 1 | 0.10% | 5.3% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.24% | 6.7% |
| BUY @ 3rd Alert (retest2) | 48 | 13 | 27.1% | 5 | 43 | 0 | -0.03% | -1.5% |
| SELL (all) | 76 | 29 | 38.2% | 2 | 63 | 11 | 0.35% | 26.8% |
| SELL @ 2nd Alert (retest1) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.51% | -10.6% |
| SELL @ 3rd Alert (retest2) | 69 | 29 | 42.0% | 2 | 56 | 11 | 0.54% | 37.3% |
| retest1 (combined) | 10 | 2 | 20.0% | 0 | 9 | 1 | -0.38% | -3.8% |
| retest2 (combined) | 117 | 42 | 35.9% | 7 | 99 | 11 | 0.31% | 35.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 13:15:00 | 915.00 | 909.93 | 909.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 916.95 | 911.33 | 910.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 11:15:00 | 909.15 | 912.24 | 911.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 11:15:00 | 909.15 | 912.24 | 911.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 11:15:00 | 909.15 | 912.24 | 911.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 11:45:00 | 910.00 | 912.24 | 911.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 12:15:00 | 905.50 | 910.89 | 910.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 12:30:00 | 905.00 | 910.89 | 910.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 13:15:00 | 905.00 | 909.71 | 910.17 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 11:15:00 | 914.85 | 910.51 | 910.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 12:15:00 | 915.95 | 911.60 | 910.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 13:15:00 | 910.25 | 911.33 | 910.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 13:15:00 | 910.25 | 911.33 | 910.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 910.25 | 911.33 | 910.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 14:00:00 | 910.25 | 911.33 | 910.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 911.55 | 911.37 | 910.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 14:30:00 | 909.05 | 911.37 | 910.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 915.90 | 912.28 | 911.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:30:00 | 918.75 | 913.31 | 911.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 14:00:00 | 917.05 | 916.41 | 914.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:45:00 | 919.45 | 921.18 | 919.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 12:15:00 | 922.80 | 924.87 | 924.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 12:15:00 | 922.80 | 924.87 | 924.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 13:15:00 | 921.90 | 924.28 | 924.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 10:15:00 | 921.90 | 920.96 | 922.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 10:15:00 | 921.90 | 920.96 | 922.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 921.90 | 920.96 | 922.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 11:00:00 | 921.90 | 920.96 | 922.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 921.00 | 920.97 | 922.52 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 09:15:00 | 930.25 | 922.92 | 922.83 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 914.00 | 922.56 | 923.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 11:15:00 | 913.75 | 919.42 | 921.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 14:15:00 | 908.55 | 908.46 | 912.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 15:00:00 | 908.55 | 908.46 | 912.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 926.00 | 911.82 | 913.48 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 926.85 | 914.83 | 914.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-04 11:15:00 | 930.65 | 920.50 | 918.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-05 09:15:00 | 925.30 | 928.74 | 924.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 09:15:00 | 925.30 | 928.74 | 924.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 925.30 | 928.74 | 924.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 09:15:00 | 951.25 | 929.65 | 927.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-18 10:15:00 | 1046.38 | 1008.93 | 997.29 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 14:15:00 | 1017.80 | 1022.01 | 1022.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 15:15:00 | 1010.05 | 1019.62 | 1021.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 12:15:00 | 1020.55 | 1018.66 | 1020.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 12:15:00 | 1020.55 | 1018.66 | 1020.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 1020.55 | 1018.66 | 1020.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 13:00:00 | 1020.55 | 1018.66 | 1020.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 13:15:00 | 1016.25 | 1018.18 | 1019.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 13:30:00 | 1019.65 | 1018.18 | 1019.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 15:15:00 | 1020.90 | 1018.54 | 1019.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 09:15:00 | 1044.50 | 1018.54 | 1019.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 1028.50 | 1020.53 | 1020.47 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 12:15:00 | 1012.60 | 1019.03 | 1019.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 12:15:00 | 1009.55 | 1014.18 | 1016.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 14:15:00 | 1015.20 | 1013.35 | 1015.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 14:15:00 | 1015.20 | 1013.35 | 1015.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 14:15:00 | 1015.20 | 1013.35 | 1015.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 14:45:00 | 1014.55 | 1013.35 | 1015.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 15:15:00 | 1012.55 | 1013.19 | 1015.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:15:00 | 1019.45 | 1013.19 | 1015.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 1007.85 | 1012.12 | 1014.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 10:30:00 | 1005.70 | 1010.70 | 1014.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 12:00:00 | 1005.00 | 1009.56 | 1013.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 09:45:00 | 1000.00 | 1004.50 | 1008.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 09:15:00 | 1036.00 | 1009.23 | 1008.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 09:15:00 | 1036.00 | 1009.23 | 1008.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 09:15:00 | 1048.05 | 1044.74 | 1041.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 09:15:00 | 1046.35 | 1047.25 | 1044.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 09:15:00 | 1046.35 | 1047.25 | 1044.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 1046.35 | 1047.25 | 1044.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:00:00 | 1046.35 | 1047.25 | 1044.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 1045.30 | 1046.86 | 1044.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 11:15:00 | 1047.00 | 1046.86 | 1044.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 12:30:00 | 1047.00 | 1046.87 | 1044.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 09:15:00 | 1047.85 | 1046.86 | 1045.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 09:15:00 | 1035.50 | 1044.59 | 1044.52 | SL hit (close<static) qty=1.00 sl=1041.70 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 1037.00 | 1043.07 | 1043.84 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 14:15:00 | 1048.00 | 1044.35 | 1044.18 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 15:15:00 | 1032.00 | 1041.88 | 1043.08 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 11:15:00 | 1047.45 | 1043.87 | 1043.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 13:15:00 | 1049.70 | 1045.77 | 1044.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 09:15:00 | 1054.35 | 1087.69 | 1077.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 09:15:00 | 1054.35 | 1087.69 | 1077.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 1054.35 | 1087.69 | 1077.92 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2024-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 12:15:00 | 1044.90 | 1067.18 | 1069.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 15:15:00 | 1038.00 | 1054.41 | 1062.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 995.25 | 978.28 | 990.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 995.25 | 978.28 | 990.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 995.25 | 978.28 | 990.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 995.25 | 978.28 | 990.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 1003.05 | 983.23 | 991.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 11:15:00 | 1006.30 | 983.23 | 991.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2024-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 14:15:00 | 1015.55 | 998.42 | 996.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 14:15:00 | 1030.05 | 1010.65 | 1004.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 15:15:00 | 1042.10 | 1047.27 | 1038.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 09:15:00 | 1042.00 | 1047.27 | 1038.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 12:15:00 | 1038.50 | 1044.41 | 1039.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 12:45:00 | 1037.55 | 1044.41 | 1039.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 13:15:00 | 1037.50 | 1043.03 | 1039.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 14:00:00 | 1037.50 | 1043.03 | 1039.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 1029.90 | 1040.40 | 1038.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 15:00:00 | 1029.90 | 1040.40 | 1038.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 09:15:00 | 1030.05 | 1036.75 | 1037.16 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 09:15:00 | 1049.45 | 1039.10 | 1037.92 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 15:15:00 | 1029.65 | 1036.79 | 1037.58 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-02 10:15:00 | 1042.30 | 1038.01 | 1038.00 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 15:15:00 | 1030.30 | 1036.97 | 1037.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 997.65 | 1029.11 | 1034.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 996.95 | 996.58 | 1011.57 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 10:30:00 | 984.55 | 995.65 | 1009.79 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 12:45:00 | 987.30 | 994.14 | 1006.62 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 13:15:00 | 988.15 | 994.14 | 1006.62 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-07 09:45:00 | 987.50 | 987.31 | 998.92 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 1007.00 | 991.24 | 995.23 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-08 09:15:00 | 1007.00 | 991.24 | 995.23 | SL hit (close>ema400) qty=1.00 sl=995.23 alert=retest1 |

### Cycle 23 — BUY (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 11:15:00 | 1011.90 | 998.08 | 997.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 09:15:00 | 1017.00 | 1002.25 | 1000.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 13:15:00 | 998.00 | 1003.11 | 1001.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 13:15:00 | 998.00 | 1003.11 | 1001.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 998.00 | 1003.11 | 1001.42 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2024-08-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 15:15:00 | 996.05 | 1000.24 | 1000.32 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 09:15:00 | 1002.95 | 1000.78 | 1000.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 10:15:00 | 1023.15 | 1005.26 | 1002.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 15:15:00 | 1015.00 | 1016.04 | 1010.04 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 09:15:00 | 1033.35 | 1016.04 | 1010.04 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 1016.00 | 1028.10 | 1020.68 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-13 14:15:00 | 1016.00 | 1028.10 | 1020.68 | SL hit (close<ema400) qty=1.00 sl=1020.68 alert=retest1 |

### Cycle 26 — SELL (started 2024-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 13:15:00 | 1130.00 | 1144.24 | 1145.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 10:15:00 | 1113.65 | 1132.79 | 1139.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 11:15:00 | 1131.00 | 1121.10 | 1127.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 11:15:00 | 1131.00 | 1121.10 | 1127.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 1131.00 | 1121.10 | 1127.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 12:00:00 | 1131.00 | 1121.10 | 1127.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 12:15:00 | 1129.20 | 1122.72 | 1127.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 12:45:00 | 1130.00 | 1122.72 | 1127.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 13:15:00 | 1129.00 | 1123.98 | 1127.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 13:45:00 | 1128.05 | 1123.98 | 1127.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 1128.90 | 1124.96 | 1128.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 14:45:00 | 1144.50 | 1124.96 | 1128.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 1145.05 | 1128.98 | 1129.59 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2024-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 09:15:00 | 1149.25 | 1133.03 | 1131.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 13:15:00 | 1189.55 | 1152.76 | 1142.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 14:15:00 | 1185.45 | 1188.32 | 1177.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-04 15:00:00 | 1185.45 | 1188.32 | 1177.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 1258.85 | 1273.01 | 1257.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:30:00 | 1263.25 | 1273.01 | 1257.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 1264.95 | 1271.40 | 1257.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 14:45:00 | 1261.10 | 1271.40 | 1257.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 1264.90 | 1270.10 | 1258.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 09:15:00 | 1280.10 | 1270.10 | 1258.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 15:15:00 | 1270.00 | 1275.66 | 1276.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 15:15:00 | 1270.00 | 1275.66 | 1276.26 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 1293.40 | 1279.21 | 1277.82 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 10:15:00 | 1268.40 | 1280.30 | 1280.78 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 14:15:00 | 1295.15 | 1280.21 | 1278.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 09:15:00 | 1310.50 | 1288.00 | 1282.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 10:15:00 | 1283.95 | 1287.19 | 1282.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 10:15:00 | 1283.95 | 1287.19 | 1282.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 1283.95 | 1287.19 | 1282.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 10:30:00 | 1289.70 | 1287.19 | 1282.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 1315.00 | 1292.75 | 1285.58 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-20 09:15:00 | 1284.70 | 1289.16 | 1289.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-20 13:15:00 | 1279.65 | 1285.85 | 1287.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 11:15:00 | 1285.95 | 1283.50 | 1285.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-23 11:15:00 | 1285.95 | 1283.50 | 1285.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 11:15:00 | 1285.95 | 1283.50 | 1285.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 11:45:00 | 1286.00 | 1283.50 | 1285.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 12:15:00 | 1289.40 | 1284.68 | 1285.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 12:45:00 | 1291.00 | 1284.68 | 1285.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 13:15:00 | 1285.85 | 1284.91 | 1285.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 13:45:00 | 1291.00 | 1284.91 | 1285.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 14:15:00 | 1286.30 | 1285.19 | 1285.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 14:45:00 | 1288.70 | 1285.19 | 1285.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 15:15:00 | 1279.50 | 1284.05 | 1285.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-24 09:15:00 | 1283.10 | 1284.05 | 1285.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 1281.65 | 1283.57 | 1285.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 09:30:00 | 1240.00 | 1266.84 | 1273.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 10:00:00 | 1256.85 | 1251.23 | 1259.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 12:00:00 | 1256.25 | 1254.49 | 1259.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 09:30:00 | 1251.05 | 1251.05 | 1255.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 1225.45 | 1239.29 | 1246.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 09:15:00 | 1205.45 | 1234.31 | 1238.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 11:00:00 | 1207.10 | 1226.92 | 1234.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 12:30:00 | 1209.70 | 1225.12 | 1232.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 14:45:00 | 1210.00 | 1223.64 | 1230.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 1188.40 | 1216.01 | 1225.50 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 1194.01 | 1216.01 | 1225.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 1193.44 | 1216.01 | 1225.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 1188.50 | 1216.01 | 1225.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 1178.00 | 1211.13 | 1222.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 09:15:00 | 1177.25 | 1203.98 | 1214.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 14:15:00 | 1199.50 | 1195.81 | 1204.51 | SL hit (close>ema200) qty=0.50 sl=1195.81 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 1231.00 | 1209.04 | 1208.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 14:15:00 | 1239.40 | 1221.78 | 1215.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 14:15:00 | 1228.45 | 1233.76 | 1226.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 15:00:00 | 1228.45 | 1233.76 | 1226.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 1225.75 | 1231.88 | 1226.79 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 13:15:00 | 1215.00 | 1222.81 | 1223.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 15:15:00 | 1212.05 | 1219.41 | 1221.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 09:15:00 | 1236.95 | 1208.16 | 1211.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 09:15:00 | 1236.95 | 1208.16 | 1211.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 1236.95 | 1208.16 | 1211.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 09:45:00 | 1237.75 | 1208.16 | 1211.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 1235.00 | 1213.53 | 1213.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 11:15:00 | 1232.80 | 1213.53 | 1213.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 11:15:00 | 1232.65 | 1217.35 | 1215.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 11:15:00 | 1232.65 | 1217.35 | 1215.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 09:15:00 | 1298.45 | 1239.53 | 1227.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 11:15:00 | 1300.50 | 1303.16 | 1276.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-17 11:45:00 | 1299.75 | 1303.16 | 1276.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 1282.00 | 1296.29 | 1283.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 09:15:00 | 1310.65 | 1283.97 | 1281.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 15:15:00 | 1274.50 | 1282.63 | 1282.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-10-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 15:15:00 | 1274.50 | 1282.63 | 1282.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 1267.70 | 1279.64 | 1281.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 09:15:00 | 1196.30 | 1175.83 | 1197.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-28 09:15:00 | 1196.30 | 1175.83 | 1197.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 1196.30 | 1175.83 | 1197.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 10:30:00 | 1175.20 | 1188.30 | 1194.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 09:15:00 | 1254.75 | 1206.90 | 1200.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 1254.75 | 1206.90 | 1200.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 12:15:00 | 1290.00 | 1237.37 | 1217.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 14:15:00 | 1325.05 | 1341.96 | 1317.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 15:00:00 | 1325.05 | 1341.96 | 1317.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 1322.25 | 1332.81 | 1325.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 09:15:00 | 1342.40 | 1332.81 | 1325.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 10:45:00 | 1332.60 | 1333.32 | 1326.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 13:15:00 | 1326.05 | 1339.96 | 1340.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 1326.05 | 1339.96 | 1340.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 1320.95 | 1336.16 | 1338.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 1358.25 | 1322.22 | 1326.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 1358.25 | 1322.22 | 1326.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 1358.25 | 1322.22 | 1326.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:45:00 | 1360.65 | 1322.22 | 1326.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 1336.00 | 1324.98 | 1327.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:00:00 | 1319.20 | 1323.82 | 1326.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-12 13:15:00 | 1345.25 | 1330.48 | 1329.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 13:15:00 | 1345.25 | 1330.48 | 1329.60 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 1306.45 | 1328.02 | 1329.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 10:15:00 | 1298.95 | 1322.21 | 1326.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1318.85 | 1297.00 | 1309.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 1318.85 | 1297.00 | 1309.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 1318.85 | 1297.00 | 1309.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 1318.85 | 1297.00 | 1309.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 1300.55 | 1297.71 | 1308.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:15:00 | 1274.00 | 1303.52 | 1307.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 10:30:00 | 1285.50 | 1281.00 | 1289.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 15:15:00 | 1290.00 | 1281.72 | 1283.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 09:15:00 | 1316.30 | 1289.96 | 1287.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 09:15:00 | 1316.30 | 1289.96 | 1287.15 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 14:15:00 | 1288.00 | 1294.32 | 1294.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-25 15:15:00 | 1283.05 | 1292.07 | 1293.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 11:15:00 | 1281.45 | 1281.13 | 1285.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-27 12:00:00 | 1281.45 | 1281.13 | 1285.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 12:15:00 | 1285.55 | 1282.01 | 1285.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 12:30:00 | 1288.65 | 1282.01 | 1285.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 1289.85 | 1283.58 | 1285.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 13:30:00 | 1294.05 | 1283.58 | 1285.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 14:15:00 | 1300.40 | 1286.95 | 1286.84 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 15:15:00 | 1281.05 | 1286.86 | 1287.47 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 10:15:00 | 1293.00 | 1288.07 | 1287.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 14:15:00 | 1299.20 | 1292.01 | 1289.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 10:15:00 | 1291.60 | 1293.92 | 1291.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 10:15:00 | 1291.60 | 1293.92 | 1291.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 1291.60 | 1293.92 | 1291.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:00:00 | 1291.60 | 1293.92 | 1291.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 1290.25 | 1293.19 | 1291.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 12:00:00 | 1290.25 | 1293.19 | 1291.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 12:15:00 | 1298.55 | 1294.26 | 1292.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 13:15:00 | 1300.10 | 1294.26 | 1292.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 09:15:00 | 1330.00 | 1295.13 | 1293.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 09:15:00 | 1348.65 | 1368.91 | 1370.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 1348.65 | 1368.91 | 1370.38 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 14:15:00 | 1369.10 | 1360.35 | 1360.22 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 09:15:00 | 1335.55 | 1355.97 | 1358.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 11:15:00 | 1329.45 | 1348.05 | 1354.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 14:15:00 | 1231.95 | 1231.04 | 1258.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 15:00:00 | 1231.95 | 1231.04 | 1258.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 1226.00 | 1225.43 | 1234.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 1232.20 | 1225.43 | 1234.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 1238.05 | 1227.95 | 1234.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:45:00 | 1241.55 | 1227.95 | 1234.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 1236.95 | 1229.75 | 1234.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:30:00 | 1237.40 | 1229.75 | 1234.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 1239.85 | 1231.77 | 1235.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:30:00 | 1239.00 | 1231.77 | 1235.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 1244.35 | 1234.29 | 1235.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:00:00 | 1244.35 | 1234.29 | 1235.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 1244.70 | 1236.37 | 1236.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:30:00 | 1245.45 | 1236.37 | 1236.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 14:15:00 | 1251.00 | 1239.30 | 1238.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 10:15:00 | 1262.50 | 1247.71 | 1242.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 10:15:00 | 1373.00 | 1374.21 | 1356.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 10:30:00 | 1372.65 | 1374.21 | 1356.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 1361.00 | 1371.39 | 1360.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 1361.00 | 1371.39 | 1360.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 1370.70 | 1371.25 | 1361.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 1355.25 | 1371.25 | 1361.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 1333.00 | 1363.60 | 1358.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 1336.20 | 1363.60 | 1358.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 1306.35 | 1352.15 | 1354.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 1297.95 | 1330.01 | 1342.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 12:15:00 | 1309.00 | 1307.43 | 1322.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 13:00:00 | 1309.00 | 1307.43 | 1322.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 1285.00 | 1300.67 | 1314.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 10:30:00 | 1274.90 | 1296.34 | 1311.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 11:15:00 | 1269.95 | 1296.34 | 1311.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 13:15:00 | 1211.15 | 1236.58 | 1264.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 13:15:00 | 1206.45 | 1236.58 | 1264.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-10 12:15:00 | 1206.55 | 1205.68 | 1234.31 | SL hit (close>ema200) qty=0.50 sl=1205.68 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 10:15:00 | 1242.50 | 1204.22 | 1201.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 11:15:00 | 1251.00 | 1213.58 | 1206.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 11:15:00 | 1244.45 | 1246.74 | 1231.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-16 12:00:00 | 1244.45 | 1246.74 | 1231.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 1219.25 | 1240.19 | 1233.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:30:00 | 1217.00 | 1240.19 | 1233.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 1223.65 | 1236.88 | 1232.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:45:00 | 1207.00 | 1236.88 | 1232.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 12:15:00 | 1221.25 | 1232.41 | 1231.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 12:30:00 | 1223.15 | 1232.41 | 1231.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-01-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 13:15:00 | 1220.00 | 1229.93 | 1230.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 14:15:00 | 1214.55 | 1226.85 | 1229.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 09:15:00 | 1227.35 | 1224.90 | 1227.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 09:15:00 | 1227.35 | 1224.90 | 1227.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 1227.35 | 1224.90 | 1227.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:45:00 | 1229.00 | 1224.90 | 1227.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 1233.95 | 1226.71 | 1228.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 11:00:00 | 1233.95 | 1226.71 | 1228.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 11:15:00 | 1232.10 | 1227.79 | 1228.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 11:30:00 | 1232.05 | 1227.79 | 1228.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 12:15:00 | 1244.00 | 1231.03 | 1229.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 13:15:00 | 1246.80 | 1234.18 | 1231.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 1233.50 | 1238.23 | 1234.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 1233.50 | 1238.23 | 1234.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 1233.50 | 1238.23 | 1234.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 1233.50 | 1238.23 | 1234.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 1226.00 | 1235.78 | 1233.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:30:00 | 1228.95 | 1235.78 | 1233.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 12:15:00 | 1221.55 | 1231.45 | 1232.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 1213.10 | 1226.29 | 1229.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 1231.55 | 1219.05 | 1222.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 1231.55 | 1219.05 | 1222.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 1231.55 | 1219.05 | 1222.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 1231.55 | 1219.05 | 1222.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 1225.00 | 1220.24 | 1223.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:15:00 | 1220.65 | 1220.24 | 1223.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 1237.60 | 1226.25 | 1225.50 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 14:15:00 | 1215.90 | 1225.86 | 1225.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 15:15:00 | 1207.00 | 1222.09 | 1224.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 14:15:00 | 1070.30 | 1063.24 | 1089.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 15:00:00 | 1070.30 | 1063.24 | 1089.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 1054.55 | 1035.25 | 1044.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:30:00 | 1056.70 | 1035.25 | 1044.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1031.30 | 1037.23 | 1043.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 1039.30 | 1037.23 | 1043.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1053.65 | 1040.51 | 1044.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:45:00 | 1054.40 | 1040.51 | 1044.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 1053.75 | 1043.16 | 1044.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 15:00:00 | 1053.75 | 1043.16 | 1044.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 1026.70 | 1040.80 | 1043.63 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2025-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 12:15:00 | 1048.65 | 1037.55 | 1037.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 13:15:00 | 1053.10 | 1040.66 | 1038.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 1073.40 | 1078.36 | 1064.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 1073.40 | 1078.36 | 1064.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 1073.40 | 1078.36 | 1064.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:30:00 | 1076.25 | 1078.36 | 1064.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 1066.20 | 1075.93 | 1065.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:45:00 | 1065.30 | 1075.93 | 1065.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 1068.00 | 1074.35 | 1065.35 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 09:15:00 | 1049.25 | 1061.14 | 1061.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 1030.25 | 1050.42 | 1055.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 1010.70 | 1002.66 | 1014.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 13:00:00 | 1010.70 | 1002.66 | 1014.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 1005.25 | 1002.86 | 1010.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:30:00 | 1006.25 | 1002.86 | 1010.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 950.10 | 929.00 | 939.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 950.10 | 929.00 | 939.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 938.85 | 930.97 | 939.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 950.25 | 930.97 | 939.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 11:15:00 | 949.75 | 934.73 | 940.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 12:00:00 | 949.75 | 934.73 | 940.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 12:15:00 | 946.20 | 937.02 | 940.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 13:15:00 | 941.35 | 937.02 | 940.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-20 09:15:00 | 965.05 | 946.89 | 944.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 09:15:00 | 965.05 | 946.89 | 944.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 10:15:00 | 975.45 | 952.60 | 947.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 10:15:00 | 962.65 | 965.61 | 958.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 11:00:00 | 962.65 | 965.61 | 958.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 12:15:00 | 957.50 | 964.28 | 958.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:00:00 | 957.50 | 964.28 | 958.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 955.85 | 962.59 | 958.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:45:00 | 955.35 | 962.59 | 958.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 952.50 | 960.58 | 958.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 15:00:00 | 952.50 | 960.58 | 958.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 957.95 | 964.69 | 961.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:00:00 | 957.95 | 964.69 | 961.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 974.45 | 966.64 | 962.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:30:00 | 957.35 | 966.64 | 962.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 967.95 | 966.96 | 963.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:15:00 | 964.05 | 966.96 | 963.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 973.15 | 968.20 | 964.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 12:00:00 | 979.20 | 971.49 | 966.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 14:45:00 | 979.55 | 975.56 | 969.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-27 09:15:00 | 980.30 | 975.95 | 970.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-27 10:00:00 | 980.50 | 976.86 | 971.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 943.50 | 972.87 | 972.81 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-28 09:15:00 | 943.50 | 972.87 | 972.81 | SL hit (close<static) qty=1.00 sl=961.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 10:15:00 | 935.25 | 965.34 | 969.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 09:15:00 | 929.60 | 945.11 | 956.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 14:15:00 | 942.85 | 940.02 | 948.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 14:45:00 | 942.95 | 940.02 | 948.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 952.80 | 942.57 | 948.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 952.80 | 942.57 | 948.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 947.70 | 943.60 | 948.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:45:00 | 948.15 | 943.60 | 948.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 944.35 | 943.75 | 947.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 12:15:00 | 948.15 | 943.75 | 947.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 12:15:00 | 946.15 | 944.23 | 947.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 12:30:00 | 944.85 | 944.23 | 947.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 945.45 | 944.88 | 947.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 15:15:00 | 949.00 | 944.88 | 947.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 15:15:00 | 949.00 | 945.71 | 947.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 09:15:00 | 923.35 | 945.71 | 947.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-06 11:15:00 | 954.75 | 941.87 | 941.91 | SL hit (close>static) qty=1.00 sl=949.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 12:15:00 | 958.15 | 945.13 | 943.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 13:15:00 | 965.95 | 949.29 | 945.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 10:15:00 | 966.45 | 971.61 | 963.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 11:00:00 | 966.45 | 971.61 | 963.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 960.00 | 969.28 | 963.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 960.00 | 969.28 | 963.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 964.25 | 968.28 | 963.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:30:00 | 957.65 | 968.28 | 963.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 959.80 | 966.58 | 963.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 13:45:00 | 964.05 | 966.58 | 963.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 943.30 | 961.93 | 961.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 15:00:00 | 943.30 | 961.93 | 961.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 942.00 | 957.94 | 959.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 928.25 | 952.00 | 956.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 12:15:00 | 932.00 | 929.72 | 938.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 13:00:00 | 932.00 | 929.72 | 938.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 945.35 | 932.84 | 938.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 13:45:00 | 945.45 | 932.84 | 938.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 941.85 | 934.64 | 939.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 14:45:00 | 946.95 | 934.64 | 939.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 934.10 | 934.91 | 938.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:45:00 | 926.55 | 932.54 | 936.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 13:30:00 | 927.40 | 930.62 | 933.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 946.70 | 931.62 | 932.78 | SL hit (close>static) qty=1.00 sl=946.45 alert=retest2 |

### Cycle 63 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 947.00 | 934.70 | 934.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 10:15:00 | 958.25 | 950.83 | 944.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 15:15:00 | 984.00 | 986.92 | 973.86 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:15:00 | 1022.85 | 986.92 | 973.86 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 14:15:00 | 1073.99 | 1018.91 | 996.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-03-25 11:15:00 | 1057.75 | 1066.32 | 1044.89 | SL hit (close<ema200) qty=0.50 sl=1066.32 alert=retest1 |

### Cycle 64 — SELL (started 2025-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 15:15:00 | 1040.00 | 1044.27 | 1044.54 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 11:15:00 | 1059.90 | 1046.73 | 1045.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 13:15:00 | 1067.00 | 1052.63 | 1048.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 11:15:00 | 1057.00 | 1058.25 | 1053.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-28 11:45:00 | 1057.05 | 1058.25 | 1053.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 1050.25 | 1056.65 | 1053.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 12:45:00 | 1051.15 | 1056.65 | 1053.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 1054.85 | 1056.29 | 1053.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 13:15:00 | 1063.00 | 1057.15 | 1054.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 14:15:00 | 1063.40 | 1056.64 | 1056.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 14:45:00 | 1067.95 | 1058.71 | 1057.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 09:15:00 | 1071.90 | 1059.37 | 1057.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 1062.45 | 1059.99 | 1058.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 15:00:00 | 1083.95 | 1068.84 | 1063.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 14:15:00 | 1057.00 | 1061.28 | 1061.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-04-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 14:15:00 | 1057.00 | 1061.28 | 1061.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 15:15:00 | 1056.00 | 1060.22 | 1061.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 1031.45 | 1026.63 | 1037.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 1031.45 | 1026.63 | 1037.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1031.45 | 1026.63 | 1037.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:15:00 | 1021.00 | 1027.53 | 1037.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 12:30:00 | 1028.00 | 1028.19 | 1035.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 13:30:00 | 1025.35 | 1026.40 | 1034.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 13:15:00 | 1017.40 | 1010.93 | 1010.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 13:15:00 | 1017.40 | 1010.93 | 1010.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 14:15:00 | 1019.80 | 1012.70 | 1011.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 1019.90 | 1023.78 | 1019.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 1019.90 | 1023.78 | 1019.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1019.90 | 1023.78 | 1019.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:15:00 | 1020.10 | 1023.78 | 1019.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 1025.20 | 1024.06 | 1020.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:45:00 | 1026.60 | 1023.99 | 1020.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 14:00:00 | 1028.70 | 1024.93 | 1021.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-22 10:15:00 | 1129.26 | 1081.46 | 1058.34 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 12:15:00 | 1084.20 | 1102.28 | 1103.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 13:15:00 | 1079.90 | 1097.80 | 1101.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 15:15:00 | 1085.00 | 1082.57 | 1088.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-29 09:15:00 | 1096.40 | 1082.57 | 1088.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 1092.90 | 1084.64 | 1089.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:30:00 | 1089.80 | 1084.64 | 1089.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 1081.00 | 1083.91 | 1088.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 11:15:00 | 1077.50 | 1083.91 | 1088.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 11:45:00 | 1078.00 | 1082.71 | 1087.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 14:15:00 | 1023.62 | 1051.61 | 1067.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 14:15:00 | 1024.10 | 1051.61 | 1067.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 1027.60 | 1026.15 | 1041.49 | SL hit (close>ema200) qty=0.50 sl=1026.15 alert=retest2 |

### Cycle 69 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1024.00 | 997.61 | 997.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 1045.00 | 1018.79 | 1008.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 09:15:00 | 1177.00 | 1177.93 | 1165.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 11:15:00 | 1153.00 | 1171.69 | 1165.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 1153.00 | 1171.69 | 1165.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:00:00 | 1153.00 | 1171.69 | 1165.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 1162.90 | 1169.93 | 1164.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 13:30:00 | 1170.00 | 1169.75 | 1165.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 10:30:00 | 1172.50 | 1173.29 | 1168.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 11:15:00 | 1188.00 | 1194.42 | 1194.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 11:15:00 | 1188.00 | 1194.42 | 1194.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 12:15:00 | 1182.50 | 1192.04 | 1193.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 10:15:00 | 1191.80 | 1184.50 | 1188.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 10:15:00 | 1191.80 | 1184.50 | 1188.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1191.80 | 1184.50 | 1188.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:00:00 | 1191.80 | 1184.50 | 1188.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 1198.50 | 1187.30 | 1189.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:00:00 | 1198.50 | 1187.30 | 1189.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 1188.50 | 1187.97 | 1189.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 13:30:00 | 1191.40 | 1187.97 | 1189.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 1191.50 | 1188.68 | 1189.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 1191.50 | 1188.68 | 1189.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 1189.00 | 1188.74 | 1189.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 1182.90 | 1188.74 | 1189.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 1193.00 | 1189.59 | 1189.72 | SL hit (close>static) qty=1.00 sl=1192.10 alert=retest2 |

### Cycle 71 — BUY (started 2025-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 14:15:00 | 1196.00 | 1190.76 | 1190.07 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 1171.40 | 1187.70 | 1188.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 11:15:00 | 1155.10 | 1176.91 | 1183.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 11:15:00 | 1162.10 | 1161.54 | 1170.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 11:45:00 | 1164.00 | 1161.54 | 1170.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 1157.80 | 1158.59 | 1165.38 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 09:15:00 | 1234.90 | 1179.60 | 1172.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 10:15:00 | 1259.50 | 1229.17 | 1206.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 1252.90 | 1256.13 | 1232.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 09:45:00 | 1252.00 | 1256.13 | 1232.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1248.10 | 1254.88 | 1243.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:45:00 | 1243.00 | 1254.88 | 1243.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 1257.40 | 1263.18 | 1255.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 1257.40 | 1263.18 | 1255.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1249.70 | 1260.49 | 1254.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 1249.70 | 1260.49 | 1254.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1257.40 | 1259.87 | 1254.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 13:15:00 | 1258.80 | 1259.87 | 1254.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 10:15:00 | 1245.20 | 1253.18 | 1253.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 10:15:00 | 1245.20 | 1253.18 | 1253.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 12:15:00 | 1236.50 | 1248.60 | 1251.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 1244.40 | 1241.38 | 1246.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 1244.40 | 1241.38 | 1246.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1244.40 | 1241.38 | 1246.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:00:00 | 1244.40 | 1241.38 | 1246.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 1257.00 | 1244.51 | 1247.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 1257.00 | 1244.51 | 1247.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 1258.20 | 1247.25 | 1248.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:30:00 | 1271.90 | 1247.25 | 1248.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 12:15:00 | 1258.60 | 1249.52 | 1249.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 13:15:00 | 1270.00 | 1253.61 | 1251.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 10:15:00 | 1279.90 | 1280.88 | 1271.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 11:00:00 | 1279.90 | 1280.88 | 1271.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 1273.40 | 1279.38 | 1271.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:45:00 | 1269.80 | 1279.38 | 1271.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 1267.30 | 1276.97 | 1271.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:00:00 | 1267.30 | 1276.97 | 1271.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 1253.80 | 1272.33 | 1269.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 14:00:00 | 1253.80 | 1272.33 | 1269.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2025-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 14:15:00 | 1250.00 | 1267.87 | 1267.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 13:15:00 | 1237.80 | 1254.62 | 1260.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 1252.00 | 1249.80 | 1256.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 10:00:00 | 1252.00 | 1249.80 | 1256.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1261.70 | 1252.18 | 1256.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 1261.70 | 1252.18 | 1256.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1250.70 | 1251.88 | 1256.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 1246.20 | 1251.88 | 1256.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 1244.30 | 1253.32 | 1255.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 10:15:00 | 1245.60 | 1252.28 | 1255.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 11:00:00 | 1247.70 | 1251.36 | 1254.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 1245.00 | 1250.09 | 1253.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:45:00 | 1242.90 | 1250.09 | 1253.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 1275.00 | 1252.93 | 1253.15 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 1275.00 | 1252.93 | 1253.15 | SL hit (close>static) qty=1.00 sl=1265.00 alert=retest2 |

### Cycle 77 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 1266.80 | 1255.70 | 1254.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 12:15:00 | 1283.30 | 1264.08 | 1258.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 1264.80 | 1264.97 | 1260.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 15:00:00 | 1264.80 | 1264.97 | 1260.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 1267.00 | 1265.37 | 1260.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 1282.80 | 1265.37 | 1260.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 1283.30 | 1277.62 | 1270.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:00:00 | 1282.80 | 1275.46 | 1272.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 13:15:00 | 1251.00 | 1268.85 | 1270.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 13:15:00 | 1251.00 | 1268.85 | 1270.58 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 11:15:00 | 1287.60 | 1271.68 | 1270.68 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 11:15:00 | 1273.90 | 1277.08 | 1277.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 12:15:00 | 1264.70 | 1274.61 | 1276.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 10:15:00 | 1277.40 | 1270.93 | 1273.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 10:15:00 | 1277.40 | 1270.93 | 1273.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 1277.40 | 1270.93 | 1273.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:00:00 | 1277.40 | 1270.93 | 1273.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 1282.90 | 1273.32 | 1274.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:00:00 | 1282.90 | 1273.32 | 1274.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 13:15:00 | 1285.10 | 1276.71 | 1275.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 14:15:00 | 1285.90 | 1278.55 | 1276.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 09:15:00 | 1332.00 | 1332.14 | 1318.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 09:45:00 | 1329.60 | 1332.14 | 1318.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1378.70 | 1389.21 | 1379.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:00:00 | 1378.70 | 1389.21 | 1379.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1383.10 | 1387.98 | 1379.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:30:00 | 1378.30 | 1387.98 | 1379.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 1386.10 | 1387.61 | 1380.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:30:00 | 1379.30 | 1387.61 | 1380.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 1382.40 | 1386.57 | 1380.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:00:00 | 1382.40 | 1386.57 | 1380.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 1381.00 | 1385.45 | 1380.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:45:00 | 1379.20 | 1385.45 | 1380.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 1400.80 | 1388.52 | 1382.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 15:15:00 | 1411.00 | 1388.52 | 1382.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 10:00:00 | 1410.00 | 1396.41 | 1387.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 10:45:00 | 1408.60 | 1398.61 | 1388.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 1390.40 | 1455.63 | 1459.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 09:15:00 | 1390.40 | 1455.63 | 1459.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 1357.20 | 1407.12 | 1428.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 1381.70 | 1368.38 | 1392.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 10:00:00 | 1381.70 | 1368.38 | 1392.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 1346.30 | 1329.66 | 1340.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 15:15:00 | 1328.60 | 1339.22 | 1341.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 09:45:00 | 1328.50 | 1313.87 | 1317.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 11:15:00 | 1332.40 | 1318.90 | 1319.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 11:15:00 | 1321.40 | 1319.40 | 1319.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 11:15:00 | 1321.40 | 1319.40 | 1319.27 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 12:15:00 | 1313.10 | 1318.14 | 1318.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 14:15:00 | 1303.90 | 1313.88 | 1316.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 13:15:00 | 1305.00 | 1304.97 | 1309.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 13:15:00 | 1305.00 | 1304.97 | 1309.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 1305.00 | 1304.97 | 1309.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 13:45:00 | 1309.60 | 1304.97 | 1309.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 1302.00 | 1304.38 | 1309.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:30:00 | 1306.00 | 1304.38 | 1309.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 1321.00 | 1307.70 | 1310.25 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 11:15:00 | 1326.60 | 1314.47 | 1312.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 14:15:00 | 1340.00 | 1321.87 | 1316.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 13:15:00 | 1328.40 | 1329.27 | 1323.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-08 13:45:00 | 1329.80 | 1329.27 | 1323.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 1323.40 | 1328.10 | 1323.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 1323.40 | 1328.10 | 1323.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 1318.80 | 1326.24 | 1323.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:15:00 | 1324.40 | 1326.24 | 1323.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 1313.60 | 1323.71 | 1322.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:45:00 | 1311.20 | 1323.71 | 1322.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 1316.40 | 1322.25 | 1321.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:15:00 | 1312.20 | 1322.25 | 1321.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 11:15:00 | 1311.70 | 1320.14 | 1320.89 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 15:15:00 | 1330.00 | 1322.55 | 1321.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 10:15:00 | 1333.00 | 1325.66 | 1323.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 13:15:00 | 1335.80 | 1340.42 | 1334.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 13:45:00 | 1337.40 | 1340.42 | 1334.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 1328.70 | 1338.08 | 1333.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 1328.70 | 1338.08 | 1333.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 1331.00 | 1336.66 | 1333.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 1333.00 | 1336.66 | 1333.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 10:15:00 | 1322.90 | 1332.25 | 1331.94 | SL hit (close<static) qty=1.00 sl=1325.90 alert=retest2 |

### Cycle 88 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 1326.00 | 1331.00 | 1331.40 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 15:15:00 | 1333.20 | 1331.81 | 1331.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 1347.60 | 1334.96 | 1333.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 1387.70 | 1394.04 | 1377.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 09:30:00 | 1393.50 | 1394.04 | 1377.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1383.80 | 1389.38 | 1382.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:45:00 | 1382.50 | 1389.38 | 1382.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1386.80 | 1388.86 | 1383.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 10:00:00 | 1393.00 | 1384.43 | 1382.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 14:15:00 | 1375.30 | 1382.35 | 1382.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 1375.30 | 1382.35 | 1382.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 15:15:00 | 1369.00 | 1379.68 | 1381.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 14:15:00 | 1355.00 | 1345.12 | 1356.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 14:15:00 | 1355.00 | 1345.12 | 1356.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 1355.00 | 1345.12 | 1356.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 1355.00 | 1345.12 | 1356.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 1336.90 | 1343.48 | 1354.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 1325.90 | 1343.48 | 1354.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 09:15:00 | 1339.20 | 1310.53 | 1307.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 1339.20 | 1310.53 | 1307.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 14:15:00 | 1348.80 | 1326.73 | 1317.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 1343.60 | 1346.96 | 1334.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 14:45:00 | 1345.30 | 1346.96 | 1334.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1336.90 | 1344.40 | 1335.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:15:00 | 1335.30 | 1344.40 | 1335.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1331.20 | 1341.76 | 1335.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:45:00 | 1330.10 | 1341.76 | 1335.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 1327.30 | 1338.87 | 1334.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:00:00 | 1327.30 | 1338.87 | 1334.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 1322.50 | 1332.34 | 1332.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:30:00 | 1330.60 | 1332.34 | 1332.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 15:15:00 | 1322.50 | 1330.38 | 1331.44 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 1343.20 | 1332.94 | 1332.51 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 09:15:00 | 1325.00 | 1332.48 | 1332.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 10:15:00 | 1317.60 | 1329.50 | 1331.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 11:15:00 | 1332.80 | 1330.16 | 1331.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 11:15:00 | 1332.80 | 1330.16 | 1331.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 1332.80 | 1330.16 | 1331.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:45:00 | 1332.90 | 1330.16 | 1331.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 1337.10 | 1331.55 | 1332.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:30:00 | 1337.90 | 1331.55 | 1332.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 13:15:00 | 1339.50 | 1333.14 | 1332.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 14:15:00 | 1341.50 | 1334.81 | 1333.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 12:15:00 | 1342.60 | 1344.09 | 1339.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 13:00:00 | 1342.60 | 1344.09 | 1339.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 1339.00 | 1342.53 | 1339.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:15:00 | 1329.60 | 1342.53 | 1339.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 1342.10 | 1342.44 | 1339.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:30:00 | 1329.30 | 1342.44 | 1339.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1342.20 | 1342.39 | 1340.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:30:00 | 1341.70 | 1342.39 | 1340.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 1353.40 | 1349.96 | 1346.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 15:15:00 | 1355.00 | 1350.48 | 1347.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 13:15:00 | 1344.90 | 1348.81 | 1348.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 1344.90 | 1348.81 | 1348.99 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 14:15:00 | 1364.00 | 1351.85 | 1350.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 15:15:00 | 1375.00 | 1356.48 | 1352.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 09:15:00 | 1365.10 | 1366.04 | 1360.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 1365.10 | 1366.04 | 1360.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1365.10 | 1366.04 | 1360.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:30:00 | 1365.10 | 1366.04 | 1360.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 1362.00 | 1365.23 | 1361.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 11:00:00 | 1362.00 | 1365.23 | 1361.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 1362.70 | 1364.72 | 1361.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 11:30:00 | 1362.40 | 1364.72 | 1361.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 1360.80 | 1363.94 | 1361.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:30:00 | 1360.90 | 1363.94 | 1361.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 1359.40 | 1363.03 | 1360.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:00:00 | 1359.40 | 1363.03 | 1360.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1356.20 | 1361.67 | 1360.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 09:45:00 | 1376.00 | 1365.37 | 1362.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 1362.10 | 1381.78 | 1383.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 1362.10 | 1381.78 | 1383.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 10:15:00 | 1352.30 | 1375.89 | 1380.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 15:15:00 | 1306.00 | 1305.44 | 1317.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-01 09:15:00 | 1312.90 | 1305.44 | 1317.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 1313.50 | 1308.56 | 1316.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 13:30:00 | 1310.00 | 1314.58 | 1317.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 14:45:00 | 1310.00 | 1313.66 | 1316.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 1328.80 | 1314.82 | 1316.80 | SL hit (close>static) qty=1.00 sl=1321.40 alert=retest2 |

### Cycle 99 — BUY (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 11:15:00 | 1320.70 | 1313.92 | 1313.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 14:15:00 | 1326.70 | 1318.40 | 1315.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 1315.00 | 1319.82 | 1316.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 1315.00 | 1319.82 | 1316.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1315.00 | 1319.82 | 1316.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 1315.00 | 1319.82 | 1316.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1308.10 | 1317.47 | 1316.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:45:00 | 1309.40 | 1317.47 | 1316.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 1306.50 | 1315.28 | 1315.25 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 12:15:00 | 1300.40 | 1312.30 | 1313.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 1289.10 | 1305.47 | 1310.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 1300.60 | 1287.67 | 1295.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 1300.60 | 1287.67 | 1295.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1300.60 | 1287.67 | 1295.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 1305.40 | 1287.67 | 1295.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 1305.20 | 1291.18 | 1296.16 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 1308.30 | 1300.05 | 1299.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 09:15:00 | 1352.00 | 1313.29 | 1305.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 14:15:00 | 1364.50 | 1369.61 | 1351.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 15:00:00 | 1364.50 | 1369.61 | 1351.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1356.80 | 1366.04 | 1354.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 1354.30 | 1366.04 | 1354.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 1366.00 | 1366.03 | 1355.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 13:30:00 | 1375.80 | 1368.50 | 1358.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 1373.30 | 1368.81 | 1360.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1300.00 | 1375.22 | 1374.76 | SL hit (close<static) qty=1.00 sl=1354.30 alert=retest2 |

### Cycle 102 — SELL (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 10:15:00 | 1297.00 | 1359.58 | 1367.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 09:15:00 | 1272.90 | 1302.18 | 1312.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 15:15:00 | 1257.00 | 1254.28 | 1272.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-31 09:15:00 | 1254.40 | 1254.28 | 1272.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1244.90 | 1247.25 | 1258.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 10:15:00 | 1242.20 | 1247.25 | 1258.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 1180.09 | 1203.75 | 1217.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 13:15:00 | 1210.50 | 1200.40 | 1210.66 | SL hit (close>ema200) qty=0.50 sl=1200.40 alert=retest2 |

### Cycle 103 — BUY (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 15:15:00 | 1176.00 | 1170.26 | 1169.84 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 09:15:00 | 1158.90 | 1167.99 | 1168.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 11:15:00 | 1153.30 | 1163.21 | 1166.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 15:15:00 | 1152.80 | 1151.14 | 1155.45 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 09:15:00 | 1143.80 | 1151.14 | 1155.45 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 12:15:00 | 1148.00 | 1147.45 | 1152.42 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 1150.00 | 1147.96 | 1152.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 13:00:00 | 1150.00 | 1147.96 | 1152.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 1150.60 | 1148.49 | 1152.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 13:30:00 | 1152.70 | 1148.49 | 1152.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 1150.20 | 1148.83 | 1151.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:45:00 | 1144.60 | 1148.13 | 1151.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 1153.60 | 1149.23 | 1151.28 | SL hit (close>ema400) qty=1.00 sl=1151.28 alert=retest1 |

### Cycle 105 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 1153.80 | 1148.03 | 1147.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 1156.80 | 1151.05 | 1148.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 1147.40 | 1151.11 | 1149.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 1147.40 | 1151.11 | 1149.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 1147.40 | 1151.11 | 1149.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:30:00 | 1145.50 | 1151.11 | 1149.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 1141.60 | 1149.21 | 1148.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 1141.60 | 1149.21 | 1148.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 11:15:00 | 1140.30 | 1147.43 | 1147.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 12:15:00 | 1134.10 | 1144.76 | 1146.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 15:15:00 | 1145.00 | 1144.23 | 1145.90 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-28 09:15:00 | 1136.90 | 1144.23 | 1145.90 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1149.00 | 1145.18 | 1146.18 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-28 09:15:00 | 1149.00 | 1145.18 | 1146.18 | SL hit (close>ema400) qty=1.00 sl=1146.18 alert=retest1 |

### Cycle 107 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 1152.70 | 1146.64 | 1145.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 15:15:00 | 1154.90 | 1149.98 | 1147.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 1138.80 | 1147.75 | 1147.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 1138.80 | 1147.75 | 1147.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1138.80 | 1147.75 | 1147.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:45:00 | 1139.10 | 1147.75 | 1147.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 1142.50 | 1146.70 | 1146.73 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 15:15:00 | 1151.60 | 1146.47 | 1146.29 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 1133.00 | 1143.77 | 1145.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 1128.00 | 1140.62 | 1143.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 10:15:00 | 1135.80 | 1134.99 | 1138.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:45:00 | 1133.90 | 1134.99 | 1138.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 1139.10 | 1135.81 | 1138.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:00:00 | 1139.10 | 1135.81 | 1138.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 1134.50 | 1135.55 | 1138.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 1130.60 | 1137.03 | 1138.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 1131.90 | 1119.28 | 1119.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 1131.90 | 1119.28 | 1119.25 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 10:15:00 | 1118.40 | 1119.10 | 1119.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 11:15:00 | 1113.20 | 1117.92 | 1118.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 10:15:00 | 1115.60 | 1110.57 | 1113.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 10:15:00 | 1115.60 | 1110.57 | 1113.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 1115.60 | 1110.57 | 1113.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 1115.60 | 1110.57 | 1113.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 1120.10 | 1112.48 | 1114.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:30:00 | 1120.20 | 1112.48 | 1114.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 1119.00 | 1115.10 | 1115.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:30:00 | 1125.70 | 1115.10 | 1115.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 1123.30 | 1116.74 | 1116.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 15:15:00 | 1128.00 | 1118.99 | 1117.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 10:15:00 | 1135.20 | 1138.74 | 1131.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 10:45:00 | 1135.20 | 1138.74 | 1131.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 1135.70 | 1138.06 | 1132.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:45:00 | 1134.30 | 1138.06 | 1132.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 13:15:00 | 1132.70 | 1136.99 | 1132.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 13:30:00 | 1131.30 | 1136.99 | 1132.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 1126.30 | 1134.85 | 1132.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:45:00 | 1123.90 | 1134.85 | 1132.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 1129.10 | 1133.70 | 1131.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 1122.30 | 1133.70 | 1131.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 1125.50 | 1130.41 | 1130.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 15:15:00 | 1119.00 | 1126.51 | 1128.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 09:15:00 | 1150.70 | 1124.93 | 1125.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 09:15:00 | 1150.70 | 1124.93 | 1125.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 1150.70 | 1124.93 | 1125.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:30:00 | 1155.80 | 1124.93 | 1125.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 10:15:00 | 1137.00 | 1127.35 | 1126.63 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 10:15:00 | 1115.20 | 1128.46 | 1128.68 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 1137.00 | 1128.78 | 1128.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 11:15:00 | 1138.10 | 1134.05 | 1132.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 12:15:00 | 1133.00 | 1133.84 | 1132.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 12:15:00 | 1133.00 | 1133.84 | 1132.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 1133.00 | 1133.84 | 1132.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 13:00:00 | 1133.00 | 1133.84 | 1132.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 1131.00 | 1133.27 | 1132.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:00:00 | 1131.00 | 1133.27 | 1132.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 1125.40 | 1131.70 | 1132.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 1115.10 | 1127.47 | 1130.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 1112.00 | 1108.28 | 1115.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 14:00:00 | 1112.00 | 1108.28 | 1115.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1103.00 | 1107.22 | 1114.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:45:00 | 1108.30 | 1107.22 | 1114.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1119.40 | 1109.22 | 1113.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:45:00 | 1119.70 | 1109.22 | 1113.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1125.00 | 1112.38 | 1114.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 1125.00 | 1112.38 | 1114.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 1125.00 | 1116.92 | 1116.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 1132.00 | 1121.28 | 1118.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 1122.50 | 1122.60 | 1119.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 10:00:00 | 1122.50 | 1122.60 | 1119.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 1122.00 | 1122.48 | 1120.01 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 13:15:00 | 1112.20 | 1117.77 | 1118.25 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 1125.60 | 1119.47 | 1118.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 13:15:00 | 1129.00 | 1122.23 | 1120.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 1127.20 | 1131.43 | 1127.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 1127.20 | 1131.43 | 1127.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1127.20 | 1131.43 | 1127.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:00:00 | 1127.20 | 1131.43 | 1127.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 1130.90 | 1131.32 | 1127.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:15:00 | 1129.00 | 1131.32 | 1127.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 1128.70 | 1130.80 | 1128.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 13:15:00 | 1131.60 | 1130.66 | 1128.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 14:45:00 | 1132.10 | 1131.05 | 1128.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 1121.10 | 1129.53 | 1128.52 | SL hit (close<static) qty=1.00 sl=1126.30 alert=retest2 |

### Cycle 122 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 1118.00 | 1128.26 | 1128.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 1115.60 | 1125.73 | 1127.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 1073.20 | 1068.46 | 1083.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 10:00:00 | 1073.20 | 1068.46 | 1083.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 1078.30 | 1073.65 | 1080.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:45:00 | 1081.60 | 1073.65 | 1080.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 1081.80 | 1075.28 | 1080.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 1072.90 | 1074.86 | 1079.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:00:00 | 1073.60 | 1075.03 | 1079.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:30:00 | 1076.80 | 1076.14 | 1078.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:15:00 | 1075.00 | 1077.13 | 1079.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 1075.00 | 1076.71 | 1078.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:15:00 | 1097.00 | 1076.71 | 1078.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 1100.90 | 1081.54 | 1080.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 1100.90 | 1081.54 | 1080.77 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 1080.30 | 1086.74 | 1087.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 1065.50 | 1082.49 | 1085.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1059.10 | 1046.89 | 1059.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 1059.10 | 1046.89 | 1059.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1059.10 | 1046.89 | 1059.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 1058.30 | 1046.89 | 1059.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1048.70 | 1047.25 | 1058.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 1048.70 | 1047.25 | 1058.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 988.40 | 972.39 | 985.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:15:00 | 986.00 | 972.39 | 985.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 987.90 | 975.49 | 985.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 13:15:00 | 984.50 | 975.49 | 985.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 990.80 | 980.36 | 986.48 | SL hit (close>static) qty=1.00 sl=990.00 alert=retest2 |

### Cycle 125 — BUY (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 14:15:00 | 987.65 | 975.40 | 975.39 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 973.05 | 975.84 | 975.88 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 12:15:00 | 977.45 | 976.17 | 976.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 13:15:00 | 980.90 | 977.11 | 976.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1046.10 | 1052.32 | 1035.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:45:00 | 1047.45 | 1052.32 | 1035.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1045.00 | 1060.54 | 1049.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:30:00 | 1042.00 | 1060.54 | 1049.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1050.00 | 1058.43 | 1049.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:30:00 | 1056.05 | 1057.39 | 1049.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 1060.00 | 1082.18 | 1084.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 1060.00 | 1082.18 | 1084.04 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 1082.80 | 1074.33 | 1073.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 15:15:00 | 1085.50 | 1078.50 | 1075.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 1078.05 | 1078.41 | 1076.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 1078.05 | 1078.41 | 1076.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1078.05 | 1078.41 | 1076.16 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 1063.90 | 1074.51 | 1075.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 1060.40 | 1068.90 | 1072.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 1065.00 | 1064.54 | 1069.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 10:15:00 | 1065.00 | 1064.54 | 1069.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 1065.00 | 1064.54 | 1069.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:30:00 | 1069.00 | 1064.54 | 1069.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 1066.00 | 1065.02 | 1068.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:45:00 | 1068.35 | 1065.02 | 1068.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 1064.90 | 1064.99 | 1068.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:45:00 | 1067.40 | 1064.99 | 1068.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 1068.45 | 1065.69 | 1068.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 15:00:00 | 1068.45 | 1065.69 | 1068.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 1070.00 | 1066.55 | 1068.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 1073.55 | 1066.55 | 1068.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1075.85 | 1068.41 | 1069.11 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 1073.40 | 1069.86 | 1069.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 12:15:00 | 1083.70 | 1072.63 | 1070.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 13:15:00 | 1069.40 | 1071.98 | 1070.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 13:15:00 | 1069.40 | 1071.98 | 1070.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 1069.40 | 1071.98 | 1070.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:00:00 | 1069.40 | 1071.98 | 1070.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 1082.95 | 1074.18 | 1071.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:30:00 | 1071.45 | 1074.18 | 1071.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 1063.80 | 1073.03 | 1071.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 1061.55 | 1073.03 | 1071.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 10:15:00 | 1057.40 | 1069.91 | 1070.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 11:15:00 | 1051.15 | 1066.16 | 1068.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 1064.60 | 1058.35 | 1063.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 1064.60 | 1058.35 | 1063.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1064.60 | 1058.35 | 1063.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 1064.60 | 1058.35 | 1063.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 1058.30 | 1058.34 | 1062.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:15:00 | 1056.90 | 1058.34 | 1062.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:45:00 | 1052.20 | 1057.64 | 1061.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 13:15:00 | 1004.06 | 1018.03 | 1031.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 14:15:00 | 999.59 | 1011.21 | 1027.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 951.21 | 975.41 | 995.94 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 133 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 992.10 | 965.76 | 965.26 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 962.85 | 974.16 | 975.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 954.75 | 968.06 | 972.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 13:15:00 | 952.45 | 951.37 | 959.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:00:00 | 952.45 | 951.37 | 959.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 960.20 | 953.14 | 959.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 961.55 | 953.14 | 959.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 962.00 | 954.91 | 959.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 966.05 | 954.91 | 959.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 973.90 | 958.71 | 961.13 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 976.35 | 964.64 | 963.56 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 13:15:00 | 954.50 | 961.98 | 962.50 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 971.85 | 963.73 | 963.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 975.35 | 967.08 | 964.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 955.00 | 967.16 | 965.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 955.00 | 967.16 | 965.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 955.00 | 967.16 | 965.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:15:00 | 955.70 | 967.16 | 965.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 951.85 | 964.09 | 964.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 950.00 | 959.76 | 962.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 922.15 | 915.49 | 926.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 922.15 | 915.49 | 926.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 928.80 | 918.15 | 926.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 929.55 | 918.15 | 926.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 934.60 | 921.44 | 927.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 934.60 | 921.44 | 927.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 928.80 | 924.04 | 927.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 951.45 | 924.04 | 927.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 958.00 | 930.83 | 930.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 974.25 | 939.52 | 934.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 948.00 | 955.41 | 946.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:00:00 | 948.00 | 955.41 | 946.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 944.65 | 953.26 | 946.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 944.65 | 953.26 | 946.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 941.60 | 950.93 | 946.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 941.60 | 950.93 | 946.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 945.90 | 949.92 | 946.01 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 924.85 | 940.45 | 942.40 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 960.55 | 939.52 | 939.48 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 932.35 | 943.06 | 943.22 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 963.25 | 940.47 | 940.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 993.20 | 965.97 | 961.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 13:15:00 | 1032.15 | 1033.03 | 1019.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 14:00:00 | 1032.15 | 1033.03 | 1019.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 1039.15 | 1034.33 | 1023.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:15:00 | 1042.80 | 1034.33 | 1023.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 15:15:00 | 1042.65 | 1055.54 | 1054.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 15:15:00 | 1042.65 | 1052.96 | 1053.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 1042.65 | 1052.96 | 1053.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 982.00 | 1038.77 | 1046.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 966.45 | 962.19 | 994.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 10:00:00 | 966.45 | 962.19 | 994.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 958.00 | 951.79 | 956.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:30:00 | 962.35 | 951.79 | 956.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 959.20 | 953.27 | 956.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:30:00 | 957.00 | 953.27 | 956.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 958.95 | 954.41 | 957.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:45:00 | 960.80 | 954.41 | 957.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 957.65 | 954.56 | 956.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 15:00:00 | 957.65 | 954.56 | 956.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 15:15:00 | 956.00 | 954.85 | 956.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:15:00 | 950.00 | 954.85 | 956.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 953.65 | 954.61 | 956.21 | EMA400 retest candle locked (from downside) |

### Cycle 145 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 964.30 | 955.85 | 955.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 10:15:00 | 974.15 | 962.74 | 959.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 12:15:00 | 972.50 | 973.49 | 968.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 13:00:00 | 972.50 | 973.49 | 968.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-17 09:30:00 | 918.75 | 2024-05-27 12:15:00 | 922.80 | STOP_HIT | 1.00 | 0.44% |
| BUY | retest2 | 2024-05-17 14:00:00 | 917.05 | 2024-05-27 12:15:00 | 922.80 | STOP_HIT | 1.00 | 0.63% |
| BUY | retest2 | 2024-05-22 09:45:00 | 919.45 | 2024-05-27 12:15:00 | 922.80 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2024-06-06 09:15:00 | 951.25 | 2024-06-18 10:15:00 | 1046.38 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-27 10:30:00 | 1005.70 | 2024-07-01 09:15:00 | 1036.00 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2024-06-27 12:00:00 | 1005.00 | 2024-07-01 09:15:00 | 1036.00 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2024-06-28 09:45:00 | 1000.00 | 2024-07-01 09:15:00 | 1036.00 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2024-07-09 11:15:00 | 1047.00 | 2024-07-10 09:15:00 | 1035.50 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-07-09 12:30:00 | 1047.00 | 2024-07-10 09:15:00 | 1035.50 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-07-10 09:15:00 | 1047.85 | 2024-07-10 09:15:00 | 1035.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest1 | 2024-08-06 10:30:00 | 984.55 | 2024-08-08 09:15:00 | 1007.00 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest1 | 2024-08-06 12:45:00 | 987.30 | 2024-08-08 09:15:00 | 1007.00 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest1 | 2024-08-06 13:15:00 | 988.15 | 2024-08-08 09:15:00 | 1007.00 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest1 | 2024-08-07 09:45:00 | 987.50 | 2024-08-08 09:15:00 | 1007.00 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest1 | 2024-08-13 09:15:00 | 1033.35 | 2024-08-13 14:15:00 | 1016.00 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-08-14 09:15:00 | 1039.05 | 2024-08-20 15:15:00 | 1128.55 | TARGET_HIT | 1.00 | 8.61% |
| BUY | retest2 | 2024-08-14 11:45:00 | 1025.95 | 2024-08-21 09:15:00 | 1142.96 | TARGET_HIT | 1.00 | 11.40% |
| BUY | retest2 | 2024-09-10 09:15:00 | 1280.10 | 2024-09-12 15:15:00 | 1270.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-09-26 09:30:00 | 1240.00 | 2024-10-07 09:15:00 | 1194.01 | PARTIAL | 0.50 | 3.71% |
| SELL | retest2 | 2024-09-27 10:00:00 | 1256.85 | 2024-10-07 09:15:00 | 1193.44 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2024-09-27 12:00:00 | 1256.25 | 2024-10-07 09:15:00 | 1188.50 | PARTIAL | 0.50 | 5.39% |
| SELL | retest2 | 2024-09-30 09:30:00 | 1251.05 | 2024-10-07 10:15:00 | 1178.00 | PARTIAL | 0.50 | 5.84% |
| SELL | retest2 | 2024-09-26 09:30:00 | 1240.00 | 2024-10-08 14:15:00 | 1199.50 | STOP_HIT | 0.50 | 3.27% |
| SELL | retest2 | 2024-09-27 10:00:00 | 1256.85 | 2024-10-08 14:15:00 | 1199.50 | STOP_HIT | 0.50 | 4.56% |
| SELL | retest2 | 2024-09-27 12:00:00 | 1256.25 | 2024-10-08 14:15:00 | 1199.50 | STOP_HIT | 0.50 | 4.52% |
| SELL | retest2 | 2024-09-30 09:30:00 | 1251.05 | 2024-10-08 14:15:00 | 1199.50 | STOP_HIT | 0.50 | 4.12% |
| SELL | retest2 | 2024-10-04 09:15:00 | 1205.45 | 2024-10-09 11:15:00 | 1231.00 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2024-10-04 11:00:00 | 1207.10 | 2024-10-09 11:15:00 | 1231.00 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2024-10-04 12:30:00 | 1209.70 | 2024-10-09 11:15:00 | 1231.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-10-04 14:45:00 | 1210.00 | 2024-10-09 11:15:00 | 1231.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-10-08 09:15:00 | 1177.25 | 2024-10-09 11:15:00 | 1231.00 | STOP_HIT | 1.00 | -4.57% |
| SELL | retest2 | 2024-10-15 11:15:00 | 1232.80 | 2024-10-15 11:15:00 | 1232.65 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2024-10-21 09:15:00 | 1310.65 | 2024-10-21 15:15:00 | 1274.50 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2024-10-29 10:30:00 | 1175.20 | 2024-10-30 09:15:00 | 1254.75 | STOP_HIT | 1.00 | -6.77% |
| BUY | retest2 | 2024-11-06 09:15:00 | 1342.40 | 2024-11-08 13:15:00 | 1326.05 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-11-06 10:45:00 | 1332.60 | 2024-11-08 13:15:00 | 1326.05 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2024-11-12 12:00:00 | 1319.20 | 2024-11-12 13:15:00 | 1345.25 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2024-11-18 09:15:00 | 1274.00 | 2024-11-22 09:15:00 | 1316.30 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2024-11-19 10:30:00 | 1285.50 | 2024-11-22 09:15:00 | 1316.30 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2024-11-21 15:15:00 | 1290.00 | 2024-11-22 09:15:00 | 1316.30 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-12-02 13:15:00 | 1300.10 | 2024-12-13 09:15:00 | 1348.65 | STOP_HIT | 1.00 | 3.73% |
| BUY | retest2 | 2024-12-03 09:15:00 | 1330.00 | 2024-12-13 09:15:00 | 1348.65 | STOP_HIT | 1.00 | 1.40% |
| SELL | retest2 | 2025-01-08 10:30:00 | 1274.90 | 2025-01-09 13:15:00 | 1211.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 11:15:00 | 1269.95 | 2025-01-09 13:15:00 | 1206.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 10:30:00 | 1274.90 | 2025-01-10 12:15:00 | 1206.55 | STOP_HIT | 0.50 | 5.36% |
| SELL | retest2 | 2025-01-08 11:15:00 | 1269.95 | 2025-01-10 12:15:00 | 1206.55 | STOP_HIT | 0.50 | 4.99% |
| SELL | retest2 | 2025-02-19 13:15:00 | 941.35 | 2025-02-20 09:15:00 | 965.05 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-02-25 12:00:00 | 979.20 | 2025-02-28 09:15:00 | 943.50 | STOP_HIT | 1.00 | -3.65% |
| BUY | retest2 | 2025-02-25 14:45:00 | 979.55 | 2025-02-28 09:15:00 | 943.50 | STOP_HIT | 1.00 | -3.68% |
| BUY | retest2 | 2025-02-27 09:15:00 | 980.30 | 2025-02-28 09:15:00 | 943.50 | STOP_HIT | 1.00 | -3.75% |
| BUY | retest2 | 2025-02-27 10:00:00 | 980.50 | 2025-02-28 09:15:00 | 943.50 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest2 | 2025-03-05 09:15:00 | 923.35 | 2025-03-06 11:15:00 | 954.75 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2025-03-13 11:45:00 | 926.55 | 2025-03-18 09:15:00 | 946.70 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-03-17 13:30:00 | 927.40 | 2025-03-18 09:15:00 | 946.70 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest1 | 2025-03-21 09:15:00 | 1022.85 | 2025-03-21 14:15:00 | 1073.99 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-03-21 09:15:00 | 1022.85 | 2025-03-25 11:15:00 | 1057.75 | STOP_HIT | 0.50 | 3.41% |
| BUY | retest2 | 2025-04-01 13:15:00 | 1063.00 | 2025-04-04 14:15:00 | 1057.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-04-02 14:15:00 | 1063.40 | 2025-04-04 14:15:00 | 1057.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-04-02 14:45:00 | 1067.95 | 2025-04-04 14:15:00 | 1057.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-04-03 09:15:00 | 1071.90 | 2025-04-04 14:15:00 | 1057.00 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-04-03 15:00:00 | 1083.95 | 2025-04-04 14:15:00 | 1057.00 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2025-04-08 11:15:00 | 1021.00 | 2025-04-15 13:15:00 | 1017.40 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2025-04-08 12:30:00 | 1028.00 | 2025-04-15 13:15:00 | 1017.40 | STOP_HIT | 1.00 | 1.03% |
| SELL | retest2 | 2025-04-08 13:30:00 | 1025.35 | 2025-04-15 13:15:00 | 1017.40 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2025-04-17 12:45:00 | 1026.60 | 2025-04-22 10:15:00 | 1129.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-17 14:00:00 | 1028.70 | 2025-04-22 10:15:00 | 1131.57 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-29 11:15:00 | 1077.50 | 2025-04-30 14:15:00 | 1023.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-29 11:45:00 | 1078.00 | 2025-04-30 14:15:00 | 1024.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-29 11:15:00 | 1077.50 | 2025-05-05 09:15:00 | 1027.60 | STOP_HIT | 0.50 | 4.63% |
| SELL | retest2 | 2025-04-29 11:45:00 | 1078.00 | 2025-05-05 09:15:00 | 1027.60 | STOP_HIT | 0.50 | 4.68% |
| BUY | retest2 | 2025-05-22 13:30:00 | 1170.00 | 2025-05-29 11:15:00 | 1188.00 | STOP_HIT | 1.00 | 1.54% |
| BUY | retest2 | 2025-05-23 10:30:00 | 1172.50 | 2025-05-29 11:15:00 | 1188.00 | STOP_HIT | 1.00 | 1.32% |
| SELL | retest2 | 2025-06-02 09:15:00 | 1182.90 | 2025-06-02 09:15:00 | 1193.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-06-02 12:00:00 | 1183.00 | 2025-06-02 13:15:00 | 1193.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-06-12 13:15:00 | 1258.80 | 2025-06-13 10:15:00 | 1245.20 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-06-20 12:15:00 | 1246.20 | 2025-06-24 09:15:00 | 1275.00 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-06-23 09:15:00 | 1244.30 | 2025-06-24 09:15:00 | 1275.00 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-06-23 10:15:00 | 1245.60 | 2025-06-24 09:15:00 | 1275.00 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-06-23 11:00:00 | 1247.70 | 2025-06-24 09:15:00 | 1275.00 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-06-25 09:15:00 | 1282.80 | 2025-06-27 13:15:00 | 1251.00 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-06-26 09:15:00 | 1283.30 | 2025-06-27 13:15:00 | 1251.00 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-06-27 10:00:00 | 1282.80 | 2025-06-27 13:15:00 | 1251.00 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-07-14 15:15:00 | 1411.00 | 2025-07-24 09:15:00 | 1390.40 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-07-15 10:00:00 | 1410.00 | 2025-07-24 09:15:00 | 1390.40 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-07-15 10:45:00 | 1408.60 | 2025-07-24 09:15:00 | 1390.40 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-07-31 15:15:00 | 1328.60 | 2025-08-05 11:15:00 | 1321.40 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2025-08-05 09:45:00 | 1328.50 | 2025-08-05 11:15:00 | 1321.40 | STOP_HIT | 1.00 | 0.53% |
| SELL | retest2 | 2025-08-05 11:15:00 | 1332.40 | 2025-08-05 11:15:00 | 1321.40 | STOP_HIT | 1.00 | 0.83% |
| BUY | retest2 | 2025-08-14 09:15:00 | 1333.00 | 2025-08-14 10:15:00 | 1322.90 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-08-22 10:00:00 | 1393.00 | 2025-08-22 14:15:00 | 1375.30 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-08-28 09:15:00 | 1325.90 | 2025-09-03 09:15:00 | 1339.20 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-09-12 15:15:00 | 1355.00 | 2025-09-17 13:15:00 | 1344.90 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-09-22 09:45:00 | 1376.00 | 2025-09-25 09:15:00 | 1362.10 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-10-01 13:30:00 | 1310.00 | 2025-10-03 09:15:00 | 1328.80 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-10-01 14:45:00 | 1310.00 | 2025-10-03 09:15:00 | 1328.80 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-10-03 12:15:00 | 1309.20 | 2025-10-07 11:15:00 | 1320.70 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-10-06 09:45:00 | 1309.70 | 2025-10-07 11:15:00 | 1320.70 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-10-15 13:30:00 | 1375.80 | 2025-10-20 09:15:00 | 1300.00 | STOP_HIT | 1.00 | -5.51% |
| BUY | retest2 | 2025-10-16 09:15:00 | 1373.30 | 2025-10-20 09:15:00 | 1300.00 | STOP_HIT | 1.00 | -5.34% |
| SELL | retest2 | 2025-11-03 10:15:00 | 1242.20 | 2025-11-07 09:15:00 | 1180.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 10:15:00 | 1242.20 | 2025-11-07 13:15:00 | 1210.50 | STOP_HIT | 0.50 | 2.55% |
| SELL | retest1 | 2025-11-21 09:15:00 | 1143.80 | 2025-11-24 10:15:00 | 1153.60 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest1 | 2025-11-21 12:15:00 | 1148.00 | 2025-11-24 10:15:00 | 1153.60 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-11-24 09:45:00 | 1144.60 | 2025-11-24 10:15:00 | 1153.60 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-11-24 12:45:00 | 1145.60 | 2025-11-26 11:15:00 | 1155.50 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-11-24 13:15:00 | 1145.40 | 2025-11-26 11:15:00 | 1155.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-11-24 14:30:00 | 1143.90 | 2025-11-26 11:15:00 | 1155.50 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest1 | 2025-11-28 09:15:00 | 1136.90 | 2025-11-28 09:15:00 | 1149.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-11-28 11:15:00 | 1144.30 | 2025-12-01 09:15:00 | 1153.80 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-12-05 09:15:00 | 1130.60 | 2025-12-10 09:15:00 | 1131.90 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2026-01-06 13:15:00 | 1131.60 | 2026-01-07 09:15:00 | 1121.10 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2026-01-06 14:45:00 | 1132.10 | 2026-01-07 09:15:00 | 1121.10 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-01-07 12:30:00 | 1131.40 | 2026-01-08 10:15:00 | 1118.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2026-01-08 09:30:00 | 1132.90 | 2026-01-08 10:15:00 | 1118.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-01-14 09:30:00 | 1072.90 | 2026-01-16 09:15:00 | 1100.90 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2026-01-14 12:00:00 | 1073.60 | 2026-01-16 09:15:00 | 1100.90 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2026-01-14 13:30:00 | 1076.80 | 2026-01-16 09:15:00 | 1100.90 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2026-01-14 15:15:00 | 1075.00 | 2026-01-16 09:15:00 | 1100.90 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2026-01-28 13:15:00 | 984.50 | 2026-01-28 14:15:00 | 990.80 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2026-01-29 10:00:00 | 984.50 | 2026-02-01 14:15:00 | 987.65 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2026-02-06 11:30:00 | 1056.05 | 2026-02-13 09:15:00 | 1060.00 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2026-02-25 11:15:00 | 1056.90 | 2026-02-27 13:15:00 | 1004.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 12:45:00 | 1052.20 | 2026-02-27 14:15:00 | 999.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:15:00 | 1056.90 | 2026-03-04 09:15:00 | 951.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 12:45:00 | 1052.20 | 2026-03-04 09:15:00 | 946.98 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-04-21 10:15:00 | 1042.80 | 2026-04-23 15:15:00 | 1042.65 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2026-04-23 15:15:00 | 1042.65 | 2026-04-23 15:15:00 | 1042.65 | STOP_HIT | 1.00 | 0.00% |
