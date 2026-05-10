# Vijaya Diagnostic Centre Ltd. (VIJAYA)

## Backtest Summary

- **Window:** 2024-10-14 09:15:00 → 2026-05-08 15:15:00 (2706 bars)
- **Last close:** 1275.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 20 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 30 |
| PARTIAL | 11 |
| TARGET_HIT | 9 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 40 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 21
- **Target hits / Stop hits / Partials:** 8 / 22 / 10
- **Avg / median % per leg:** 1.92% / -0.20%
- **Sum % (uncompounded):** 76.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.91% | -11.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.91% | -11.7% |
| SELL (all) | 36 | 19 | 52.8% | 8 | 18 | 10 | 2.45% | 88.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 36 | 19 | 52.8% | 8 | 18 | 10 | 2.45% | 88.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 40 | 19 | 47.5% | 8 | 22 | 10 | 1.92% | 76.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 09:15:00 | 1002.70 | 981.13 | 981.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 11:15:00 | 1016.70 | 981.76 | 981.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 09:15:00 | 1042.70 | 1044.87 | 1022.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-11 09:45:00 | 1042.00 | 1044.87 | 1022.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1035.90 | 1047.99 | 1029.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 1036.00 | 1047.99 | 1029.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 1029.00 | 1047.67 | 1029.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 1029.00 | 1047.67 | 1029.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 1018.40 | 1047.38 | 1029.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:00:00 | 1018.40 | 1047.38 | 1029.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 1019.60 | 1047.11 | 1029.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:30:00 | 1013.00 | 1047.11 | 1029.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1018.80 | 1046.12 | 1028.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 1019.20 | 1046.12 | 1028.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 1015.30 | 1045.28 | 1028.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 13:45:00 | 1015.40 | 1045.28 | 1028.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 1020.30 | 1044.83 | 1028.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 10:00:00 | 1034.20 | 1044.72 | 1028.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 10:15:00 | 1006.00 | 1042.65 | 1028.28 | SL hit (close<static) qty=1.00 sl=1010.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 09:30:00 | 1030.20 | 1035.30 | 1025.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 14:15:00 | 1026.00 | 1034.73 | 1026.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-26 14:30:00 | 1029.60 | 1047.39 | 1039.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 998.00 | 1044.52 | 1038.54 | SL hit (close<static) qty=1.00 sl=1010.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 998.00 | 1044.52 | 1038.54 | SL hit (close<static) qty=1.00 sl=1010.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 998.00 | 1044.52 | 1038.54 | SL hit (close<static) qty=1.00 sl=1010.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 14:15:00 | 1000.00 | 1033.15 | 1033.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 994.50 | 1032.14 | 1032.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 11:15:00 | 1013.55 | 1006.76 | 1016.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 11:15:00 | 1013.55 | 1006.76 | 1016.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 1013.55 | 1006.76 | 1016.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:30:00 | 1011.25 | 1006.76 | 1016.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1001.25 | 1006.82 | 1016.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 13:30:00 | 999.80 | 1006.60 | 1016.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 13:15:00 | 1017.45 | 1005.20 | 1014.70 | SL hit (close>static) qty=1.00 sl=1016.95 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 11:45:00 | 999.15 | 1008.70 | 1015.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 12:15:00 | 997.00 | 1008.70 | 1015.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 14:15:00 | 1021.50 | 1008.59 | 1015.27 | SL hit (close>static) qty=1.00 sl=1016.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-10 14:15:00 | 1021.50 | 1008.59 | 1015.27 | SL hit (close>static) qty=1.00 sl=1016.95 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 982.10 | 1016.10 | 1017.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 1034.00 | 1015.10 | 1017.39 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 1034.00 | 1015.10 | 1017.39 | SL hit (close>static) qty=1.00 sl=1016.95 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 1034.00 | 1015.10 | 1017.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1012.80 | 1015.08 | 1017.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 1006.65 | 1015.08 | 1017.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:30:00 | 1011.20 | 1011.15 | 1014.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 11:00:00 | 1012.70 | 1011.09 | 1014.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 12:15:00 | 1051.45 | 1011.34 | 1014.64 | SL hit (close>static) qty=1.00 sl=1039.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 12:15:00 | 1051.45 | 1011.34 | 1014.64 | SL hit (close>static) qty=1.00 sl=1039.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 12:15:00 | 1051.45 | 1011.34 | 1014.64 | SL hit (close>static) qty=1.00 sl=1039.30 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:45:00 | 1010.95 | 1014.18 | 1015.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 1013.35 | 1014.17 | 1015.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:45:00 | 1015.70 | 1014.17 | 1015.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 1010.90 | 1014.14 | 1015.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 13:15:00 | 1008.80 | 1014.14 | 1015.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 15:15:00 | 1018.45 | 1014.07 | 1015.84 | SL hit (close>static) qty=1.00 sl=1016.20 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 09:30:00 | 1003.00 | 1013.98 | 1015.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 1026.50 | 1014.07 | 1015.81 | SL hit (close>static) qty=1.00 sl=1016.20 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 1006.85 | 1014.67 | 1016.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 09:15:00 | 960.40 | 1009.49 | 1013.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:15:00 | 956.51 | 1005.39 | 1010.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 15:15:00 | 1005.00 | 1002.50 | 1008.76 | SL hit (close>ema200) qty=0.50 sl=1002.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 15:15:00 | 1005.00 | 1002.50 | 1008.76 | SL hit (close>ema200) qty=0.50 sl=1002.50 alert=retest2 |

### Cycle 3 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 1063.20 | 1013.77 | 1013.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 12:15:00 | 1067.20 | 1017.90 | 1015.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 11:15:00 | 1022.60 | 1023.80 | 1019.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 12:00:00 | 1022.60 | 1023.80 | 1019.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 1028.70 | 1023.84 | 1019.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:00:00 | 1028.70 | 1023.84 | 1019.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1014.00 | 1023.87 | 1019.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 1014.00 | 1023.87 | 1019.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1010.10 | 1023.73 | 1019.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 1010.10 | 1023.73 | 1019.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 10:15:00 | 990.00 | 1015.45 | 1015.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 11:15:00 | 976.50 | 1015.06 | 1015.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 990.00 | 981.67 | 994.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 10:30:00 | 987.70 | 981.67 | 994.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 994.75 | 981.80 | 994.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:00:00 | 994.75 | 981.80 | 994.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 991.35 | 981.89 | 994.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 983.65 | 987.34 | 995.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 10:00:00 | 987.70 | 987.34 | 995.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 11:15:00 | 995.65 | 987.44 | 995.61 | SL hit (close>static) qty=1.00 sl=994.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 11:15:00 | 995.65 | 987.44 | 995.61 | SL hit (close>static) qty=1.00 sl=994.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 14:45:00 | 987.80 | 987.59 | 995.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:15:00 | 987.00 | 987.59 | 995.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 1003.15 | 987.74 | 995.55 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 1003.15 | 987.74 | 995.55 | SL hit (close>static) qty=1.00 sl=994.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 1003.15 | 987.74 | 995.55 | SL hit (close>static) qty=1.00 sl=994.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 12:15:00 | 985.65 | 995.93 | 998.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 13:15:00 | 982.60 | 995.84 | 998.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 09:30:00 | 985.20 | 995.51 | 998.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:45:00 | 982.10 | 995.36 | 998.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 996.05 | 995.17 | 998.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:15:00 | 989.00 | 995.08 | 997.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 13:30:00 | 986.90 | 994.96 | 997.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 972.20 | 994.05 | 997.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 13:15:00 | 936.37 | 990.44 | 995.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 13:15:00 | 933.47 | 990.44 | 995.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 13:15:00 | 935.94 | 990.44 | 995.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 13:15:00 | 933.00 | 990.44 | 995.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 13:15:00 | 939.55 | 990.44 | 995.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 13:15:00 | 937.55 | 990.44 | 995.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 923.59 | 984.06 | 991.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-20 15:15:00 | 890.10 | 959.46 | 975.23 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-23 09:15:00 | 887.09 | 958.76 | 974.81 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-23 09:15:00 | 884.34 | 958.76 | 974.81 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-23 09:15:00 | 886.68 | 958.76 | 974.81 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-23 09:15:00 | 883.89 | 958.76 | 974.81 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-23 09:15:00 | 888.21 | 958.76 | 974.81 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-23 09:15:00 | 874.98 | 958.76 | 974.81 | Target hit (10%) qty=0.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 11:00:00 | 985.55 | 935.66 | 953.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1010.00 | 939.09 | 954.97 | SL hit (close>static) qty=1.00 sl=999.80 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 1035.20 | 968.00 | 967.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 13:15:00 | 1050.45 | 974.64 | 971.27 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 14:15:00 | 1002.70 | 2025-05-14 09:15:00 | 955.89 | PARTIAL | 0.50 | 4.67% |
| SELL | retest2 | 2025-05-12 14:15:00 | 1002.70 | 2025-05-15 11:15:00 | 905.58 | TARGET_HIT | 0.50 | 9.69% |
| SELL | retest2 | 2025-07-03 12:45:00 | 1005.35 | 2025-07-09 09:15:00 | 1036.55 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-07-07 10:15:00 | 1006.45 | 2025-07-09 09:15:00 | 1036.55 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2025-08-25 10:00:00 | 1034.20 | 2025-08-26 10:15:00 | 1006.00 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2025-09-01 09:30:00 | 1030.20 | 2025-09-30 09:15:00 | 998.00 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2025-09-02 14:15:00 | 1026.00 | 2025-09-30 09:15:00 | 998.00 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2025-09-26 14:30:00 | 1029.60 | 2025-09-30 09:15:00 | 998.00 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2025-10-30 13:30:00 | 999.80 | 2025-11-03 13:15:00 | 1017.45 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-11-10 11:45:00 | 999.15 | 2025-11-10 14:15:00 | 1021.50 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-11-10 12:15:00 | 997.00 | 2025-11-10 14:15:00 | 1021.50 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-11-24 09:15:00 | 982.10 | 2025-11-24 14:15:00 | 1034.00 | STOP_HIT | 1.00 | -5.28% |
| SELL | retest2 | 2025-11-25 09:15:00 | 1006.65 | 2025-12-04 12:15:00 | 1051.45 | STOP_HIT | 1.00 | -4.45% |
| SELL | retest2 | 2025-12-01 14:30:00 | 1011.20 | 2025-12-04 12:15:00 | 1051.45 | STOP_HIT | 1.00 | -3.98% |
| SELL | retest2 | 2025-12-02 11:00:00 | 1012.70 | 2025-12-04 12:15:00 | 1051.45 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2025-12-08 09:45:00 | 1010.95 | 2025-12-08 15:15:00 | 1018.45 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-12-08 13:15:00 | 1008.80 | 2025-12-09 11:15:00 | 1026.50 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-12-09 09:30:00 | 1003.00 | 2025-12-16 09:15:00 | 960.40 | PARTIAL | 0.50 | 4.25% |
| SELL | retest2 | 2025-12-11 09:15:00 | 1006.85 | 2025-12-18 09:15:00 | 956.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-09 09:30:00 | 1003.00 | 2025-12-22 15:15:00 | 1005.00 | STOP_HIT | 0.50 | -0.20% |
| SELL | retest2 | 2025-12-11 09:15:00 | 1006.85 | 2025-12-22 15:15:00 | 1005.00 | STOP_HIT | 0.50 | 0.18% |
| SELL | retest2 | 2026-02-13 09:15:00 | 983.65 | 2026-02-13 11:15:00 | 995.65 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2026-02-13 10:00:00 | 987.70 | 2026-02-13 11:15:00 | 995.65 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-02-13 14:45:00 | 987.80 | 2026-02-16 09:15:00 | 1003.15 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-02-13 15:15:00 | 987.00 | 2026-02-16 09:15:00 | 1003.15 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-02-24 12:15:00 | 985.65 | 2026-03-04 13:15:00 | 936.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 13:15:00 | 982.60 | 2026-03-04 13:15:00 | 933.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 09:30:00 | 985.20 | 2026-03-04 13:15:00 | 935.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 12:45:00 | 982.10 | 2026-03-04 13:15:00 | 933.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 12:15:00 | 989.00 | 2026-03-04 13:15:00 | 939.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 13:30:00 | 986.90 | 2026-03-04 13:15:00 | 937.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 972.20 | 2026-03-09 09:15:00 | 923.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 12:15:00 | 985.65 | 2026-03-20 15:15:00 | 890.10 | TARGET_HIT | 0.50 | 9.69% |
| SELL | retest2 | 2026-02-24 13:15:00 | 982.60 | 2026-03-23 09:15:00 | 887.09 | TARGET_HIT | 0.50 | 9.72% |
| SELL | retest2 | 2026-02-25 09:30:00 | 985.20 | 2026-03-23 09:15:00 | 884.34 | TARGET_HIT | 0.50 | 10.24% |
| SELL | retest2 | 2026-02-25 12:45:00 | 982.10 | 2026-03-23 09:15:00 | 886.68 | TARGET_HIT | 0.50 | 9.72% |
| SELL | retest2 | 2026-02-26 12:15:00 | 989.00 | 2026-03-23 09:15:00 | 883.89 | TARGET_HIT | 0.50 | 10.63% |
| SELL | retest2 | 2026-02-26 13:30:00 | 986.90 | 2026-03-23 09:15:00 | 888.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 972.20 | 2026-03-23 09:15:00 | 874.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-13 11:00:00 | 985.55 | 2026-04-15 09:15:00 | 1010.00 | STOP_HIT | 1.00 | -2.48% |
