# DCM Shriram Ltd. (DCMSHRIRAM)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 1237.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 148 |
| ALERT1 | 104 |
| ALERT2 | 98 |
| ALERT2_SKIP | 50 |
| ALERT3 | 275 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 140 |
| PARTIAL | 14 |
| TARGET_HIT | 12 |
| STOP_HIT | 134 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 160 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 50 / 110
- **Target hits / Stop hits / Partials:** 12 / 134 / 14
- **Avg / median % per leg:** 0.26% / -0.84%
- **Sum % (uncompounded):** 41.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 61 | 13 | 21.3% | 4 | 57 | 0 | -0.39% | -24.0% |
| BUY @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 6 | 0 | -1.04% | -6.2% |
| BUY @ 3rd Alert (retest2) | 55 | 12 | 21.8% | 4 | 51 | 0 | -0.32% | -17.8% |
| SELL (all) | 99 | 37 | 37.4% | 8 | 77 | 14 | 0.66% | 65.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 99 | 37 | 37.4% | 8 | 77 | 14 | 0.66% | 65.5% |
| retest1 (combined) | 6 | 1 | 16.7% | 0 | 6 | 0 | -1.04% | -6.2% |
| retest2 (combined) | 154 | 49 | 31.8% | 12 | 128 | 14 | 0.31% | 47.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 14:15:00 | 990.50 | 982.36 | 981.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 14:15:00 | 1013.00 | 996.52 | 993.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 12:15:00 | 997.65 | 999.38 | 996.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 12:15:00 | 997.65 | 999.38 | 996.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 997.65 | 999.38 | 996.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 12:45:00 | 990.75 | 999.38 | 996.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 993.15 | 998.13 | 996.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 13:30:00 | 993.95 | 998.13 | 996.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 995.40 | 997.59 | 996.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 15:00:00 | 995.40 | 997.59 | 996.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 999.90 | 998.05 | 996.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 09:30:00 | 1000.85 | 996.45 | 995.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 10:15:00 | 988.00 | 994.76 | 995.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 10:15:00 | 988.00 | 994.76 | 995.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 09:15:00 | 979.80 | 989.27 | 992.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 13:15:00 | 987.00 | 985.31 | 988.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-24 14:00:00 | 987.00 | 985.31 | 988.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 985.40 | 985.22 | 987.98 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 14:15:00 | 994.70 | 989.78 | 989.48 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 11:15:00 | 985.70 | 988.99 | 989.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 12:15:00 | 985.55 | 988.30 | 988.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 12:15:00 | 986.90 | 985.00 | 986.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 12:15:00 | 986.90 | 985.00 | 986.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 12:15:00 | 986.90 | 985.00 | 986.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 12:30:00 | 987.10 | 985.00 | 986.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 986.55 | 985.31 | 986.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 15:15:00 | 981.10 | 985.84 | 986.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 15:15:00 | 990.50 | 985.23 | 985.33 | SL hit (close>static) qty=1.00 sl=987.85 alert=retest2 |

### Cycle 5 — BUY (started 2024-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 09:15:00 | 986.10 | 985.41 | 985.40 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 10:15:00 | 985.20 | 985.37 | 985.38 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 11:15:00 | 986.95 | 985.68 | 985.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 13:15:00 | 997.95 | 988.09 | 986.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-31 15:15:00 | 979.40 | 988.42 | 987.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 15:15:00 | 979.40 | 988.42 | 987.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 979.40 | 988.42 | 987.17 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2024-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-03 11:15:00 | 981.95 | 985.72 | 986.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 09:15:00 | 953.00 | 977.01 | 981.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 941.85 | 924.01 | 935.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 941.85 | 924.01 | 935.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 941.85 | 924.01 | 935.93 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2024-06-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 10:15:00 | 970.50 | 944.48 | 941.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 11:15:00 | 983.10 | 952.20 | 945.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 13:15:00 | 993.00 | 994.43 | 984.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 14:00:00 | 993.00 | 994.43 | 984.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 999.70 | 994.12 | 986.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 11:45:00 | 1014.00 | 1006.64 | 997.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-18 14:15:00 | 1016.20 | 1027.20 | 1027.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 14:15:00 | 1016.20 | 1027.20 | 1027.97 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 11:15:00 | 1052.20 | 1029.45 | 1028.39 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 11:15:00 | 1024.15 | 1035.09 | 1036.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 12:15:00 | 1018.90 | 1031.85 | 1034.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 09:15:00 | 1011.55 | 1009.97 | 1017.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 09:15:00 | 1011.55 | 1009.97 | 1017.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 1011.55 | 1009.97 | 1017.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 09:45:00 | 1016.05 | 1009.97 | 1017.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 1007.40 | 1009.45 | 1016.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 11:30:00 | 1003.40 | 1007.82 | 1015.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 14:45:00 | 1003.70 | 1001.39 | 1006.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 09:15:00 | 999.05 | 1002.26 | 1006.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 11:15:00 | 1004.20 | 994.53 | 994.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 1004.20 | 994.53 | 994.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 14:15:00 | 1005.00 | 999.08 | 996.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 15:15:00 | 1003.95 | 1004.62 | 1001.49 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:15:00 | 1011.00 | 1004.62 | 1001.49 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 10:45:00 | 1009.65 | 1005.45 | 1002.42 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 12:15:00 | 1005.75 | 1005.75 | 1003.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 12:45:00 | 1003.55 | 1005.75 | 1003.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 1007.55 | 1008.11 | 1005.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:30:00 | 1006.80 | 1008.11 | 1005.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 1009.25 | 1008.33 | 1005.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 11:00:00 | 1009.25 | 1008.33 | 1005.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 1011.00 | 1011.11 | 1008.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 09:15:00 | 1012.05 | 1011.11 | 1008.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 12:00:00 | 1013.00 | 1011.14 | 1009.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 12:30:00 | 1014.75 | 1011.23 | 1009.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 14:00:00 | 1018.00 | 1012.59 | 1010.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 1013.45 | 1012.76 | 1010.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 14:45:00 | 1011.70 | 1012.76 | 1010.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 1015.10 | 1014.07 | 1011.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:45:00 | 1010.25 | 1014.07 | 1011.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 1002.45 | 1011.74 | 1010.65 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-08 10:15:00 | 1002.45 | 1011.74 | 1010.65 | SL hit (close<ema400) qty=1.00 sl=1010.65 alert=retest1 |

### Cycle 14 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 997.70 | 1008.93 | 1009.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 13:15:00 | 995.35 | 1005.07 | 1007.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 10:15:00 | 1007.90 | 1001.44 | 1004.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 10:15:00 | 1007.90 | 1001.44 | 1004.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 1007.90 | 1001.44 | 1004.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 11:00:00 | 1007.90 | 1001.44 | 1004.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 1002.20 | 1001.59 | 1004.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 09:45:00 | 993.15 | 998.54 | 1001.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-11 13:15:00 | 1002.60 | 995.17 | 994.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 13:15:00 | 1002.60 | 995.17 | 994.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 13:15:00 | 1007.90 | 1001.11 | 999.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 14:15:00 | 996.75 | 1000.24 | 999.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 14:15:00 | 996.75 | 1000.24 | 999.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 996.75 | 1000.24 | 999.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 15:00:00 | 996.75 | 1000.24 | 999.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 995.30 | 999.25 | 998.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 09:15:00 | 1012.25 | 999.25 | 998.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 09:15:00 | 993.05 | 1003.28 | 1004.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 993.05 | 1003.28 | 1004.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 987.60 | 1000.14 | 1002.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 10:15:00 | 991.20 | 987.68 | 993.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 10:15:00 | 991.20 | 987.68 | 993.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 991.20 | 987.68 | 993.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:00:00 | 991.20 | 987.68 | 993.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 1000.75 | 990.29 | 994.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:45:00 | 1000.15 | 990.29 | 994.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 1002.00 | 992.64 | 995.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:15:00 | 1003.75 | 992.64 | 995.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 09:15:00 | 998.00 | 996.63 | 996.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 10:15:00 | 1017.60 | 1000.82 | 998.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 12:15:00 | 1001.25 | 1001.29 | 999.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 12:15:00 | 1001.25 | 1001.29 | 999.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 1001.25 | 1001.29 | 999.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 989.95 | 1001.29 | 999.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 1006.25 | 1002.28 | 999.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:45:00 | 1005.55 | 1002.28 | 999.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 1032.10 | 1023.36 | 1014.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 09:15:00 | 1045.50 | 1034.04 | 1029.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 1038.80 | 1056.06 | 1056.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 1038.80 | 1056.06 | 1056.88 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 09:15:00 | 1122.90 | 1063.54 | 1058.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 09:15:00 | 1151.40 | 1105.22 | 1089.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 12:15:00 | 1131.95 | 1143.67 | 1126.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 13:00:00 | 1131.95 | 1143.67 | 1126.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 1149.95 | 1143.27 | 1130.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 1155.10 | 1143.27 | 1130.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 10:00:00 | 1152.25 | 1145.06 | 1132.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 13:45:00 | 1150.50 | 1148.27 | 1138.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 15:15:00 | 1127.80 | 1142.12 | 1137.41 | SL hit (close<static) qty=1.00 sl=1129.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 11:15:00 | 1115.60 | 1131.16 | 1133.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 12:15:00 | 1106.95 | 1126.32 | 1130.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 13:15:00 | 1093.95 | 1092.99 | 1103.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 13:45:00 | 1094.50 | 1092.99 | 1103.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 1108.50 | 1093.90 | 1100.94 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2024-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 12:15:00 | 1120.55 | 1107.77 | 1106.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 13:15:00 | 1125.15 | 1111.25 | 1108.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 11:15:00 | 1144.35 | 1151.24 | 1139.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 11:30:00 | 1150.25 | 1151.24 | 1139.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 13:15:00 | 1139.35 | 1148.39 | 1140.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 13:30:00 | 1136.85 | 1148.39 | 1140.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 14:15:00 | 1151.00 | 1148.91 | 1141.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 09:15:00 | 1158.60 | 1147.17 | 1141.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 09:45:00 | 1158.00 | 1150.74 | 1143.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 15:00:00 | 1163.40 | 1159.46 | 1151.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 10:30:00 | 1163.20 | 1156.08 | 1151.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 1157.25 | 1156.32 | 1152.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 13:30:00 | 1166.60 | 1158.05 | 1153.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 14:15:00 | 1143.75 | 1158.54 | 1157.76 | SL hit (close<static) qty=1.00 sl=1151.10 alert=retest2 |

### Cycle 22 — SELL (started 2024-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 15:15:00 | 1147.95 | 1156.42 | 1156.87 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-08-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 12:15:00 | 1185.45 | 1162.10 | 1159.25 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-08-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 15:15:00 | 1147.20 | 1156.46 | 1157.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 14:15:00 | 1143.80 | 1151.94 | 1154.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 1136.00 | 1128.04 | 1137.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 1136.00 | 1128.04 | 1137.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 1136.00 | 1128.04 | 1137.02 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 14:15:00 | 1152.00 | 1141.17 | 1140.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 09:15:00 | 1190.00 | 1153.47 | 1146.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 13:15:00 | 1153.90 | 1159.40 | 1152.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 14:00:00 | 1153.90 | 1159.40 | 1152.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 1149.35 | 1157.39 | 1151.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 14:45:00 | 1145.90 | 1157.39 | 1151.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 15:15:00 | 1151.05 | 1156.12 | 1151.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 09:15:00 | 1158.60 | 1156.12 | 1151.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 11:15:00 | 1139.50 | 1150.75 | 1150.35 | SL hit (close<static) qty=1.00 sl=1143.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 12:15:00 | 1146.15 | 1149.83 | 1149.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 13:15:00 | 1131.65 | 1143.94 | 1146.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 1142.90 | 1141.29 | 1144.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 09:15:00 | 1142.90 | 1141.29 | 1144.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 1142.90 | 1141.29 | 1144.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:30:00 | 1149.70 | 1141.29 | 1144.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 1144.25 | 1141.88 | 1144.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 11:00:00 | 1144.25 | 1141.88 | 1144.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 11:15:00 | 1140.45 | 1141.59 | 1143.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 12:15:00 | 1139.00 | 1141.59 | 1143.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 13:00:00 | 1136.85 | 1140.64 | 1143.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 10:15:00 | 1138.10 | 1130.37 | 1131.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 14:15:00 | 1143.00 | 1131.62 | 1131.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 14:15:00 | 1143.00 | 1131.62 | 1131.40 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 15:15:00 | 1127.55 | 1130.80 | 1131.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 12:15:00 | 1120.60 | 1126.39 | 1128.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 09:15:00 | 1104.60 | 1103.84 | 1110.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-13 10:00:00 | 1104.60 | 1103.84 | 1110.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 1084.65 | 1090.29 | 1099.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 11:30:00 | 1082.40 | 1088.27 | 1097.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 12:45:00 | 1080.05 | 1086.95 | 1095.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 09:30:00 | 1082.15 | 1085.08 | 1091.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 12:00:00 | 1081.70 | 1084.60 | 1090.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 12:15:00 | 1091.85 | 1086.05 | 1090.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 13:00:00 | 1091.85 | 1086.05 | 1090.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 13:15:00 | 1085.75 | 1085.99 | 1090.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 14:30:00 | 1080.05 | 1084.85 | 1089.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 09:15:00 | 1080.10 | 1085.08 | 1089.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 11:45:00 | 1081.85 | 1084.56 | 1087.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 12:45:00 | 1080.55 | 1083.64 | 1087.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 1085.70 | 1084.05 | 1086.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:45:00 | 1088.10 | 1084.05 | 1086.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 1087.15 | 1084.67 | 1086.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 15:00:00 | 1087.15 | 1084.67 | 1086.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 15:15:00 | 1089.45 | 1085.63 | 1087.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:15:00 | 1092.00 | 1085.63 | 1087.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 1079.15 | 1084.33 | 1086.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 11:00:00 | 1079.05 | 1083.27 | 1085.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 10:15:00 | 1092.10 | 1085.17 | 1084.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 1092.10 | 1085.17 | 1084.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 12:15:00 | 1121.10 | 1094.14 | 1089.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 14:15:00 | 1092.45 | 1094.90 | 1090.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-23 14:15:00 | 1092.45 | 1094.90 | 1090.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 14:15:00 | 1092.45 | 1094.90 | 1090.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 15:00:00 | 1092.45 | 1094.90 | 1090.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 15:15:00 | 1095.00 | 1094.92 | 1090.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 09:15:00 | 1088.00 | 1094.92 | 1090.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 1087.35 | 1093.41 | 1090.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 09:45:00 | 1083.05 | 1093.41 | 1090.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 1081.75 | 1091.07 | 1089.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:00:00 | 1081.75 | 1091.07 | 1089.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 12:15:00 | 1080.45 | 1087.27 | 1088.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 13:15:00 | 1072.65 | 1084.35 | 1086.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 10:15:00 | 1079.40 | 1076.63 | 1081.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-25 11:00:00 | 1079.40 | 1076.63 | 1081.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 1085.20 | 1078.35 | 1082.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 09:45:00 | 1071.00 | 1080.26 | 1082.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 11:00:00 | 1075.15 | 1067.21 | 1072.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 14:00:00 | 1059.10 | 1067.43 | 1071.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 15:15:00 | 1017.45 | 1055.61 | 1064.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 15:15:00 | 1021.39 | 1055.61 | 1064.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-01 09:15:00 | 1044.00 | 1040.60 | 1049.99 | SL hit (close>ema200) qty=0.50 sl=1040.60 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 14:15:00 | 1009.45 | 1003.59 | 1002.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 15:15:00 | 1015.20 | 1005.91 | 1003.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 15:15:00 | 1011.90 | 1013.69 | 1009.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 09:15:00 | 1011.70 | 1013.69 | 1009.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 1033.15 | 1017.58 | 1011.91 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 14:15:00 | 1012.75 | 1015.53 | 1015.74 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 10:15:00 | 1022.05 | 1016.32 | 1015.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 13:15:00 | 1034.40 | 1023.10 | 1019.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 15:15:00 | 1114.90 | 1117.58 | 1095.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-18 09:15:00 | 1095.35 | 1117.58 | 1095.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 1096.75 | 1113.42 | 1095.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 10:30:00 | 1112.40 | 1112.16 | 1096.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 11:30:00 | 1113.30 | 1111.94 | 1098.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 09:15:00 | 1118.05 | 1105.37 | 1099.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 10:00:00 | 1123.40 | 1108.98 | 1101.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 1103.05 | 1107.79 | 1101.49 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-22 09:15:00 | 1061.65 | 1096.21 | 1098.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-10-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 09:15:00 | 1061.65 | 1096.21 | 1098.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 12:15:00 | 1041.70 | 1078.04 | 1089.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 12:15:00 | 1061.60 | 1060.06 | 1072.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 12:45:00 | 1061.60 | 1060.06 | 1072.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 1059.30 | 1059.33 | 1067.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 09:30:00 | 1056.00 | 1059.33 | 1067.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 1057.85 | 1059.03 | 1066.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:30:00 | 1065.05 | 1059.03 | 1066.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 12:15:00 | 1053.90 | 1057.42 | 1064.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 12:30:00 | 1056.65 | 1057.42 | 1064.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 1029.95 | 1014.96 | 1016.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:45:00 | 1035.40 | 1014.96 | 1016.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 1040.15 | 1020.00 | 1018.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 09:15:00 | 1077.10 | 1043.31 | 1031.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 14:15:00 | 1054.15 | 1057.36 | 1044.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 15:00:00 | 1054.15 | 1057.36 | 1044.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 15:15:00 | 1035.00 | 1052.88 | 1043.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:00:00 | 1058.80 | 1054.07 | 1045.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 1022.45 | 1050.77 | 1045.32 | SL hit (close<static) qty=1.00 sl=1028.95 alert=retest2 |

### Cycle 36 — SELL (started 2024-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 12:15:00 | 1214.00 | 1231.32 | 1231.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 13:15:00 | 1203.75 | 1225.81 | 1229.33 | Break + close below crossover candle low |

### Cycle 37 — BUY (started 2024-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 09:15:00 | 1300.55 | 1229.78 | 1229.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-18 09:15:00 | 1321.90 | 1276.10 | 1255.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 13:15:00 | 1327.20 | 1331.65 | 1307.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 13:45:00 | 1327.55 | 1331.65 | 1307.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 1294.60 | 1324.24 | 1306.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 15:00:00 | 1294.60 | 1324.24 | 1306.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 1295.00 | 1318.40 | 1305.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:30:00 | 1299.90 | 1308.82 | 1301.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 1298.80 | 1306.81 | 1301.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 13:15:00 | 1301.30 | 1303.79 | 1301.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-21 15:15:00 | 1294.00 | 1299.40 | 1299.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 15:15:00 | 1294.00 | 1299.40 | 1299.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-22 09:15:00 | 1271.35 | 1293.79 | 1296.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 10:15:00 | 1264.85 | 1263.63 | 1276.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-25 11:00:00 | 1264.85 | 1263.63 | 1276.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 1221.80 | 1254.20 | 1267.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 15:00:00 | 1221.80 | 1254.20 | 1267.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 1134.05 | 1125.56 | 1141.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:30:00 | 1145.00 | 1125.56 | 1141.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 1139.20 | 1128.29 | 1141.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 13:30:00 | 1142.50 | 1128.29 | 1141.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 1153.85 | 1133.40 | 1142.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 15:00:00 | 1153.85 | 1133.40 | 1142.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 1156.80 | 1138.08 | 1144.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:15:00 | 1137.75 | 1138.08 | 1144.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 13:15:00 | 1136.80 | 1134.58 | 1139.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 14:00:00 | 1136.80 | 1134.58 | 1139.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 1137.55 | 1135.18 | 1139.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 15:00:00 | 1137.55 | 1135.18 | 1139.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 1136.90 | 1135.52 | 1139.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:15:00 | 1154.30 | 1135.52 | 1139.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 1150.55 | 1138.53 | 1140.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:30:00 | 1158.00 | 1138.53 | 1140.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 1149.25 | 1140.67 | 1141.11 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2024-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 11:15:00 | 1146.45 | 1141.83 | 1141.60 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 12:15:00 | 1128.10 | 1140.16 | 1141.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 14:15:00 | 1115.55 | 1133.69 | 1138.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 09:15:00 | 1098.30 | 1093.34 | 1103.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 09:15:00 | 1098.30 | 1093.34 | 1103.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 1098.30 | 1093.34 | 1103.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:30:00 | 1108.85 | 1093.34 | 1103.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 1121.50 | 1098.97 | 1104.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:30:00 | 1123.85 | 1098.97 | 1104.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 11:15:00 | 1111.00 | 1101.38 | 1105.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 12:30:00 | 1110.05 | 1103.27 | 1105.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 13:15:00 | 1110.00 | 1103.27 | 1105.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 14:15:00 | 1109.95 | 1104.85 | 1106.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 15:00:00 | 1104.10 | 1104.70 | 1106.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 1105.80 | 1104.92 | 1106.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:15:00 | 1121.70 | 1104.92 | 1106.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 1113.05 | 1106.55 | 1106.70 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-11 10:15:00 | 1113.75 | 1107.99 | 1107.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 10:15:00 | 1113.75 | 1107.99 | 1107.34 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 14:15:00 | 1104.85 | 1106.95 | 1107.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 09:15:00 | 1089.00 | 1103.12 | 1105.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 13:15:00 | 1097.10 | 1097.07 | 1101.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-12 14:00:00 | 1097.10 | 1097.07 | 1101.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 14:15:00 | 1103.00 | 1098.26 | 1101.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 15:00:00 | 1103.00 | 1098.26 | 1101.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 1097.00 | 1098.01 | 1100.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:15:00 | 1091.95 | 1098.01 | 1100.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 1085.35 | 1095.47 | 1099.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 10:15:00 | 1080.80 | 1095.47 | 1099.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 12:30:00 | 1081.20 | 1090.29 | 1095.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 13:45:00 | 1078.05 | 1087.82 | 1094.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 15:15:00 | 1081.00 | 1087.45 | 1093.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1092.85 | 1087.50 | 1092.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:00:00 | 1092.85 | 1087.50 | 1092.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 1100.00 | 1090.00 | 1093.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:30:00 | 1101.05 | 1090.00 | 1093.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 1093.55 | 1090.71 | 1093.14 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-16 15:15:00 | 1102.30 | 1094.28 | 1094.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2024-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 15:15:00 | 1102.30 | 1094.28 | 1094.15 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 09:15:00 | 1090.05 | 1093.43 | 1093.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 12:15:00 | 1087.05 | 1091.99 | 1093.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 09:15:00 | 1085.85 | 1085.74 | 1089.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 09:15:00 | 1085.85 | 1085.74 | 1089.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 1085.85 | 1085.74 | 1089.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 10:00:00 | 1085.85 | 1085.74 | 1089.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 1085.35 | 1085.66 | 1088.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 10:30:00 | 1094.65 | 1085.66 | 1088.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 14:15:00 | 1089.50 | 1083.33 | 1086.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 15:00:00 | 1089.50 | 1083.33 | 1086.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 15:15:00 | 1084.00 | 1083.46 | 1086.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:15:00 | 1068.60 | 1083.46 | 1086.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 14:00:00 | 1082.00 | 1082.58 | 1084.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 15:00:00 | 1082.35 | 1082.54 | 1084.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 09:45:00 | 1080.65 | 1080.18 | 1082.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 1085.40 | 1081.23 | 1083.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 11:00:00 | 1085.40 | 1081.23 | 1083.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 1090.20 | 1083.02 | 1083.82 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-20 11:15:00 | 1090.20 | 1083.02 | 1083.82 | SL hit (close>static) qty=1.00 sl=1089.90 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 11:15:00 | 1091.00 | 1073.87 | 1072.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 14:15:00 | 1097.00 | 1083.77 | 1077.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 09:15:00 | 1079.00 | 1084.61 | 1079.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 09:15:00 | 1079.00 | 1084.61 | 1079.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 1079.00 | 1084.61 | 1079.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 10:00:00 | 1079.00 | 1084.61 | 1079.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 1085.60 | 1084.81 | 1079.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-26 11:15:00 | 1087.05 | 1084.81 | 1079.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-26 11:45:00 | 1086.25 | 1085.05 | 1080.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 09:15:00 | 1086.95 | 1086.82 | 1082.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 09:45:00 | 1093.65 | 1088.56 | 1083.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 1094.90 | 1092.08 | 1088.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:15:00 | 1076.85 | 1092.08 | 1088.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 1093.50 | 1092.37 | 1088.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:30:00 | 1082.55 | 1092.37 | 1088.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 1091.30 | 1092.15 | 1088.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:45:00 | 1091.80 | 1092.15 | 1088.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 11:15:00 | 1088.95 | 1091.51 | 1088.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 12:00:00 | 1088.95 | 1091.51 | 1088.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 12:15:00 | 1091.50 | 1091.51 | 1089.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 12:30:00 | 1091.40 | 1091.51 | 1089.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 1090.00 | 1091.21 | 1089.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 13:45:00 | 1086.90 | 1091.21 | 1089.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 1084.85 | 1089.94 | 1088.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 1084.85 | 1089.94 | 1088.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 1086.50 | 1089.25 | 1088.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:30:00 | 1075.65 | 1089.33 | 1088.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 1105.00 | 1092.46 | 1090.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 14:15:00 | 1115.10 | 1096.68 | 1092.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 15:00:00 | 1149.80 | 1107.31 | 1097.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 11:15:00 | 1077.15 | 1118.07 | 1122.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 1077.15 | 1118.07 | 1122.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 12:15:00 | 1073.70 | 1109.19 | 1118.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 12:15:00 | 1086.95 | 1083.12 | 1096.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 12:45:00 | 1084.85 | 1083.12 | 1096.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 1062.40 | 1078.52 | 1090.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 10:15:00 | 1058.25 | 1078.52 | 1090.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 11:00:00 | 1058.85 | 1074.59 | 1087.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 11:30:00 | 1057.55 | 1070.50 | 1084.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 14:30:00 | 1056.05 | 1065.07 | 1078.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 1072.95 | 1065.86 | 1076.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:45:00 | 1076.05 | 1065.86 | 1076.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 11:15:00 | 1073.45 | 1068.31 | 1075.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 11:30:00 | 1075.35 | 1068.31 | 1075.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 13:15:00 | 1071.50 | 1069.55 | 1074.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 13:30:00 | 1073.20 | 1069.55 | 1074.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 15:15:00 | 1078.50 | 1071.89 | 1075.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 1060.00 | 1071.89 | 1075.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 13:15:00 | 1070.80 | 1071.20 | 1073.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 13:45:00 | 1070.90 | 1070.56 | 1072.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 13:15:00 | 1017.26 | 1052.30 | 1062.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 13:15:00 | 1017.36 | 1052.30 | 1062.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 09:15:00 | 1045.25 | 1044.93 | 1055.75 | SL hit (close>ema200) qty=0.50 sl=1044.93 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 1058.15 | 1051.77 | 1051.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 10:15:00 | 1065.55 | 1054.53 | 1052.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 12:15:00 | 1064.05 | 1064.07 | 1060.13 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 14:00:00 | 1076.25 | 1066.51 | 1061.59 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 1121.00 | 1088.48 | 1073.84 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-22 10:15:00 | 1101.65 | 1110.51 | 1103.41 | SL hit (close<ema400) qty=1.00 sl=1103.41 alert=retest1 |

### Cycle 48 — SELL (started 2025-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 11:15:00 | 1092.05 | 1121.20 | 1124.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 12:15:00 | 1084.30 | 1113.82 | 1121.28 | Break + close below crossover candle low |

### Cycle 49 — BUY (started 2025-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-27 14:15:00 | 1201.70 | 1125.51 | 1124.93 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 1135.20 | 1158.52 | 1159.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 14:15:00 | 1133.05 | 1153.42 | 1157.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 14:15:00 | 1173.50 | 1140.28 | 1146.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 14:15:00 | 1173.50 | 1140.28 | 1146.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 1173.50 | 1140.28 | 1146.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 15:00:00 | 1173.50 | 1140.28 | 1146.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 15:15:00 | 1168.85 | 1145.99 | 1148.48 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 10:15:00 | 1164.55 | 1152.73 | 1151.30 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 1139.15 | 1150.01 | 1150.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 1112.15 | 1139.14 | 1144.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 1109.70 | 1109.30 | 1123.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 09:30:00 | 1112.50 | 1109.30 | 1123.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 1074.05 | 1064.33 | 1074.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:00:00 | 1074.05 | 1064.33 | 1074.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 1075.00 | 1066.47 | 1074.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 1058.25 | 1066.47 | 1074.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1069.35 | 1067.04 | 1073.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 14:00:00 | 1050.10 | 1063.19 | 1069.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 10:00:00 | 1044.35 | 1059.64 | 1066.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 09:15:00 | 1042.10 | 1061.88 | 1064.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 10:30:00 | 1048.85 | 1057.78 | 1062.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 14:15:00 | 1052.00 | 1054.67 | 1059.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 15:00:00 | 1052.00 | 1054.67 | 1059.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 1050.50 | 1044.44 | 1050.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 15:00:00 | 1050.50 | 1044.44 | 1050.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 1062.00 | 1047.95 | 1051.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 09:15:00 | 1038.75 | 1047.95 | 1051.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-13 11:15:00 | 1062.60 | 1053.62 | 1053.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 11:15:00 | 1062.60 | 1053.62 | 1053.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 14:15:00 | 1065.90 | 1058.14 | 1055.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 09:15:00 | 1042.05 | 1055.38 | 1054.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-14 09:15:00 | 1042.05 | 1055.38 | 1054.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 1042.05 | 1055.38 | 1054.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 10:15:00 | 1040.00 | 1055.38 | 1054.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 10:15:00 | 1046.20 | 1053.54 | 1054.02 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-14 15:15:00 | 1060.00 | 1054.05 | 1053.88 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 09:15:00 | 1046.85 | 1052.61 | 1053.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 11:15:00 | 1020.75 | 1035.24 | 1042.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 15:15:00 | 1021.50 | 1021.31 | 1027.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 09:15:00 | 1024.10 | 1021.87 | 1027.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 1024.10 | 1021.87 | 1027.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 09:30:00 | 1026.80 | 1021.87 | 1027.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 10:15:00 | 1032.90 | 1024.07 | 1027.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 11:00:00 | 1032.90 | 1024.07 | 1027.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 11:15:00 | 1030.90 | 1025.44 | 1028.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 11:30:00 | 1031.45 | 1025.44 | 1028.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 15:15:00 | 1034.80 | 1029.47 | 1029.44 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 09:15:00 | 1024.30 | 1028.44 | 1028.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 12:15:00 | 1020.60 | 1025.40 | 1027.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 14:15:00 | 966.70 | 941.37 | 952.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 14:15:00 | 966.70 | 941.37 | 952.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 966.70 | 941.37 | 952.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 966.70 | 941.37 | 952.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 965.50 | 946.19 | 953.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:15:00 | 955.80 | 946.19 | 953.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 10:15:00 | 937.40 | 944.99 | 951.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 11:15:00 | 931.45 | 944.99 | 951.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 13:15:00 | 930.50 | 939.13 | 947.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-04 12:15:00 | 1028.00 | 947.33 | 945.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 12:15:00 | 1028.00 | 947.33 | 945.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 1052.50 | 1022.60 | 998.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 1033.10 | 1035.30 | 1016.55 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 09:15:00 | 1069.35 | 1035.30 | 1016.55 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 12:15:00 | 1052.95 | 1044.77 | 1026.10 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 1040.00 | 1043.43 | 1031.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 1026.10 | 1043.43 | 1031.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 1032.80 | 1041.30 | 1031.54 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-10 10:15:00 | 1029.05 | 1038.85 | 1031.31 | SL hit (close<ema400) qty=1.00 sl=1031.31 alert=retest1 |

### Cycle 60 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 1013.95 | 1027.71 | 1027.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 09:15:00 | 1006.35 | 1019.66 | 1022.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 10:15:00 | 1022.20 | 1020.16 | 1022.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 10:15:00 | 1022.20 | 1020.16 | 1022.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 1022.20 | 1020.16 | 1022.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:45:00 | 1024.90 | 1020.16 | 1022.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 1022.75 | 1020.68 | 1022.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:30:00 | 1021.95 | 1020.68 | 1022.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 1018.55 | 1020.26 | 1022.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:30:00 | 1025.75 | 1020.26 | 1022.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 1021.45 | 1020.49 | 1022.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 13:45:00 | 1021.25 | 1020.49 | 1022.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 1022.75 | 1020.95 | 1022.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 1022.75 | 1020.95 | 1022.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 1021.85 | 1021.13 | 1022.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:15:00 | 973.25 | 1021.13 | 1022.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 986.15 | 1014.13 | 1019.06 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2025-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 14:15:00 | 1006.00 | 1002.61 | 1002.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 1023.65 | 1007.20 | 1004.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 14:15:00 | 1108.00 | 1111.26 | 1088.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 15:00:00 | 1108.00 | 1111.26 | 1088.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 1089.00 | 1104.22 | 1088.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 1089.00 | 1104.22 | 1088.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 1092.35 | 1101.85 | 1089.18 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 14:15:00 | 1061.00 | 1083.00 | 1083.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 10:15:00 | 1049.25 | 1070.07 | 1076.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 1123.20 | 1058.69 | 1060.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 1123.20 | 1058.69 | 1060.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 1123.20 | 1058.69 | 1060.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 1123.20 | 1058.69 | 1060.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 1077.10 | 1062.37 | 1062.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 10:15:00 | 1198.55 | 1099.73 | 1079.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 11:15:00 | 1084.75 | 1096.74 | 1080.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 11:15:00 | 1084.75 | 1096.74 | 1080.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 11:15:00 | 1084.75 | 1096.74 | 1080.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 12:00:00 | 1084.75 | 1096.74 | 1080.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 1073.25 | 1092.04 | 1079.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 12:45:00 | 1076.25 | 1092.04 | 1079.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 1096.00 | 1092.83 | 1081.24 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 11:15:00 | 1057.65 | 1075.13 | 1076.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 13:15:00 | 1049.10 | 1066.18 | 1071.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 09:15:00 | 1073.10 | 1062.12 | 1068.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 1073.10 | 1062.12 | 1068.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 1073.10 | 1062.12 | 1068.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:45:00 | 1070.40 | 1062.12 | 1068.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 1072.30 | 1064.15 | 1068.43 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 1078.80 | 1071.63 | 1070.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 10:15:00 | 1087.85 | 1074.87 | 1072.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 12:15:00 | 1072.60 | 1075.18 | 1072.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 12:15:00 | 1072.60 | 1075.18 | 1072.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 12:15:00 | 1072.60 | 1075.18 | 1072.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 12:45:00 | 1076.55 | 1075.18 | 1072.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 13:15:00 | 1073.10 | 1074.77 | 1072.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 13:30:00 | 1073.00 | 1074.77 | 1072.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 14:15:00 | 1082.10 | 1076.23 | 1073.80 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 1065.20 | 1072.71 | 1072.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 14:15:00 | 1049.70 | 1063.49 | 1068.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 1037.80 | 1030.10 | 1042.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 13:15:00 | 1037.75 | 1034.67 | 1041.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 1037.75 | 1034.67 | 1041.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 13:30:00 | 1039.85 | 1034.67 | 1041.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 1041.15 | 1035.97 | 1041.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 15:00:00 | 1041.15 | 1035.97 | 1041.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 15:15:00 | 1032.00 | 1035.17 | 1040.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 1014.95 | 1035.17 | 1040.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 1041.80 | 1024.34 | 1029.66 | SL hit (close>static) qty=1.00 sl=1041.15 alert=retest2 |

### Cycle 67 — BUY (started 2025-04-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 13:15:00 | 1037.70 | 1031.76 | 1031.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1044.20 | 1035.45 | 1033.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 13:15:00 | 1064.90 | 1065.51 | 1054.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 13:45:00 | 1063.50 | 1065.51 | 1054.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 15:15:00 | 1072.70 | 1067.57 | 1057.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:15:00 | 1065.80 | 1067.57 | 1057.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1068.40 | 1067.74 | 1058.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 10:00:00 | 1082.10 | 1070.34 | 1064.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 15:00:00 | 1080.90 | 1081.95 | 1077.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 11:00:00 | 1078.40 | 1081.13 | 1078.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 15:15:00 | 1073.00 | 1076.28 | 1076.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 15:15:00 | 1073.00 | 1076.28 | 1076.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 1036.60 | 1065.49 | 1071.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 10:15:00 | 1035.00 | 1032.88 | 1046.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 11:00:00 | 1035.00 | 1032.88 | 1046.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 1000.90 | 988.07 | 997.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 1000.90 | 988.07 | 997.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1001.90 | 990.84 | 997.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 11:45:00 | 995.45 | 992.03 | 997.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 12:30:00 | 995.50 | 992.46 | 997.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 14:15:00 | 1013.85 | 996.22 | 998.24 | SL hit (close>static) qty=1.00 sl=1008.25 alert=retest2 |

### Cycle 69 — BUY (started 2025-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 15:15:00 | 1020.00 | 1000.98 | 1000.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 09:15:00 | 1090.00 | 1018.78 | 1008.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 13:15:00 | 1032.80 | 1034.47 | 1020.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 14:00:00 | 1032.80 | 1034.47 | 1020.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 1012.90 | 1030.15 | 1019.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 15:00:00 | 1012.90 | 1030.15 | 1019.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 996.90 | 1023.50 | 1017.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 995.90 | 1023.50 | 1017.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 10:15:00 | 994.80 | 1012.88 | 1013.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 990.40 | 998.60 | 1003.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 1022.25 | 999.50 | 1000.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 1022.25 | 999.50 | 1000.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1022.25 | 999.50 | 1000.59 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1030.50 | 1005.70 | 1003.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 10:15:00 | 1043.30 | 1026.54 | 1016.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 1041.00 | 1041.22 | 1034.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 1041.00 | 1041.22 | 1034.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 1041.00 | 1041.22 | 1034.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 10:15:00 | 1053.00 | 1043.02 | 1040.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 15:15:00 | 1055.00 | 1059.60 | 1055.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 14:15:00 | 1052.60 | 1058.50 | 1059.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 14:15:00 | 1052.60 | 1058.50 | 1059.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 10:15:00 | 1050.60 | 1056.43 | 1057.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 09:15:00 | 1053.30 | 1047.31 | 1051.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 1053.30 | 1047.31 | 1051.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1053.30 | 1047.31 | 1051.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:30:00 | 1057.05 | 1047.31 | 1051.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1034.50 | 1044.74 | 1049.70 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 1054.85 | 1052.08 | 1051.94 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 11:15:00 | 1038.75 | 1050.15 | 1051.13 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 11:15:00 | 1061.85 | 1051.96 | 1051.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 11:15:00 | 1068.85 | 1061.03 | 1056.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 14:15:00 | 1059.45 | 1062.51 | 1058.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 15:00:00 | 1059.45 | 1062.51 | 1058.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 1085.00 | 1067.01 | 1061.12 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 11:15:00 | 1062.10 | 1068.62 | 1068.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 13:15:00 | 1056.60 | 1064.92 | 1067.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 09:15:00 | 1060.20 | 1059.54 | 1063.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-05 09:30:00 | 1059.50 | 1059.54 | 1063.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 1061.40 | 1060.56 | 1063.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 12:45:00 | 1060.00 | 1060.31 | 1063.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 09:15:00 | 1082.20 | 1062.39 | 1062.88 | SL hit (close>static) qty=1.00 sl=1066.40 alert=retest2 |

### Cycle 77 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 1081.60 | 1066.23 | 1064.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 1086.50 | 1074.71 | 1070.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 1096.60 | 1101.25 | 1093.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 1096.60 | 1101.25 | 1093.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1096.60 | 1101.25 | 1093.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 11:45:00 | 1114.50 | 1103.82 | 1095.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:45:00 | 1115.30 | 1121.44 | 1115.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 15:15:00 | 1190.00 | 1193.85 | 1194.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 15:15:00 | 1190.00 | 1193.85 | 1194.17 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 09:15:00 | 1226.40 | 1200.36 | 1197.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 12:15:00 | 1248.70 | 1217.45 | 1206.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 1216.10 | 1223.53 | 1213.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 10:00:00 | 1216.10 | 1223.53 | 1213.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1215.70 | 1221.96 | 1213.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:45:00 | 1211.10 | 1221.96 | 1213.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 1211.60 | 1219.89 | 1213.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:45:00 | 1209.00 | 1219.89 | 1213.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 1208.40 | 1217.59 | 1213.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:45:00 | 1208.80 | 1217.59 | 1213.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1209.20 | 1215.77 | 1213.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 1209.20 | 1215.77 | 1213.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 1217.40 | 1216.10 | 1213.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:45:00 | 1220.70 | 1216.10 | 1213.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 1236.50 | 1221.82 | 1217.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 1223.20 | 1221.82 | 1217.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1380.50 | 1392.76 | 1382.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:00:00 | 1380.50 | 1392.76 | 1382.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 1377.00 | 1389.61 | 1381.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:00:00 | 1377.00 | 1389.61 | 1381.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 1379.60 | 1387.61 | 1381.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:00:00 | 1379.60 | 1387.61 | 1381.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 1368.20 | 1383.72 | 1380.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:00:00 | 1368.20 | 1383.72 | 1380.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 15:15:00 | 1359.00 | 1374.55 | 1376.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 1350.00 | 1367.03 | 1372.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 13:15:00 | 1359.70 | 1354.20 | 1360.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 13:15:00 | 1359.70 | 1354.20 | 1360.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 1359.70 | 1354.20 | 1360.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:00:00 | 1359.70 | 1354.20 | 1360.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 1366.00 | 1356.56 | 1360.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:00:00 | 1366.00 | 1356.56 | 1360.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 1365.00 | 1358.25 | 1361.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 1379.70 | 1358.25 | 1361.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 1406.00 | 1367.80 | 1365.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 1419.90 | 1397.48 | 1383.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 15:15:00 | 1409.00 | 1409.32 | 1397.07 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 09:15:00 | 1413.00 | 1409.32 | 1397.07 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 1399.00 | 1406.44 | 1401.46 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-17 15:15:00 | 1399.00 | 1406.44 | 1401.46 | SL hit (close<ema400) qty=1.00 sl=1401.46 alert=retest1 |

### Cycle 82 — SELL (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 14:15:00 | 1392.10 | 1402.98 | 1403.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 15:15:00 | 1382.00 | 1398.79 | 1401.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 12:15:00 | 1413.80 | 1396.26 | 1398.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 12:15:00 | 1413.80 | 1396.26 | 1398.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 1413.80 | 1396.26 | 1398.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:00:00 | 1413.80 | 1396.26 | 1398.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 1401.50 | 1397.31 | 1398.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:30:00 | 1403.70 | 1397.31 | 1398.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 1398.70 | 1397.59 | 1398.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:45:00 | 1405.70 | 1397.59 | 1398.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 1396.00 | 1397.27 | 1398.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 1385.20 | 1397.27 | 1398.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 09:15:00 | 1413.10 | 1400.44 | 1399.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 14:15:00 | 1438.60 | 1413.98 | 1407.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 11:15:00 | 1427.60 | 1428.96 | 1417.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 11:45:00 | 1428.90 | 1428.96 | 1417.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1429.40 | 1430.50 | 1421.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:45:00 | 1433.20 | 1430.50 | 1421.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 1402.70 | 1425.18 | 1420.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:45:00 | 1404.60 | 1425.18 | 1420.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 1392.80 | 1414.58 | 1416.34 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 1418.90 | 1400.63 | 1399.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 15:15:00 | 1425.00 | 1408.44 | 1403.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 1426.30 | 1435.55 | 1425.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 14:15:00 | 1426.30 | 1435.55 | 1425.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 1426.30 | 1435.55 | 1425.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 1426.30 | 1435.55 | 1425.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 1427.10 | 1433.86 | 1425.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 1423.80 | 1433.86 | 1425.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1419.50 | 1430.99 | 1425.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 1419.50 | 1430.99 | 1425.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 1409.40 | 1426.67 | 1423.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:00:00 | 1409.40 | 1426.67 | 1423.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2025-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 12:15:00 | 1418.00 | 1421.45 | 1421.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 1371.70 | 1409.11 | 1415.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 12:15:00 | 1380.00 | 1379.97 | 1390.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-05 12:45:00 | 1382.10 | 1379.97 | 1390.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 1382.00 | 1380.62 | 1387.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 1367.10 | 1380.62 | 1387.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:30:00 | 1377.00 | 1380.42 | 1386.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 12:15:00 | 1378.80 | 1380.36 | 1385.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 14:00:00 | 1379.80 | 1380.36 | 1385.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1334.40 | 1346.55 | 1360.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:30:00 | 1330.60 | 1343.84 | 1358.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 11:15:00 | 1327.60 | 1343.84 | 1358.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 09:30:00 | 1332.00 | 1347.99 | 1354.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 14:15:00 | 1365.40 | 1356.70 | 1356.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 1365.40 | 1356.70 | 1356.40 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 12:15:00 | 1354.10 | 1356.23 | 1356.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 13:15:00 | 1347.10 | 1354.40 | 1355.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 10:15:00 | 1280.00 | 1279.03 | 1294.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 10:30:00 | 1283.90 | 1279.03 | 1294.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 1268.10 | 1263.48 | 1275.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:00:00 | 1268.10 | 1263.48 | 1275.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 1258.20 | 1262.91 | 1272.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 09:15:00 | 1249.60 | 1269.21 | 1271.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 14:15:00 | 1241.70 | 1239.29 | 1239.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 14:15:00 | 1241.70 | 1239.29 | 1239.28 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 15:15:00 | 1237.20 | 1238.87 | 1239.09 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 1244.60 | 1240.02 | 1239.59 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 13:15:00 | 1235.00 | 1238.84 | 1239.24 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 1245.20 | 1240.12 | 1239.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 15:15:00 | 1252.00 | 1247.38 | 1244.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 09:15:00 | 1239.30 | 1245.77 | 1243.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 1239.30 | 1245.77 | 1243.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1239.30 | 1245.77 | 1243.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 1239.30 | 1245.77 | 1243.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1242.60 | 1245.13 | 1243.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:30:00 | 1246.50 | 1245.13 | 1243.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 1238.50 | 1243.81 | 1243.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:00:00 | 1238.50 | 1243.81 | 1243.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 1243.00 | 1243.65 | 1243.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 15:15:00 | 1248.30 | 1243.25 | 1242.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 10:15:00 | 1238.30 | 1242.08 | 1242.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 10:15:00 | 1238.30 | 1242.08 | 1242.54 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 12:15:00 | 1245.00 | 1243.20 | 1243.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 09:15:00 | 1251.10 | 1245.60 | 1244.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 14:15:00 | 1234.80 | 1245.26 | 1244.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 14:15:00 | 1234.80 | 1245.26 | 1244.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 1234.80 | 1245.26 | 1244.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 1234.80 | 1245.26 | 1244.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 15:15:00 | 1236.90 | 1243.59 | 1244.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 12:15:00 | 1232.80 | 1240.49 | 1242.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 1243.70 | 1237.66 | 1240.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 1243.70 | 1237.66 | 1240.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 1243.70 | 1237.66 | 1240.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:30:00 | 1244.70 | 1237.66 | 1240.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1242.50 | 1238.62 | 1240.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:30:00 | 1240.00 | 1241.50 | 1241.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 1248.10 | 1242.82 | 1242.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 12:15:00 | 1248.10 | 1242.82 | 1242.13 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 14:15:00 | 1232.60 | 1241.44 | 1241.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 14:15:00 | 1224.40 | 1234.53 | 1237.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 1226.20 | 1225.40 | 1230.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 1226.20 | 1225.40 | 1230.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1226.20 | 1225.40 | 1230.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 1226.30 | 1225.40 | 1230.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 1235.10 | 1227.60 | 1229.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 10:00:00 | 1235.10 | 1227.60 | 1229.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 1240.00 | 1230.08 | 1230.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 11:15:00 | 1255.30 | 1235.12 | 1232.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 09:15:00 | 1278.10 | 1305.68 | 1289.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 1278.10 | 1305.68 | 1289.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1278.10 | 1305.68 | 1289.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:30:00 | 1280.80 | 1305.68 | 1289.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 1277.00 | 1299.94 | 1288.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 11:00:00 | 1277.00 | 1299.94 | 1288.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 1253.10 | 1279.06 | 1280.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 10:15:00 | 1248.50 | 1265.50 | 1273.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 13:15:00 | 1241.40 | 1239.54 | 1251.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 14:00:00 | 1241.40 | 1239.54 | 1251.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 1209.80 | 1233.59 | 1247.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 15:15:00 | 1206.00 | 1233.59 | 1247.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 13:15:00 | 1192.70 | 1221.75 | 1235.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 1145.70 | 1175.88 | 1197.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 1158.90 | 1156.95 | 1174.79 | SL hit (close>ema200) qty=0.50 sl=1156.95 alert=retest2 |

### Cycle 101 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 1170.10 | 1160.13 | 1159.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 1183.20 | 1164.74 | 1161.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 14:15:00 | 1189.80 | 1192.65 | 1183.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 15:00:00 | 1189.80 | 1192.65 | 1183.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 1191.20 | 1191.56 | 1186.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 1193.00 | 1191.09 | 1187.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 10:15:00 | 1183.50 | 1189.22 | 1187.30 | SL hit (close<static) qty=1.00 sl=1185.70 alert=retest2 |

### Cycle 102 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 1171.10 | 1183.66 | 1185.12 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 13:15:00 | 1193.50 | 1186.07 | 1185.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 1199.50 | 1190.36 | 1187.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 1177.20 | 1192.46 | 1190.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 1177.20 | 1192.46 | 1190.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1177.20 | 1192.46 | 1190.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:15:00 | 1175.40 | 1192.46 | 1190.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 1175.00 | 1188.97 | 1189.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 12:15:00 | 1165.20 | 1181.69 | 1185.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1172.00 | 1168.27 | 1173.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 1172.00 | 1168.27 | 1173.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1172.00 | 1168.27 | 1173.26 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 1201.20 | 1176.42 | 1175.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 15:15:00 | 1211.00 | 1183.33 | 1178.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 14:15:00 | 1233.90 | 1236.57 | 1226.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-21 14:30:00 | 1233.90 | 1236.57 | 1226.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1241.50 | 1237.56 | 1227.79 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 09:15:00 | 1219.00 | 1230.48 | 1231.44 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 11:15:00 | 1244.80 | 1233.64 | 1232.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 13:15:00 | 1275.90 | 1243.38 | 1237.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 14:15:00 | 1315.00 | 1326.50 | 1301.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 15:00:00 | 1315.00 | 1326.50 | 1301.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1310.80 | 1322.05 | 1313.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:45:00 | 1313.20 | 1322.05 | 1313.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1294.70 | 1316.58 | 1312.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 1294.70 | 1316.58 | 1312.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 1294.50 | 1312.16 | 1310.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:45:00 | 1293.10 | 1312.16 | 1310.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 12:15:00 | 1295.70 | 1308.87 | 1309.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 1275.50 | 1300.55 | 1305.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 15:15:00 | 1281.00 | 1279.08 | 1289.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-04 09:15:00 | 1266.10 | 1279.08 | 1289.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 1252.40 | 1233.68 | 1243.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:00:00 | 1252.40 | 1233.68 | 1243.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 1242.70 | 1235.48 | 1243.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 1231.00 | 1239.19 | 1244.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 12:15:00 | 1255.80 | 1248.14 | 1247.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 1255.80 | 1248.14 | 1247.18 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 1240.90 | 1246.44 | 1246.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 14:15:00 | 1235.30 | 1242.16 | 1244.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 1242.10 | 1241.33 | 1243.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 1242.10 | 1241.33 | 1243.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 1242.10 | 1241.33 | 1243.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:45:00 | 1244.20 | 1241.33 | 1243.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 1244.00 | 1241.86 | 1243.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:30:00 | 1244.30 | 1241.86 | 1243.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 1240.20 | 1241.53 | 1243.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:30:00 | 1243.00 | 1241.53 | 1243.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1239.80 | 1239.29 | 1241.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:15:00 | 1242.60 | 1239.29 | 1241.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1240.90 | 1239.61 | 1241.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 1240.90 | 1239.61 | 1241.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 1237.80 | 1239.25 | 1240.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:30:00 | 1241.30 | 1239.25 | 1240.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 1236.20 | 1236.53 | 1239.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:45:00 | 1237.50 | 1236.53 | 1239.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1232.00 | 1235.38 | 1238.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 10:15:00 | 1208.40 | 1237.29 | 1238.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:00:00 | 1224.30 | 1234.69 | 1236.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 10:15:00 | 1242.40 | 1191.85 | 1191.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 10:15:00 | 1242.40 | 1191.85 | 1191.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 11:15:00 | 1257.00 | 1204.88 | 1197.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 09:15:00 | 1223.40 | 1234.07 | 1217.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 1223.40 | 1234.07 | 1217.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 1223.40 | 1234.07 | 1217.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:45:00 | 1220.20 | 1234.07 | 1217.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 1215.30 | 1228.54 | 1217.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:45:00 | 1214.30 | 1228.54 | 1217.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 1206.90 | 1224.22 | 1216.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 12:45:00 | 1210.40 | 1224.22 | 1216.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 1200.20 | 1219.41 | 1215.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:00:00 | 1200.20 | 1219.41 | 1215.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2025-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 15:15:00 | 1198.00 | 1212.01 | 1212.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 09:15:00 | 1192.60 | 1208.13 | 1210.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 10:15:00 | 1204.00 | 1198.12 | 1202.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 10:15:00 | 1204.00 | 1198.12 | 1202.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1204.00 | 1198.12 | 1202.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 1200.70 | 1198.12 | 1202.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 1208.50 | 1200.20 | 1203.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:00:00 | 1208.50 | 1200.20 | 1203.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 1203.50 | 1200.86 | 1203.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:15:00 | 1212.80 | 1200.86 | 1203.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 1206.00 | 1201.89 | 1203.47 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 1266.50 | 1214.81 | 1209.20 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 13:15:00 | 1214.50 | 1218.21 | 1218.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 14:15:00 | 1206.70 | 1215.91 | 1217.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 10:15:00 | 1232.70 | 1216.57 | 1217.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 10:15:00 | 1232.70 | 1216.57 | 1217.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1232.70 | 1216.57 | 1217.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:30:00 | 1225.70 | 1216.57 | 1217.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 1217.50 | 1216.76 | 1217.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 12:15:00 | 1215.00 | 1216.76 | 1217.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 12:15:00 | 1223.60 | 1218.13 | 1217.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 12:15:00 | 1223.60 | 1218.13 | 1217.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 12:15:00 | 1230.30 | 1224.61 | 1221.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 11:15:00 | 1227.20 | 1227.44 | 1224.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-05 12:00:00 | 1227.20 | 1227.44 | 1224.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 1230.00 | 1227.95 | 1225.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:30:00 | 1226.60 | 1227.95 | 1225.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 1222.00 | 1226.76 | 1224.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:00:00 | 1222.00 | 1226.76 | 1224.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 1221.10 | 1225.63 | 1224.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:30:00 | 1220.00 | 1225.63 | 1224.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 1221.90 | 1224.88 | 1224.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 1212.00 | 1224.88 | 1224.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 1225.40 | 1226.06 | 1225.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 1225.40 | 1226.06 | 1225.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 1227.90 | 1226.43 | 1225.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 1227.90 | 1226.43 | 1225.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 1227.20 | 1226.58 | 1225.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:30:00 | 1222.00 | 1226.58 | 1225.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 1226.40 | 1226.55 | 1225.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:00:00 | 1226.40 | 1226.55 | 1225.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 14:15:00 | 1202.00 | 1221.64 | 1223.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 15:15:00 | 1200.20 | 1217.35 | 1221.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 1217.30 | 1214.95 | 1218.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 1217.30 | 1214.95 | 1218.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 1221.30 | 1216.39 | 1218.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:45:00 | 1221.00 | 1216.39 | 1218.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 1228.00 | 1218.71 | 1219.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 1228.00 | 1218.71 | 1219.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 1229.00 | 1220.77 | 1220.60 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 12:15:00 | 1209.70 | 1220.01 | 1220.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 1208.80 | 1216.10 | 1218.56 | Break + close below crossover candle low |

### Cycle 119 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 1290.10 | 1229.89 | 1224.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 10:15:00 | 1307.60 | 1245.43 | 1231.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 15:15:00 | 1260.00 | 1268.86 | 1250.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-12 09:15:00 | 1255.00 | 1268.86 | 1250.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1257.80 | 1266.65 | 1251.49 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 1242.20 | 1250.27 | 1251.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 1240.30 | 1246.02 | 1248.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 1225.10 | 1224.92 | 1233.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:00:00 | 1225.10 | 1224.92 | 1233.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 1234.60 | 1217.97 | 1221.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:00:00 | 1234.60 | 1217.97 | 1221.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 1232.50 | 1220.88 | 1222.76 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2025-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 12:15:00 | 1239.80 | 1227.07 | 1225.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 13:15:00 | 1245.90 | 1230.83 | 1227.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 11:15:00 | 1247.00 | 1247.79 | 1241.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 11:45:00 | 1246.00 | 1247.79 | 1241.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 1258.20 | 1255.55 | 1248.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:45:00 | 1252.90 | 1255.55 | 1248.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 1245.10 | 1254.06 | 1249.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 15:00:00 | 1245.10 | 1254.06 | 1249.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 1245.00 | 1252.25 | 1249.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 1233.70 | 1252.25 | 1249.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1246.20 | 1247.96 | 1247.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:45:00 | 1244.10 | 1247.96 | 1247.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 1246.30 | 1247.63 | 1247.63 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 09:15:00 | 1253.30 | 1247.78 | 1247.57 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 15:15:00 | 1241.10 | 1247.24 | 1247.75 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 15:15:00 | 1250.90 | 1247.47 | 1247.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 1261.10 | 1250.19 | 1248.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 11:15:00 | 1262.50 | 1267.38 | 1261.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 11:15:00 | 1262.50 | 1267.38 | 1261.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 1262.50 | 1267.38 | 1261.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:00:00 | 1262.50 | 1267.38 | 1261.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 1257.90 | 1265.49 | 1260.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:00:00 | 1257.90 | 1265.49 | 1260.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 1256.80 | 1263.75 | 1260.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:45:00 | 1258.40 | 1263.75 | 1260.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1249.40 | 1259.10 | 1258.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 1244.70 | 1259.10 | 1258.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 10:15:00 | 1251.00 | 1257.48 | 1258.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 1242.80 | 1252.94 | 1255.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 1179.00 | 1177.94 | 1190.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 13:00:00 | 1179.00 | 1177.94 | 1190.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1198.00 | 1182.34 | 1189.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:30:00 | 1175.20 | 1183.45 | 1189.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 1175.10 | 1182.34 | 1186.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:30:00 | 1174.10 | 1179.22 | 1181.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 1116.44 | 1138.88 | 1154.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 1116.34 | 1138.88 | 1154.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 1115.39 | 1138.88 | 1154.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-21 10:15:00 | 1057.68 | 1098.59 | 1124.04 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 127 — BUY (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 10:15:00 | 1132.00 | 1112.34 | 1111.54 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 15:15:00 | 1099.50 | 1110.51 | 1111.67 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2026-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 13:15:00 | 1121.40 | 1113.43 | 1112.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 14:15:00 | 1129.60 | 1116.66 | 1114.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-28 09:15:00 | 1116.30 | 1118.28 | 1115.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 09:15:00 | 1116.30 | 1118.28 | 1115.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1116.30 | 1118.28 | 1115.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:45:00 | 1115.80 | 1118.28 | 1115.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 1112.10 | 1117.04 | 1115.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 11:30:00 | 1133.60 | 1118.29 | 1115.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 09:15:00 | 1147.30 | 1160.22 | 1161.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 1147.30 | 1160.22 | 1161.27 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1173.40 | 1157.85 | 1157.77 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 1147.70 | 1160.76 | 1162.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 1144.50 | 1154.76 | 1158.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 12:15:00 | 1161.10 | 1155.49 | 1157.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 12:15:00 | 1161.10 | 1155.49 | 1157.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 1161.10 | 1155.49 | 1157.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 1161.10 | 1155.49 | 1157.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 1159.30 | 1156.26 | 1157.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 14:15:00 | 1155.00 | 1156.26 | 1157.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 1148.10 | 1157.48 | 1158.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 1171.90 | 1155.94 | 1155.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 1171.90 | 1155.94 | 1155.18 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 1148.60 | 1157.39 | 1158.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 13:15:00 | 1143.20 | 1151.89 | 1155.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 1126.00 | 1122.42 | 1131.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 09:45:00 | 1127.00 | 1122.42 | 1131.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1123.90 | 1123.18 | 1127.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:00:00 | 1108.10 | 1116.30 | 1122.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:30:00 | 1108.30 | 1114.62 | 1121.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 13:30:00 | 1108.60 | 1112.13 | 1116.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 14:30:00 | 1108.20 | 1111.01 | 1115.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 1107.40 | 1110.09 | 1114.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:00:00 | 1105.70 | 1109.21 | 1113.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 1052.69 | 1064.11 | 1070.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 1052.88 | 1064.11 | 1070.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 1053.17 | 1064.11 | 1070.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 1052.79 | 1064.11 | 1070.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 1050.41 | 1064.11 | 1070.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-02 09:15:00 | 997.29 | 1040.77 | 1054.35 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 135 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 1004.30 | 989.22 | 987.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 1012.50 | 998.97 | 993.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 1002.70 | 1002.72 | 996.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 13:00:00 | 1002.70 | 1002.72 | 996.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 987.70 | 998.91 | 996.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 987.70 | 998.91 | 996.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 986.50 | 996.43 | 995.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 977.40 | 996.43 | 995.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 975.00 | 992.15 | 993.31 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 13:15:00 | 1001.00 | 995.05 | 994.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 14:15:00 | 1016.00 | 999.24 | 996.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 1001.40 | 1002.63 | 998.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 1001.40 | 1002.63 | 998.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 1001.40 | 1002.63 | 998.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 09:30:00 | 1025.00 | 1014.29 | 1006.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 09:15:00 | 1044.10 | 1060.49 | 1062.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 09:15:00 | 1044.10 | 1060.49 | 1062.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 10:15:00 | 1025.70 | 1053.53 | 1059.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 13:15:00 | 1036.00 | 1033.97 | 1042.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-23 13:45:00 | 1033.80 | 1033.97 | 1042.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 1026.50 | 1032.48 | 1041.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 09:45:00 | 1019.30 | 1029.46 | 1038.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 1048.50 | 1033.36 | 1038.44 | SL hit (close>static) qty=1.00 sl=1041.90 alert=retest2 |

### Cycle 139 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 1075.90 | 1041.87 | 1041.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 1095.00 | 1064.50 | 1053.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 09:15:00 | 1107.90 | 1110.57 | 1098.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 1107.90 | 1110.57 | 1098.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 1107.90 | 1110.57 | 1098.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 13:45:00 | 1116.90 | 1111.79 | 1102.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:45:00 | 1116.70 | 1125.94 | 1125.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 13:15:00 | 1124.50 | 1137.45 | 1137.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2026-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 13:15:00 | 1124.50 | 1137.45 | 1137.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-08 15:15:00 | 1117.00 | 1131.11 | 1134.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 09:15:00 | 1121.10 | 1116.14 | 1123.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 09:15:00 | 1121.10 | 1116.14 | 1123.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 1121.10 | 1116.14 | 1123.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 09:45:00 | 1121.60 | 1116.14 | 1123.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 1121.50 | 1117.22 | 1122.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 14:15:00 | 1112.60 | 1118.15 | 1121.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 15:15:00 | 1115.00 | 1118.44 | 1121.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 13:15:00 | 1114.70 | 1118.46 | 1120.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 14:00:00 | 1115.80 | 1117.92 | 1119.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 1119.70 | 1118.28 | 1119.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 14:45:00 | 1117.00 | 1118.28 | 1119.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 1121.00 | 1118.82 | 1120.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:15:00 | 1126.50 | 1118.82 | 1120.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1155.00 | 1126.06 | 1123.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 1155.00 | 1126.06 | 1123.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 10:15:00 | 1160.70 | 1132.99 | 1126.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 14:15:00 | 1191.30 | 1195.44 | 1172.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 15:00:00 | 1191.30 | 1195.44 | 1172.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 15:15:00 | 1180.00 | 1192.35 | 1173.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 1193.20 | 1192.35 | 1173.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 1202.90 | 1199.34 | 1198.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 10:45:00 | 1192.70 | 1201.01 | 1200.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 11:15:00 | 1192.20 | 1199.25 | 1200.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 11:15:00 | 1192.20 | 1199.25 | 1200.10 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 15:15:00 | 1205.00 | 1200.64 | 1200.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 09:15:00 | 1205.90 | 1201.69 | 1200.92 | Break + close above crossover candle high |

### Cycle 144 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 1193.80 | 1200.11 | 1200.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 1188.60 | 1197.81 | 1199.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 15:15:00 | 1194.00 | 1193.55 | 1196.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:15:00 | 1194.60 | 1193.55 | 1196.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1198.10 | 1194.46 | 1196.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1200.30 | 1194.46 | 1196.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1199.00 | 1195.37 | 1196.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 1193.00 | 1195.37 | 1196.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:00:00 | 1195.00 | 1195.85 | 1196.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 15:15:00 | 1204.00 | 1197.57 | 1197.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 15:15:00 | 1204.00 | 1197.57 | 1197.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 1204.20 | 1198.90 | 1197.98 | Break + close above crossover candle high |

### Cycle 146 — SELL (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 13:15:00 | 1186.00 | 1197.49 | 1197.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 14:15:00 | 1181.00 | 1194.19 | 1196.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 15:15:00 | 1185.00 | 1179.79 | 1186.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 09:15:00 | 1190.60 | 1179.79 | 1186.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1212.00 | 1186.23 | 1188.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:45:00 | 1227.00 | 1186.23 | 1188.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 1196.00 | 1188.19 | 1189.13 | EMA400 retest candle locked (from downside) |

### Cycle 147 — BUY (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 11:15:00 | 1200.90 | 1190.73 | 1190.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 1242.10 | 1208.35 | 1199.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 10:15:00 | 1229.50 | 1237.42 | 1223.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 10:15:00 | 1229.50 | 1237.42 | 1223.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 1229.50 | 1237.42 | 1223.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 11:45:00 | 1242.80 | 1238.52 | 1225.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 10:15:00 | 1247.80 | 1263.51 | 1263.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 10:15:00 | 1247.80 | 1263.51 | 1263.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 11:15:00 | 1245.50 | 1259.91 | 1262.18 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-23 09:30:00 | 1000.85 | 2024-05-23 10:15:00 | 988.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-05-29 15:15:00 | 981.10 | 2024-05-30 15:15:00 | 990.50 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-05-31 09:15:00 | 978.25 | 2024-05-31 09:15:00 | 986.10 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-06-13 11:45:00 | 1014.00 | 2024-06-18 14:15:00 | 1016.20 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2024-06-25 11:30:00 | 1003.40 | 2024-07-01 11:15:00 | 1004.20 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2024-06-26 14:45:00 | 1003.70 | 2024-07-01 11:15:00 | 1004.20 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2024-06-27 09:15:00 | 999.05 | 2024-07-01 11:15:00 | 1004.20 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2024-07-03 09:15:00 | 1011.00 | 2024-07-08 10:15:00 | 1002.45 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest1 | 2024-07-03 10:45:00 | 1009.65 | 2024-07-08 10:15:00 | 1002.45 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-07-05 09:15:00 | 1012.05 | 2024-07-08 10:15:00 | 1002.45 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-07-05 12:00:00 | 1013.00 | 2024-07-08 10:15:00 | 1002.45 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-07-05 12:30:00 | 1014.75 | 2024-07-08 10:15:00 | 1002.45 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-07-05 14:00:00 | 1018.00 | 2024-07-08 10:15:00 | 1002.45 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-07-10 09:45:00 | 993.15 | 2024-07-11 13:15:00 | 1002.60 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-07-16 09:15:00 | 1012.25 | 2024-07-19 09:15:00 | 993.05 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-07-29 09:15:00 | 1045.50 | 2024-08-05 09:15:00 | 1038.80 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-08-12 09:15:00 | 1155.10 | 2024-08-12 15:15:00 | 1127.80 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2024-08-12 10:00:00 | 1152.25 | 2024-08-12 15:15:00 | 1127.80 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-08-12 13:45:00 | 1150.50 | 2024-08-12 15:15:00 | 1127.80 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-08-22 09:15:00 | 1158.60 | 2024-08-26 14:15:00 | 1143.75 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-08-22 09:45:00 | 1158.00 | 2024-08-26 15:15:00 | 1147.95 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-08-22 15:00:00 | 1163.40 | 2024-08-26 15:15:00 | 1147.95 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-08-23 10:30:00 | 1163.20 | 2024-08-26 15:15:00 | 1147.95 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-08-23 13:30:00 | 1166.60 | 2024-08-26 15:15:00 | 1147.95 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-09-03 09:15:00 | 1158.60 | 2024-09-03 11:15:00 | 1139.50 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-09-05 12:15:00 | 1139.00 | 2024-09-10 14:15:00 | 1143.00 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2024-09-05 13:00:00 | 1136.85 | 2024-09-10 14:15:00 | 1143.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-09-10 10:15:00 | 1138.10 | 2024-09-10 14:15:00 | 1143.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2024-09-16 11:30:00 | 1082.40 | 2024-09-23 10:15:00 | 1092.10 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-09-16 12:45:00 | 1080.05 | 2024-09-23 10:15:00 | 1092.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-09-17 09:30:00 | 1082.15 | 2024-09-23 10:15:00 | 1092.10 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2024-09-17 12:00:00 | 1081.70 | 2024-09-23 10:15:00 | 1092.10 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-09-17 14:30:00 | 1080.05 | 2024-09-23 10:15:00 | 1092.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-09-18 09:15:00 | 1080.10 | 2024-09-23 10:15:00 | 1092.10 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-09-18 11:45:00 | 1081.85 | 2024-09-23 10:15:00 | 1092.10 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-09-18 12:45:00 | 1080.55 | 2024-09-23 10:15:00 | 1092.10 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-09-19 11:00:00 | 1079.05 | 2024-09-23 10:15:00 | 1092.10 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-09-26 09:45:00 | 1071.00 | 2024-09-27 15:15:00 | 1017.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 11:00:00 | 1075.15 | 2024-09-27 15:15:00 | 1021.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-26 09:45:00 | 1071.00 | 2024-10-01 09:15:00 | 1044.00 | STOP_HIT | 0.50 | 2.52% |
| SELL | retest2 | 2024-09-27 11:00:00 | 1075.15 | 2024-10-01 09:15:00 | 1044.00 | STOP_HIT | 0.50 | 2.90% |
| SELL | retest2 | 2024-09-27 14:00:00 | 1059.10 | 2024-10-07 09:15:00 | 1006.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 14:00:00 | 1059.10 | 2024-10-08 10:15:00 | 996.00 | STOP_HIT | 0.50 | 5.96% |
| BUY | retest2 | 2024-10-18 10:30:00 | 1112.40 | 2024-10-22 09:15:00 | 1061.65 | STOP_HIT | 1.00 | -4.56% |
| BUY | retest2 | 2024-10-18 11:30:00 | 1113.30 | 2024-10-22 09:15:00 | 1061.65 | STOP_HIT | 1.00 | -4.64% |
| BUY | retest2 | 2024-10-21 09:15:00 | 1118.05 | 2024-10-22 09:15:00 | 1061.65 | STOP_HIT | 1.00 | -5.04% |
| BUY | retest2 | 2024-10-21 10:00:00 | 1123.40 | 2024-10-22 09:15:00 | 1061.65 | STOP_HIT | 1.00 | -5.50% |
| BUY | retest2 | 2024-11-01 18:00:00 | 1058.80 | 2024-11-04 09:15:00 | 1022.45 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest2 | 2024-11-04 15:00:00 | 1055.00 | 2024-11-07 09:15:00 | 1160.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-05 09:45:00 | 1060.70 | 2024-11-07 10:15:00 | 1166.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-06 09:15:00 | 1066.30 | 2024-11-07 10:15:00 | 1172.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-07 09:15:00 | 1144.80 | 2024-11-07 14:15:00 | 1259.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-21 13:15:00 | 1301.30 | 2024-11-21 15:15:00 | 1294.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2024-12-10 12:30:00 | 1110.05 | 2024-12-11 10:15:00 | 1113.75 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2024-12-10 13:15:00 | 1110.00 | 2024-12-11 10:15:00 | 1113.75 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2024-12-10 14:15:00 | 1109.95 | 2024-12-11 10:15:00 | 1113.75 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2024-12-10 15:00:00 | 1104.10 | 2024-12-11 10:15:00 | 1113.75 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-12-13 10:15:00 | 1080.80 | 2024-12-16 15:15:00 | 1102.30 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2024-12-13 12:30:00 | 1081.20 | 2024-12-16 15:15:00 | 1102.30 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2024-12-13 13:45:00 | 1078.05 | 2024-12-16 15:15:00 | 1102.30 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2024-12-13 15:15:00 | 1081.00 | 2024-12-16 15:15:00 | 1102.30 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2024-12-19 09:15:00 | 1068.60 | 2024-12-20 11:15:00 | 1090.20 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-12-19 14:00:00 | 1082.00 | 2024-12-20 11:15:00 | 1090.20 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-12-19 15:00:00 | 1082.35 | 2024-12-20 11:15:00 | 1090.20 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-12-20 09:45:00 | 1080.65 | 2024-12-20 11:15:00 | 1090.20 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-12-26 11:15:00 | 1087.05 | 2025-01-06 11:15:00 | 1077.15 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-12-26 11:45:00 | 1086.25 | 2025-01-06 11:15:00 | 1077.15 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-12-27 09:15:00 | 1086.95 | 2025-01-06 11:15:00 | 1077.15 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-12-27 09:45:00 | 1093.65 | 2025-01-06 11:15:00 | 1077.15 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-12-31 14:15:00 | 1115.10 | 2025-01-06 11:15:00 | 1077.15 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2024-12-31 15:00:00 | 1149.80 | 2025-01-06 11:15:00 | 1077.15 | STOP_HIT | 1.00 | -6.32% |
| SELL | retest2 | 2025-01-08 10:15:00 | 1058.25 | 2025-01-13 13:15:00 | 1017.26 | PARTIAL | 0.50 | 3.87% |
| SELL | retest2 | 2025-01-08 11:00:00 | 1058.85 | 2025-01-13 13:15:00 | 1017.36 | PARTIAL | 0.50 | 3.92% |
| SELL | retest2 | 2025-01-08 10:15:00 | 1058.25 | 2025-01-14 09:15:00 | 1045.25 | STOP_HIT | 0.50 | 1.23% |
| SELL | retest2 | 2025-01-08 11:00:00 | 1058.85 | 2025-01-14 09:15:00 | 1045.25 | STOP_HIT | 0.50 | 1.28% |
| SELL | retest2 | 2025-01-08 11:30:00 | 1057.55 | 2025-01-16 09:15:00 | 1058.15 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-01-08 14:30:00 | 1056.05 | 2025-01-16 09:15:00 | 1058.15 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-01-10 09:15:00 | 1060.00 | 2025-01-16 09:15:00 | 1058.15 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2025-01-10 13:15:00 | 1070.80 | 2025-01-16 09:15:00 | 1058.15 | STOP_HIT | 1.00 | 1.18% |
| SELL | retest2 | 2025-01-10 13:45:00 | 1070.90 | 2025-01-16 09:15:00 | 1058.15 | STOP_HIT | 1.00 | 1.19% |
| BUY | retest1 | 2025-01-17 14:00:00 | 1076.25 | 2025-01-22 10:15:00 | 1101.65 | STOP_HIT | 1.00 | 2.36% |
| BUY | retest2 | 2025-01-23 14:45:00 | 1161.00 | 2025-01-27 11:15:00 | 1092.05 | STOP_HIT | 1.00 | -5.94% |
| SELL | retest2 | 2025-02-07 14:00:00 | 1050.10 | 2025-02-13 11:15:00 | 1062.60 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-02-10 10:00:00 | 1044.35 | 2025-02-13 11:15:00 | 1062.60 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-02-11 09:15:00 | 1042.10 | 2025-02-13 11:15:00 | 1062.60 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-02-11 10:30:00 | 1048.85 | 2025-02-13 11:15:00 | 1062.60 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-02-13 09:15:00 | 1038.75 | 2025-02-13 11:15:00 | 1062.60 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-03-03 11:15:00 | 931.45 | 2025-03-04 12:15:00 | 1028.00 | STOP_HIT | 1.00 | -10.37% |
| SELL | retest2 | 2025-03-03 13:15:00 | 930.50 | 2025-03-04 12:15:00 | 1028.00 | STOP_HIT | 1.00 | -10.48% |
| BUY | retest1 | 2025-03-07 09:15:00 | 1069.35 | 2025-03-10 10:15:00 | 1029.05 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest1 | 2025-03-07 12:15:00 | 1052.95 | 2025-03-10 10:15:00 | 1029.05 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-04-09 09:15:00 | 1014.95 | 2025-04-11 09:15:00 | 1041.80 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-04-11 09:30:00 | 1027.15 | 2025-04-11 12:15:00 | 1042.45 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-04-11 11:00:00 | 1022.50 | 2025-04-11 12:15:00 | 1042.45 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-04-21 10:00:00 | 1082.10 | 2025-04-23 15:15:00 | 1073.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-04-22 15:00:00 | 1080.90 | 2025-04-23 15:15:00 | 1073.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-04-23 11:00:00 | 1078.40 | 2025-04-23 15:15:00 | 1073.00 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-05-05 11:45:00 | 995.45 | 2025-05-05 14:15:00 | 1013.85 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-05-05 12:30:00 | 995.50 | 2025-05-05 14:15:00 | 1013.85 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-05-19 10:15:00 | 1053.00 | 2025-05-23 14:15:00 | 1052.60 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-05-20 15:15:00 | 1055.00 | 2025-05-23 14:15:00 | 1052.60 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-06-05 12:45:00 | 1060.00 | 2025-06-06 09:15:00 | 1082.20 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-06-12 11:45:00 | 1114.50 | 2025-06-27 15:15:00 | 1190.00 | STOP_HIT | 1.00 | 6.77% |
| BUY | retest2 | 2025-06-16 10:45:00 | 1115.30 | 2025-06-27 15:15:00 | 1190.00 | STOP_HIT | 1.00 | 6.70% |
| BUY | retest1 | 2025-07-17 09:15:00 | 1413.00 | 2025-07-17 15:15:00 | 1399.00 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-07-18 09:15:00 | 1408.90 | 2025-07-18 09:15:00 | 1395.50 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-07-18 13:00:00 | 1409.20 | 2025-07-21 14:15:00 | 1392.10 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-07-18 14:30:00 | 1410.40 | 2025-07-21 14:15:00 | 1392.10 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-07-18 15:15:00 | 1410.30 | 2025-07-21 14:15:00 | 1392.10 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-08-06 09:15:00 | 1367.10 | 2025-08-11 14:15:00 | 1365.40 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2025-08-06 10:30:00 | 1377.00 | 2025-08-11 14:15:00 | 1365.40 | STOP_HIT | 1.00 | 0.84% |
| SELL | retest2 | 2025-08-06 12:15:00 | 1378.80 | 2025-08-11 14:15:00 | 1365.40 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2025-08-06 14:00:00 | 1379.80 | 2025-08-11 14:15:00 | 1365.40 | STOP_HIT | 1.00 | 1.04% |
| SELL | retest2 | 2025-08-08 10:30:00 | 1330.60 | 2025-08-11 14:15:00 | 1365.40 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-08-08 11:15:00 | 1327.60 | 2025-08-11 14:15:00 | 1365.40 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-08-11 09:30:00 | 1332.00 | 2025-08-11 14:15:00 | 1365.40 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-08-25 09:15:00 | 1249.60 | 2025-09-02 14:15:00 | 1241.70 | STOP_HIT | 1.00 | 0.63% |
| BUY | retest2 | 2025-09-05 15:15:00 | 1248.30 | 2025-09-08 10:15:00 | 1238.30 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-09-11 11:30:00 | 1240.00 | 2025-09-11 12:15:00 | 1248.10 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-09-23 15:15:00 | 1206.00 | 2025-09-26 09:15:00 | 1145.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 15:15:00 | 1206.00 | 2025-09-29 09:15:00 | 1158.90 | STOP_HIT | 0.50 | 3.91% |
| SELL | retest2 | 2025-09-24 13:15:00 | 1192.70 | 2025-10-01 15:15:00 | 1170.10 | STOP_HIT | 1.00 | 1.89% |
| BUY | retest2 | 2025-10-08 09:15:00 | 1193.00 | 2025-10-08 10:15:00 | 1183.50 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-11-10 09:15:00 | 1231.00 | 2025-11-10 12:15:00 | 1255.80 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-11-17 10:15:00 | 1208.40 | 2025-11-21 10:15:00 | 1242.40 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-11-17 11:00:00 | 1224.30 | 2025-11-21 10:15:00 | 1242.40 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-12-02 12:15:00 | 1215.00 | 2025-12-02 12:15:00 | 1223.60 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-01-13 09:30:00 | 1175.20 | 2026-01-20 10:15:00 | 1116.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 14:00:00 | 1175.10 | 2026-01-20 10:15:00 | 1116.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 12:30:00 | 1174.10 | 2026-01-20 10:15:00 | 1115.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 09:30:00 | 1175.20 | 2026-01-21 10:15:00 | 1057.68 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-13 14:00:00 | 1175.10 | 2026-01-21 10:15:00 | 1057.59 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-16 12:30:00 | 1174.10 | 2026-01-21 10:15:00 | 1056.69 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-01-28 11:30:00 | 1133.60 | 2026-02-02 09:15:00 | 1147.30 | STOP_HIT | 1.00 | 1.21% |
| SELL | retest2 | 2026-02-06 14:15:00 | 1155.00 | 2026-02-10 09:15:00 | 1171.90 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-02-09 09:15:00 | 1148.10 | 2026-02-10 09:15:00 | 1171.90 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2026-02-17 14:00:00 | 1108.10 | 2026-02-27 09:15:00 | 1052.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 14:30:00 | 1108.30 | 2026-02-27 09:15:00 | 1052.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 13:30:00 | 1108.60 | 2026-02-27 09:15:00 | 1053.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 14:30:00 | 1108.20 | 2026-02-27 09:15:00 | 1052.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 11:00:00 | 1105.70 | 2026-02-27 09:15:00 | 1050.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 14:00:00 | 1108.10 | 2026-03-02 09:15:00 | 997.29 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-17 14:30:00 | 1108.30 | 2026-03-02 09:15:00 | 997.47 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-18 13:30:00 | 1108.60 | 2026-03-02 09:15:00 | 997.74 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-18 14:30:00 | 1108.20 | 2026-03-02 09:15:00 | 997.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-19 11:00:00 | 1105.70 | 2026-03-02 09:15:00 | 995.13 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-03-16 09:30:00 | 1025.00 | 2026-03-20 09:15:00 | 1044.10 | STOP_HIT | 1.00 | 1.86% |
| SELL | retest2 | 2026-03-24 09:45:00 | 1019.30 | 2026-03-24 11:15:00 | 1048.50 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2026-03-30 13:45:00 | 1116.90 | 2026-04-08 13:15:00 | 1124.50 | STOP_HIT | 1.00 | 0.68% |
| BUY | retest2 | 2026-04-06 13:45:00 | 1116.70 | 2026-04-08 13:15:00 | 1124.50 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2026-04-10 14:15:00 | 1112.60 | 2026-04-15 09:15:00 | 1155.00 | STOP_HIT | 1.00 | -3.81% |
| SELL | retest2 | 2026-04-10 15:15:00 | 1115.00 | 2026-04-15 09:15:00 | 1155.00 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2026-04-13 13:15:00 | 1114.70 | 2026-04-15 09:15:00 | 1155.00 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2026-04-13 14:00:00 | 1115.80 | 2026-04-15 09:15:00 | 1155.00 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest2 | 2026-04-17 09:15:00 | 1193.20 | 2026-04-23 11:15:00 | 1192.20 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2026-04-22 09:15:00 | 1202.90 | 2026-04-23 11:15:00 | 1192.20 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2026-04-23 10:45:00 | 1192.70 | 2026-04-23 11:15:00 | 1192.20 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2026-04-27 11:15:00 | 1193.00 | 2026-04-27 15:15:00 | 1204.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-04-27 14:00:00 | 1195.00 | 2026-04-27 15:15:00 | 1204.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2026-05-05 11:45:00 | 1242.80 | 2026-05-08 10:15:00 | 1247.80 | STOP_HIT | 1.00 | 0.40% |
