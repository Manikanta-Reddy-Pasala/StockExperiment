# E.I.D. Parry (India) Ltd. (EIDPARRY)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 834.95
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 82 |
| ALERT1 | 53 |
| ALERT2 | 51 |
| ALERT2_SKIP | 30 |
| ALERT3 | 134 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 65 |
| PARTIAL | 12 |
| TARGET_HIT | 3 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 78 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 32 / 46
- **Target hits / Stop hits / Partials:** 3 / 63 / 12
- **Avg / median % per leg:** 1.27% / -0.17%
- **Sum % (uncompounded):** 99.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 8 | 34.8% | 3 | 20 | 0 | 0.82% | 19.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 23 | 8 | 34.8% | 3 | 20 | 0 | 0.82% | 19.0% |
| SELL (all) | 55 | 24 | 43.6% | 0 | 43 | 12 | 1.46% | 80.4% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.22% | -1.2% |
| SELL @ 3rd Alert (retest2) | 54 | 24 | 44.4% | 0 | 42 | 12 | 1.51% | 81.6% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.22% | -1.2% |
| retest2 (combined) | 77 | 32 | 41.6% | 3 | 62 | 12 | 1.31% | 100.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 871.00 | 843.08 | 841.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 876.30 | 849.73 | 844.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 13:15:00 | 980.80 | 980.99 | 967.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 13:45:00 | 981.75 | 980.99 | 967.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 972.25 | 978.79 | 970.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:00:00 | 981.75 | 977.59 | 972.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 15:00:00 | 984.95 | 979.06 | 973.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 10:30:00 | 981.70 | 978.61 | 974.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 12:00:00 | 979.80 | 978.85 | 974.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 973.30 | 977.74 | 974.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:00:00 | 973.30 | 977.74 | 974.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 980.00 | 978.19 | 975.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 14:30:00 | 982.50 | 979.02 | 975.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 09:15:00 | 990.65 | 979.39 | 976.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 10:45:00 | 985.80 | 989.49 | 989.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 11:15:00 | 984.45 | 988.48 | 988.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 11:15:00 | 984.45 | 988.48 | 988.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 12:15:00 | 978.75 | 986.54 | 987.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 09:15:00 | 989.00 | 985.57 | 986.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 09:15:00 | 989.00 | 985.57 | 986.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 989.00 | 985.57 | 986.83 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 12:15:00 | 989.00 | 987.56 | 987.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 13:15:00 | 993.05 | 988.66 | 988.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 14:15:00 | 986.95 | 988.32 | 987.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 14:15:00 | 986.95 | 988.32 | 987.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 986.95 | 988.32 | 987.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 15:00:00 | 986.95 | 988.32 | 987.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 989.00 | 988.45 | 988.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:15:00 | 975.30 | 988.45 | 988.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 09:15:00 | 970.00 | 984.76 | 986.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 14:15:00 | 964.15 | 975.17 | 980.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 12:15:00 | 957.60 | 957.41 | 964.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 12:45:00 | 958.55 | 957.41 | 964.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 986.75 | 961.59 | 964.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 10:00:00 | 986.75 | 961.59 | 964.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 10:15:00 | 984.75 | 966.22 | 965.95 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 12:15:00 | 973.00 | 975.35 | 975.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 14:15:00 | 970.90 | 974.09 | 974.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 10:15:00 | 974.55 | 972.99 | 974.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 10:15:00 | 974.55 | 972.99 | 974.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 974.55 | 972.99 | 974.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 11:00:00 | 974.55 | 972.99 | 974.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 11:15:00 | 973.80 | 973.15 | 974.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 15:00:00 | 969.70 | 972.35 | 973.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 09:15:00 | 977.30 | 972.33 | 973.19 | SL hit (close>static) qty=1.00 sl=976.70 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 11:15:00 | 977.35 | 973.78 | 973.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 09:15:00 | 999.05 | 979.02 | 976.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 989.45 | 999.73 | 992.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 13:15:00 | 989.45 | 999.73 | 992.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 989.45 | 999.73 | 992.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 989.45 | 999.73 | 992.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 989.75 | 997.73 | 992.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:00:00 | 989.75 | 997.73 | 992.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 984.50 | 995.09 | 991.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 978.65 | 995.09 | 991.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 985.00 | 990.27 | 990.23 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 12:15:00 | 982.00 | 988.62 | 989.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 965.95 | 982.65 | 986.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 09:15:00 | 979.70 | 972.87 | 978.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 979.70 | 972.87 | 978.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 979.70 | 972.87 | 978.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 980.95 | 972.87 | 978.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 967.90 | 971.87 | 977.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:30:00 | 966.15 | 969.83 | 975.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:15:00 | 962.25 | 958.36 | 962.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 974.60 | 962.13 | 960.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 974.60 | 962.13 | 960.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 979.05 | 965.51 | 962.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 964.25 | 968.43 | 965.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 964.25 | 968.43 | 965.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 964.25 | 968.43 | 965.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:45:00 | 966.35 | 968.43 | 965.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 971.90 | 969.12 | 966.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 11:15:00 | 975.55 | 969.12 | 966.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 15:15:00 | 974.00 | 971.57 | 968.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-26 09:15:00 | 1073.11 | 1028.36 | 1004.34 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 11:15:00 | 1080.90 | 1095.54 | 1097.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 09:15:00 | 1073.60 | 1088.40 | 1092.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 13:15:00 | 1080.20 | 1078.30 | 1085.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 14:00:00 | 1080.20 | 1078.30 | 1085.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1085.40 | 1080.24 | 1084.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:15:00 | 1088.10 | 1080.24 | 1084.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 1085.90 | 1081.37 | 1084.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:30:00 | 1089.30 | 1081.37 | 1084.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 1080.00 | 1081.10 | 1084.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:45:00 | 1077.00 | 1083.45 | 1084.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 11:15:00 | 1090.50 | 1084.74 | 1084.98 | SL hit (close>static) qty=1.00 sl=1088.70 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 12:15:00 | 1101.50 | 1088.09 | 1086.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 13:15:00 | 1105.30 | 1091.53 | 1088.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 13:15:00 | 1156.00 | 1156.84 | 1144.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 13:30:00 | 1154.20 | 1156.84 | 1144.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1162.00 | 1158.97 | 1148.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 10:15:00 | 1166.30 | 1158.97 | 1148.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 11:15:00 | 1163.20 | 1159.78 | 1150.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 1145.50 | 1151.07 | 1151.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 1145.50 | 1151.07 | 1151.46 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 09:15:00 | 1155.10 | 1151.81 | 1151.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 12:15:00 | 1173.90 | 1159.37 | 1155.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 13:15:00 | 1162.40 | 1167.62 | 1162.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 13:15:00 | 1162.40 | 1167.62 | 1162.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 1162.40 | 1167.62 | 1162.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:00:00 | 1162.40 | 1167.62 | 1162.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 1168.90 | 1167.88 | 1163.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 13:00:00 | 1178.60 | 1171.89 | 1166.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 13:15:00 | 1162.60 | 1167.66 | 1167.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 13:15:00 | 1162.60 | 1167.66 | 1167.98 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 14:15:00 | 1183.80 | 1170.89 | 1169.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 09:15:00 | 1206.60 | 1184.11 | 1178.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 13:15:00 | 1228.00 | 1232.15 | 1219.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 14:00:00 | 1228.00 | 1232.15 | 1219.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1228.50 | 1231.12 | 1222.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:30:00 | 1219.60 | 1231.12 | 1222.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 1224.70 | 1229.44 | 1224.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:00:00 | 1224.70 | 1229.44 | 1224.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 1196.30 | 1222.81 | 1221.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 1196.30 | 1222.81 | 1221.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 1203.00 | 1218.85 | 1220.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 1195.70 | 1214.22 | 1217.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 14:15:00 | 1199.90 | 1199.29 | 1204.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-05 15:00:00 | 1199.90 | 1199.29 | 1204.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1179.20 | 1192.38 | 1200.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 13:15:00 | 1143.40 | 1185.86 | 1191.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 15:15:00 | 1086.23 | 1135.46 | 1156.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 09:15:00 | 1137.00 | 1135.77 | 1154.58 | SL hit (close>ema200) qty=0.50 sl=1135.77 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 13:15:00 | 1161.20 | 1152.24 | 1151.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 14:15:00 | 1171.10 | 1156.01 | 1153.61 | Break + close above crossover candle high |

### Cycle 18 — SELL (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 09:15:00 | 1134.70 | 1152.39 | 1152.42 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 12:15:00 | 1157.00 | 1152.76 | 1152.47 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 14:15:00 | 1138.70 | 1150.45 | 1151.50 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 10:15:00 | 1159.80 | 1153.07 | 1152.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 1166.10 | 1159.65 | 1156.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 13:15:00 | 1156.00 | 1167.45 | 1164.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 13:15:00 | 1156.00 | 1167.45 | 1164.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 1156.00 | 1167.45 | 1164.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:00:00 | 1156.00 | 1167.45 | 1164.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 1146.30 | 1163.22 | 1162.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 1146.30 | 1163.22 | 1162.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 15:15:00 | 1146.80 | 1159.93 | 1161.34 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 1171.50 | 1161.92 | 1160.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 11:15:00 | 1175.60 | 1170.91 | 1166.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 14:15:00 | 1182.90 | 1183.05 | 1177.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-25 15:00:00 | 1182.90 | 1183.05 | 1177.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 1165.20 | 1179.48 | 1176.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 1142.00 | 1179.48 | 1176.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1144.50 | 1172.48 | 1173.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 1130.80 | 1150.58 | 1160.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1126.10 | 1125.18 | 1137.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 1126.10 | 1125.18 | 1137.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 1125.00 | 1125.20 | 1132.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 09:15:00 | 1115.00 | 1125.20 | 1132.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:45:00 | 1123.20 | 1125.15 | 1131.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 13:15:00 | 1135.40 | 1126.26 | 1130.31 | SL hit (close>static) qty=1.00 sl=1134.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 14:15:00 | 1128.10 | 1124.26 | 1124.09 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 1120.30 | 1123.47 | 1123.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 14:15:00 | 1112.90 | 1119.98 | 1121.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 1096.50 | 1077.53 | 1085.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 1096.50 | 1077.53 | 1085.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1096.50 | 1077.53 | 1085.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 1096.50 | 1077.53 | 1085.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 1088.10 | 1079.64 | 1085.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 14:15:00 | 1087.20 | 1085.40 | 1087.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 15:15:00 | 1085.10 | 1086.00 | 1087.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 09:45:00 | 1087.40 | 1085.44 | 1086.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 12:00:00 | 1085.40 | 1084.42 | 1086.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 1085.00 | 1084.53 | 1086.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:30:00 | 1085.40 | 1084.53 | 1086.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 1085.10 | 1084.65 | 1085.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:45:00 | 1085.90 | 1084.65 | 1085.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 1085.10 | 1084.74 | 1085.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:30:00 | 1085.80 | 1084.74 | 1085.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 1079.00 | 1083.59 | 1085.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 1079.00 | 1083.59 | 1085.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1078.10 | 1082.49 | 1084.58 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-16 11:15:00 | 1089.00 | 1082.66 | 1081.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 11:15:00 | 1089.00 | 1082.66 | 1081.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 12:15:00 | 1089.70 | 1084.07 | 1082.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 1086.20 | 1092.19 | 1089.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 1086.20 | 1092.19 | 1089.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1086.20 | 1092.19 | 1089.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 1086.20 | 1092.19 | 1089.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 1095.70 | 1092.89 | 1089.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 10:00:00 | 1101.10 | 1094.55 | 1092.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 12:00:00 | 1099.00 | 1096.07 | 1093.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 13:15:00 | 1083.90 | 1094.17 | 1092.86 | SL hit (close<static) qty=1.00 sl=1084.60 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 1065.20 | 1088.38 | 1090.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 14:15:00 | 1059.20 | 1069.17 | 1075.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 1073.10 | 1069.32 | 1074.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 1073.10 | 1069.32 | 1074.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1073.10 | 1069.32 | 1074.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:45:00 | 1073.60 | 1069.32 | 1074.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 1067.80 | 1069.01 | 1073.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 13:30:00 | 1065.00 | 1070.20 | 1071.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 14:30:00 | 1064.90 | 1068.62 | 1070.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 09:15:00 | 1011.75 | 1040.34 | 1052.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 09:15:00 | 1011.66 | 1040.34 | 1052.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 11:15:00 | 1024.90 | 1020.89 | 1032.30 | SL hit (close>ema200) qty=0.50 sl=1020.89 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 1036.00 | 1030.79 | 1030.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 1040.90 | 1032.81 | 1031.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 12:15:00 | 1038.30 | 1043.76 | 1039.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 12:15:00 | 1038.30 | 1043.76 | 1039.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 1038.30 | 1043.76 | 1039.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:30:00 | 1040.10 | 1043.76 | 1039.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 1043.00 | 1043.61 | 1040.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:30:00 | 1037.10 | 1043.61 | 1040.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 1041.00 | 1043.52 | 1040.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:45:00 | 1051.00 | 1045.15 | 1041.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 1049.50 | 1047.18 | 1044.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:00:00 | 1053.80 | 1048.48 | 1046.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 10:00:00 | 1050.20 | 1053.56 | 1050.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 1043.60 | 1051.56 | 1049.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:45:00 | 1042.00 | 1051.56 | 1049.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 1035.80 | 1048.41 | 1048.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-09 11:15:00 | 1035.80 | 1048.41 | 1048.18 | SL hit (close<static) qty=1.00 sl=1040.10 alert=retest2 |

### Cycle 30 — SELL (started 2025-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 12:15:00 | 1043.10 | 1047.35 | 1047.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 14:15:00 | 1035.30 | 1043.48 | 1045.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 10:15:00 | 1035.30 | 1033.39 | 1037.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-13 10:30:00 | 1033.80 | 1033.39 | 1037.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 1049.50 | 1037.00 | 1038.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:00:00 | 1049.50 | 1037.00 | 1038.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 1035.70 | 1036.74 | 1038.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:45:00 | 1032.00 | 1037.44 | 1038.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 14:15:00 | 1041.00 | 1038.30 | 1038.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 14:15:00 | 1041.00 | 1038.30 | 1038.27 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 15:15:00 | 1035.50 | 1037.74 | 1038.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 13:15:00 | 1029.90 | 1034.76 | 1036.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 1035.40 | 1034.38 | 1035.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 1035.40 | 1034.38 | 1035.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1035.40 | 1034.38 | 1035.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:45:00 | 1036.50 | 1034.38 | 1035.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 1040.00 | 1035.50 | 1036.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:45:00 | 1043.00 | 1035.50 | 1036.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 1033.80 | 1035.16 | 1035.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 12:15:00 | 1032.70 | 1035.16 | 1035.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 13:30:00 | 1031.50 | 1034.90 | 1035.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 14:30:00 | 1032.90 | 1034.38 | 1035.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 15:00:00 | 1032.30 | 1034.38 | 1035.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 1025.90 | 1022.21 | 1025.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 12:30:00 | 1025.00 | 1022.21 | 1025.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 1025.00 | 1022.77 | 1025.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:30:00 | 1025.50 | 1022.77 | 1025.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 1026.90 | 1023.60 | 1025.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 15:00:00 | 1026.90 | 1023.60 | 1025.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 1030.00 | 1024.88 | 1026.35 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 1037.80 | 1027.46 | 1027.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 1037.80 | 1027.46 | 1027.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 14:15:00 | 1041.00 | 1030.17 | 1028.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 1043.10 | 1043.39 | 1037.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 15:00:00 | 1043.10 | 1043.39 | 1037.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1044.70 | 1044.37 | 1038.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:45:00 | 1043.50 | 1044.37 | 1038.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 1034.20 | 1042.34 | 1038.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:00:00 | 1034.20 | 1042.34 | 1038.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 1038.80 | 1041.63 | 1038.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:30:00 | 1031.40 | 1041.63 | 1038.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 1036.90 | 1040.69 | 1038.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 1036.90 | 1040.69 | 1038.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 1035.60 | 1039.67 | 1038.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:30:00 | 1035.90 | 1039.67 | 1038.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 1039.00 | 1039.53 | 1038.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 15:15:00 | 1043.00 | 1039.53 | 1038.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 1072.90 | 1077.38 | 1077.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 1072.90 | 1077.38 | 1077.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 11:15:00 | 1060.60 | 1072.24 | 1074.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 09:15:00 | 1028.10 | 1027.91 | 1036.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 09:15:00 | 1028.10 | 1027.91 | 1036.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1028.10 | 1027.91 | 1036.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 1015.70 | 1027.92 | 1033.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 12:15:00 | 1050.10 | 1025.01 | 1029.28 | SL hit (close>static) qty=1.00 sl=1039.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 1071.80 | 1032.22 | 1031.06 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 1026.80 | 1038.77 | 1039.23 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 11:15:00 | 1050.50 | 1036.86 | 1036.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 12:15:00 | 1059.10 | 1041.31 | 1038.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1027.80 | 1042.99 | 1040.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 1027.80 | 1042.99 | 1040.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1027.80 | 1042.99 | 1040.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 1027.80 | 1042.99 | 1040.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1025.50 | 1039.49 | 1039.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 11:00:00 | 1025.50 | 1039.49 | 1039.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 11:15:00 | 1026.80 | 1036.95 | 1038.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 13:15:00 | 1020.10 | 1025.51 | 1028.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 15:15:00 | 1020.00 | 1018.46 | 1021.97 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-25 09:15:00 | 1013.50 | 1018.46 | 1021.97 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1025.90 | 1019.94 | 1022.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 1025.90 | 1019.94 | 1022.33 | SL hit (close>ema400) qty=1.00 sl=1022.33 alert=retest1 |

### Cycle 39 — BUY (started 2025-11-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 12:15:00 | 1029.90 | 1024.53 | 1024.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 1034.40 | 1027.28 | 1025.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 1030.30 | 1031.35 | 1029.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:00:00 | 1030.30 | 1031.35 | 1029.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 1029.10 | 1030.90 | 1029.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 1029.10 | 1030.90 | 1029.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 1029.10 | 1030.54 | 1029.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:30:00 | 1029.40 | 1030.54 | 1029.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1029.90 | 1030.41 | 1029.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:30:00 | 1029.10 | 1030.41 | 1029.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 1029.90 | 1030.31 | 1029.26 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 1026.30 | 1029.20 | 1029.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 1024.20 | 1028.20 | 1028.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 09:15:00 | 1033.40 | 1024.63 | 1026.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 1033.40 | 1024.63 | 1026.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1033.40 | 1024.63 | 1026.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 1033.40 | 1024.63 | 1026.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1039.00 | 1027.50 | 1027.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:00:00 | 1039.00 | 1027.50 | 1027.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 11:15:00 | 1035.00 | 1029.00 | 1028.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 14:15:00 | 1042.80 | 1033.89 | 1030.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 1029.50 | 1033.75 | 1031.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 1029.50 | 1033.75 | 1031.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1029.50 | 1033.75 | 1031.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:30:00 | 1024.80 | 1033.75 | 1031.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 1029.00 | 1032.80 | 1031.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 12:00:00 | 1031.30 | 1032.50 | 1031.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 09:45:00 | 1034.60 | 1035.40 | 1033.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 09:45:00 | 1035.00 | 1034.99 | 1034.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 13:15:00 | 1030.60 | 1033.82 | 1033.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 13:15:00 | 1030.60 | 1033.82 | 1033.89 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 14:15:00 | 1034.90 | 1034.04 | 1033.99 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 1019.90 | 1031.36 | 1032.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 1014.40 | 1026.38 | 1030.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 1006.20 | 1003.20 | 1012.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 15:00:00 | 1006.20 | 1003.20 | 1012.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 1019.00 | 1006.36 | 1013.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 1028.00 | 1006.36 | 1013.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 1031.50 | 1011.39 | 1014.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 1032.20 | 1011.39 | 1014.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 1045.30 | 1018.17 | 1017.58 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 1025.50 | 1032.02 | 1032.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 1019.70 | 1025.46 | 1028.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 1020.50 | 1018.82 | 1023.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 1020.50 | 1018.82 | 1023.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 1020.50 | 1018.82 | 1023.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 1020.50 | 1018.82 | 1023.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 1024.00 | 1020.38 | 1023.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:45:00 | 1023.00 | 1020.38 | 1023.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 1027.80 | 1021.86 | 1023.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 1027.80 | 1021.86 | 1023.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 1027.00 | 1022.89 | 1024.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 09:15:00 | 1022.10 | 1022.89 | 1024.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:00:00 | 1022.30 | 1022.77 | 1023.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 1041.60 | 1016.89 | 1018.46 | SL hit (close>static) qty=1.00 sl=1029.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 1043.00 | 1022.11 | 1020.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 1051.70 | 1036.23 | 1029.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 10:15:00 | 1053.40 | 1055.63 | 1046.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 11:00:00 | 1053.40 | 1055.63 | 1046.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 1052.80 | 1055.06 | 1046.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:45:00 | 1050.30 | 1055.06 | 1046.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1063.40 | 1067.03 | 1060.66 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 09:15:00 | 1051.40 | 1058.29 | 1058.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 10:15:00 | 1041.00 | 1054.83 | 1056.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 15:15:00 | 1018.00 | 1015.10 | 1022.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 09:15:00 | 1009.40 | 1015.10 | 1022.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1006.40 | 1013.36 | 1021.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 10:30:00 | 1005.40 | 1011.81 | 1019.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 11:15:00 | 1003.40 | 1011.81 | 1019.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 13:30:00 | 1005.50 | 1009.65 | 1016.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 14:45:00 | 1005.40 | 1009.60 | 1016.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1001.90 | 1001.99 | 1007.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 12:30:00 | 995.60 | 1000.07 | 1005.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 14:30:00 | 997.20 | 998.77 | 1003.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 993.40 | 999.01 | 1003.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 15:15:00 | 955.13 | 976.39 | 988.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 15:15:00 | 953.23 | 976.39 | 988.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 15:15:00 | 955.22 | 976.39 | 988.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 15:15:00 | 955.13 | 976.39 | 988.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 947.34 | 973.54 | 985.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 13:15:00 | 945.82 | 960.48 | 975.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 13:15:00 | 943.73 | 960.48 | 975.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 13:15:00 | 942.20 | 941.55 | 956.10 | SL hit (close>ema200) qty=0.50 sl=941.55 alert=retest2 |

### Cycle 49 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 900.60 | 894.28 | 893.94 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 888.80 | 893.81 | 893.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 10:15:00 | 885.00 | 892.05 | 893.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 13:15:00 | 895.20 | 890.83 | 892.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 13:15:00 | 895.20 | 890.83 | 892.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 895.20 | 890.83 | 892.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:00:00 | 895.20 | 890.83 | 892.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 14:15:00 | 904.60 | 893.59 | 893.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 912.30 | 898.42 | 895.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 900.00 | 912.16 | 905.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 900.00 | 912.16 | 905.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 900.00 | 912.16 | 905.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:00:00 | 900.00 | 912.16 | 905.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 899.05 | 909.54 | 905.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:45:00 | 897.05 | 909.54 | 905.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 891.45 | 901.60 | 902.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 885.40 | 898.36 | 900.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 890.90 | 886.48 | 891.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 890.90 | 886.48 | 891.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 890.90 | 886.48 | 891.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 890.90 | 886.48 | 891.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 890.00 | 887.18 | 891.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 906.75 | 887.18 | 891.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 908.35 | 891.42 | 893.28 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 907.85 | 894.70 | 894.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 910.25 | 897.81 | 896.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 902.25 | 903.51 | 900.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 11:00:00 | 902.25 | 903.51 | 900.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 906.00 | 904.01 | 900.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:30:00 | 901.15 | 904.01 | 900.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 896.30 | 905.86 | 903.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 897.20 | 905.86 | 903.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 890.40 | 902.77 | 902.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:45:00 | 889.00 | 902.77 | 902.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 888.40 | 899.90 | 900.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 13:15:00 | 886.35 | 895.36 | 898.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 891.30 | 886.19 | 890.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 891.30 | 886.19 | 890.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 891.30 | 886.19 | 890.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:15:00 | 900.45 | 886.19 | 890.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 911.70 | 891.29 | 892.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 911.70 | 891.29 | 892.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 921.35 | 897.30 | 894.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 13:15:00 | 927.15 | 906.66 | 899.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 921.60 | 923.63 | 915.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:45:00 | 922.70 | 923.63 | 915.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 908.75 | 920.36 | 917.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:45:00 | 908.20 | 920.36 | 917.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 911.85 | 918.66 | 917.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:45:00 | 907.35 | 918.66 | 917.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 925.95 | 918.29 | 917.20 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 14:15:00 | 909.05 | 916.44 | 916.46 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 10:15:00 | 925.00 | 917.23 | 916.71 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 908.30 | 917.14 | 917.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 892.20 | 910.69 | 914.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 13:15:00 | 907.25 | 904.10 | 909.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 13:45:00 | 907.55 | 904.10 | 909.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 918.00 | 905.63 | 908.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 918.00 | 905.63 | 908.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 912.00 | 906.90 | 908.90 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 918.85 | 910.08 | 910.05 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 12:15:00 | 907.90 | 910.48 | 910.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 13:15:00 | 905.90 | 909.57 | 910.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 14:15:00 | 909.80 | 909.61 | 910.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 14:15:00 | 909.80 | 909.61 | 910.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 909.80 | 909.61 | 910.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:30:00 | 911.25 | 909.61 | 910.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 908.00 | 909.29 | 910.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 903.00 | 909.29 | 910.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 893.80 | 906.19 | 908.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 884.90 | 899.52 | 903.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:30:00 | 885.35 | 887.58 | 894.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 13:15:00 | 886.25 | 874.21 | 872.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 13:15:00 | 886.25 | 874.21 | 872.96 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 14:15:00 | 863.20 | 872.01 | 872.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 857.85 | 866.56 | 869.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 870.00 | 864.92 | 867.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 14:15:00 | 870.00 | 864.92 | 867.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 870.00 | 864.92 | 867.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 14:45:00 | 872.00 | 864.92 | 867.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 871.00 | 866.13 | 867.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 841.60 | 866.13 | 867.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 11:00:00 | 863.50 | 862.93 | 865.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 12:15:00 | 820.32 | 831.43 | 841.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 799.52 | 825.32 | 835.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 818.85 | 806.67 | 817.24 | SL hit (close>ema200) qty=0.50 sl=806.67 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 813.00 | 798.90 | 797.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 820.30 | 807.42 | 802.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 808.85 | 815.50 | 810.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 808.85 | 815.50 | 810.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 808.85 | 815.50 | 810.29 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 794.90 | 806.46 | 807.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 12:15:00 | 793.00 | 800.49 | 803.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 770.50 | 770.35 | 781.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 770.50 | 770.35 | 781.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 770.50 | 770.35 | 781.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:30:00 | 788.90 | 770.35 | 781.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 789.65 | 776.30 | 781.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 789.65 | 776.30 | 781.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 796.00 | 780.24 | 782.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 796.00 | 780.24 | 782.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 793.00 | 784.30 | 784.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 810.75 | 789.59 | 786.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 794.00 | 802.04 | 796.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 794.00 | 802.04 | 796.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 794.00 | 802.04 | 796.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 794.00 | 802.04 | 796.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 789.10 | 799.45 | 795.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 788.60 | 799.45 | 795.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 787.65 | 793.50 | 793.63 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 14:15:00 | 795.65 | 793.93 | 793.81 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 789.80 | 793.10 | 793.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 785.45 | 791.57 | 792.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 12:15:00 | 790.10 | 788.52 | 790.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-30 13:00:00 | 790.10 | 788.52 | 790.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 13:15:00 | 784.35 | 787.69 | 790.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 14:30:00 | 779.80 | 785.12 | 788.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 808.70 | 788.38 | 789.56 | SL hit (close>static) qty=1.00 sl=790.45 alert=retest2 |

### Cycle 69 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 812.25 | 793.15 | 791.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 11:15:00 | 816.45 | 797.81 | 793.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 784.70 | 803.16 | 799.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 784.70 | 803.16 | 799.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 784.70 | 803.16 | 799.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 784.70 | 803.16 | 799.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 788.50 | 800.22 | 798.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:15:00 | 792.05 | 797.89 | 797.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 871.25 | 858.89 | 852.26 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 14:15:00 | 850.80 | 855.11 | 855.53 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 860.50 | 855.85 | 855.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 12:15:00 | 870.25 | 859.33 | 857.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 854.80 | 860.97 | 859.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 09:15:00 | 854.80 | 860.97 | 859.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 854.80 | 860.97 | 859.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:45:00 | 851.25 | 860.97 | 859.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 856.70 | 860.12 | 858.88 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2026-04-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 13:15:00 | 851.55 | 857.53 | 857.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 15:15:00 | 850.00 | 855.15 | 856.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 11:15:00 | 855.30 | 854.58 | 856.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 11:15:00 | 855.30 | 854.58 | 856.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 855.30 | 854.58 | 856.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 11:45:00 | 854.90 | 854.58 | 856.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 855.70 | 854.81 | 855.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:30:00 | 855.80 | 854.81 | 855.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 853.20 | 854.49 | 855.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 13:30:00 | 852.75 | 854.49 | 855.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 856.25 | 853.72 | 854.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:15:00 | 864.45 | 853.72 | 854.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 863.00 | 855.57 | 855.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:30:00 | 867.60 | 855.57 | 855.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 11:15:00 | 861.50 | 856.76 | 856.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 14:15:00 | 871.70 | 861.14 | 858.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 14:15:00 | 872.15 | 876.65 | 869.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-23 15:00:00 | 872.15 | 876.65 | 869.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 869.30 | 875.18 | 869.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 863.75 | 875.18 | 869.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 858.30 | 871.80 | 868.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:45:00 | 853.35 | 871.80 | 868.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 848.60 | 867.16 | 866.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:00:00 | 848.60 | 867.16 | 866.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 848.45 | 863.42 | 864.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 845.25 | 859.79 | 863.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 10:15:00 | 855.95 | 851.78 | 856.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 11:00:00 | 855.95 | 851.78 | 856.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 858.55 | 853.13 | 857.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:00:00 | 858.55 | 853.13 | 857.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 862.00 | 854.90 | 857.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:30:00 | 861.15 | 854.90 | 857.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 859.15 | 855.52 | 857.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:45:00 | 857.85 | 855.52 | 857.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 855.15 | 855.45 | 856.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 13:30:00 | 847.80 | 852.51 | 855.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 09:45:00 | 849.15 | 849.37 | 852.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 11:15:00 | 849.70 | 849.82 | 852.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 12:30:00 | 849.65 | 850.26 | 852.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 845.40 | 844.74 | 848.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 845.40 | 844.74 | 848.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 854.25 | 846.64 | 849.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:00:00 | 854.25 | 846.64 | 849.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 855.95 | 848.50 | 849.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:30:00 | 853.55 | 848.50 | 849.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-30 13:15:00 | 857.25 | 851.29 | 850.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 857.25 | 851.29 | 850.90 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 14:15:00 | 844.15 | 849.86 | 850.28 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 855.30 | 851.21 | 850.79 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 09:15:00 | 843.40 | 850.48 | 850.73 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 859.75 | 852.33 | 851.55 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 10:15:00 | 849.00 | 851.91 | 851.94 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 858.00 | 852.89 | 852.25 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2026-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 14:15:00 | 847.40 | 851.41 | 851.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 09:15:00 | 839.90 | 848.73 | 850.51 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-21 14:00:00 | 981.75 | 2025-05-27 11:15:00 | 984.45 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2025-05-21 15:00:00 | 984.95 | 2025-05-27 11:15:00 | 984.45 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-05-22 10:30:00 | 981.70 | 2025-05-27 11:15:00 | 984.45 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2025-05-22 12:00:00 | 979.80 | 2025-05-27 11:15:00 | 984.45 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest2 | 2025-05-22 14:30:00 | 982.50 | 2025-05-27 11:15:00 | 984.45 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-05-23 09:15:00 | 990.65 | 2025-05-27 11:15:00 | 984.45 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-05-27 10:45:00 | 985.80 | 2025-05-27 11:15:00 | 984.45 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-06-09 15:00:00 | 969.70 | 2025-06-10 09:15:00 | 977.30 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-06-17 11:30:00 | 966.15 | 2025-06-23 10:15:00 | 974.60 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-06-19 10:15:00 | 962.25 | 2025-06-23 10:15:00 | 974.60 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-06-24 11:15:00 | 975.55 | 2025-06-26 09:15:00 | 1073.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-24 15:15:00 | 974.00 | 2025-06-26 09:15:00 | 1071.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-10 09:45:00 | 1077.00 | 2025-07-10 11:15:00 | 1090.50 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-07-16 10:15:00 | 1166.30 | 2025-07-18 09:15:00 | 1145.50 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-07-16 11:15:00 | 1163.20 | 2025-07-18 09:15:00 | 1145.50 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-07-23 13:00:00 | 1178.60 | 2025-07-24 13:15:00 | 1162.60 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-08-07 13:15:00 | 1143.40 | 2025-08-08 15:15:00 | 1086.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-07 13:15:00 | 1143.40 | 2025-08-11 09:15:00 | 1137.00 | STOP_HIT | 0.50 | 0.56% |
| SELL | retest2 | 2025-09-01 09:15:00 | 1115.00 | 2025-09-01 13:15:00 | 1135.40 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-09-01 10:45:00 | 1123.20 | 2025-09-01 13:15:00 | 1135.40 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-09-02 12:30:00 | 1120.60 | 2025-09-04 12:15:00 | 1130.10 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-09-03 10:45:00 | 1124.60 | 2025-09-04 12:15:00 | 1130.10 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-09-04 09:15:00 | 1113.00 | 2025-09-04 14:15:00 | 1128.10 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-09-04 10:15:00 | 1116.90 | 2025-09-04 14:15:00 | 1128.10 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-10 14:15:00 | 1087.20 | 2025-09-16 11:15:00 | 1089.00 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-09-10 15:15:00 | 1085.10 | 2025-09-16 11:15:00 | 1089.00 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-09-11 09:45:00 | 1087.40 | 2025-09-16 11:15:00 | 1089.00 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-09-11 12:00:00 | 1085.40 | 2025-09-16 11:15:00 | 1089.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-09-19 10:00:00 | 1101.10 | 2025-09-19 13:15:00 | 1083.90 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-09-19 12:00:00 | 1099.00 | 2025-09-19 13:15:00 | 1083.90 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-09-25 13:30:00 | 1065.00 | 2025-09-29 09:15:00 | 1011.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 14:30:00 | 1064.90 | 2025-09-29 09:15:00 | 1011.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 13:30:00 | 1065.00 | 2025-09-30 11:15:00 | 1024.90 | STOP_HIT | 0.50 | 3.77% |
| SELL | retest2 | 2025-09-25 14:30:00 | 1064.90 | 2025-09-30 11:15:00 | 1024.90 | STOP_HIT | 0.50 | 3.76% |
| BUY | retest2 | 2025-10-07 09:45:00 | 1051.00 | 2025-10-09 11:15:00 | 1035.80 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-10-08 09:15:00 | 1049.50 | 2025-10-09 11:15:00 | 1035.80 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-10-08 12:00:00 | 1053.80 | 2025-10-09 11:15:00 | 1035.80 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-10-09 10:00:00 | 1050.20 | 2025-10-09 11:15:00 | 1035.80 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-10-14 11:45:00 | 1032.00 | 2025-10-14 14:15:00 | 1041.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-10-16 12:15:00 | 1032.70 | 2025-10-21 13:15:00 | 1037.80 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-10-16 13:30:00 | 1031.50 | 2025-10-21 13:15:00 | 1037.80 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-10-16 14:30:00 | 1032.90 | 2025-10-21 13:15:00 | 1037.80 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-10-16 15:00:00 | 1032.30 | 2025-10-21 13:15:00 | 1037.80 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-10-24 15:15:00 | 1043.00 | 2025-10-31 14:15:00 | 1072.90 | STOP_HIT | 1.00 | 2.87% |
| SELL | retest2 | 2025-11-11 09:15:00 | 1015.70 | 2025-11-11 12:15:00 | 1050.10 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2025-11-11 14:45:00 | 1016.20 | 2025-11-12 09:15:00 | 1071.80 | STOP_HIT | 1.00 | -5.47% |
| SELL | retest1 | 2025-11-25 09:15:00 | 1013.50 | 2025-11-25 09:15:00 | 1025.90 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-12-03 12:00:00 | 1031.30 | 2025-12-05 13:15:00 | 1030.60 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-12-04 09:45:00 | 1034.60 | 2025-12-05 13:15:00 | 1030.60 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-12-05 09:45:00 | 1035.00 | 2025-12-05 13:15:00 | 1030.60 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-12-19 09:15:00 | 1022.10 | 2025-12-22 09:15:00 | 1041.60 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-12-19 10:00:00 | 1022.30 | 2025-12-22 09:15:00 | 1041.60 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2026-01-05 10:30:00 | 1005.40 | 2026-01-08 15:15:00 | 955.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 11:15:00 | 1003.40 | 2026-01-08 15:15:00 | 953.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 13:30:00 | 1005.50 | 2026-01-08 15:15:00 | 955.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 14:45:00 | 1005.40 | 2026-01-08 15:15:00 | 955.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 12:30:00 | 995.60 | 2026-01-09 09:15:00 | 947.34 | PARTIAL | 0.50 | 4.85% |
| SELL | retest2 | 2026-01-07 14:30:00 | 997.20 | 2026-01-09 13:15:00 | 945.82 | PARTIAL | 0.50 | 5.15% |
| SELL | retest2 | 2026-01-08 09:15:00 | 993.40 | 2026-01-09 13:15:00 | 943.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 10:30:00 | 1005.40 | 2026-01-12 13:15:00 | 942.20 | STOP_HIT | 0.50 | 6.29% |
| SELL | retest2 | 2026-01-05 11:15:00 | 1003.40 | 2026-01-12 13:15:00 | 942.20 | STOP_HIT | 0.50 | 6.10% |
| SELL | retest2 | 2026-01-05 13:30:00 | 1005.50 | 2026-01-12 13:15:00 | 942.20 | STOP_HIT | 0.50 | 6.30% |
| SELL | retest2 | 2026-01-05 14:45:00 | 1005.40 | 2026-01-12 13:15:00 | 942.20 | STOP_HIT | 0.50 | 6.29% |
| SELL | retest2 | 2026-01-07 12:30:00 | 995.60 | 2026-01-12 13:15:00 | 942.20 | STOP_HIT | 0.50 | 5.36% |
| SELL | retest2 | 2026-01-07 14:30:00 | 997.20 | 2026-01-12 13:15:00 | 942.20 | STOP_HIT | 0.50 | 5.52% |
| SELL | retest2 | 2026-01-08 09:15:00 | 993.40 | 2026-01-12 13:15:00 | 942.20 | STOP_HIT | 0.50 | 5.15% |
| SELL | retest2 | 2026-02-20 09:15:00 | 884.90 | 2026-02-27 13:15:00 | 886.25 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2026-02-23 09:30:00 | 885.35 | 2026-02-27 13:15:00 | 886.25 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2026-03-04 09:15:00 | 841.60 | 2026-03-06 12:15:00 | 820.32 | PARTIAL | 0.50 | 2.53% |
| SELL | retest2 | 2026-03-04 11:00:00 | 863.50 | 2026-03-09 09:15:00 | 799.52 | PARTIAL | 0.50 | 7.41% |
| SELL | retest2 | 2026-03-04 09:15:00 | 841.60 | 2026-03-10 10:15:00 | 818.85 | STOP_HIT | 0.50 | 2.70% |
| SELL | retest2 | 2026-03-04 11:00:00 | 863.50 | 2026-03-10 10:15:00 | 818.85 | STOP_HIT | 0.50 | 5.17% |
| SELL | retest2 | 2026-03-30 14:30:00 | 779.80 | 2026-04-01 09:15:00 | 808.70 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2026-04-02 12:15:00 | 792.05 | 2026-04-15 09:15:00 | 871.25 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-28 13:30:00 | 847.80 | 2026-04-30 13:15:00 | 857.25 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2026-04-29 09:45:00 | 849.15 | 2026-04-30 13:15:00 | 857.25 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-04-29 11:15:00 | 849.70 | 2026-04-30 13:15:00 | 857.25 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2026-04-29 12:30:00 | 849.65 | 2026-04-30 13:15:00 | 857.25 | STOP_HIT | 1.00 | -0.89% |
