# Adani Green Energy Ltd. (ADANIGREEN)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 1350.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 199 |
| ALERT1 | 133 |
| ALERT2 | 133 |
| ALERT2_SKIP | 66 |
| ALERT3 | 361 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 11 |
| ENTRY2 | 149 |
| PARTIAL | 36 |
| TARGET_HIT | 26 |
| STOP_HIT | 135 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 195 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 92 / 103
- **Target hits / Stop hits / Partials:** 26 / 134 / 35
- **Avg / median % per leg:** 1.47% / -0.24%
- **Sum % (uncompounded):** 287.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 73 | 17 | 23.3% | 6 | 64 | 3 | -0.40% | -29.0% |
| BUY @ 2nd Alert (retest1) | 14 | 6 | 42.9% | 0 | 11 | 3 | 0.97% | 13.6% |
| BUY @ 3rd Alert (retest2) | 59 | 11 | 18.6% | 6 | 53 | 0 | -0.72% | -42.7% |
| SELL (all) | 122 | 75 | 61.5% | 20 | 70 | 32 | 2.59% | 316.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 122 | 75 | 61.5% | 20 | 70 | 32 | 2.59% | 316.4% |
| retest1 (combined) | 14 | 6 | 42.9% | 0 | 11 | 3 | 0.97% | 13.6% |
| retest2 (combined) | 181 | 86 | 47.5% | 26 | 123 | 32 | 1.51% | 273.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 15:15:00 | 903.55 | 871.68 | 870.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 09:15:00 | 941.75 | 885.69 | 876.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 11:15:00 | 970.45 | 984.50 | 958.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-24 11:15:00 | 970.45 | 984.50 | 958.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 11:15:00 | 970.45 | 984.50 | 958.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 12:00:00 | 970.45 | 984.50 | 958.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 12:15:00 | 969.00 | 981.40 | 959.19 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 14:15:00 | 966.65 | 969.52 | 969.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 09:15:00 | 944.55 | 963.80 | 967.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-31 14:15:00 | 977.80 | 961.40 | 963.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 14:15:00 | 977.80 | 961.40 | 963.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 14:15:00 | 977.80 | 961.40 | 963.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-31 15:00:00 | 977.80 | 961.40 | 963.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2023-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 15:15:00 | 984.00 | 965.92 | 965.81 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-06-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-07 15:15:00 | 984.50 | 987.00 | 987.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-08 11:15:00 | 975.20 | 983.96 | 985.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 13:15:00 | 963.65 | 963.50 | 969.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-12 14:00:00 | 963.65 | 963.50 | 969.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 09:15:00 | 951.60 | 954.30 | 959.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-14 10:30:00 | 950.05 | 953.54 | 958.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-14 11:00:00 | 950.50 | 953.54 | 958.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-14 11:30:00 | 949.90 | 953.41 | 958.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-15 09:15:00 | 987.05 | 961.10 | 960.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2023-06-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-15 09:15:00 | 987.05 | 961.10 | 960.09 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2023-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 15:15:00 | 959.95 | 965.15 | 965.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-20 11:15:00 | 958.35 | 962.50 | 964.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 14:15:00 | 962.85 | 961.19 | 963.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 14:15:00 | 962.85 | 961.19 | 963.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 14:15:00 | 962.85 | 961.19 | 963.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-20 15:00:00 | 962.85 | 961.19 | 963.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 15:15:00 | 965.00 | 961.95 | 963.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-21 09:15:00 | 959.50 | 961.95 | 963.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-21 09:15:00 | 981.25 | 965.81 | 964.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2023-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 09:15:00 | 981.25 | 965.81 | 964.88 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 09:15:00 | 956.75 | 969.42 | 969.72 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 15:15:00 | 968.80 | 964.06 | 963.74 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-27 10:15:00 | 960.00 | 963.21 | 963.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-27 15:15:00 | 957.85 | 960.72 | 961.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-28 14:15:00 | 955.95 | 950.36 | 954.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 14:15:00 | 955.95 | 950.36 | 954.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 14:15:00 | 955.95 | 950.36 | 954.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-28 15:00:00 | 955.95 | 950.36 | 954.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 15:15:00 | 959.00 | 952.09 | 955.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-30 09:15:00 | 956.90 | 952.09 | 955.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 09:15:00 | 943.90 | 950.45 | 954.25 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 13:15:00 | 950.70 | 947.91 | 947.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 13:15:00 | 956.85 | 952.58 | 950.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 10:15:00 | 953.95 | 954.74 | 952.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 10:15:00 | 953.95 | 954.74 | 952.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 10:15:00 | 953.95 | 954.74 | 952.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 10:45:00 | 955.40 | 954.74 | 952.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 11:15:00 | 954.00 | 954.59 | 952.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 11:45:00 | 955.95 | 954.59 | 952.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 12:15:00 | 952.15 | 954.10 | 952.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 12:45:00 | 953.60 | 954.10 | 952.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 13:15:00 | 956.45 | 954.57 | 952.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-07 14:30:00 | 959.25 | 953.87 | 952.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-07 15:15:00 | 950.00 | 953.09 | 952.41 | SL hit (close<static) qty=1.00 sl=951.00 alert=retest2 |

### Cycle 12 — SELL (started 2023-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 09:15:00 | 941.95 | 950.86 | 951.46 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 13:15:00 | 963.50 | 953.84 | 952.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 11:15:00 | 968.80 | 960.96 | 956.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-11 15:15:00 | 963.10 | 963.33 | 959.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-12 09:15:00 | 961.55 | 963.33 | 959.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 957.40 | 962.15 | 959.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 10:00:00 | 957.40 | 962.15 | 959.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 10:15:00 | 959.80 | 961.68 | 959.33 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2023-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 13:15:00 | 955.50 | 957.73 | 957.86 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-07-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 10:15:00 | 962.80 | 958.50 | 958.05 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 13:15:00 | 949.00 | 957.00 | 957.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 15:15:00 | 945.35 | 953.23 | 955.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 11:15:00 | 956.80 | 952.86 | 954.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-14 11:15:00 | 956.80 | 952.86 | 954.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 11:15:00 | 956.80 | 952.86 | 954.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 12:00:00 | 956.80 | 952.86 | 954.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 12:15:00 | 956.80 | 953.65 | 954.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 13:00:00 | 956.80 | 953.65 | 954.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 13:15:00 | 955.00 | 953.92 | 954.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 14:15:00 | 955.85 | 953.92 | 954.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2023-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 14:15:00 | 963.75 | 955.89 | 955.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 09:15:00 | 969.90 | 960.16 | 957.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-17 14:15:00 | 963.85 | 964.67 | 961.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-17 15:00:00 | 963.85 | 964.67 | 961.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 09:15:00 | 965.65 | 964.87 | 961.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 09:30:00 | 958.70 | 964.87 | 961.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 992.55 | 985.59 | 980.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-21 11:15:00 | 997.40 | 987.31 | 981.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-21 13:00:00 | 998.50 | 991.23 | 984.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-21 13:45:00 | 998.75 | 992.68 | 985.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-24 10:00:00 | 997.00 | 993.91 | 988.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 14:15:00 | 987.40 | 994.06 | 990.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 15:00:00 | 987.40 | 994.06 | 990.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 15:15:00 | 986.15 | 992.48 | 990.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-25 09:15:00 | 1002.00 | 992.48 | 990.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-07-26 09:15:00 | 1097.14 | 1074.33 | 1038.35 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2023-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-31 14:15:00 | 1092.50 | 1102.28 | 1103.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 09:15:00 | 1077.40 | 1088.03 | 1093.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-08 09:15:00 | 1011.30 | 986.87 | 1006.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 09:15:00 | 1011.30 | 986.87 | 1006.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 09:15:00 | 1011.30 | 986.87 | 1006.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-08 10:00:00 | 1011.30 | 986.87 | 1006.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 10:15:00 | 994.80 | 988.46 | 1005.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-08 12:00:00 | 990.70 | 988.91 | 1004.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-08 14:30:00 | 988.05 | 990.01 | 1001.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-09 09:30:00 | 990.95 | 989.89 | 999.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-14 09:15:00 | 941.16 | 969.36 | 975.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-14 09:15:00 | 941.40 | 969.36 | 975.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-17 11:15:00 | 938.65 | 944.95 | 953.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-08-18 09:15:00 | 970.15 | 944.65 | 949.03 | SL hit (close>ema200) qty=0.50 sl=944.65 alert=retest2 |

### Cycle 19 — BUY (started 2023-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 11:15:00 | 1008.05 | 961.46 | 956.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 09:15:00 | 1028.80 | 1015.06 | 996.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 10:15:00 | 1015.00 | 1015.05 | 998.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-22 10:45:00 | 1013.60 | 1015.05 | 998.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 09:15:00 | 1010.30 | 1014.61 | 1005.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 09:30:00 | 1007.90 | 1014.61 | 1005.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 10:15:00 | 1003.10 | 1012.31 | 1005.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 11:00:00 | 1003.10 | 1012.31 | 1005.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 11:15:00 | 1005.05 | 1010.86 | 1005.23 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 14:15:00 | 973.05 | 999.23 | 1001.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 09:15:00 | 924.90 | 962.19 | 969.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 11:15:00 | 940.45 | 936.32 | 948.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-01 12:00:00 | 940.45 | 936.32 | 948.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 13:15:00 | 947.35 | 939.77 | 948.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 14:00:00 | 947.35 | 939.77 | 948.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 14:15:00 | 946.70 | 941.15 | 948.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 14:30:00 | 955.30 | 941.15 | 948.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 15:15:00 | 948.00 | 942.52 | 948.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 09:15:00 | 944.85 | 942.52 | 948.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 945.55 | 943.13 | 947.79 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 13:15:00 | 958.30 | 951.05 | 950.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 13:15:00 | 962.00 | 957.64 | 954.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-07 15:15:00 | 996.00 | 998.08 | 985.76 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 09:30:00 | 1006.45 | 999.36 | 987.47 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 11:00:00 | 1005.50 | 1000.59 | 989.11 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 12:45:00 | 1005.80 | 1002.27 | 991.90 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 14:30:00 | 1005.50 | 1002.50 | 993.84 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 994.30 | 1007.03 | 1001.96 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-09-12 09:15:00 | 994.30 | 1007.03 | 1001.96 | SL hit (close<ema400) qty=1.00 sl=1001.96 alert=retest1 |

### Cycle 22 — SELL (started 2023-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-13 09:15:00 | 972.85 | 997.50 | 999.41 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 11:15:00 | 1007.00 | 993.52 | 991.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-21 09:15:00 | 1011.60 | 1003.93 | 1001.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-22 09:15:00 | 1004.20 | 1012.01 | 1007.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 09:15:00 | 1004.20 | 1012.01 | 1007.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 1004.20 | 1012.01 | 1007.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-22 10:00:00 | 1004.20 | 1012.01 | 1007.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 1012.35 | 1012.08 | 1008.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-22 11:30:00 | 1015.40 | 1012.95 | 1009.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-22 13:45:00 | 1017.20 | 1015.19 | 1010.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-22 14:30:00 | 1018.45 | 1016.03 | 1011.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-26 09:15:00 | 1004.80 | 1009.81 | 1010.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2023-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 09:15:00 | 1004.80 | 1009.81 | 1010.29 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 12:15:00 | 1014.35 | 1011.29 | 1010.89 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 15:15:00 | 1003.50 | 1010.28 | 1010.59 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 11:15:00 | 1015.75 | 1011.72 | 1011.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 14:15:00 | 1018.80 | 1014.21 | 1012.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 11:15:00 | 1014.35 | 1015.08 | 1013.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-28 12:00:00 | 1014.35 | 1015.08 | 1013.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 12:15:00 | 1019.40 | 1015.94 | 1014.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 12:30:00 | 1014.45 | 1015.94 | 1014.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 1013.95 | 1015.92 | 1014.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 14:45:00 | 1011.60 | 1015.92 | 1014.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 1010.00 | 1014.74 | 1014.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-29 09:15:00 | 1001.15 | 1014.74 | 1014.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2023-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-29 09:15:00 | 997.95 | 1011.38 | 1012.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-29 11:15:00 | 990.50 | 1004.54 | 1009.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-04 10:15:00 | 988.15 | 980.01 | 987.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 10:15:00 | 988.15 | 980.01 | 987.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 10:15:00 | 988.15 | 980.01 | 987.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-04 11:00:00 | 988.15 | 980.01 | 987.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 11:15:00 | 979.05 | 979.82 | 987.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-04 14:45:00 | 972.20 | 977.77 | 984.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 10:00:00 | 972.85 | 975.61 | 982.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-11 11:15:00 | 968.30 | 954.59 | 953.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2023-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 11:15:00 | 968.30 | 954.59 | 953.59 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-10-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 14:15:00 | 947.25 | 955.31 | 955.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-13 09:15:00 | 943.95 | 952.35 | 954.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-13 14:15:00 | 949.25 | 948.30 | 951.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-13 15:00:00 | 949.25 | 948.30 | 951.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 15:15:00 | 951.00 | 948.84 | 951.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-16 09:15:00 | 950.90 | 948.84 | 951.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 946.40 | 948.35 | 950.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-16 13:30:00 | 943.85 | 946.00 | 948.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 11:00:00 | 942.00 | 946.58 | 947.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 11:45:00 | 942.80 | 945.74 | 946.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 14:30:00 | 939.40 | 944.65 | 946.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 10:15:00 | 942.20 | 937.49 | 940.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 10:45:00 | 943.50 | 937.49 | 940.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 11:15:00 | 936.75 | 937.34 | 940.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 12:30:00 | 933.70 | 936.63 | 939.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 13:00:00 | 933.80 | 936.63 | 939.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 10:15:00 | 896.66 | 918.45 | 929.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 10:15:00 | 894.90 | 918.45 | 929.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 10:15:00 | 895.66 | 918.45 | 929.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 10:15:00 | 892.43 | 918.45 | 929.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 14:15:00 | 887.01 | 902.91 | 917.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 14:15:00 | 887.11 | 902.91 | 917.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2023-10-26 09:15:00 | 849.47 | 870.60 | 890.69 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 31 — BUY (started 2023-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 11:15:00 | 883.00 | 876.20 | 875.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 12:15:00 | 890.45 | 879.05 | 876.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 14:15:00 | 911.70 | 912.27 | 900.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-31 15:00:00 | 911.70 | 912.27 | 900.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 12:15:00 | 896.40 | 908.59 | 903.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 13:00:00 | 896.40 | 908.59 | 903.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 13:15:00 | 892.90 | 905.45 | 902.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 14:00:00 | 892.90 | 905.45 | 902.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2023-11-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 15:15:00 | 883.00 | 896.75 | 898.50 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 11:15:00 | 899.70 | 897.47 | 897.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 12:15:00 | 903.30 | 898.64 | 897.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 12:15:00 | 920.95 | 924.75 | 916.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 12:15:00 | 920.95 | 924.75 | 916.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 12:15:00 | 920.95 | 924.75 | 916.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 13:00:00 | 920.95 | 924.75 | 916.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 14:15:00 | 933.30 | 925.82 | 918.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 10:45:00 | 939.55 | 930.91 | 923.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-09 15:00:00 | 938.70 | 940.58 | 935.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-10 09:15:00 | 941.55 | 940.06 | 935.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-10 15:00:00 | 939.20 | 940.32 | 937.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 15:15:00 | 939.05 | 940.07 | 937.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-12 18:15:00 | 951.00 | 940.07 | 937.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-13 11:15:00 | 934.50 | 938.94 | 938.11 | SL hit (close<static) qty=1.00 sl=936.05 alert=retest2 |

### Cycle 34 — SELL (started 2023-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 10:15:00 | 938.00 | 940.78 | 941.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-17 12:15:00 | 935.75 | 939.17 | 940.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-20 15:15:00 | 928.00 | 926.13 | 930.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-20 15:15:00 | 928.00 | 926.13 | 930.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 15:15:00 | 928.00 | 926.13 | 930.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 09:15:00 | 931.75 | 926.13 | 930.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 928.20 | 926.54 | 930.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 09:30:00 | 933.15 | 926.54 | 930.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 14:15:00 | 926.05 | 926.93 | 929.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 15:00:00 | 926.05 | 926.93 | 929.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 922.50 | 925.25 | 928.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-22 11:15:00 | 920.40 | 924.30 | 927.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-23 14:15:00 | 931.55 | 918.29 | 920.41 | SL hit (close>static) qty=1.00 sl=929.95 alert=retest2 |

### Cycle 35 — BUY (started 2023-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 10:15:00 | 924.10 | 921.97 | 921.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-24 11:15:00 | 934.40 | 924.45 | 922.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-29 14:15:00 | 1050.10 | 1050.73 | 1019.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-29 15:15:00 | 1038.00 | 1050.73 | 1019.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 1015.35 | 1041.61 | 1020.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 09:45:00 | 1011.05 | 1041.61 | 1020.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 10:15:00 | 1014.20 | 1036.13 | 1020.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 11:00:00 | 1014.20 | 1036.13 | 1020.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 11:15:00 | 1011.85 | 1031.27 | 1019.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 11:45:00 | 1011.30 | 1031.27 | 1019.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 15:15:00 | 1021.95 | 1028.17 | 1024.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 09:15:00 | 1110.15 | 1028.17 | 1024.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-12-05 09:15:00 | 1221.17 | 1131.50 | 1086.64 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2023-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 11:15:00 | 1497.00 | 1517.66 | 1517.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 12:15:00 | 1487.60 | 1511.65 | 1515.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-14 09:15:00 | 1482.45 | 1441.24 | 1461.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 09:15:00 | 1482.45 | 1441.24 | 1461.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 1482.45 | 1441.24 | 1461.52 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-12-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 13:15:00 | 1505.55 | 1477.14 | 1474.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 09:15:00 | 1535.00 | 1496.96 | 1484.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 15:15:00 | 1517.00 | 1517.18 | 1502.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-18 09:15:00 | 1522.95 | 1517.18 | 1502.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 09:15:00 | 1518.15 | 1517.37 | 1503.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 10:30:00 | 1527.45 | 1519.69 | 1506.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 15:15:00 | 1529.00 | 1525.47 | 1513.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 11:45:00 | 1531.00 | 1524.43 | 1516.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 15:15:00 | 1535.00 | 1529.18 | 1521.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 11:15:00 | 1520.00 | 1531.02 | 1525.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 11:45:00 | 1514.70 | 1531.02 | 1525.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 12:15:00 | 1511.00 | 1527.02 | 1523.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 13:00:00 | 1511.00 | 1527.02 | 1523.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-12-20 13:15:00 | 1464.60 | 1514.54 | 1518.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 1464.60 | 1514.54 | 1518.36 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 10:15:00 | 1519.85 | 1500.77 | 1498.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 14:15:00 | 1535.05 | 1513.57 | 1505.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 15:15:00 | 1597.00 | 1597.27 | 1572.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-28 09:15:00 | 1593.00 | 1597.27 | 1572.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 13:15:00 | 1581.95 | 1589.28 | 1577.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 13:45:00 | 1577.90 | 1589.28 | 1577.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 14:15:00 | 1560.10 | 1583.44 | 1576.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 15:00:00 | 1560.10 | 1583.44 | 1576.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 15:15:00 | 1564.85 | 1579.72 | 1575.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 09:15:00 | 1557.25 | 1579.72 | 1575.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 10:15:00 | 1576.00 | 1576.36 | 1574.26 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2023-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-29 13:15:00 | 1563.05 | 1571.70 | 1572.51 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-29 14:15:00 | 1599.00 | 1577.16 | 1574.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-29 15:15:00 | 1603.00 | 1582.33 | 1577.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-01 14:15:00 | 1594.00 | 1597.18 | 1588.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-01 15:00:00 | 1594.00 | 1597.18 | 1588.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 09:15:00 | 1592.25 | 1595.85 | 1589.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:00:00 | 1592.25 | 1595.85 | 1589.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 1578.75 | 1592.43 | 1588.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:30:00 | 1577.35 | 1592.43 | 1588.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 11:15:00 | 1595.00 | 1592.94 | 1588.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 14:45:00 | 1604.50 | 1594.87 | 1590.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-15 09:15:00 | 1702.00 | 1715.10 | 1715.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 09:15:00 | 1702.00 | 1715.10 | 1715.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 09:15:00 | 1680.15 | 1696.80 | 1704.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-20 09:15:00 | 1584.70 | 1583.96 | 1602.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-20 09:15:00 | 1584.70 | 1583.96 | 1602.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 09:15:00 | 1584.70 | 1583.96 | 1602.56 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2024-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 15:15:00 | 1674.50 | 1619.26 | 1611.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 09:15:00 | 1726.65 | 1669.88 | 1656.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 09:15:00 | 1692.80 | 1701.41 | 1683.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 09:15:00 | 1692.80 | 1701.41 | 1683.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 09:15:00 | 1692.80 | 1701.41 | 1683.14 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2024-01-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-31 12:15:00 | 1671.55 | 1682.40 | 1682.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 15:15:00 | 1663.50 | 1672.66 | 1676.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-02 09:15:00 | 1675.50 | 1673.23 | 1676.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-02 09:15:00 | 1675.50 | 1673.23 | 1676.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 1675.50 | 1673.23 | 1676.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 10:15:00 | 1686.95 | 1673.23 | 1676.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 10:15:00 | 1690.65 | 1676.71 | 1677.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 11:00:00 | 1690.65 | 1676.71 | 1677.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2024-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 11:15:00 | 1696.50 | 1680.67 | 1679.58 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 15:15:00 | 1672.00 | 1677.94 | 1678.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 09:15:00 | 1662.00 | 1674.75 | 1677.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-05 13:15:00 | 1667.00 | 1666.49 | 1671.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-05 14:00:00 | 1667.00 | 1666.49 | 1671.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 14:15:00 | 1667.40 | 1666.67 | 1671.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-05 15:00:00 | 1667.40 | 1666.67 | 1671.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 1682.00 | 1669.79 | 1672.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 09:45:00 | 1686.00 | 1669.79 | 1672.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2024-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 10:15:00 | 1726.85 | 1681.20 | 1676.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 09:15:00 | 1919.60 | 1750.38 | 1714.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 10:15:00 | 1832.05 | 1840.10 | 1793.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-08 11:00:00 | 1832.05 | 1840.10 | 1793.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 1875.60 | 1843.87 | 1815.14 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2024-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-13 10:15:00 | 1804.95 | 1839.89 | 1841.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-13 11:15:00 | 1779.10 | 1827.73 | 1836.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 14:15:00 | 1818.80 | 1812.89 | 1826.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-13 15:00:00 | 1818.80 | 1812.89 | 1826.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 1842.10 | 1819.87 | 1826.98 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2024-02-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 11:15:00 | 1854.20 | 1831.62 | 1831.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 1921.95 | 1859.21 | 1845.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 15:15:00 | 1900.00 | 1905.06 | 1888.92 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-19 09:15:00 | 1934.65 | 1905.06 | 1888.92 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 13:15:00 | 1926.35 | 1936.59 | 1925.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 14:00:00 | 1926.35 | 1936.59 | 1925.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 14:15:00 | 1927.00 | 1934.67 | 1925.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 14:30:00 | 1925.40 | 1934.67 | 1925.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 15:15:00 | 1927.00 | 1933.14 | 1925.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 09:15:00 | 1920.00 | 1933.14 | 1925.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 09:15:00 | 1940.60 | 1934.63 | 1927.32 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-02-21 13:15:00 | 1920.25 | 1932.04 | 1928.57 | SL hit (close<ema400) qty=1.00 sl=1928.57 alert=retest1 |

### Cycle 50 — SELL (started 2024-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 15:15:00 | 1901.10 | 1922.18 | 1924.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 09:15:00 | 1883.55 | 1914.46 | 1920.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 13:15:00 | 1907.00 | 1905.95 | 1913.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 13:15:00 | 1907.00 | 1905.95 | 1913.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 13:15:00 | 1907.00 | 1905.95 | 1913.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 13:45:00 | 1908.90 | 1905.95 | 1913.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 1921.65 | 1909.09 | 1914.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 15:00:00 | 1921.65 | 1909.09 | 1914.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 1923.55 | 1911.98 | 1915.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:15:00 | 1920.95 | 1911.98 | 1915.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 10:15:00 | 1925.00 | 1916.81 | 1917.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 11:00:00 | 1925.00 | 1916.81 | 1917.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 11:15:00 | 1914.25 | 1916.30 | 1916.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 11:30:00 | 1925.95 | 1916.30 | 1916.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2024-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 12:15:00 | 1935.70 | 1920.18 | 1918.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-26 09:15:00 | 1959.65 | 1929.71 | 1923.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-27 13:15:00 | 1956.10 | 1972.95 | 1958.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 13:15:00 | 1956.10 | 1972.95 | 1958.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 13:15:00 | 1956.10 | 1972.95 | 1958.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 13:45:00 | 1962.25 | 1972.95 | 1958.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 14:15:00 | 1981.65 | 1974.69 | 1960.35 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2024-02-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 11:15:00 | 1915.00 | 1953.05 | 1954.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 1902.55 | 1942.95 | 1949.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 15:15:00 | 1920.00 | 1906.31 | 1921.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 15:15:00 | 1920.00 | 1906.31 | 1921.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 1920.00 | 1906.31 | 1921.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:15:00 | 1966.10 | 1906.31 | 1921.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 1966.00 | 1918.25 | 1925.48 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2024-03-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 11:15:00 | 1959.70 | 1934.18 | 1931.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 12:15:00 | 1977.00 | 1942.74 | 1936.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 14:15:00 | 1946.55 | 1962.05 | 1956.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 14:15:00 | 1946.55 | 1962.05 | 1956.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 14:15:00 | 1946.55 | 1962.05 | 1956.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 15:00:00 | 1946.55 | 1962.05 | 1956.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 15:15:00 | 1937.90 | 1957.22 | 1954.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 09:15:00 | 1962.35 | 1957.22 | 1954.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-05 13:15:00 | 1949.25 | 1952.73 | 1953.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 13:15:00 | 1949.25 | 1952.73 | 1953.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 1919.25 | 1943.76 | 1948.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 12:15:00 | 1918.80 | 1918.53 | 1927.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-07 13:00:00 | 1918.80 | 1918.53 | 1927.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 14:15:00 | 1926.15 | 1921.17 | 1927.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 15:00:00 | 1926.15 | 1921.17 | 1927.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 15:15:00 | 1923.00 | 1921.54 | 1927.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-11 09:15:00 | 1916.70 | 1921.54 | 1927.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 09:15:00 | 1919.10 | 1921.05 | 1926.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-11 10:15:00 | 1956.75 | 1921.05 | 1926.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 10:15:00 | 1936.25 | 1924.09 | 1927.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 12:00:00 | 1917.80 | 1922.83 | 1926.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-12 09:15:00 | 1903.00 | 1926.33 | 1927.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-12 09:45:00 | 1916.00 | 1924.75 | 1926.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-03-13 09:15:00 | 1726.02 | 1885.55 | 1904.92 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2024-03-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 15:15:00 | 1902.00 | 1860.42 | 1855.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-15 09:15:00 | 1914.55 | 1871.24 | 1860.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-18 09:15:00 | 1873.35 | 1890.89 | 1879.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-18 09:15:00 | 1873.35 | 1890.89 | 1879.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 09:15:00 | 1873.35 | 1890.89 | 1879.12 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2024-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 15:15:00 | 1860.05 | 1872.32 | 1873.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 09:15:00 | 1842.40 | 1866.34 | 1870.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 12:15:00 | 1845.05 | 1831.99 | 1843.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 12:15:00 | 1845.05 | 1831.99 | 1843.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 12:15:00 | 1845.05 | 1831.99 | 1843.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 13:00:00 | 1845.05 | 1831.99 | 1843.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 13:15:00 | 1846.95 | 1834.98 | 1843.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 14:00:00 | 1846.95 | 1834.98 | 1843.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 14:15:00 | 1846.85 | 1837.36 | 1843.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 15:00:00 | 1846.85 | 1837.36 | 1843.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 15:15:00 | 1840.00 | 1837.88 | 1843.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 09:15:00 | 1857.35 | 1837.88 | 1843.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 1861.80 | 1842.67 | 1845.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-21 12:00:00 | 1841.80 | 1845.55 | 1846.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-21 13:30:00 | 1848.00 | 1846.05 | 1846.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-21 14:15:00 | 1852.50 | 1847.34 | 1846.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2024-03-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 14:15:00 | 1852.50 | 1847.34 | 1846.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-26 09:15:00 | 1886.85 | 1856.08 | 1851.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 14:15:00 | 1862.75 | 1865.80 | 1858.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 14:15:00 | 1862.75 | 1865.80 | 1858.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 14:15:00 | 1862.75 | 1865.80 | 1858.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 15:00:00 | 1862.75 | 1865.80 | 1858.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 15:15:00 | 1853.95 | 1863.43 | 1858.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 09:15:00 | 1884.65 | 1863.43 | 1858.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 11:30:00 | 1866.20 | 1867.36 | 1861.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 12:45:00 | 1866.25 | 1867.06 | 1862.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-27 14:15:00 | 1825.85 | 1858.97 | 1859.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2024-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 14:15:00 | 1825.85 | 1858.97 | 1859.40 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 10:15:00 | 1891.60 | 1856.81 | 1852.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-04 09:15:00 | 1905.30 | 1887.35 | 1881.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-04 13:15:00 | 1885.25 | 1889.00 | 1884.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-04 14:00:00 | 1885.25 | 1889.00 | 1884.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 14:15:00 | 1892.00 | 1889.60 | 1885.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 14:30:00 | 1883.65 | 1889.60 | 1885.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 1913.10 | 1894.67 | 1888.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-08 09:15:00 | 1959.85 | 1901.95 | 1895.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-10 10:15:00 | 1901.10 | 1913.56 | 1914.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2024-04-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 10:15:00 | 1901.10 | 1913.56 | 1914.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 12:15:00 | 1894.20 | 1903.16 | 1907.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-18 12:15:00 | 1817.95 | 1816.37 | 1832.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-18 12:45:00 | 1818.00 | 1816.37 | 1832.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 1798.60 | 1778.92 | 1793.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 11:00:00 | 1784.20 | 1779.97 | 1792.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 13:00:00 | 1784.45 | 1782.35 | 1791.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-23 09:15:00 | 1823.50 | 1794.92 | 1795.07 | SL hit (close>static) qty=1.00 sl=1814.60 alert=retest2 |

### Cycle 61 — BUY (started 2024-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 10:15:00 | 1821.85 | 1800.31 | 1797.50 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 11:15:00 | 1795.70 | 1805.91 | 1806.67 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 09:15:00 | 1817.80 | 1807.13 | 1805.69 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 15:15:00 | 1799.00 | 1805.96 | 1806.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-02 09:15:00 | 1779.65 | 1800.70 | 1803.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-02 13:15:00 | 1802.50 | 1796.13 | 1799.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-02 13:15:00 | 1802.50 | 1796.13 | 1799.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 13:15:00 | 1802.50 | 1796.13 | 1799.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 14:00:00 | 1802.50 | 1796.13 | 1799.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 14:15:00 | 1785.10 | 1793.93 | 1798.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 14:00:00 | 1754.55 | 1786.27 | 1793.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-03 14:15:00 | 1805.00 | 1790.02 | 1794.11 | SL hit (close>static) qty=1.00 sl=1803.90 alert=retest2 |

### Cycle 65 — BUY (started 2024-05-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-09 12:15:00 | 1770.15 | 1760.62 | 1760.35 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-05-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 14:15:00 | 1708.95 | 1750.04 | 1755.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 09:15:00 | 1674.40 | 1711.34 | 1728.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-13 14:15:00 | 1705.00 | 1691.86 | 1710.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-13 14:15:00 | 1705.00 | 1691.86 | 1710.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 14:15:00 | 1705.00 | 1691.86 | 1710.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 14:45:00 | 1717.90 | 1691.86 | 1710.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 15:15:00 | 1714.00 | 1696.29 | 1710.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 09:30:00 | 1693.35 | 1697.25 | 1709.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-14 11:15:00 | 1745.00 | 1710.43 | 1713.68 | SL hit (close>static) qty=1.00 sl=1735.00 alert=retest2 |

### Cycle 67 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 1756.00 | 1719.55 | 1717.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 1785.70 | 1739.55 | 1727.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 09:15:00 | 1809.10 | 1813.16 | 1782.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 10:00:00 | 1809.10 | 1813.16 | 1782.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 1841.30 | 1833.70 | 1824.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 10:15:00 | 1854.50 | 1833.70 | 1824.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 14:15:00 | 1898.25 | 1903.30 | 1903.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 14:15:00 | 1898.25 | 1903.30 | 1903.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 15:15:00 | 1885.75 | 1899.79 | 1902.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 13:15:00 | 1885.80 | 1875.63 | 1882.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 13:15:00 | 1885.80 | 1875.63 | 1882.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 13:15:00 | 1885.80 | 1875.63 | 1882.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 14:00:00 | 1885.80 | 1875.63 | 1882.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 1875.50 | 1875.60 | 1882.10 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2024-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 11:15:00 | 1898.05 | 1886.90 | 1885.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 12:15:00 | 1902.55 | 1890.03 | 1887.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-31 14:15:00 | 1889.70 | 1901.43 | 1893.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 14:15:00 | 1889.70 | 1901.43 | 1893.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 1889.70 | 1901.43 | 1893.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 1889.70 | 1901.43 | 1893.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 1912.90 | 1903.72 | 1895.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 09:15:00 | 2046.45 | 1903.72 | 1895.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 09:45:00 | 1921.00 | 1986.87 | 1954.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 1687.65 | 1927.03 | 1930.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 1687.65 | 1927.03 | 1930.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 1667.85 | 1875.19 | 1906.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 1747.20 | 1739.32 | 1812.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 11:00:00 | 1747.20 | 1739.32 | 1812.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 1784.00 | 1757.91 | 1803.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:45:00 | 1824.85 | 1757.91 | 1803.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 1818.10 | 1769.95 | 1804.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 15:00:00 | 1818.10 | 1769.95 | 1804.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 1827.00 | 1781.36 | 1806.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 1888.55 | 1781.36 | 1806.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 1895.00 | 1823.26 | 1822.56 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 11:15:00 | 1856.70 | 1862.70 | 1863.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-12 12:15:00 | 1849.70 | 1860.10 | 1861.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-14 09:15:00 | 1815.70 | 1814.95 | 1830.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-14 09:45:00 | 1816.05 | 1814.95 | 1830.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 1811.15 | 1807.72 | 1818.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 09:15:00 | 1781.15 | 1811.71 | 1815.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 09:15:00 | 1800.85 | 1801.58 | 1802.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 14:15:00 | 1803.60 | 1801.40 | 1801.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 13:15:00 | 1812.30 | 1799.25 | 1798.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2024-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 13:15:00 | 1812.30 | 1799.25 | 1798.59 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 12:15:00 | 1793.00 | 1798.54 | 1799.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 13:15:00 | 1787.65 | 1796.36 | 1798.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 14:15:00 | 1799.95 | 1797.08 | 1798.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 14:15:00 | 1799.95 | 1797.08 | 1798.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 1799.95 | 1797.08 | 1798.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 12:00:00 | 1787.00 | 1792.74 | 1795.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 12:30:00 | 1786.35 | 1791.62 | 1794.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 14:15:00 | 1812.55 | 1786.61 | 1788.33 | SL hit (close>static) qty=1.00 sl=1801.40 alert=retest2 |

### Cycle 75 — BUY (started 2024-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 15:15:00 | 1808.00 | 1790.89 | 1790.11 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 09:15:00 | 1779.80 | 1788.91 | 1789.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 14:15:00 | 1772.05 | 1778.46 | 1782.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-02 15:15:00 | 1779.75 | 1778.71 | 1781.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 15:15:00 | 1779.75 | 1778.71 | 1781.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 15:15:00 | 1779.75 | 1778.71 | 1781.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 09:15:00 | 1768.25 | 1778.71 | 1781.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 1772.25 | 1777.42 | 1781.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 10:30:00 | 1760.70 | 1766.70 | 1771.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 10:15:00 | 1776.65 | 1760.68 | 1759.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2024-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 10:15:00 | 1776.65 | 1760.68 | 1759.91 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 1753.90 | 1760.75 | 1760.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 10:15:00 | 1743.05 | 1757.21 | 1759.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 14:15:00 | 1758.80 | 1755.26 | 1757.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 14:15:00 | 1758.80 | 1755.26 | 1757.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 1758.80 | 1755.26 | 1757.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 15:00:00 | 1758.80 | 1755.26 | 1757.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 1752.35 | 1754.68 | 1757.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 1750.85 | 1754.68 | 1757.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 1747.00 | 1753.14 | 1756.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:45:00 | 1753.00 | 1753.14 | 1756.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 1752.00 | 1752.41 | 1755.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:00:00 | 1752.00 | 1752.41 | 1755.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 1752.00 | 1750.38 | 1753.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:15:00 | 1747.45 | 1750.38 | 1753.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 1742.55 | 1748.82 | 1752.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 12:45:00 | 1739.90 | 1745.40 | 1749.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 13:45:00 | 1738.75 | 1743.49 | 1748.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 11:00:00 | 1739.50 | 1739.33 | 1744.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 13:00:00 | 1739.00 | 1739.28 | 1743.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 1741.00 | 1739.61 | 1743.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 15:00:00 | 1741.00 | 1739.61 | 1743.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 1737.20 | 1739.13 | 1742.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:15:00 | 1798.05 | 1739.13 | 1742.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-16 09:15:00 | 1784.50 | 1748.20 | 1746.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 1784.50 | 1748.20 | 1746.36 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 11:15:00 | 1739.45 | 1752.25 | 1752.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 1728.50 | 1743.72 | 1747.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 09:15:00 | 1766.85 | 1732.54 | 1734.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 09:15:00 | 1766.85 | 1732.54 | 1734.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 1766.85 | 1732.54 | 1734.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:30:00 | 1775.75 | 1732.54 | 1734.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 10:15:00 | 1788.00 | 1743.63 | 1739.31 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 15:15:00 | 1722.00 | 1736.80 | 1738.10 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 14:15:00 | 1821.15 | 1750.02 | 1741.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 15:15:00 | 1848.00 | 1769.61 | 1750.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 14:15:00 | 1807.80 | 1812.23 | 1784.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 15:00:00 | 1807.80 | 1812.23 | 1784.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 1830.00 | 1844.66 | 1828.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 15:00:00 | 1830.00 | 1844.66 | 1828.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 1830.00 | 1841.73 | 1828.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:30:00 | 1819.00 | 1839.46 | 1828.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 1846.55 | 1840.88 | 1830.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 12:45:00 | 1856.55 | 1845.28 | 1834.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 1854.60 | 1847.04 | 1837.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 10:15:00 | 1794.25 | 1858.94 | 1863.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 1794.25 | 1858.94 | 1863.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 12:15:00 | 1780.50 | 1833.18 | 1850.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 1774.60 | 1770.96 | 1794.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 10:45:00 | 1778.00 | 1770.96 | 1794.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 1781.15 | 1777.04 | 1784.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 14:00:00 | 1778.65 | 1777.36 | 1784.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 15:00:00 | 1775.10 | 1776.91 | 1783.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 14:15:00 | 1778.80 | 1780.42 | 1782.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 15:15:00 | 1774.00 | 1780.38 | 1782.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 1774.00 | 1779.11 | 1781.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 1722.10 | 1779.11 | 1781.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1689.72 | 1767.37 | 1776.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1686.34 | 1767.37 | 1776.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1689.86 | 1767.37 | 1776.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1685.30 | 1767.37 | 1776.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 11:00:00 | 1752.80 | 1764.46 | 1774.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 13:15:00 | 1786.70 | 1767.26 | 1772.78 | SL hit (close>ema200) qty=0.50 sl=1767.26 alert=retest2 |

### Cycle 85 — BUY (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 09:15:00 | 1825.35 | 1785.73 | 1780.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 1871.05 | 1824.66 | 1812.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 12:15:00 | 1913.45 | 1917.70 | 1893.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 13:00:00 | 1913.45 | 1917.70 | 1893.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 1909.35 | 1913.44 | 1900.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 09:45:00 | 1914.80 | 1900.66 | 1898.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 12:15:00 | 1899.05 | 1902.70 | 1899.91 | SL hit (close<static) qty=1.00 sl=1900.20 alert=retest2 |

### Cycle 86 — SELL (started 2024-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 10:15:00 | 1884.15 | 1896.62 | 1898.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 13:15:00 | 1877.90 | 1889.04 | 1893.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 14:15:00 | 1890.75 | 1889.38 | 1893.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-26 15:00:00 | 1890.75 | 1889.38 | 1893.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 1891.45 | 1889.57 | 1892.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 14:00:00 | 1882.30 | 1886.87 | 1890.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 09:15:00 | 1873.00 | 1886.21 | 1889.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 10:15:00 | 1882.10 | 1845.89 | 1847.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-02 10:15:00 | 1870.85 | 1850.88 | 1849.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2024-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 10:15:00 | 1870.85 | 1850.88 | 1849.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 12:15:00 | 1892.95 | 1861.86 | 1854.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 14:15:00 | 1897.50 | 1904.47 | 1889.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-03 15:00:00 | 1897.50 | 1904.47 | 1889.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 1892.90 | 1900.48 | 1889.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 12:45:00 | 1907.90 | 1902.12 | 1892.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 09:15:00 | 1920.40 | 1902.09 | 1900.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 09:15:00 | 1882.00 | 1898.07 | 1898.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 1882.00 | 1898.07 | 1898.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 1857.10 | 1885.66 | 1892.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 1875.75 | 1865.10 | 1874.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 1875.75 | 1865.10 | 1874.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1875.75 | 1865.10 | 1874.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:30:00 | 1876.45 | 1865.10 | 1874.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 1886.00 | 1869.28 | 1875.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 11:00:00 | 1886.00 | 1869.28 | 1875.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 11:15:00 | 1877.65 | 1870.96 | 1875.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 15:00:00 | 1845.25 | 1870.12 | 1874.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-16 09:15:00 | 1892.60 | 1817.37 | 1822.92 | SL hit (close>static) qty=1.00 sl=1887.95 alert=retest2 |

### Cycle 89 — BUY (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 10:15:00 | 1903.15 | 1834.52 | 1830.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 11:15:00 | 1918.70 | 1851.36 | 1838.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 09:15:00 | 1935.95 | 1938.31 | 1909.15 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 10:30:00 | 1959.15 | 1943.36 | 1914.09 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 11:00:00 | 1963.55 | 1943.36 | 1914.09 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 1912.15 | 1940.55 | 1926.44 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-19 09:15:00 | 1912.15 | 1940.55 | 1926.44 | SL hit (close<ema400) qty=1.00 sl=1926.44 alert=retest1 |

### Cycle 90 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 1973.75 | 2048.12 | 2049.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 1957.60 | 2018.80 | 2035.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 1766.40 | 1765.65 | 1797.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 09:45:00 | 1772.25 | 1765.65 | 1797.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 1804.50 | 1777.91 | 1795.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:00:00 | 1804.50 | 1777.91 | 1795.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 1802.20 | 1782.77 | 1795.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:45:00 | 1807.75 | 1782.77 | 1795.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 1813.45 | 1793.89 | 1798.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 1816.05 | 1793.89 | 1798.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 1812.45 | 1803.38 | 1802.52 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 14:15:00 | 1783.55 | 1800.73 | 1801.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 10:15:00 | 1777.20 | 1792.39 | 1797.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-10 14:15:00 | 1782.10 | 1780.69 | 1789.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-10 15:00:00 | 1782.10 | 1780.69 | 1789.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 1787.85 | 1782.12 | 1789.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:15:00 | 1776.55 | 1782.12 | 1789.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 1788.00 | 1783.30 | 1789.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:45:00 | 1787.30 | 1783.30 | 1789.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 1772.80 | 1781.20 | 1787.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 11:15:00 | 1768.50 | 1781.20 | 1787.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-11 14:15:00 | 1791.50 | 1783.00 | 1786.35 | SL hit (close>static) qty=1.00 sl=1788.60 alert=retest2 |

### Cycle 93 — BUY (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 15:15:00 | 1639.40 | 1624.99 | 1623.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 1654.70 | 1630.93 | 1626.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 1672.20 | 1682.83 | 1659.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:00:00 | 1672.20 | 1682.83 | 1659.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 1665.00 | 1674.24 | 1662.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:45:00 | 1663.95 | 1674.24 | 1662.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 1642.40 | 1667.87 | 1660.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 1642.40 | 1667.87 | 1660.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 1650.00 | 1664.30 | 1659.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 1632.15 | 1664.30 | 1659.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 1623.70 | 1650.94 | 1654.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 1615.00 | 1643.76 | 1650.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1502.80 | 1498.75 | 1528.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 1502.80 | 1498.75 | 1528.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 14:15:00 | 1481.40 | 1495.25 | 1515.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 14:45:00 | 1511.85 | 1495.25 | 1515.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 1486.50 | 1493.14 | 1511.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 11:00:00 | 1470.10 | 1488.53 | 1507.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 13:15:00 | 1467.40 | 1484.27 | 1502.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 14:15:00 | 1462.95 | 1481.59 | 1499.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-21 09:15:00 | 1323.09 | 1381.79 | 1436.20 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 10:15:00 | 1087.20 | 1010.58 | 1010.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 09:15:00 | 1216.00 | 1092.87 | 1055.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 13:15:00 | 1316.00 | 1316.62 | 1270.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 09:15:00 | 1274.90 | 1306.95 | 1277.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 1274.90 | 1306.95 | 1277.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:30:00 | 1281.90 | 1306.95 | 1277.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 1264.80 | 1298.52 | 1276.10 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 09:15:00 | 1234.95 | 1267.50 | 1268.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 11:15:00 | 1226.70 | 1236.36 | 1247.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 11:15:00 | 1214.95 | 1211.93 | 1227.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 11:15:00 | 1214.95 | 1211.93 | 1227.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 11:15:00 | 1214.95 | 1211.93 | 1227.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 11:45:00 | 1216.05 | 1211.93 | 1227.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 13:15:00 | 1218.75 | 1214.91 | 1226.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 13:30:00 | 1221.30 | 1214.91 | 1226.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 1194.50 | 1211.88 | 1222.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 11:30:00 | 1188.00 | 1203.43 | 1216.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 09:15:00 | 1128.60 | 1152.59 | 1173.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-12 10:15:00 | 1244.50 | 1170.97 | 1180.14 | SL hit (close>ema200) qty=0.50 sl=1170.97 alert=retest2 |

### Cycle 97 — BUY (started 2024-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 12:15:00 | 1242.00 | 1193.99 | 1189.56 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 10:15:00 | 1174.35 | 1194.74 | 1196.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 11:15:00 | 1170.95 | 1189.98 | 1194.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 11:15:00 | 1088.05 | 1083.42 | 1102.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 11:15:00 | 1088.05 | 1083.42 | 1102.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 1088.05 | 1083.42 | 1102.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 11:45:00 | 1096.55 | 1083.42 | 1102.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 1038.50 | 1037.74 | 1055.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:30:00 | 1046.95 | 1037.74 | 1055.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 1042.50 | 1034.66 | 1044.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 11:00:00 | 1042.50 | 1034.66 | 1044.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 11:15:00 | 1035.75 | 1034.88 | 1043.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 11:30:00 | 1040.25 | 1034.88 | 1043.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 12:15:00 | 1069.00 | 1041.70 | 1045.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 13:00:00 | 1069.00 | 1041.70 | 1045.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 1071.90 | 1047.74 | 1048.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 13:45:00 | 1077.00 | 1047.74 | 1048.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2024-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 14:15:00 | 1063.35 | 1050.86 | 1049.70 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 10:15:00 | 1037.85 | 1053.87 | 1054.72 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 09:15:00 | 1055.85 | 1047.59 | 1047.46 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 14:15:00 | 1039.95 | 1046.51 | 1047.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 1017.00 | 1039.41 | 1043.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 1004.15 | 1002.35 | 1018.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 09:30:00 | 1012.15 | 1002.35 | 1018.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 973.35 | 925.52 | 942.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 10:00:00 | 973.35 | 925.52 | 942.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 1001.85 | 940.79 | 947.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 1001.85 | 940.79 | 947.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 11:15:00 | 1009.70 | 954.57 | 953.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 09:15:00 | 1041.45 | 997.25 | 977.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 11:15:00 | 1065.45 | 1067.49 | 1046.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 12:00:00 | 1065.45 | 1067.49 | 1046.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 1061.40 | 1069.33 | 1055.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:00:00 | 1061.40 | 1069.33 | 1055.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 1044.55 | 1063.49 | 1059.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 1044.55 | 1063.49 | 1059.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 1050.40 | 1060.87 | 1058.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 11:30:00 | 1054.65 | 1060.17 | 1058.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 12:30:00 | 1052.00 | 1058.77 | 1058.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 14:00:00 | 1054.50 | 1057.92 | 1057.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 14:15:00 | 1046.55 | 1055.65 | 1056.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 14:15:00 | 1046.55 | 1055.65 | 1056.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 1016.80 | 1046.02 | 1052.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 1032.75 | 1028.82 | 1039.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 1032.75 | 1028.82 | 1039.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 1032.75 | 1028.82 | 1039.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 1032.75 | 1028.82 | 1039.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1041.55 | 1031.72 | 1038.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 1041.55 | 1031.72 | 1038.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1033.70 | 1032.11 | 1038.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 11:30:00 | 1022.05 | 1030.84 | 1037.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:45:00 | 1024.30 | 1031.28 | 1035.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 15:15:00 | 1022.65 | 1031.28 | 1035.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 10:00:00 | 1017.75 | 1027.20 | 1033.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 1026.45 | 1027.05 | 1032.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:45:00 | 1024.00 | 1027.05 | 1032.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 11:15:00 | 1025.75 | 1026.79 | 1031.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 11:45:00 | 1028.10 | 1026.79 | 1031.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 971.55 | 996.15 | 1008.60 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 970.95 | 996.15 | 1008.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 973.08 | 996.15 | 1008.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 971.52 | 996.15 | 1008.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 10:15:00 | 966.86 | 992.69 | 1005.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 12:15:00 | 1003.00 | 994.00 | 1004.15 | SL hit (close>ema200) qty=0.50 sl=994.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 14:15:00 | 1000.45 | 990.69 | 990.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 15:15:00 | 1006.00 | 993.75 | 991.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 995.90 | 1008.06 | 1000.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 995.90 | 1008.06 | 1000.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 995.90 | 1008.06 | 1000.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 995.90 | 1008.06 | 1000.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1007.95 | 1008.03 | 1001.50 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 982.30 | 996.08 | 997.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 13:15:00 | 971.20 | 985.58 | 991.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 983.00 | 980.62 | 987.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 983.00 | 980.62 | 987.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 983.00 | 980.62 | 987.62 | EMA400 retest candle locked (from downside) |

### Cycle 107 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 1004.90 | 989.04 | 987.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 12:15:00 | 1008.80 | 994.59 | 990.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 11:15:00 | 1001.45 | 1004.49 | 998.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 12:00:00 | 1001.45 | 1004.49 | 998.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 1002.30 | 1004.05 | 999.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 1002.30 | 1004.05 | 999.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 996.90 | 1002.62 | 998.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 997.10 | 1002.62 | 998.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 997.90 | 1001.68 | 998.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:15:00 | 995.40 | 1001.68 | 998.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 995.40 | 1000.42 | 998.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 991.55 | 1000.42 | 998.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 996.90 | 999.72 | 998.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:45:00 | 989.85 | 999.72 | 998.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 1001.65 | 1000.10 | 998.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 1006.55 | 1000.10 | 998.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 13:15:00 | 989.45 | 996.90 | 997.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 989.45 | 996.90 | 997.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 971.30 | 989.78 | 993.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 09:15:00 | 972.85 | 966.49 | 977.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-11 09:15:00 | 972.85 | 966.49 | 977.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 972.85 | 966.49 | 977.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 09:30:00 | 983.25 | 966.49 | 977.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 10:15:00 | 970.00 | 967.19 | 976.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 10:30:00 | 971.80 | 967.19 | 976.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 936.80 | 928.09 | 942.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:30:00 | 922.45 | 927.90 | 938.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 13:15:00 | 876.33 | 895.16 | 914.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 12:15:00 | 875.05 | 872.98 | 893.22 | SL hit (close>ema200) qty=0.50 sl=872.98 alert=retest2 |

### Cycle 109 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 827.85 | 797.87 | 796.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 12:15:00 | 839.20 | 806.14 | 800.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 843.20 | 843.62 | 829.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 09:15:00 | 837.40 | 843.62 | 829.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 827.40 | 838.38 | 830.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:00:00 | 827.40 | 838.38 | 830.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 829.75 | 836.65 | 830.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 14:30:00 | 836.50 | 835.36 | 830.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:30:00 | 839.80 | 837.51 | 832.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 09:15:00 | 821.40 | 832.39 | 833.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 821.40 | 832.39 | 833.06 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 14:15:00 | 857.00 | 829.68 | 828.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 09:15:00 | 888.45 | 845.47 | 836.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 15:15:00 | 909.00 | 909.07 | 899.69 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 09:15:00 | 917.05 | 909.07 | 899.69 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 10:15:00 | 913.70 | 907.92 | 900.02 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 12:00:00 | 912.70 | 910.02 | 902.42 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 10:15:00 | 962.90 | 930.47 | 916.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 10:15:00 | 959.39 | 930.47 | 916.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 10:15:00 | 958.34 | 930.47 | 916.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-03-24 10:15:00 | 951.00 | 951.32 | 936.24 | SL hit (close<ema200) qty=0.50 sl=951.32 alert=retest1 |

### Cycle 112 — SELL (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 14:15:00 | 926.00 | 938.58 | 939.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 15:15:00 | 918.00 | 934.46 | 937.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 09:15:00 | 937.40 | 935.05 | 937.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 09:15:00 | 937.40 | 935.05 | 937.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 937.40 | 935.05 | 937.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:00:00 | 937.40 | 935.05 | 937.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 933.10 | 934.66 | 937.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 11:30:00 | 926.55 | 932.44 | 936.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 12:00:00 | 928.55 | 926.59 | 930.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 13:15:00 | 951.30 | 933.77 | 932.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 13:15:00 | 951.30 | 933.77 | 932.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 14:15:00 | 958.50 | 938.71 | 935.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 13:15:00 | 952.00 | 953.29 | 945.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 13:15:00 | 952.00 | 953.29 | 945.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 952.00 | 953.29 | 945.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:00:00 | 952.00 | 953.29 | 945.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 948.40 | 952.32 | 945.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 15:00:00 | 948.40 | 952.32 | 945.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 941.05 | 949.37 | 945.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:00:00 | 941.05 | 949.37 | 945.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 930.25 | 945.55 | 944.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:00:00 | 930.25 | 945.55 | 944.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 11:15:00 | 931.50 | 942.74 | 943.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 09:15:00 | 916.35 | 928.91 | 935.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 10:15:00 | 937.40 | 930.60 | 935.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 10:15:00 | 937.40 | 930.60 | 935.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 937.40 | 930.60 | 935.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:00:00 | 937.40 | 930.60 | 935.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 938.10 | 932.10 | 935.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 938.10 | 932.10 | 935.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 938.95 | 933.47 | 936.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 13:30:00 | 933.35 | 933.80 | 936.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 09:45:00 | 931.70 | 937.17 | 937.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 10:15:00 | 951.40 | 940.02 | 938.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 10:15:00 | 951.40 | 940.02 | 938.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 11:15:00 | 959.15 | 943.85 | 940.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 928.50 | 946.68 | 944.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 928.50 | 946.68 | 944.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 928.50 | 946.68 | 944.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 928.50 | 946.68 | 944.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 939.00 | 945.14 | 943.56 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 925.75 | 939.31 | 941.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 924.00 | 936.25 | 939.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 890.50 | 884.38 | 900.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 890.50 | 884.38 | 900.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 892.85 | 869.83 | 878.03 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2025-04-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 13:15:00 | 886.30 | 882.96 | 882.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 15:15:00 | 894.45 | 886.78 | 884.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 14:15:00 | 947.60 | 949.29 | 937.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 15:00:00 | 947.60 | 949.29 | 937.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 13:15:00 | 950.95 | 954.79 | 950.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 13:45:00 | 949.15 | 954.79 | 950.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 944.00 | 952.63 | 949.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 14:45:00 | 944.60 | 952.63 | 949.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 944.25 | 950.95 | 949.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 951.80 | 950.95 | 949.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 930.65 | 946.89 | 947.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2025-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 09:15:00 | 930.65 | 946.89 | 947.75 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 15:15:00 | 951.00 | 946.78 | 946.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 09:15:00 | 961.10 | 949.64 | 948.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 12:15:00 | 950.00 | 951.14 | 949.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 12:15:00 | 950.00 | 951.14 | 949.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 950.00 | 951.14 | 949.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 12:45:00 | 952.20 | 951.14 | 949.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 966.00 | 954.11 | 950.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 15:00:00 | 969.80 | 957.25 | 952.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 924.50 | 952.96 | 951.55 | SL hit (close<static) qty=1.00 sl=949.50 alert=retest2 |

### Cycle 120 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 910.50 | 944.47 | 947.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 14:15:00 | 902.05 | 914.03 | 922.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 922.85 | 912.91 | 920.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 922.85 | 912.91 | 920.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 922.85 | 912.91 | 920.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:30:00 | 920.40 | 912.91 | 920.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 918.60 | 914.05 | 920.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:15:00 | 915.45 | 914.05 | 920.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 936.75 | 915.06 | 917.52 | SL hit (close>static) qty=1.00 sl=931.95 alert=retest2 |

### Cycle 121 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 957.25 | 923.49 | 921.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 11:15:00 | 991.00 | 937.00 | 927.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 947.30 | 953.94 | 941.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 10:00:00 | 947.30 | 953.94 | 941.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 938.90 | 950.94 | 940.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:00:00 | 938.90 | 950.94 | 940.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 936.65 | 948.08 | 940.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:30:00 | 928.90 | 948.08 | 940.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 15:15:00 | 921.90 | 935.17 | 936.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 911.30 | 930.40 | 933.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-08 11:15:00 | 922.55 | 921.31 | 925.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 11:15:00 | 922.55 | 921.31 | 925.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 11:15:00 | 922.55 | 921.31 | 925.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 11:30:00 | 922.05 | 921.31 | 925.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 934.95 | 890.30 | 897.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:45:00 | 939.10 | 890.30 | 897.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 933.65 | 906.76 | 904.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 941.55 | 921.61 | 912.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 959.40 | 962.65 | 951.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 09:45:00 | 958.20 | 962.65 | 951.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1016.00 | 1014.07 | 1000.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 1013.35 | 1014.07 | 1000.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 1006.00 | 1012.27 | 1002.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 1004.95 | 1012.27 | 1002.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 1000.50 | 1009.92 | 1002.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:45:00 | 1000.30 | 1009.92 | 1002.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 990.90 | 1006.12 | 1001.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 990.90 | 1006.12 | 1001.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 995.05 | 998.40 | 998.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 995.05 | 998.40 | 998.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 986.00 | 995.92 | 997.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 13:15:00 | 981.35 | 989.65 | 992.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 10:15:00 | 990.20 | 988.49 | 991.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 11:00:00 | 990.20 | 988.49 | 991.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 991.00 | 988.99 | 991.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:00:00 | 991.00 | 988.99 | 991.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 992.10 | 989.62 | 991.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 13:30:00 | 987.45 | 989.05 | 990.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 09:15:00 | 1002.25 | 991.24 | 991.39 | SL hit (close>static) qty=1.00 sl=994.80 alert=retest2 |

### Cycle 125 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 1017.90 | 996.57 | 993.80 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 10:15:00 | 1006.00 | 1007.38 | 1007.42 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 11:15:00 | 1008.70 | 1007.65 | 1007.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 1021.85 | 1010.49 | 1008.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 1014.00 | 1014.34 | 1011.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 10:00:00 | 1014.00 | 1014.34 | 1011.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1009.20 | 1013.31 | 1011.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 1009.75 | 1013.31 | 1011.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 1011.20 | 1012.89 | 1011.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 13:30:00 | 1017.70 | 1013.64 | 1011.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 14:45:00 | 1018.00 | 1013.56 | 1012.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 09:30:00 | 1017.50 | 1014.54 | 1012.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 12:00:00 | 1017.50 | 1015.12 | 1013.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 1010.70 | 1014.19 | 1013.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:45:00 | 1009.50 | 1014.19 | 1013.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 1009.50 | 1013.25 | 1012.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 1004.00 | 1013.25 | 1012.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-03 10:15:00 | 1001.50 | 1010.85 | 1011.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 1001.50 | 1010.85 | 1011.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 995.50 | 1006.51 | 1009.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 12:15:00 | 1003.80 | 999.04 | 1003.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 12:15:00 | 1003.80 | 999.04 | 1003.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 1003.80 | 999.04 | 1003.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:00:00 | 1003.80 | 999.04 | 1003.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 1005.60 | 1000.35 | 1003.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:00:00 | 1005.60 | 1000.35 | 1003.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 1002.90 | 1000.86 | 1003.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:15:00 | 1005.90 | 1000.86 | 1003.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 1005.90 | 1001.87 | 1003.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:15:00 | 1016.70 | 1001.87 | 1003.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 1014.90 | 1004.48 | 1004.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:00:00 | 1014.90 | 1004.48 | 1004.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 10:15:00 | 1014.50 | 1006.48 | 1005.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 12:15:00 | 1016.90 | 1010.56 | 1008.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 12:15:00 | 1047.70 | 1053.49 | 1044.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 13:00:00 | 1047.70 | 1053.49 | 1044.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 1039.60 | 1050.71 | 1043.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 1040.20 | 1050.71 | 1043.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1047.70 | 1050.11 | 1044.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 15:15:00 | 1049.00 | 1050.11 | 1044.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 1033.70 | 1044.99 | 1043.10 | SL hit (close<static) qty=1.00 sl=1039.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 1036.90 | 1040.89 | 1041.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 1013.00 | 1035.31 | 1038.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 996.90 | 993.97 | 1005.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 996.90 | 993.97 | 1005.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 963.10 | 951.61 | 956.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:00:00 | 963.10 | 951.61 | 956.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 963.80 | 954.05 | 956.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:00:00 | 963.80 | 954.05 | 956.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 959.40 | 955.91 | 957.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:30:00 | 960.40 | 955.91 | 957.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 960.70 | 956.87 | 957.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:45:00 | 960.40 | 956.87 | 957.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 960.00 | 957.49 | 957.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 979.70 | 957.49 | 957.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 987.40 | 963.48 | 960.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 995.50 | 969.88 | 963.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 10:15:00 | 983.10 | 984.21 | 975.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 10:45:00 | 984.50 | 984.21 | 975.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 980.60 | 983.71 | 979.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:45:00 | 981.70 | 983.71 | 979.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 978.90 | 982.75 | 979.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 978.90 | 982.75 | 979.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 983.50 | 982.90 | 979.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 13:45:00 | 990.00 | 984.62 | 981.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 14:45:00 | 991.20 | 985.94 | 982.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 12:15:00 | 1011.10 | 1015.71 | 1016.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 12:15:00 | 1011.10 | 1015.71 | 1016.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 15:15:00 | 1007.30 | 1012.55 | 1014.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 992.50 | 990.31 | 994.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 15:00:00 | 992.50 | 990.31 | 994.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 998.00 | 992.39 | 995.13 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 09:15:00 | 1000.40 | 997.09 | 996.82 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 995.70 | 996.72 | 996.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 988.90 | 995.15 | 996.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 11:15:00 | 996.00 | 995.32 | 996.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 11:15:00 | 996.00 | 995.32 | 996.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 996.00 | 995.32 | 996.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:45:00 | 998.10 | 995.32 | 996.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 997.60 | 995.78 | 996.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:30:00 | 997.70 | 995.78 | 996.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 997.30 | 996.08 | 996.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:30:00 | 998.40 | 996.08 | 996.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 993.70 | 995.64 | 996.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 1006.00 | 995.64 | 996.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 09:15:00 | 1006.90 | 997.89 | 997.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 10:15:00 | 1021.60 | 1002.63 | 999.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 1040.00 | 1044.05 | 1037.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 10:00:00 | 1040.00 | 1044.05 | 1037.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1039.50 | 1043.14 | 1038.06 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 1030.00 | 1035.42 | 1035.72 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 11:15:00 | 1037.60 | 1035.93 | 1035.87 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 09:15:00 | 1031.40 | 1035.44 | 1035.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 11:15:00 | 1027.30 | 1032.70 | 1034.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 1035.30 | 1026.36 | 1029.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 1035.30 | 1026.36 | 1029.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1035.30 | 1026.36 | 1029.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:30:00 | 1031.30 | 1026.36 | 1029.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1030.50 | 1027.19 | 1029.98 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 15:15:00 | 1034.40 | 1031.29 | 1031.20 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 09:15:00 | 1026.80 | 1030.39 | 1030.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 1006.40 | 1021.16 | 1025.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 999.50 | 995.30 | 1007.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 10:00:00 | 999.50 | 995.30 | 1007.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 14:15:00 | 1006.40 | 997.43 | 1004.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 14:45:00 | 1004.70 | 997.43 | 1004.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 1009.00 | 999.75 | 1004.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:15:00 | 1017.70 | 999.75 | 1004.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 1008.10 | 1001.42 | 1004.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 10:15:00 | 996.80 | 1001.42 | 1004.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 10:45:00 | 1002.00 | 1001.41 | 1004.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 14:15:00 | 1014.20 | 1006.52 | 1006.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 1014.20 | 1006.52 | 1006.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 1018.80 | 1010.56 | 1008.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 14:15:00 | 1011.70 | 1013.53 | 1010.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-30 15:00:00 | 1011.70 | 1013.53 | 1010.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 1009.00 | 1012.63 | 1010.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 1002.40 | 1012.63 | 1010.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 997.00 | 1009.50 | 1009.43 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 999.20 | 1007.44 | 1008.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 981.90 | 999.07 | 1004.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 09:15:00 | 999.70 | 997.25 | 1002.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 999.70 | 997.25 | 1002.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 999.70 | 997.25 | 1002.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:30:00 | 1002.05 | 997.25 | 1002.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 987.75 | 984.14 | 990.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:30:00 | 988.35 | 984.14 | 990.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 988.85 | 985.08 | 990.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:45:00 | 989.65 | 985.08 | 990.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 988.70 | 985.80 | 990.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 988.50 | 985.80 | 990.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 991.50 | 986.94 | 990.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 09:15:00 | 987.15 | 986.94 | 990.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 09:15:00 | 937.79 | 958.65 | 969.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 938.40 | 938.09 | 953.62 | SL hit (close>ema200) qty=0.50 sl=938.09 alert=retest2 |

### Cycle 143 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 947.00 | 930.14 | 927.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 11:15:00 | 955.35 | 944.43 | 937.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 975.70 | 976.48 | 967.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 13:15:00 | 973.80 | 976.48 | 967.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 966.10 | 973.65 | 967.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 966.10 | 973.65 | 967.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 966.00 | 972.12 | 967.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 964.15 | 972.12 | 967.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 968.65 | 971.43 | 967.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 967.85 | 971.43 | 967.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 959.15 | 968.97 | 966.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 959.15 | 968.97 | 966.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 961.80 | 967.54 | 966.26 | EMA400 retest candle locked (from upside) |

### Cycle 144 — SELL (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 13:15:00 | 956.55 | 963.97 | 964.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 955.70 | 962.31 | 963.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 963.85 | 962.49 | 963.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 963.85 | 962.49 | 963.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 963.85 | 962.49 | 963.74 | EMA400 retest candle locked (from downside) |

### Cycle 145 — BUY (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 13:15:00 | 966.70 | 964.55 | 964.41 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 948.80 | 961.32 | 962.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 934.00 | 945.87 | 953.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 924.50 | 920.29 | 928.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 924.50 | 920.29 | 928.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 924.50 | 920.29 | 928.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 926.15 | 920.29 | 928.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 927.95 | 923.47 | 927.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 927.95 | 923.47 | 927.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 931.60 | 925.09 | 927.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 931.60 | 925.09 | 927.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 931.00 | 926.27 | 927.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 930.85 | 926.27 | 927.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 938.60 | 930.87 | 929.85 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 931.15 | 934.11 | 934.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 924.65 | 931.07 | 932.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 932.10 | 928.03 | 930.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 932.10 | 928.03 | 930.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 932.10 | 928.03 | 930.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 932.10 | 928.03 | 930.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 928.90 | 928.20 | 930.52 | EMA400 retest candle locked (from downside) |

### Cycle 149 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 936.65 | 931.77 | 931.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 12:15:00 | 941.60 | 934.41 | 932.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 12:15:00 | 977.50 | 978.50 | 970.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 12:45:00 | 976.45 | 978.50 | 970.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 985.85 | 987.73 | 985.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:45:00 | 985.85 | 987.73 | 985.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 981.00 | 986.39 | 985.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 981.00 | 986.39 | 985.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 979.25 | 984.96 | 984.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:30:00 | 977.95 | 984.96 | 984.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — SELL (started 2025-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 15:15:00 | 980.00 | 983.97 | 984.19 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 10:15:00 | 989.05 | 984.79 | 984.51 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 977.75 | 983.25 | 983.85 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 09:15:00 | 1008.45 | 986.70 | 985.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 12:15:00 | 1051.80 | 1005.86 | 994.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 09:15:00 | 1097.80 | 1125.05 | 1101.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 1097.80 | 1125.05 | 1101.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1097.80 | 1125.05 | 1101.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:45:00 | 1105.80 | 1125.05 | 1101.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 1108.95 | 1121.83 | 1101.83 | EMA400 retest candle locked (from upside) |

### Cycle 154 — SELL (started 2025-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 13:15:00 | 1091.10 | 1098.38 | 1099.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 1077.95 | 1094.30 | 1097.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 15:15:00 | 1032.00 | 1028.23 | 1041.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-01 09:15:00 | 1048.10 | 1028.23 | 1041.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1040.80 | 1030.75 | 1041.35 | EMA400 retest candle locked (from downside) |

### Cycle 155 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 1067.30 | 1048.22 | 1045.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 1073.40 | 1053.25 | 1048.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 1063.00 | 1064.57 | 1057.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 09:30:00 | 1064.20 | 1064.57 | 1057.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 1058.60 | 1063.38 | 1057.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 1059.60 | 1063.38 | 1057.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 1059.40 | 1062.58 | 1057.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:45:00 | 1058.00 | 1062.58 | 1057.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 1057.70 | 1061.60 | 1057.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:45:00 | 1059.50 | 1061.60 | 1057.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 1058.00 | 1060.88 | 1057.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 14:15:00 | 1058.20 | 1060.88 | 1057.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 1059.70 | 1060.65 | 1058.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:15:00 | 1086.90 | 1060.36 | 1058.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:30:00 | 1063.00 | 1064.81 | 1061.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:00:00 | 1064.30 | 1064.81 | 1061.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 1046.20 | 1059.41 | 1059.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 1046.20 | 1059.41 | 1059.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 1037.70 | 1055.07 | 1057.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 1049.80 | 1048.46 | 1052.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 1049.80 | 1048.46 | 1052.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1049.80 | 1048.46 | 1052.57 | EMA400 retest candle locked (from downside) |

### Cycle 157 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 1060.00 | 1053.79 | 1053.13 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 1053.50 | 1054.98 | 1055.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 1041.00 | 1050.54 | 1052.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1041.90 | 1037.98 | 1043.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 1041.90 | 1037.98 | 1043.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1041.90 | 1037.98 | 1043.54 | EMA400 retest candle locked (from downside) |

### Cycle 159 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 1058.40 | 1045.04 | 1044.50 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 1040.40 | 1048.54 | 1049.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 09:15:00 | 1030.10 | 1041.34 | 1045.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 1041.60 | 1036.84 | 1040.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 1041.60 | 1036.84 | 1040.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 1041.60 | 1036.84 | 1040.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 1038.60 | 1036.84 | 1040.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 1036.30 | 1036.73 | 1039.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:15:00 | 1044.90 | 1036.73 | 1039.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1051.30 | 1039.65 | 1040.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:15:00 | 1057.40 | 1039.65 | 1040.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 1053.80 | 1042.48 | 1042.15 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 1036.30 | 1043.04 | 1043.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 12:15:00 | 1032.50 | 1039.60 | 1041.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 1050.70 | 1018.42 | 1021.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 1050.70 | 1018.42 | 1021.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1050.70 | 1018.42 | 1021.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 1050.70 | 1018.42 | 1021.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 1104.60 | 1035.65 | 1029.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 12:15:00 | 1115.30 | 1061.91 | 1043.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 15:15:00 | 1136.00 | 1138.14 | 1117.06 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 09:15:00 | 1157.30 | 1138.14 | 1117.06 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 1122.60 | 1133.85 | 1123.19 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-03 13:15:00 | 1122.60 | 1133.85 | 1123.19 | SL hit (close<ema400) qty=1.00 sl=1123.19 alert=retest1 |

### Cycle 164 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 1094.50 | 1113.92 | 1116.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 13:15:00 | 1086.20 | 1102.90 | 1109.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 14:15:00 | 1062.90 | 1057.14 | 1070.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 14:15:00 | 1062.90 | 1057.14 | 1070.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 1062.90 | 1057.14 | 1070.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 1069.20 | 1057.14 | 1070.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1069.10 | 1059.83 | 1069.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 12:45:00 | 1064.10 | 1062.76 | 1068.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:30:00 | 1061.10 | 1062.47 | 1067.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 1088.30 | 1053.17 | 1055.45 | SL hit (close>static) qty=1.00 sl=1088.00 alert=retest2 |

### Cycle 165 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 1076.10 | 1057.76 | 1057.32 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 10:15:00 | 1055.00 | 1074.39 | 1076.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 1048.90 | 1064.12 | 1069.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 14:15:00 | 1013.50 | 1011.63 | 1024.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 15:00:00 | 1013.50 | 1011.63 | 1024.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1028.80 | 1015.12 | 1023.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 1028.80 | 1015.12 | 1023.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1031.80 | 1018.46 | 1024.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 1031.80 | 1018.46 | 1024.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 1035.50 | 1027.13 | 1027.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 09:15:00 | 1040.60 | 1032.27 | 1029.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 10:15:00 | 1036.20 | 1040.87 | 1036.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 10:15:00 | 1036.20 | 1040.87 | 1036.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1036.20 | 1040.87 | 1036.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:00:00 | 1036.20 | 1040.87 | 1036.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 1033.60 | 1039.41 | 1036.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 1033.60 | 1039.41 | 1036.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 1043.50 | 1040.23 | 1037.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:30:00 | 1030.80 | 1040.23 | 1037.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 1039.00 | 1039.98 | 1037.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:30:00 | 1040.50 | 1039.98 | 1037.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 1039.60 | 1039.91 | 1037.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:45:00 | 1036.20 | 1039.91 | 1037.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 1042.30 | 1040.39 | 1038.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 1035.10 | 1040.39 | 1038.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1028.40 | 1037.99 | 1037.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 1028.40 | 1037.99 | 1037.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 1030.50 | 1036.49 | 1036.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 13:15:00 | 1021.70 | 1030.08 | 1033.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 1020.70 | 1018.17 | 1023.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:00:00 | 1020.70 | 1018.17 | 1023.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1018.50 | 1018.23 | 1022.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 1021.40 | 1018.23 | 1022.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 1018.20 | 1013.82 | 1017.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:00:00 | 1018.20 | 1013.82 | 1017.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 1016.30 | 1014.32 | 1016.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:15:00 | 1018.10 | 1014.32 | 1016.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 1018.10 | 1015.07 | 1017.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 1013.60 | 1015.07 | 1017.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:00:00 | 1013.90 | 1005.77 | 1006.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:30:00 | 1014.20 | 1006.15 | 1006.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 1013.40 | 1004.59 | 1004.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — BUY (started 2025-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 13:15:00 | 1013.40 | 1004.59 | 1004.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 15:15:00 | 1024.00 | 1010.75 | 1007.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 15:15:00 | 1043.90 | 1044.34 | 1034.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:15:00 | 1034.60 | 1044.34 | 1034.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1030.80 | 1041.63 | 1034.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 1031.70 | 1041.63 | 1034.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1032.50 | 1039.81 | 1034.13 | EMA400 retest candle locked (from upside) |

### Cycle 170 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 1024.00 | 1030.61 | 1031.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 1021.60 | 1028.81 | 1030.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 11:15:00 | 1028.30 | 1027.84 | 1029.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 11:15:00 | 1028.30 | 1027.84 | 1029.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 1028.30 | 1027.84 | 1029.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:45:00 | 1029.40 | 1027.84 | 1029.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 1022.10 | 1026.69 | 1028.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 13:30:00 | 1021.00 | 1025.44 | 1028.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 14:00:00 | 1020.40 | 1025.44 | 1028.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 15:15:00 | 1018.00 | 1014.76 | 1017.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 1022.00 | 1019.17 | 1018.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 1022.00 | 1019.17 | 1018.80 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 15:15:00 | 1017.00 | 1019.18 | 1019.40 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 10:15:00 | 1021.30 | 1019.69 | 1019.60 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 1017.20 | 1019.29 | 1019.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 1014.90 | 1018.41 | 1019.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 09:15:00 | 1022.50 | 1018.55 | 1018.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 09:15:00 | 1022.50 | 1018.55 | 1018.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1022.50 | 1018.55 | 1018.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:30:00 | 1021.10 | 1018.55 | 1018.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — BUY (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 10:15:00 | 1022.60 | 1019.36 | 1019.28 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 1014.40 | 1019.29 | 1019.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 1011.90 | 1017.81 | 1018.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 09:15:00 | 1007.40 | 1007.39 | 1011.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 10:00:00 | 1007.40 | 1007.39 | 1011.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 1002.00 | 1006.31 | 1011.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 11:15:00 | 1001.50 | 1006.31 | 1011.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 13:15:00 | 1001.70 | 1004.37 | 1009.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 1013.60 | 1007.99 | 1009.56 | SL hit (close>static) qty=1.00 sl=1011.60 alert=retest2 |

### Cycle 177 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 1016.80 | 1010.95 | 1010.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 1033.20 | 1017.32 | 1014.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 10:15:00 | 1033.60 | 1034.17 | 1028.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 10:30:00 | 1031.70 | 1034.17 | 1028.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 1030.00 | 1033.33 | 1028.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:45:00 | 1029.50 | 1033.33 | 1028.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 1031.00 | 1033.24 | 1029.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 1031.00 | 1033.24 | 1029.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 1031.30 | 1032.86 | 1029.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:30:00 | 1030.20 | 1032.86 | 1029.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 1028.70 | 1032.02 | 1029.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 1024.80 | 1032.02 | 1029.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1031.30 | 1031.88 | 1029.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 1024.00 | 1031.88 | 1029.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 1027.40 | 1030.98 | 1029.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 1027.40 | 1030.98 | 1029.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 1028.10 | 1030.41 | 1029.52 | EMA400 retest candle locked (from upside) |

### Cycle 178 — SELL (started 2026-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 13:15:00 | 1015.80 | 1026.52 | 1027.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 1012.30 | 1018.67 | 1021.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 936.80 | 936.72 | 947.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 10:00:00 | 936.80 | 936.72 | 947.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 936.70 | 937.37 | 942.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:30:00 | 941.30 | 937.37 | 942.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 907.80 | 917.38 | 926.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 12:00:00 | 898.80 | 911.19 | 921.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 12:30:00 | 895.90 | 908.07 | 919.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:45:00 | 900.60 | 890.79 | 897.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 897.30 | 892.91 | 897.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 896.50 | 893.63 | 897.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:15:00 | 892.50 | 899.18 | 899.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 12:15:00 | 853.86 | 875.78 | 887.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 12:15:00 | 851.10 | 875.78 | 887.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 12:15:00 | 855.57 | 875.78 | 887.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 12:15:00 | 852.43 | 875.78 | 887.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 12:15:00 | 847.88 | 875.78 | 887.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-23 13:15:00 | 808.92 | 858.97 | 879.02 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 179 — BUY (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 13:15:00 | 858.00 | 827.56 | 826.07 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 806.55 | 835.45 | 838.97 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 926.40 | 854.06 | 844.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 11:15:00 | 966.60 | 930.01 | 898.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 957.80 | 958.71 | 940.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 09:45:00 | 960.80 | 958.71 | 940.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 978.95 | 977.21 | 967.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:30:00 | 969.05 | 977.21 | 967.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 982.50 | 987.10 | 980.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:45:00 | 985.00 | 987.10 | 980.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 982.60 | 986.20 | 981.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:45:00 | 982.00 | 986.20 | 981.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 988.50 | 986.66 | 981.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:30:00 | 989.80 | 987.17 | 982.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:15:00 | 990.55 | 987.17 | 982.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 950.80 | 983.45 | 982.70 | SL hit (close<static) qty=1.00 sl=980.65 alert=retest2 |

### Cycle 182 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 954.95 | 977.75 | 980.18 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2026-02-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 13:15:00 | 992.45 | 975.47 | 973.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 1008.25 | 989.25 | 982.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 998.10 | 1000.68 | 990.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 10:00:00 | 998.10 | 1000.68 | 990.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 994.00 | 999.35 | 991.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:45:00 | 992.50 | 999.35 | 991.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 993.80 | 998.10 | 994.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 993.80 | 998.10 | 994.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 987.15 | 995.91 | 993.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:45:00 | 987.70 | 995.91 | 993.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 986.50 | 994.03 | 992.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 12:15:00 | 991.15 | 994.03 | 992.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 12:45:00 | 990.30 | 993.52 | 992.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 13:15:00 | 984.50 | 991.72 | 991.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — SELL (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 13:15:00 | 984.50 | 991.72 | 991.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 976.05 | 988.58 | 990.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 982.15 | 975.25 | 980.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 982.15 | 975.25 | 980.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 982.15 | 975.25 | 980.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 981.25 | 975.25 | 980.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 978.45 | 975.89 | 980.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:15:00 | 975.40 | 975.89 | 980.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 09:30:00 | 974.65 | 973.30 | 973.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 10:30:00 | 974.10 | 972.85 | 973.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 926.63 | 945.18 | 955.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 925.92 | 945.18 | 955.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 925.39 | 945.18 | 955.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 877.86 | 905.57 | 926.52 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 185 — BUY (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 11:15:00 | 869.35 | 859.15 | 859.10 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 851.00 | 858.97 | 859.86 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2026-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 14:15:00 | 862.80 | 860.31 | 860.25 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 10:15:00 | 854.55 | 860.66 | 860.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-17 11:15:00 | 851.15 | 858.76 | 860.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 13:15:00 | 864.50 | 859.55 | 860.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 13:15:00 | 864.50 | 859.55 | 860.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 864.50 | 859.55 | 860.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 864.50 | 859.55 | 860.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 869.90 | 861.62 | 861.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 877.70 | 866.01 | 863.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 869.45 | 879.82 | 873.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 869.45 | 879.82 | 873.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 869.45 | 879.82 | 873.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 869.45 | 879.82 | 873.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 868.60 | 877.58 | 872.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:15:00 | 867.30 | 877.58 | 872.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 869.25 | 875.91 | 872.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:30:00 | 868.25 | 875.91 | 872.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 858.65 | 870.23 | 870.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 852.40 | 866.67 | 868.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 882.95 | 868.85 | 869.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 882.95 | 868.85 | 869.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 882.95 | 868.85 | 869.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 885.30 | 868.85 | 869.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 873.20 | 869.72 | 869.70 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 11:15:00 | 866.35 | 869.05 | 869.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 863.80 | 867.69 | 868.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 824.85 | 822.06 | 836.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 824.85 | 822.06 | 836.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 835.65 | 824.78 | 836.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 835.40 | 824.78 | 836.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 837.40 | 827.30 | 836.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 840.00 | 827.30 | 836.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 838.00 | 829.44 | 836.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:45:00 | 840.15 | 829.44 | 836.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 839.00 | 831.35 | 837.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 863.00 | 831.35 | 837.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 864.80 | 843.24 | 841.89 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 832.55 | 843.06 | 844.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 818.45 | 832.30 | 837.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 842.80 | 823.72 | 829.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 842.80 | 823.72 | 829.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 842.80 | 823.72 | 829.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 842.80 | 823.72 | 829.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 843.15 | 827.61 | 831.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:45:00 | 845.45 | 827.61 | 831.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 853.95 | 837.21 | 835.12 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 822.50 | 834.75 | 835.72 | EMA200 below EMA400 |

### Cycle 197 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 859.85 | 838.51 | 836.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 09:15:00 | 892.00 | 851.69 | 843.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 12:15:00 | 1115.00 | 1115.19 | 1102.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-17 13:00:00 | 1115.00 | 1115.19 | 1102.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 1191.35 | 1207.99 | 1191.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:30:00 | 1180.35 | 1207.99 | 1191.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 1168.40 | 1200.08 | 1189.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:00:00 | 1168.40 | 1200.08 | 1189.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 1160.35 | 1192.13 | 1186.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:45:00 | 1162.30 | 1192.13 | 1186.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 1240.00 | 1245.01 | 1237.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 1234.40 | 1245.01 | 1237.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1211.35 | 1238.28 | 1234.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 1211.35 | 1238.28 | 1234.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 1207.95 | 1232.21 | 1232.42 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 1259.70 | 1232.80 | 1231.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 10:15:00 | 1290.80 | 1244.40 | 1236.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 12:15:00 | 1354.20 | 1361.01 | 1346.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 12:45:00 | 1356.80 | 1361.01 | 1346.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-06-14 10:30:00 | 950.05 | 2023-06-15 09:15:00 | 987.05 | STOP_HIT | 1.00 | -3.89% |
| SELL | retest2 | 2023-06-14 11:00:00 | 950.50 | 2023-06-15 09:15:00 | 987.05 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2023-06-14 11:30:00 | 949.90 | 2023-06-15 09:15:00 | 987.05 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2023-06-21 09:15:00 | 959.50 | 2023-06-21 09:15:00 | 981.25 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2023-07-07 14:30:00 | 959.25 | 2023-07-07 15:15:00 | 950.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2023-07-21 11:15:00 | 997.40 | 2023-07-26 09:15:00 | 1097.14 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-07-21 13:00:00 | 998.50 | 2023-07-26 09:15:00 | 1098.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-07-21 13:45:00 | 998.75 | 2023-07-26 09:15:00 | 1098.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-07-24 10:00:00 | 997.00 | 2023-07-26 09:15:00 | 1096.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-07-25 09:15:00 | 1002.00 | 2023-07-26 09:15:00 | 1102.20 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-08-08 12:00:00 | 990.70 | 2023-08-14 09:15:00 | 941.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-08-08 14:30:00 | 988.05 | 2023-08-14 09:15:00 | 941.40 | PARTIAL | 0.50 | 4.72% |
| SELL | retest2 | 2023-08-09 09:30:00 | 990.95 | 2023-08-17 11:15:00 | 938.65 | PARTIAL | 0.50 | 5.28% |
| SELL | retest2 | 2023-08-08 12:00:00 | 990.70 | 2023-08-18 09:15:00 | 970.15 | STOP_HIT | 0.50 | 2.07% |
| SELL | retest2 | 2023-08-08 14:30:00 | 988.05 | 2023-08-18 09:15:00 | 970.15 | STOP_HIT | 0.50 | 1.81% |
| SELL | retest2 | 2023-08-09 09:30:00 | 990.95 | 2023-08-18 09:15:00 | 970.15 | STOP_HIT | 0.50 | 2.10% |
| BUY | retest1 | 2023-09-08 09:30:00 | 1006.45 | 2023-09-12 09:15:00 | 994.30 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest1 | 2023-09-08 11:00:00 | 1005.50 | 2023-09-12 09:15:00 | 994.30 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest1 | 2023-09-08 12:45:00 | 1005.80 | 2023-09-12 09:15:00 | 994.30 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest1 | 2023-09-08 14:30:00 | 1005.50 | 2023-09-12 09:15:00 | 994.30 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2023-09-22 11:30:00 | 1015.40 | 2023-09-26 09:15:00 | 1004.80 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2023-09-22 13:45:00 | 1017.20 | 2023-09-26 09:15:00 | 1004.80 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2023-09-22 14:30:00 | 1018.45 | 2023-09-26 09:15:00 | 1004.80 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2023-10-04 14:45:00 | 972.20 | 2023-10-11 11:15:00 | 968.30 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2023-10-05 10:00:00 | 972.85 | 2023-10-11 11:15:00 | 968.30 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2023-10-16 13:30:00 | 943.85 | 2023-10-23 10:15:00 | 896.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-18 11:00:00 | 942.00 | 2023-10-23 10:15:00 | 894.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-18 11:45:00 | 942.80 | 2023-10-23 10:15:00 | 895.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-18 14:30:00 | 939.40 | 2023-10-23 10:15:00 | 892.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 12:30:00 | 933.70 | 2023-10-23 14:15:00 | 887.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 13:00:00 | 933.80 | 2023-10-23 14:15:00 | 887.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-16 13:30:00 | 943.85 | 2023-10-26 09:15:00 | 849.47 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-10-18 11:00:00 | 942.00 | 2023-10-26 09:15:00 | 847.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-10-18 11:45:00 | 942.80 | 2023-10-26 09:15:00 | 848.52 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-10-18 14:30:00 | 939.40 | 2023-10-26 09:15:00 | 845.46 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-10-20 12:30:00 | 933.70 | 2023-10-26 09:15:00 | 840.33 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-10-20 13:00:00 | 933.80 | 2023-10-26 09:15:00 | 840.42 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2023-11-08 10:45:00 | 939.55 | 2023-11-13 11:15:00 | 934.50 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2023-11-09 15:00:00 | 938.70 | 2023-11-17 10:15:00 | 938.00 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2023-11-10 09:15:00 | 941.55 | 2023-11-17 10:15:00 | 938.00 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2023-11-10 15:00:00 | 939.20 | 2023-11-17 10:15:00 | 938.00 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2023-11-12 18:15:00 | 951.00 | 2023-11-17 10:15:00 | 938.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2023-11-15 09:15:00 | 944.90 | 2023-11-17 10:15:00 | 938.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2023-11-15 11:30:00 | 941.60 | 2023-11-17 10:15:00 | 938.00 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2023-11-15 12:45:00 | 943.50 | 2023-11-17 10:15:00 | 938.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2023-11-22 11:15:00 | 920.40 | 2023-11-23 14:15:00 | 931.55 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2023-12-04 09:15:00 | 1110.15 | 2023-12-05 09:15:00 | 1221.17 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-18 10:30:00 | 1527.45 | 2023-12-20 13:15:00 | 1464.60 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest2 | 2023-12-18 15:15:00 | 1529.00 | 2023-12-20 13:15:00 | 1464.60 | STOP_HIT | 1.00 | -4.21% |
| BUY | retest2 | 2023-12-19 11:45:00 | 1531.00 | 2023-12-20 13:15:00 | 1464.60 | STOP_HIT | 1.00 | -4.34% |
| BUY | retest2 | 2023-12-19 15:15:00 | 1535.00 | 2023-12-20 13:15:00 | 1464.60 | STOP_HIT | 1.00 | -4.59% |
| BUY | retest2 | 2024-01-02 14:45:00 | 1604.50 | 2024-01-15 09:15:00 | 1702.00 | STOP_HIT | 1.00 | 6.08% |
| BUY | retest1 | 2024-02-19 09:15:00 | 1934.65 | 2024-02-21 13:15:00 | 1920.25 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-03-05 09:15:00 | 1962.35 | 2024-03-05 13:15:00 | 1949.25 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-03-11 12:00:00 | 1917.80 | 2024-03-13 09:15:00 | 1726.02 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-03-12 09:15:00 | 1903.00 | 2024-03-13 09:15:00 | 1712.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-03-12 09:45:00 | 1916.00 | 2024-03-13 09:15:00 | 1724.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-03-21 12:00:00 | 1841.80 | 2024-03-21 14:15:00 | 1852.50 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-03-21 13:30:00 | 1848.00 | 2024-03-21 14:15:00 | 1852.50 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2024-03-27 09:15:00 | 1884.65 | 2024-03-27 14:15:00 | 1825.85 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2024-03-27 11:30:00 | 1866.20 | 2024-03-27 14:15:00 | 1825.85 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2024-03-27 12:45:00 | 1866.25 | 2024-03-27 14:15:00 | 1825.85 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2024-04-08 09:15:00 | 1959.85 | 2024-04-10 10:15:00 | 1901.10 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2024-04-22 11:00:00 | 1784.20 | 2024-04-23 09:15:00 | 1823.50 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2024-04-22 13:00:00 | 1784.45 | 2024-04-23 09:15:00 | 1823.50 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-05-03 14:00:00 | 1754.55 | 2024-05-03 14:15:00 | 1805.00 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2024-05-06 09:15:00 | 1770.05 | 2024-05-09 12:15:00 | 1770.15 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2024-05-09 09:30:00 | 1783.50 | 2024-05-09 12:15:00 | 1770.15 | STOP_HIT | 1.00 | 0.75% |
| SELL | retest2 | 2024-05-09 10:00:00 | 1773.00 | 2024-05-09 12:15:00 | 1770.15 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2024-05-14 09:30:00 | 1693.35 | 2024-05-14 11:15:00 | 1745.00 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2024-05-21 10:15:00 | 1854.50 | 2024-05-28 14:15:00 | 1898.25 | STOP_HIT | 1.00 | 2.36% |
| BUY | retest2 | 2024-06-03 09:15:00 | 2046.45 | 2024-06-04 10:15:00 | 1687.65 | STOP_HIT | 1.00 | -17.53% |
| BUY | retest2 | 2024-06-04 09:45:00 | 1921.00 | 2024-06-04 10:15:00 | 1687.65 | STOP_HIT | 1.00 | -12.15% |
| SELL | retest2 | 2024-06-19 09:15:00 | 1781.15 | 2024-06-24 13:15:00 | 1812.30 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-06-21 09:15:00 | 1800.85 | 2024-06-24 13:15:00 | 1812.30 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-06-21 14:15:00 | 1803.60 | 2024-06-24 13:15:00 | 1812.30 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2024-06-26 12:00:00 | 1787.00 | 2024-06-27 14:15:00 | 1812.55 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-06-26 12:30:00 | 1786.35 | 2024-06-27 14:15:00 | 1812.55 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-07-05 10:30:00 | 1760.70 | 2024-07-09 10:15:00 | 1776.65 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-07-12 12:45:00 | 1739.90 | 2024-07-16 09:15:00 | 1784.50 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2024-07-12 13:45:00 | 1738.75 | 2024-07-16 09:15:00 | 1784.50 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2024-07-15 11:00:00 | 1739.50 | 2024-07-16 09:15:00 | 1784.50 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2024-07-15 13:00:00 | 1739.00 | 2024-07-16 09:15:00 | 1784.50 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2024-07-31 12:45:00 | 1856.55 | 2024-08-05 10:15:00 | 1794.25 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2024-08-01 09:15:00 | 1854.60 | 2024-08-05 10:15:00 | 1794.25 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2024-08-08 14:00:00 | 1778.65 | 2024-08-12 09:15:00 | 1689.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-08 15:00:00 | 1775.10 | 2024-08-12 09:15:00 | 1686.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-09 14:15:00 | 1778.80 | 2024-08-12 09:15:00 | 1689.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-09 15:15:00 | 1774.00 | 2024-08-12 09:15:00 | 1685.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-08 14:00:00 | 1778.65 | 2024-08-12 13:15:00 | 1786.70 | STOP_HIT | 0.50 | -0.45% |
| SELL | retest2 | 2024-08-08 15:00:00 | 1775.10 | 2024-08-12 13:15:00 | 1786.70 | STOP_HIT | 0.50 | -0.65% |
| SELL | retest2 | 2024-08-09 14:15:00 | 1778.80 | 2024-08-12 13:15:00 | 1786.70 | STOP_HIT | 0.50 | -0.44% |
| SELL | retest2 | 2024-08-09 15:15:00 | 1774.00 | 2024-08-12 13:15:00 | 1786.70 | STOP_HIT | 0.50 | -0.72% |
| SELL | retest2 | 2024-08-12 09:15:00 | 1722.10 | 2024-08-12 13:15:00 | 1786.70 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2024-08-12 11:00:00 | 1752.80 | 2024-08-12 13:15:00 | 1786.70 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-08-23 09:45:00 | 1914.80 | 2024-08-23 12:15:00 | 1899.05 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-08-27 14:00:00 | 1882.30 | 2024-09-02 10:15:00 | 1870.85 | STOP_HIT | 1.00 | 0.61% |
| SELL | retest2 | 2024-08-28 09:15:00 | 1873.00 | 2024-09-02 10:15:00 | 1870.85 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2024-09-02 10:15:00 | 1882.10 | 2024-09-02 10:15:00 | 1870.85 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest2 | 2024-09-04 12:45:00 | 1907.90 | 2024-09-06 09:15:00 | 1882.00 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-09-06 09:15:00 | 1920.40 | 2024-09-06 09:15:00 | 1882.00 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-09-10 15:00:00 | 1845.25 | 2024-09-16 09:15:00 | 1892.60 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest1 | 2024-09-18 10:30:00 | 1959.15 | 2024-09-19 09:15:00 | 1912.15 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest1 | 2024-09-18 11:00:00 | 1963.55 | 2024-09-19 09:15:00 | 1912.15 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2024-09-19 13:15:00 | 1953.90 | 2024-09-27 14:15:00 | 1973.75 | STOP_HIT | 1.00 | 1.02% |
| SELL | retest2 | 2024-10-11 11:15:00 | 1768.50 | 2024-10-11 14:15:00 | 1791.50 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-10-14 15:15:00 | 1771.00 | 2024-10-22 13:15:00 | 1682.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 09:45:00 | 1769.40 | 2024-10-22 13:15:00 | 1680.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 11:15:00 | 1768.75 | 2024-10-22 13:15:00 | 1680.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 11:45:00 | 1732.00 | 2024-10-23 09:15:00 | 1645.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-14 15:15:00 | 1771.00 | 2024-10-23 13:15:00 | 1686.00 | STOP_HIT | 0.50 | 4.80% |
| SELL | retest2 | 2024-10-15 09:45:00 | 1769.40 | 2024-10-23 13:15:00 | 1686.00 | STOP_HIT | 0.50 | 4.71% |
| SELL | retest2 | 2024-10-15 11:15:00 | 1768.75 | 2024-10-23 13:15:00 | 1686.00 | STOP_HIT | 0.50 | 4.68% |
| SELL | retest2 | 2024-10-21 11:45:00 | 1732.00 | 2024-10-23 13:15:00 | 1686.00 | STOP_HIT | 0.50 | 2.66% |
| SELL | retest2 | 2024-11-18 11:00:00 | 1470.10 | 2024-11-21 09:15:00 | 1323.09 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-18 13:15:00 | 1467.40 | 2024-11-21 09:15:00 | 1320.66 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-18 14:15:00 | 1462.95 | 2024-11-21 09:15:00 | 1316.65 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-10 11:30:00 | 1188.00 | 2024-12-12 09:15:00 | 1128.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-10 11:30:00 | 1188.00 | 2024-12-12 10:15:00 | 1244.50 | STOP_HIT | 0.50 | -4.76% |
| BUY | retest2 | 2025-01-21 11:30:00 | 1054.65 | 2025-01-21 14:15:00 | 1046.55 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-01-21 12:30:00 | 1052.00 | 2025-01-21 14:15:00 | 1046.55 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-01-21 14:00:00 | 1054.50 | 2025-01-21 14:15:00 | 1046.55 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-01-23 11:30:00 | 1022.05 | 2025-01-28 09:15:00 | 970.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 14:45:00 | 1024.30 | 2025-01-28 09:15:00 | 973.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 15:15:00 | 1022.65 | 2025-01-28 09:15:00 | 971.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 10:00:00 | 1017.75 | 2025-01-28 10:15:00 | 966.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 11:30:00 | 1022.05 | 2025-01-28 12:15:00 | 1003.00 | STOP_HIT | 0.50 | 1.86% |
| SELL | retest2 | 2025-01-23 14:45:00 | 1024.30 | 2025-01-28 12:15:00 | 1003.00 | STOP_HIT | 0.50 | 2.08% |
| SELL | retest2 | 2025-01-23 15:15:00 | 1022.65 | 2025-01-28 12:15:00 | 1003.00 | STOP_HIT | 0.50 | 1.92% |
| SELL | retest2 | 2025-01-24 10:00:00 | 1017.75 | 2025-01-28 12:15:00 | 1003.00 | STOP_HIT | 0.50 | 1.45% |
| SELL | retest2 | 2025-01-30 14:15:00 | 968.25 | 2025-01-31 14:15:00 | 1000.45 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2025-02-07 11:15:00 | 1006.55 | 2025-02-07 13:15:00 | 989.45 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-02-13 13:30:00 | 922.45 | 2025-02-14 13:15:00 | 876.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:30:00 | 922.45 | 2025-02-17 12:15:00 | 875.05 | STOP_HIT | 0.50 | 5.14% |
| BUY | retest2 | 2025-03-07 14:30:00 | 836.50 | 2025-03-11 09:15:00 | 821.40 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-03-10 09:30:00 | 839.80 | 2025-03-11 09:15:00 | 821.40 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest1 | 2025-03-20 09:15:00 | 917.05 | 2025-03-21 10:15:00 | 962.90 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-03-20 10:15:00 | 913.70 | 2025-03-21 10:15:00 | 959.39 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-03-20 12:00:00 | 912.70 | 2025-03-21 10:15:00 | 958.34 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-03-20 09:15:00 | 917.05 | 2025-03-24 10:15:00 | 951.00 | STOP_HIT | 0.50 | 3.70% |
| BUY | retest1 | 2025-03-20 10:15:00 | 913.70 | 2025-03-24 10:15:00 | 951.00 | STOP_HIT | 0.50 | 4.08% |
| BUY | retest1 | 2025-03-20 12:00:00 | 912.70 | 2025-03-24 10:15:00 | 951.00 | STOP_HIT | 0.50 | 4.20% |
| SELL | retest2 | 2025-03-26 11:30:00 | 926.55 | 2025-03-27 13:15:00 | 951.30 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-03-27 12:00:00 | 928.55 | 2025-03-27 13:15:00 | 951.30 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-04-02 13:30:00 | 933.35 | 2025-04-03 10:15:00 | 951.40 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-04-03 09:45:00 | 931.70 | 2025-04-03 10:15:00 | 951.40 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-04-23 09:15:00 | 951.80 | 2025-04-23 09:15:00 | 930.65 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-04-24 15:00:00 | 969.80 | 2025-04-25 09:15:00 | 924.50 | STOP_HIT | 1.00 | -4.67% |
| SELL | retest2 | 2025-05-02 11:15:00 | 915.45 | 2025-05-05 09:15:00 | 936.75 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-05-23 13:30:00 | 987.45 | 2025-05-26 09:15:00 | 1002.25 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-05-30 13:30:00 | 1017.70 | 2025-06-03 10:15:00 | 1001.50 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-05-30 14:45:00 | 1018.00 | 2025-06-03 10:15:00 | 1001.50 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-06-02 09:30:00 | 1017.50 | 2025-06-03 10:15:00 | 1001.50 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-06-02 12:00:00 | 1017.50 | 2025-06-03 10:15:00 | 1001.50 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-06-11 15:15:00 | 1049.00 | 2025-06-12 10:15:00 | 1033.70 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-06-26 13:45:00 | 990.00 | 2025-07-03 12:15:00 | 1011.10 | STOP_HIT | 1.00 | 2.13% |
| BUY | retest2 | 2025-06-26 14:45:00 | 991.20 | 2025-07-03 12:15:00 | 1011.10 | STOP_HIT | 1.00 | 2.01% |
| SELL | retest2 | 2025-07-29 10:15:00 | 996.80 | 2025-07-29 14:15:00 | 1014.20 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-07-29 10:45:00 | 1002.00 | 2025-07-29 14:15:00 | 1014.20 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-08-05 09:15:00 | 987.15 | 2025-08-07 09:15:00 | 937.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 09:15:00 | 987.15 | 2025-08-07 14:15:00 | 938.40 | STOP_HIT | 0.50 | 4.94% |
| BUY | retest2 | 2025-10-07 09:15:00 | 1086.90 | 2025-10-08 09:15:00 | 1046.20 | STOP_HIT | 1.00 | -3.74% |
| BUY | retest2 | 2025-10-07 13:30:00 | 1063.00 | 2025-10-08 09:15:00 | 1046.20 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-10-07 14:00:00 | 1064.30 | 2025-10-08 09:15:00 | 1046.20 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest1 | 2025-11-03 09:15:00 | 1157.30 | 2025-11-03 13:15:00 | 1122.60 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-11-10 12:45:00 | 1064.10 | 2025-11-12 09:15:00 | 1088.30 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-11-10 13:30:00 | 1061.10 | 2025-11-12 09:15:00 | 1088.30 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-12-08 09:15:00 | 1013.60 | 2025-12-11 13:15:00 | 1013.40 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2025-12-10 10:00:00 | 1013.90 | 2025-12-11 13:15:00 | 1013.40 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2025-12-10 10:30:00 | 1014.20 | 2025-12-11 13:15:00 | 1013.40 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2025-12-17 13:30:00 | 1021.00 | 2025-12-22 11:15:00 | 1022.00 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-12-17 14:00:00 | 1020.40 | 2025-12-22 11:15:00 | 1022.00 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-12-19 15:15:00 | 1018.00 | 2025-12-22 11:15:00 | 1022.00 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-12-30 11:15:00 | 1001.50 | 2025-12-31 09:15:00 | 1013.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-12-30 13:15:00 | 1001.70 | 2025-12-31 09:15:00 | 1013.60 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2026-01-20 12:00:00 | 898.80 | 2026-01-23 12:15:00 | 853.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 12:30:00 | 895.90 | 2026-01-23 12:15:00 | 851.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 10:45:00 | 900.60 | 2026-01-23 12:15:00 | 855.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 11:30:00 | 897.30 | 2026-01-23 12:15:00 | 852.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 09:15:00 | 892.50 | 2026-01-23 12:15:00 | 847.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 12:00:00 | 898.80 | 2026-01-23 13:15:00 | 808.92 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-20 12:30:00 | 895.90 | 2026-01-23 13:15:00 | 806.31 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-22 10:45:00 | 900.60 | 2026-01-23 13:15:00 | 810.54 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-22 11:30:00 | 897.30 | 2026-01-23 13:15:00 | 807.57 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-23 09:15:00 | 892.50 | 2026-01-23 13:15:00 | 803.25 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-12 12:30:00 | 989.80 | 2026-02-13 09:15:00 | 950.80 | STOP_HIT | 1.00 | -3.94% |
| BUY | retest2 | 2026-02-12 13:15:00 | 990.55 | 2026-02-13 09:15:00 | 950.80 | STOP_HIT | 1.00 | -4.01% |
| BUY | retest2 | 2026-02-19 12:15:00 | 991.15 | 2026-02-19 13:15:00 | 984.50 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2026-02-19 12:45:00 | 990.30 | 2026-02-19 13:15:00 | 984.50 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2026-02-23 11:15:00 | 975.40 | 2026-03-02 09:15:00 | 926.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 09:30:00 | 974.65 | 2026-03-02 09:15:00 | 925.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 10:30:00 | 974.10 | 2026-03-02 09:15:00 | 925.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 11:15:00 | 975.40 | 2026-03-04 09:15:00 | 877.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 09:30:00 | 974.65 | 2026-03-04 09:15:00 | 877.18 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 10:30:00 | 974.10 | 2026-03-04 09:15:00 | 876.69 | TARGET_HIT | 0.50 | 10.00% |
