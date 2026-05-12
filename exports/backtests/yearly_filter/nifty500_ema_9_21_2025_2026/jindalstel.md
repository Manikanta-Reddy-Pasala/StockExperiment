# Jindal Steel Ltd. (JINDALSTEL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1248.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 78 |
| ALERT1 | 58 |
| ALERT2 | 58 |
| ALERT2_SKIP | 29 |
| ALERT3 | 143 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 62 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 67 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 67 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 58
- **Target hits / Stop hits / Partials:** 0 / 67 / 0
- **Avg / median % per leg:** -0.95% / -1.22%
- **Sum % (uncompounded):** -63.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 43 | 8 | 18.6% | 0 | 43 | 0 | -0.84% | -36.0% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 5 | 0 | 0.29% | 1.5% |
| BUY @ 3rd Alert (retest2) | 38 | 6 | 15.8% | 0 | 38 | 0 | -0.99% | -37.4% |
| SELL (all) | 24 | 1 | 4.2% | 0 | 24 | 0 | -1.14% | -27.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 24 | 1 | 4.2% | 0 | 24 | 0 | -1.14% | -27.4% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 5 | 0 | 0.29% | 1.5% |
| retest2 (combined) | 62 | 7 | 11.3% | 0 | 62 | 0 | -1.05% | -64.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 879.25 | 867.59 | 866.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 897.30 | 873.54 | 869.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 09:15:00 | 972.60 | 974.44 | 959.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 09:30:00 | 970.15 | 974.44 | 959.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 956.40 | 969.54 | 959.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 12:00:00 | 956.40 | 969.54 | 959.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 973.20 | 970.27 | 960.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 13:15:00 | 975.10 | 970.27 | 960.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 15:00:00 | 974.40 | 971.09 | 962.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 09:15:00 | 974.05 | 970.67 | 963.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 10:15:00 | 957.15 | 962.58 | 962.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 10:15:00 | 957.15 | 962.58 | 962.99 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 15:15:00 | 966.45 | 962.92 | 962.80 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 14:15:00 | 960.00 | 963.03 | 963.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 09:15:00 | 957.15 | 961.21 | 962.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 966.00 | 958.48 | 959.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 966.00 | 958.48 | 959.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 966.00 | 958.48 | 959.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:45:00 | 964.45 | 958.48 | 959.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 964.10 | 959.61 | 960.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 12:15:00 | 960.00 | 960.20 | 960.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 12:15:00 | 964.45 | 961.05 | 960.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 964.45 | 961.05 | 960.78 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 14:15:00 | 954.95 | 959.48 | 960.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 15:15:00 | 954.00 | 958.38 | 959.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 956.00 | 948.21 | 950.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 956.00 | 948.21 | 950.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 956.00 | 948.21 | 950.84 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 12:15:00 | 967.45 | 954.56 | 953.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 14:15:00 | 970.00 | 959.18 | 955.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 10:15:00 | 956.25 | 961.72 | 958.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 10:15:00 | 956.25 | 961.72 | 958.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 956.25 | 961.72 | 958.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:00:00 | 956.25 | 961.72 | 958.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 959.05 | 961.19 | 958.16 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 14:15:00 | 948.90 | 956.31 | 956.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 15:15:00 | 947.50 | 954.55 | 955.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 09:15:00 | 946.00 | 945.07 | 948.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 10:15:00 | 949.30 | 945.91 | 948.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 949.30 | 945.91 | 948.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:00:00 | 949.30 | 945.91 | 948.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 952.55 | 947.24 | 949.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 12:00:00 | 952.55 | 947.24 | 949.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 946.90 | 947.17 | 949.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 15:15:00 | 944.20 | 946.88 | 948.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 11:00:00 | 945.25 | 945.91 | 947.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 11:30:00 | 944.35 | 946.14 | 947.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 14:15:00 | 951.40 | 948.56 | 948.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 14:15:00 | 951.40 | 948.56 | 948.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 957.55 | 950.83 | 949.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 13:15:00 | 949.40 | 952.36 | 950.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 13:15:00 | 949.40 | 952.36 | 950.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 949.40 | 952.36 | 950.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:00:00 | 949.40 | 952.36 | 950.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 951.65 | 952.22 | 950.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 956.20 | 951.99 | 950.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 14:15:00 | 956.85 | 968.04 | 968.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 956.85 | 968.04 | 968.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 15:15:00 | 953.00 | 965.04 | 967.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 924.45 | 924.09 | 935.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:45:00 | 922.85 | 924.09 | 935.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 899.55 | 892.11 | 897.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 899.55 | 892.11 | 897.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 895.30 | 892.75 | 897.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 891.00 | 892.75 | 897.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 901.85 | 895.78 | 896.90 | SL hit (close>static) qty=1.00 sl=900.20 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 906.90 | 898.01 | 897.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 12:15:00 | 913.50 | 901.11 | 899.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 15:15:00 | 920.40 | 922.18 | 914.41 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 09:15:00 | 931.70 | 922.18 | 914.41 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 922.40 | 925.77 | 920.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 15:00:00 | 922.40 | 925.77 | 920.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 918.65 | 924.34 | 920.01 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-25 15:15:00 | 918.65 | 924.34 | 920.01 | SL hit (close<ema400) qty=1.00 sl=920.01 alert=retest1 |

### Cycle 12 — SELL (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 12:15:00 | 948.95 | 957.50 | 958.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 09:15:00 | 941.70 | 948.88 | 951.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 14:15:00 | 945.45 | 944.67 | 948.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 15:00:00 | 945.45 | 944.67 | 948.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 944.85 | 944.36 | 947.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:45:00 | 939.80 | 943.69 | 945.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:15:00 | 935.20 | 939.31 | 941.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 12:15:00 | 944.40 | 939.47 | 939.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 944.40 | 939.47 | 939.30 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 13:15:00 | 935.25 | 939.21 | 939.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 14:15:00 | 931.40 | 937.65 | 938.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 11:15:00 | 935.85 | 935.61 | 937.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 11:15:00 | 935.85 | 935.61 | 937.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 935.85 | 935.61 | 937.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:45:00 | 935.00 | 935.61 | 937.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 943.45 | 937.18 | 937.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 943.45 | 937.18 | 937.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 13:15:00 | 946.40 | 939.02 | 938.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 09:15:00 | 953.90 | 945.05 | 941.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 15:15:00 | 963.70 | 964.09 | 959.92 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 09:45:00 | 966.75 | 964.37 | 960.42 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 10:15:00 | 966.05 | 964.37 | 960.42 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 987.55 | 994.67 | 988.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 987.55 | 994.67 | 988.70 | SL hit (close<ema400) qty=1.00 sl=988.70 alert=retest1 |

### Cycle 16 — SELL (started 2025-07-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 13:15:00 | 974.95 | 984.80 | 985.45 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 989.10 | 983.97 | 983.79 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 15:15:00 | 980.70 | 983.28 | 983.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 970.80 | 980.78 | 982.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 975.20 | 959.52 | 965.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 975.20 | 959.52 | 965.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 975.20 | 959.52 | 965.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:00:00 | 975.20 | 959.52 | 965.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 974.20 | 962.46 | 966.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:00:00 | 974.20 | 962.46 | 966.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 12:15:00 | 985.00 | 969.21 | 968.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 10:15:00 | 990.15 | 979.11 | 974.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 10:15:00 | 990.50 | 991.18 | 984.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-06 10:45:00 | 989.00 | 991.18 | 984.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 990.20 | 990.99 | 984.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:30:00 | 984.85 | 990.99 | 984.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 996.30 | 992.05 | 985.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:45:00 | 992.00 | 992.05 | 985.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 984.65 | 990.59 | 986.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 15:00:00 | 984.65 | 990.59 | 986.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 989.05 | 990.28 | 986.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 995.25 | 990.28 | 986.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 992.60 | 990.74 | 986.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 15:00:00 | 1000.20 | 991.07 | 988.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 10:15:00 | 981.40 | 989.26 | 988.22 | SL hit (close<static) qty=1.00 sl=983.05 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 978.30 | 987.07 | 987.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 975.50 | 983.17 | 985.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 984.90 | 982.03 | 984.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 984.90 | 982.03 | 984.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 984.90 | 982.03 | 984.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:30:00 | 987.10 | 982.03 | 984.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 984.85 | 982.59 | 984.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:30:00 | 988.15 | 982.59 | 984.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 984.90 | 983.05 | 984.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:15:00 | 988.00 | 983.05 | 984.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 988.85 | 984.21 | 984.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 13:00:00 | 988.85 | 984.21 | 984.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 989.40 | 985.25 | 985.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:00:00 | 989.40 | 985.25 | 985.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 990.75 | 986.35 | 985.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 10:15:00 | 994.65 | 988.42 | 986.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 15:15:00 | 993.25 | 993.30 | 990.31 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 09:15:00 | 1002.35 | 993.30 | 990.31 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 13:30:00 | 1004.75 | 1002.37 | 996.83 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 995.90 | 1001.08 | 996.75 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-13 14:15:00 | 995.90 | 1001.08 | 996.75 | SL hit (close<ema400) qty=1.00 sl=996.75 alert=retest1 |

### Cycle 22 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 973.70 | 990.90 | 992.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 11:15:00 | 970.80 | 986.88 | 990.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 989.75 | 981.72 | 986.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 989.75 | 981.72 | 986.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 989.75 | 981.72 | 986.07 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 15:15:00 | 994.90 | 987.76 | 987.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 1003.80 | 990.97 | 988.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 11:15:00 | 1013.40 | 1013.85 | 1007.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 12:00:00 | 1013.40 | 1013.85 | 1007.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 1012.80 | 1013.64 | 1008.41 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 999.65 | 1005.71 | 1006.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 996.45 | 1001.98 | 1004.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 1000.25 | 999.78 | 1002.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 11:00:00 | 1000.25 | 999.78 | 1002.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 998.95 | 999.61 | 1002.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:30:00 | 1000.00 | 999.61 | 1002.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 963.20 | 958.15 | 962.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 963.20 | 958.15 | 962.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 963.55 | 959.23 | 963.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 961.20 | 959.23 | 963.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 968.20 | 961.02 | 963.51 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 980.75 | 966.77 | 965.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 993.90 | 976.15 | 971.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 15:15:00 | 1039.45 | 1041.35 | 1032.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 09:15:00 | 1034.00 | 1041.35 | 1032.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1029.60 | 1039.00 | 1032.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:00:00 | 1029.60 | 1039.00 | 1032.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 1038.85 | 1038.97 | 1033.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:30:00 | 1028.45 | 1038.97 | 1033.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 1032.85 | 1037.75 | 1033.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:00:00 | 1032.85 | 1037.75 | 1033.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 1030.65 | 1036.33 | 1032.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:45:00 | 1030.15 | 1036.33 | 1032.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 1029.55 | 1034.97 | 1032.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:30:00 | 1028.65 | 1034.97 | 1032.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1036.10 | 1034.72 | 1033.08 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 11:15:00 | 1029.65 | 1034.85 | 1035.22 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 1037.90 | 1035.51 | 1035.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 10:15:00 | 1043.15 | 1037.04 | 1036.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 10:15:00 | 1040.40 | 1042.56 | 1040.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 10:15:00 | 1040.40 | 1042.56 | 1040.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1040.40 | 1042.56 | 1040.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:45:00 | 1039.30 | 1042.56 | 1040.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 1042.65 | 1042.58 | 1040.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 12:15:00 | 1047.30 | 1042.58 | 1040.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 11:45:00 | 1046.65 | 1046.02 | 1043.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 13:15:00 | 1032.25 | 1042.78 | 1042.75 | SL hit (close<static) qty=1.00 sl=1039.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 14:15:00 | 1033.50 | 1040.92 | 1041.91 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 13:15:00 | 1048.65 | 1041.68 | 1041.48 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 1039.10 | 1042.47 | 1042.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 1031.75 | 1039.68 | 1041.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 1055.25 | 1040.07 | 1040.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 12:15:00 | 1055.25 | 1040.07 | 1040.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 1055.25 | 1040.07 | 1040.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:00:00 | 1055.25 | 1040.07 | 1040.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 13:15:00 | 1063.90 | 1044.84 | 1042.56 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 11:15:00 | 1047.25 | 1049.86 | 1050.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 1036.15 | 1045.87 | 1048.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1046.85 | 1041.54 | 1045.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 1046.85 | 1041.54 | 1045.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1046.85 | 1041.54 | 1045.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 1045.00 | 1041.54 | 1045.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 1043.00 | 1041.83 | 1044.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 1041.45 | 1041.22 | 1044.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 12:00:00 | 1038.80 | 1041.22 | 1044.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 15:15:00 | 1040.00 | 1042.10 | 1044.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 1054.65 | 1044.28 | 1044.66 | SL hit (close>static) qty=1.00 sl=1049.05 alert=retest2 |

### Cycle 33 — BUY (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 10:15:00 | 1064.50 | 1048.32 | 1046.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 12:15:00 | 1067.55 | 1054.92 | 1049.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 10:15:00 | 1058.60 | 1060.01 | 1054.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-01 10:30:00 | 1057.40 | 1060.01 | 1054.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 1062.40 | 1060.49 | 1055.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:30:00 | 1057.30 | 1060.49 | 1055.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 1054.50 | 1061.11 | 1058.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:15:00 | 1051.80 | 1061.11 | 1058.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 1068.50 | 1062.59 | 1058.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 11:30:00 | 1079.00 | 1066.45 | 1061.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 10:15:00 | 1047.60 | 1065.30 | 1063.87 | SL hit (close<static) qty=1.00 sl=1049.30 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 11:15:00 | 1048.20 | 1061.88 | 1062.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 10:15:00 | 1041.80 | 1052.31 | 1056.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 1025.60 | 1024.68 | 1033.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 1025.60 | 1024.68 | 1033.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1025.60 | 1024.68 | 1033.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 1028.40 | 1024.68 | 1033.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 1031.00 | 1025.95 | 1033.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 1031.00 | 1025.95 | 1033.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 1032.70 | 1027.30 | 1033.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:30:00 | 1034.80 | 1027.30 | 1033.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 1033.00 | 1028.44 | 1033.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:45:00 | 1033.80 | 1028.44 | 1033.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 1035.10 | 1029.77 | 1033.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:00:00 | 1035.10 | 1029.77 | 1033.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 1038.60 | 1031.54 | 1034.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 1038.60 | 1031.54 | 1034.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 1008.00 | 1010.07 | 1016.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:45:00 | 1005.90 | 1008.44 | 1015.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 10:15:00 | 1019.00 | 1004.33 | 1004.71 | SL hit (close>static) qty=1.00 sl=1018.40 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 11:15:00 | 1017.00 | 1006.87 | 1005.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 13:15:00 | 1022.90 | 1012.11 | 1008.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 1007.80 | 1016.92 | 1013.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 12:15:00 | 1007.80 | 1016.92 | 1013.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 1007.80 | 1016.92 | 1013.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 1007.80 | 1016.92 | 1013.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 1009.10 | 1015.36 | 1012.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 1007.50 | 1015.36 | 1012.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 1004.40 | 1010.84 | 1011.31 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 1015.10 | 1010.34 | 1009.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 09:15:00 | 1022.70 | 1013.32 | 1011.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 12:15:00 | 1011.60 | 1013.53 | 1012.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 12:15:00 | 1011.60 | 1013.53 | 1012.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 1011.60 | 1013.53 | 1012.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 1011.60 | 1013.53 | 1012.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 1009.90 | 1012.80 | 1012.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 1006.00 | 1012.80 | 1012.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 15:15:00 | 1005.20 | 1010.48 | 1011.10 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 1020.10 | 1012.40 | 1011.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 11:15:00 | 1027.40 | 1016.27 | 1013.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 13:15:00 | 1070.80 | 1073.88 | 1058.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 14:00:00 | 1070.80 | 1073.88 | 1058.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1068.90 | 1072.26 | 1061.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:30:00 | 1064.40 | 1072.26 | 1061.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 1066.10 | 1071.03 | 1061.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:45:00 | 1058.30 | 1071.03 | 1061.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1068.90 | 1069.02 | 1064.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 1076.40 | 1068.34 | 1066.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 10:00:00 | 1076.50 | 1069.97 | 1067.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 10:45:00 | 1074.40 | 1072.00 | 1068.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 09:30:00 | 1075.90 | 1075.35 | 1072.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 1072.60 | 1074.80 | 1072.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:30:00 | 1070.90 | 1074.80 | 1072.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 1073.00 | 1074.44 | 1072.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 13:45:00 | 1077.20 | 1074.11 | 1072.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 14:15:00 | 1077.10 | 1074.11 | 1072.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 1067.20 | 1073.95 | 1072.92 | SL hit (close<static) qty=1.00 sl=1071.70 alert=retest2 |

### Cycle 40 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 1062.80 | 1071.72 | 1072.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 1052.40 | 1067.85 | 1070.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 1055.30 | 1052.94 | 1060.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:00:00 | 1055.30 | 1052.94 | 1060.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 1070.20 | 1056.02 | 1060.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 1070.20 | 1056.02 | 1060.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 1066.10 | 1058.03 | 1060.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 1073.20 | 1058.03 | 1060.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 1072.40 | 1063.97 | 1063.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 13:15:00 | 1085.90 | 1073.36 | 1069.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 13:15:00 | 1074.00 | 1077.09 | 1074.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 13:15:00 | 1074.00 | 1077.09 | 1074.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 1074.00 | 1077.09 | 1074.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:00:00 | 1074.00 | 1077.09 | 1074.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 1081.00 | 1077.87 | 1074.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 1084.00 | 1078.09 | 1075.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 12:45:00 | 1081.60 | 1083.27 | 1078.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 14:15:00 | 1083.10 | 1082.54 | 1078.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 13:00:00 | 1083.70 | 1086.81 | 1083.05 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 1084.40 | 1086.33 | 1083.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:45:00 | 1085.90 | 1086.33 | 1083.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 1076.00 | 1084.26 | 1082.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 14:45:00 | 1070.50 | 1084.26 | 1082.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 1077.10 | 1082.83 | 1082.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 1078.90 | 1082.83 | 1082.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1077.90 | 1081.84 | 1081.65 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-17 10:15:00 | 1078.20 | 1081.12 | 1081.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 10:15:00 | 1078.20 | 1081.12 | 1081.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 11:15:00 | 1075.40 | 1079.97 | 1080.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 14:15:00 | 1080.40 | 1079.04 | 1080.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 14:15:00 | 1080.40 | 1079.04 | 1080.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 1080.40 | 1079.04 | 1080.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:30:00 | 1078.00 | 1079.04 | 1080.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 1077.40 | 1078.72 | 1079.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 1065.20 | 1078.72 | 1079.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 1040.00 | 1037.46 | 1037.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 1040.00 | 1037.46 | 1037.30 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 1036.30 | 1042.81 | 1043.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 13:15:00 | 1031.30 | 1040.51 | 1041.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 12:15:00 | 1007.10 | 1006.17 | 1014.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 13:00:00 | 1007.10 | 1006.17 | 1014.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 999.90 | 1004.39 | 1010.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 12:15:00 | 996.90 | 1004.39 | 1010.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 11:30:00 | 998.00 | 999.15 | 1003.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 10:15:00 | 1010.70 | 1004.72 | 1004.86 | SL hit (close>static) qty=1.00 sl=1010.60 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 1009.20 | 1005.62 | 1005.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 13:15:00 | 1013.50 | 1007.96 | 1006.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 10:15:00 | 1024.20 | 1024.34 | 1018.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 10:45:00 | 1024.30 | 1024.34 | 1018.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1015.40 | 1026.33 | 1022.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 1015.40 | 1026.33 | 1022.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1014.00 | 1023.86 | 1021.80 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 1013.00 | 1020.29 | 1020.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 1010.20 | 1018.27 | 1019.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 991.90 | 990.99 | 998.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 991.90 | 990.99 | 998.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 991.90 | 990.99 | 998.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:45:00 | 983.50 | 989.53 | 997.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:15:00 | 986.50 | 988.97 | 996.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 13:00:00 | 986.10 | 988.39 | 995.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 1004.00 | 992.42 | 995.04 | SL hit (close>static) qty=1.00 sl=1002.50 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 1014.60 | 996.86 | 996.82 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 998.80 | 1003.76 | 1003.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 990.00 | 1000.07 | 1002.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 1003.40 | 993.77 | 996.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 1003.40 | 993.77 | 996.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1003.40 | 993.77 | 996.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:30:00 | 1006.60 | 993.77 | 996.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1004.70 | 995.95 | 997.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:30:00 | 1000.40 | 995.95 | 997.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 1001.90 | 995.48 | 996.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:00:00 | 1001.90 | 995.48 | 996.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 10:15:00 | 1006.10 | 997.61 | 997.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 12:15:00 | 1012.40 | 1001.96 | 999.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 13:15:00 | 1078.50 | 1082.06 | 1075.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 14:00:00 | 1078.50 | 1082.06 | 1075.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 1079.60 | 1081.57 | 1075.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:30:00 | 1076.30 | 1081.57 | 1075.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1081.00 | 1081.35 | 1076.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 1081.00 | 1081.35 | 1076.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 1075.00 | 1080.08 | 1076.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 1075.00 | 1080.08 | 1076.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 1073.40 | 1078.74 | 1076.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 13:00:00 | 1077.50 | 1078.49 | 1076.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:45:00 | 1078.40 | 1077.20 | 1076.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 1034.40 | 1068.13 | 1072.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 1034.40 | 1068.13 | 1072.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 12:15:00 | 1026.90 | 1049.43 | 1061.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 1024.10 | 1014.03 | 1025.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 12:15:00 | 1024.10 | 1014.03 | 1025.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 1024.10 | 1014.03 | 1025.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:00:00 | 1024.10 | 1014.03 | 1025.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 1015.80 | 1014.38 | 1024.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:45:00 | 1014.30 | 1019.29 | 1023.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 10:15:00 | 1039.10 | 1020.74 | 1021.75 | SL hit (close>static) qty=1.00 sl=1025.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 11:15:00 | 1036.00 | 1023.79 | 1023.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 1045.50 | 1035.99 | 1030.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 1041.30 | 1042.36 | 1035.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 14:00:00 | 1041.30 | 1042.36 | 1035.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1043.00 | 1042.16 | 1037.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 12:00:00 | 1056.30 | 1046.19 | 1039.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 13:15:00 | 1055.00 | 1047.37 | 1041.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 11:45:00 | 1054.90 | 1052.07 | 1046.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 10:15:00 | 1039.30 | 1045.32 | 1045.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 1039.30 | 1045.32 | 1045.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 13:15:00 | 1030.90 | 1041.21 | 1043.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 1045.80 | 1041.92 | 1043.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 15:15:00 | 1045.80 | 1041.92 | 1043.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 1045.80 | 1041.92 | 1043.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 1060.00 | 1041.92 | 1043.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 1070.40 | 1047.62 | 1045.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 1075.20 | 1064.29 | 1055.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 10:15:00 | 1071.60 | 1072.94 | 1062.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 11:00:00 | 1071.60 | 1072.94 | 1062.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 1068.10 | 1071.97 | 1062.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:45:00 | 1062.30 | 1071.97 | 1062.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 1062.60 | 1070.36 | 1064.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:45:00 | 1066.00 | 1070.36 | 1064.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 1061.00 | 1068.49 | 1064.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 1065.00 | 1068.49 | 1064.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 1069.80 | 1068.75 | 1064.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 14:45:00 | 1080.00 | 1071.51 | 1067.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 1100.10 | 1122.73 | 1123.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 1100.10 | 1122.73 | 1123.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 1090.50 | 1111.27 | 1117.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 1108.50 | 1106.40 | 1113.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:45:00 | 1108.30 | 1106.40 | 1113.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1123.10 | 1109.74 | 1114.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 1123.10 | 1109.74 | 1114.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1131.50 | 1114.09 | 1115.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 1155.20 | 1114.09 | 1115.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1140.80 | 1119.43 | 1118.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 1175.70 | 1149.37 | 1136.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1159.30 | 1162.19 | 1151.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 1159.30 | 1162.19 | 1151.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1159.30 | 1162.19 | 1151.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 11:00:00 | 1174.00 | 1164.55 | 1153.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 10:45:00 | 1169.90 | 1173.22 | 1164.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 1171.20 | 1188.54 | 1190.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 1171.20 | 1188.54 | 1190.41 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 1201.60 | 1189.11 | 1188.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 11:15:00 | 1203.90 | 1192.07 | 1190.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 09:15:00 | 1199.50 | 1202.11 | 1196.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-17 10:00:00 | 1199.50 | 1202.11 | 1196.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 1193.50 | 1200.39 | 1196.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:00:00 | 1193.50 | 1200.39 | 1196.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 1210.10 | 1202.33 | 1197.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 12:15:00 | 1212.90 | 1202.33 | 1197.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 15:15:00 | 1214.00 | 1203.69 | 1199.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 11:15:00 | 1211.20 | 1216.02 | 1216.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 11:15:00 | 1211.20 | 1216.02 | 1216.07 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 09:15:00 | 1223.10 | 1216.49 | 1216.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 13:15:00 | 1225.50 | 1220.07 | 1218.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 13:15:00 | 1260.00 | 1260.08 | 1249.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 13:45:00 | 1259.80 | 1260.08 | 1249.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1251.00 | 1258.34 | 1251.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 1251.00 | 1258.34 | 1251.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 1254.40 | 1257.55 | 1251.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:30:00 | 1250.00 | 1257.55 | 1251.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 1251.40 | 1256.32 | 1251.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:30:00 | 1251.80 | 1256.32 | 1251.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 1254.20 | 1255.90 | 1251.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 13:15:00 | 1257.70 | 1255.90 | 1251.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 13:45:00 | 1259.50 | 1256.86 | 1252.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 14:15:00 | 1245.10 | 1254.51 | 1251.77 | SL hit (close<static) qty=1.00 sl=1250.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 1241.30 | 1249.02 | 1249.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 14:15:00 | 1238.60 | 1244.20 | 1246.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 10:15:00 | 1188.30 | 1186.85 | 1206.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:30:00 | 1192.70 | 1186.85 | 1206.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 1186.00 | 1184.45 | 1193.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:30:00 | 1190.70 | 1184.45 | 1193.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1180.10 | 1157.39 | 1168.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:00:00 | 1180.10 | 1157.39 | 1168.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 1186.40 | 1163.19 | 1170.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 11:00:00 | 1186.40 | 1163.19 | 1170.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 1190.00 | 1175.12 | 1174.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 1193.50 | 1182.48 | 1178.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 1185.40 | 1188.54 | 1183.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 15:15:00 | 1185.00 | 1188.54 | 1183.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1185.00 | 1187.83 | 1183.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1179.20 | 1187.83 | 1183.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1184.40 | 1187.15 | 1183.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:15:00 | 1190.20 | 1187.15 | 1183.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 11:15:00 | 1160.90 | 1189.71 | 1191.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 1160.90 | 1189.71 | 1191.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 1154.50 | 1182.67 | 1187.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1150.40 | 1141.71 | 1155.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:00:00 | 1150.40 | 1141.71 | 1155.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1155.30 | 1144.43 | 1155.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 1155.30 | 1144.43 | 1155.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 1150.70 | 1145.68 | 1155.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 1145.20 | 1145.68 | 1155.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 14:15:00 | 1157.30 | 1150.44 | 1155.25 | SL hit (close>static) qty=1.00 sl=1157.10 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 1178.50 | 1158.52 | 1157.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 1182.10 | 1163.24 | 1159.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1146.80 | 1164.28 | 1161.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1146.80 | 1164.28 | 1161.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1146.80 | 1164.28 | 1161.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 1147.30 | 1164.28 | 1161.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 1145.20 | 1160.46 | 1159.91 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 1150.50 | 1158.47 | 1159.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 1138.70 | 1152.33 | 1156.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1178.90 | 1154.79 | 1155.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1178.90 | 1154.79 | 1155.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1178.90 | 1154.79 | 1155.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 1178.90 | 1154.79 | 1155.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 1181.40 | 1160.11 | 1158.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 11:15:00 | 1189.90 | 1166.07 | 1161.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 1128.40 | 1168.25 | 1165.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 09:15:00 | 1128.40 | 1168.25 | 1165.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 1128.40 | 1168.25 | 1165.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 1128.40 | 1168.25 | 1165.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 1115.40 | 1157.68 | 1161.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 1108.70 | 1147.88 | 1156.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1127.40 | 1122.13 | 1134.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:30:00 | 1125.60 | 1122.13 | 1134.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1127.20 | 1123.14 | 1133.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 1130.00 | 1123.14 | 1133.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1144.00 | 1127.23 | 1132.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 11:30:00 | 1131.30 | 1132.37 | 1134.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 14:15:00 | 1143.20 | 1136.94 | 1136.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 14:15:00 | 1143.20 | 1136.94 | 1136.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 09:15:00 | 1147.20 | 1139.53 | 1137.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 13:15:00 | 1144.00 | 1144.07 | 1140.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 14:00:00 | 1144.00 | 1144.07 | 1140.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 1130.20 | 1141.30 | 1139.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 1130.20 | 1141.30 | 1139.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 1129.80 | 1139.00 | 1138.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 1120.50 | 1139.00 | 1138.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 1117.60 | 1134.72 | 1136.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 1110.50 | 1121.20 | 1128.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1152.60 | 1125.69 | 1129.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1152.60 | 1125.69 | 1129.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1152.60 | 1125.69 | 1129.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 1152.60 | 1125.69 | 1129.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1149.30 | 1130.41 | 1131.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:15:00 | 1146.20 | 1130.41 | 1131.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 11:15:00 | 1149.80 | 1134.29 | 1132.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 1149.80 | 1134.29 | 1132.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 1154.10 | 1138.25 | 1134.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-01 14:15:00 | 1136.50 | 1138.74 | 1135.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 14:15:00 | 1136.50 | 1138.74 | 1135.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 14:15:00 | 1136.50 | 1138.74 | 1135.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 14:45:00 | 1136.10 | 1138.74 | 1135.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 15:15:00 | 1135.00 | 1137.99 | 1135.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 09:15:00 | 1114.50 | 1137.99 | 1135.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 1105.50 | 1131.49 | 1132.83 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 1141.90 | 1133.24 | 1132.25 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 10:15:00 | 1103.40 | 1126.58 | 1129.35 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 09:15:00 | 1145.20 | 1132.69 | 1131.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 12:15:00 | 1149.30 | 1139.62 | 1134.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 12:15:00 | 1201.50 | 1203.88 | 1186.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 12:45:00 | 1202.80 | 1203.88 | 1186.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1191.30 | 1206.04 | 1198.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 1216.90 | 1208.81 | 1200.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 1261.30 | 1277.52 | 1278.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 1261.30 | 1277.52 | 1278.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 13:15:00 | 1257.90 | 1269.37 | 1273.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 11:15:00 | 1264.50 | 1258.24 | 1265.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 11:15:00 | 1264.50 | 1258.24 | 1265.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 1264.50 | 1258.24 | 1265.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:00:00 | 1264.50 | 1258.24 | 1265.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 1270.90 | 1260.77 | 1266.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 13:00:00 | 1270.90 | 1260.77 | 1266.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 1263.90 | 1261.40 | 1265.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 14:45:00 | 1258.70 | 1259.92 | 1264.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 1281.50 | 1263.80 | 1265.72 | SL hit (close>static) qty=1.00 sl=1273.10 alert=retest2 |

### Cycle 75 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 1277.40 | 1268.95 | 1267.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 1284.50 | 1272.06 | 1269.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 11:15:00 | 1274.90 | 1277.80 | 1274.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 11:15:00 | 1274.90 | 1277.80 | 1274.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 1274.90 | 1277.80 | 1274.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:00:00 | 1274.90 | 1277.80 | 1274.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 1269.40 | 1276.12 | 1273.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:45:00 | 1269.20 | 1276.12 | 1273.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 1262.80 | 1273.45 | 1272.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:00:00 | 1262.80 | 1273.45 | 1272.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 14:15:00 | 1264.00 | 1271.56 | 1271.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 09:15:00 | 1239.80 | 1264.16 | 1268.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 14:15:00 | 1220.80 | 1220.22 | 1235.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 15:00:00 | 1220.80 | 1220.22 | 1235.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1263.00 | 1230.32 | 1237.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:45:00 | 1266.60 | 1230.32 | 1237.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 1260.00 | 1236.26 | 1239.29 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 1264.60 | 1245.89 | 1243.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 1278.40 | 1259.86 | 1251.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 11:15:00 | 1261.30 | 1262.44 | 1254.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 12:00:00 | 1261.30 | 1262.44 | 1254.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 1255.70 | 1260.97 | 1257.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:00:00 | 1255.70 | 1260.97 | 1257.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 1263.50 | 1261.48 | 1257.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 14:15:00 | 1267.00 | 1260.70 | 1257.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 09:15:00 | 1269.80 | 1261.66 | 1258.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 10:30:00 | 1269.40 | 1264.96 | 1261.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 11:45:00 | 1266.50 | 1265.73 | 1261.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 1261.60 | 1264.45 | 1261.80 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-05-08 09:15:00 | 1241.60 | 1258.39 | 1259.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 09:15:00 | 1241.60 | 1258.39 | 1259.56 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-19 13:15:00 | 975.10 | 2025-05-21 10:15:00 | 957.15 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-05-19 15:00:00 | 974.40 | 2025-05-21 10:15:00 | 957.15 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-05-20 09:15:00 | 974.05 | 2025-05-21 10:15:00 | 957.15 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-05-26 12:15:00 | 960.00 | 2025-05-26 12:15:00 | 964.45 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-06-03 15:15:00 | 944.20 | 2025-06-04 14:15:00 | 951.40 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-06-04 11:00:00 | 945.25 | 2025-06-04 14:15:00 | 951.40 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-06-04 11:30:00 | 944.35 | 2025-06-04 14:15:00 | 951.40 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-06-06 09:15:00 | 956.20 | 2025-06-11 14:15:00 | 956.85 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2025-06-20 12:15:00 | 891.00 | 2025-06-23 10:15:00 | 901.85 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest1 | 2025-06-25 09:15:00 | 931.70 | 2025-06-25 15:15:00 | 918.65 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-06-26 09:15:00 | 929.60 | 2025-07-04 12:15:00 | 948.95 | STOP_HIT | 1.00 | 2.08% |
| BUY | retest2 | 2025-06-26 11:45:00 | 926.00 | 2025-07-04 12:15:00 | 948.95 | STOP_HIT | 1.00 | 2.48% |
| SELL | retest2 | 2025-07-11 09:45:00 | 939.80 | 2025-07-15 12:15:00 | 944.40 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-07-14 11:15:00 | 935.20 | 2025-07-15 12:15:00 | 944.40 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest1 | 2025-07-23 09:45:00 | 966.75 | 2025-07-28 09:15:00 | 987.55 | STOP_HIT | 1.00 | 2.15% |
| BUY | retest1 | 2025-07-23 10:15:00 | 966.05 | 2025-07-28 09:15:00 | 987.55 | STOP_HIT | 1.00 | 2.23% |
| BUY | retest2 | 2025-08-07 15:00:00 | 1000.20 | 2025-08-08 10:15:00 | 981.40 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest1 | 2025-08-13 09:15:00 | 1002.35 | 2025-08-13 14:15:00 | 995.90 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest1 | 2025-08-13 13:30:00 | 1004.75 | 2025-08-13 14:15:00 | 995.90 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-09-16 12:15:00 | 1047.30 | 2025-09-17 13:15:00 | 1032.25 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-09-17 11:45:00 | 1046.65 | 2025-09-17 13:15:00 | 1032.25 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-09-29 11:30:00 | 1041.45 | 2025-09-30 09:15:00 | 1054.65 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-09-29 12:00:00 | 1038.80 | 2025-09-30 09:15:00 | 1054.65 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-09-29 15:15:00 | 1040.00 | 2025-09-30 09:15:00 | 1054.65 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-10-03 11:30:00 | 1079.00 | 2025-10-06 10:15:00 | 1047.60 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2025-10-14 10:45:00 | 1005.90 | 2025-10-16 10:15:00 | 1019.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-11-03 09:15:00 | 1076.40 | 2025-11-06 09:15:00 | 1067.20 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-11-03 10:00:00 | 1076.50 | 2025-11-06 09:15:00 | 1067.20 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-11-03 10:45:00 | 1074.40 | 2025-11-06 10:15:00 | 1062.80 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-11-04 09:30:00 | 1075.90 | 2025-11-06 10:15:00 | 1062.80 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-11-04 13:45:00 | 1077.20 | 2025-11-06 10:15:00 | 1062.80 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-11-04 14:15:00 | 1077.10 | 2025-11-06 10:15:00 | 1062.80 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-11-13 09:15:00 | 1084.00 | 2025-11-17 10:15:00 | 1078.20 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-11-13 12:45:00 | 1081.60 | 2025-11-17 10:15:00 | 1078.20 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-11-13 14:15:00 | 1083.10 | 2025-11-17 10:15:00 | 1078.20 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-11-14 13:00:00 | 1083.70 | 2025-11-17 10:15:00 | 1078.20 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-11-18 09:15:00 | 1065.20 | 2025-11-27 09:15:00 | 1040.00 | STOP_HIT | 1.00 | 2.37% |
| SELL | retest2 | 2025-12-08 12:15:00 | 996.90 | 2025-12-10 10:15:00 | 1010.70 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-12-09 11:30:00 | 998.00 | 2025-12-10 10:15:00 | 1010.70 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-12-19 10:45:00 | 983.50 | 2025-12-22 09:15:00 | 1004.00 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-12-19 12:15:00 | 986.50 | 2025-12-22 09:15:00 | 1004.00 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-12-19 13:00:00 | 986.10 | 2025-12-22 09:15:00 | 1004.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2026-01-07 13:00:00 | 1077.50 | 2026-01-08 09:15:00 | 1034.40 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest2 | 2026-01-07 14:45:00 | 1078.40 | 2026-01-08 09:15:00 | 1034.40 | STOP_HIT | 1.00 | -4.08% |
| SELL | retest2 | 2026-01-13 12:45:00 | 1014.30 | 2026-01-14 10:15:00 | 1039.10 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2026-01-19 12:00:00 | 1056.30 | 2026-01-21 10:15:00 | 1039.30 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2026-01-19 13:15:00 | 1055.00 | 2026-01-21 10:15:00 | 1039.30 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2026-01-20 11:45:00 | 1054.90 | 2026-01-21 10:15:00 | 1039.30 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2026-01-27 14:45:00 | 1080.00 | 2026-02-01 14:15:00 | 1100.10 | STOP_HIT | 1.00 | 1.86% |
| BUY | retest2 | 2026-02-05 11:00:00 | 1174.00 | 2026-02-13 10:15:00 | 1171.20 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2026-02-06 10:45:00 | 1169.90 | 2026-02-13 10:15:00 | 1171.20 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2026-02-17 12:15:00 | 1212.90 | 2026-02-23 11:15:00 | 1211.20 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2026-02-17 15:15:00 | 1214.00 | 2026-02-23 11:15:00 | 1211.20 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2026-02-27 13:15:00 | 1257.70 | 2026-02-27 14:15:00 | 1245.10 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2026-02-27 13:45:00 | 1259.50 | 2026-02-27 14:15:00 | 1245.10 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-03-12 10:15:00 | 1190.20 | 2026-03-13 11:15:00 | 1160.90 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2026-03-17 12:15:00 | 1145.20 | 2026-03-17 14:15:00 | 1157.30 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-03-18 09:15:00 | 1143.90 | 2026-03-18 11:15:00 | 1166.80 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2026-03-18 09:45:00 | 1144.80 | 2026-03-18 11:15:00 | 1166.80 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-03-25 11:30:00 | 1131.30 | 2026-03-25 14:15:00 | 1143.20 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2026-04-01 11:15:00 | 1146.20 | 2026-04-01 11:15:00 | 1149.80 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2026-04-13 10:45:00 | 1216.90 | 2026-04-23 10:15:00 | 1261.30 | STOP_HIT | 1.00 | 3.65% |
| SELL | retest2 | 2026-04-24 14:45:00 | 1258.70 | 2026-04-27 09:15:00 | 1281.50 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2026-05-06 14:15:00 | 1267.00 | 2026-05-08 09:15:00 | 1241.60 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2026-05-07 09:15:00 | 1269.80 | 2026-05-08 09:15:00 | 1241.60 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2026-05-07 10:30:00 | 1269.40 | 2026-05-08 09:15:00 | 1241.60 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2026-05-07 11:45:00 | 1266.50 | 2026-05-08 09:15:00 | 1241.60 | STOP_HIT | 1.00 | -1.97% |
