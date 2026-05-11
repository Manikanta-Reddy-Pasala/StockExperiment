# Apollo Hospitals Enterprise Ltd. (APOLLOHOSP)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 8100.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 73 |
| ALERT1 | 46 |
| ALERT2 | 47 |
| ALERT2_SKIP | 22 |
| ALERT3 | 123 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 57 |
| PARTIAL | 7 |
| TARGET_HIT | 0 |
| STOP_HIT | 60 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 67 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 45
- **Target hits / Stop hits / Partials:** 0 / 60 / 7
- **Avg / median % per leg:** 0.40% / -0.69%
- **Sum % (uncompounded):** 26.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 36 | 3 | 8.3% | 0 | 36 | 0 | -0.67% | -24.2% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.36% | -7.1% |
| BUY @ 3rd Alert (retest2) | 33 | 3 | 9.1% | 0 | 33 | 0 | -0.52% | -17.1% |
| SELL (all) | 31 | 19 | 61.3% | 0 | 24 | 7 | 1.65% | 51.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 31 | 19 | 61.3% | 0 | 24 | 7 | 1.65% | 51.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.36% | -7.1% |
| retest2 (combined) | 64 | 22 | 34.4% | 0 | 57 | 7 | 0.53% | 33.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 14:15:00 | 6927.00 | 6876.31 | 6870.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 6995.50 | 6908.58 | 6886.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 6929.50 | 6930.00 | 6903.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 6929.50 | 6930.00 | 6903.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 6926.00 | 6925.84 | 6907.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 09:15:00 | 6927.00 | 6925.84 | 6907.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 6967.50 | 6934.17 | 6913.37 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 10:15:00 | 6876.00 | 6908.77 | 6910.65 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 12:15:00 | 6940.00 | 6911.46 | 6911.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 13:15:00 | 7013.50 | 6931.86 | 6920.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 15:15:00 | 7000.00 | 7013.02 | 6984.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 09:15:00 | 7011.50 | 7013.02 | 6984.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 7051.50 | 7020.72 | 6991.03 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 09:15:00 | 6930.50 | 6980.27 | 6983.19 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 14:15:00 | 6979.50 | 6975.06 | 6974.61 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 6933.50 | 6968.50 | 6971.81 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 7029.00 | 6971.17 | 6968.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 10:15:00 | 7043.00 | 6985.54 | 6975.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 7054.00 | 7079.29 | 7053.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 7054.00 | 7079.29 | 7053.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 7054.00 | 7079.29 | 7053.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:45:00 | 7037.50 | 7079.29 | 7053.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 7069.50 | 7077.33 | 7054.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 7087.00 | 7077.33 | 7054.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 14:45:00 | 7073.50 | 7077.37 | 7062.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 7000.50 | 7057.62 | 7055.63 | SL hit (close<static) qty=1.00 sl=7043.50 alert=retest2 |

### Cycle 8 — SELL (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 10:15:00 | 6986.50 | 7043.39 | 7049.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 14:15:00 | 6956.00 | 6998.52 | 7023.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 6947.00 | 6938.61 | 6968.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 6947.00 | 6938.61 | 6968.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 6947.00 | 6938.61 | 6968.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 13:00:00 | 6931.50 | 6942.39 | 6963.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 14:00:00 | 6932.00 | 6940.31 | 6960.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 14:30:00 | 6928.00 | 6928.05 | 6953.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 7065.00 | 6948.71 | 6957.67 | SL hit (close>static) qty=1.00 sl=6988.50 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 13:15:00 | 6880.00 | 6870.30 | 6869.28 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 09:15:00 | 6841.00 | 6868.03 | 6868.72 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 6880.00 | 6869.28 | 6869.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 13:15:00 | 6930.00 | 6881.42 | 6874.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 09:15:00 | 6897.50 | 6901.77 | 6887.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 09:45:00 | 6915.00 | 6901.77 | 6887.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 6916.50 | 6904.71 | 6889.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 14:15:00 | 6921.50 | 6907.09 | 6894.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 13:45:00 | 6924.50 | 6934.34 | 6918.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 09:15:00 | 6878.50 | 6918.15 | 6914.68 | SL hit (close<static) qty=1.00 sl=6886.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 6899.50 | 6912.06 | 6912.73 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 14:15:00 | 6926.50 | 6914.95 | 6913.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 15:15:00 | 6932.00 | 6918.36 | 6915.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 13:15:00 | 6985.50 | 6996.54 | 6975.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 14:00:00 | 6985.50 | 6996.54 | 6975.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 7037.50 | 7066.60 | 7039.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:45:00 | 7039.00 | 7066.60 | 7039.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 7049.00 | 7063.08 | 7040.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:30:00 | 7045.50 | 7063.08 | 7040.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 7044.50 | 7059.37 | 7040.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 14:15:00 | 7036.00 | 7059.37 | 7040.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 7001.50 | 7047.79 | 7036.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 15:00:00 | 7001.50 | 7047.79 | 7036.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 7019.00 | 7042.03 | 7035.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 7011.00 | 7042.03 | 7035.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 10:15:00 | 6990.00 | 7028.58 | 7030.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 11:15:00 | 6946.50 | 7012.17 | 7022.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 7003.50 | 6975.57 | 6996.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 7003.50 | 6975.57 | 6996.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 7003.50 | 6975.57 | 6996.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:45:00 | 7009.50 | 6975.57 | 6996.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 6991.00 | 6978.66 | 6995.99 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 11:15:00 | 7052.00 | 7010.20 | 7004.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 13:15:00 | 7052.50 | 7022.07 | 7011.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 7003.00 | 7029.08 | 7018.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 7003.00 | 7029.08 | 7018.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 7003.00 | 7029.08 | 7018.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:30:00 | 7014.50 | 7029.08 | 7018.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 7034.00 | 7030.07 | 7019.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:15:00 | 7042.00 | 7030.07 | 7019.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 10:00:00 | 7046.00 | 7026.63 | 7021.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 15:15:00 | 7000.00 | 7021.19 | 7022.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 15:15:00 | 7000.00 | 7021.19 | 7022.70 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 13:15:00 | 7054.50 | 7024.85 | 7023.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 7057.00 | 7034.94 | 7028.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 11:15:00 | 7031.00 | 7036.56 | 7030.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 11:15:00 | 7031.00 | 7036.56 | 7030.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 7031.00 | 7036.56 | 7030.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:00:00 | 7031.00 | 7036.56 | 7030.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 7041.00 | 7037.45 | 7031.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 13:30:00 | 7058.00 | 7048.46 | 7037.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 11:15:00 | 7498.00 | 7548.02 | 7554.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 11:15:00 | 7498.00 | 7548.02 | 7554.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 12:15:00 | 7481.00 | 7534.61 | 7547.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 11:15:00 | 7236.00 | 7235.01 | 7306.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 12:00:00 | 7236.00 | 7235.01 | 7306.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 7294.50 | 7248.60 | 7277.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:00:00 | 7294.50 | 7248.60 | 7277.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 7314.00 | 7261.68 | 7281.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 7314.00 | 7261.68 | 7281.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 7356.00 | 7280.54 | 7287.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:45:00 | 7362.00 | 7280.54 | 7287.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 7366.00 | 7297.64 | 7295.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 7396.50 | 7325.79 | 7308.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 13:15:00 | 7343.00 | 7353.84 | 7330.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 13:15:00 | 7343.00 | 7353.84 | 7330.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 7343.00 | 7353.84 | 7330.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:30:00 | 7399.50 | 7359.76 | 7338.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 10:30:00 | 7397.00 | 7368.01 | 7344.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 11:00:00 | 7401.00 | 7368.01 | 7344.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 7276.50 | 7352.10 | 7348.71 | SL hit (close<static) qty=1.00 sl=7310.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 7304.00 | 7342.48 | 7344.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 13:15:00 | 7253.00 | 7291.63 | 7312.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 7319.00 | 7266.89 | 7279.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 7319.00 | 7266.89 | 7279.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 7319.00 | 7266.89 | 7279.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:45:00 | 7322.50 | 7266.89 | 7279.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 7317.50 | 7277.01 | 7283.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:30:00 | 7327.00 | 7277.01 | 7283.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 11:15:00 | 7329.00 | 7287.41 | 7287.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 13:15:00 | 7344.00 | 7303.06 | 7294.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 14:15:00 | 7371.50 | 7393.78 | 7358.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 14:45:00 | 7372.00 | 7393.78 | 7358.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 7366.00 | 7388.22 | 7359.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 7413.50 | 7388.22 | 7359.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 7417.00 | 7393.98 | 7364.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 11:15:00 | 7435.00 | 7400.88 | 7370.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 12:30:00 | 7435.00 | 7411.17 | 7380.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 13:15:00 | 7448.00 | 7411.17 | 7380.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 09:45:00 | 7443.50 | 7439.18 | 7405.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 7367.00 | 7424.24 | 7404.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 12:00:00 | 7367.00 | 7424.24 | 7404.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 7340.00 | 7407.39 | 7398.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-28 12:15:00 | 7340.00 | 7407.39 | 7398.51 | SL hit (close<static) qty=1.00 sl=7343.50 alert=retest2 |

### Cycle 22 — SELL (started 2025-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 14:15:00 | 7357.00 | 7387.57 | 7390.44 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 10:15:00 | 7407.50 | 7392.50 | 7391.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 12:15:00 | 7427.00 | 7402.84 | 7396.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 10:15:00 | 7432.00 | 7432.58 | 7416.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 10:15:00 | 7432.00 | 7432.58 | 7416.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 7432.00 | 7432.58 | 7416.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:45:00 | 7427.50 | 7432.58 | 7416.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 7417.00 | 7429.85 | 7418.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 12:30:00 | 7414.50 | 7429.85 | 7418.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 7450.00 | 7433.88 | 7421.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:30:00 | 7423.50 | 7433.88 | 7421.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 7448.00 | 7439.52 | 7426.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 7424.00 | 7439.52 | 7426.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 7396.00 | 7430.82 | 7423.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:00:00 | 7445.00 | 7427.43 | 7423.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 12:15:00 | 7389.00 | 7426.40 | 7429.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 12:15:00 | 7389.00 | 7426.40 | 7429.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 7334.00 | 7399.30 | 7416.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 7169.50 | 7144.14 | 7197.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 7169.50 | 7144.14 | 7197.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 7133.50 | 7149.11 | 7190.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:30:00 | 7121.50 | 7143.49 | 7184.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 12:00:00 | 7118.50 | 7138.49 | 7178.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 13:15:00 | 7215.00 | 7134.07 | 7146.53 | SL hit (close>static) qty=1.00 sl=7198.50 alert=retest2 |

### Cycle 25 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 7259.50 | 7159.16 | 7156.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 7680.00 | 7321.51 | 7247.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 14:15:00 | 7819.50 | 7836.24 | 7731.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 15:00:00 | 7819.50 | 7836.24 | 7731.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 7922.00 | 7913.07 | 7879.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 11:15:00 | 7927.00 | 7913.26 | 7882.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 14:30:00 | 7931.00 | 7921.89 | 7897.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 15:15:00 | 7930.00 | 7921.89 | 7897.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 10:15:00 | 7849.50 | 7899.75 | 7893.23 | SL hit (close<static) qty=1.00 sl=7852.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 12:15:00 | 7854.00 | 7885.44 | 7887.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 15:15:00 | 7848.50 | 7868.04 | 7878.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 7791.00 | 7786.13 | 7819.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 09:15:00 | 7791.00 | 7786.13 | 7819.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 7791.00 | 7786.13 | 7819.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:00:00 | 7791.00 | 7786.13 | 7819.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 7822.00 | 7793.31 | 7819.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:00:00 | 7822.00 | 7793.31 | 7819.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 7798.50 | 7794.35 | 7817.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:30:00 | 7785.50 | 7789.38 | 7813.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 10:15:00 | 7739.50 | 7695.68 | 7691.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 7739.50 | 7695.68 | 7691.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 13:15:00 | 7762.00 | 7717.20 | 7703.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 11:15:00 | 7866.00 | 7867.73 | 7816.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-05 11:45:00 | 7872.00 | 7867.73 | 7816.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 7846.00 | 7860.23 | 7821.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:30:00 | 7824.00 | 7860.23 | 7821.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 7800.50 | 7848.28 | 7819.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:00:00 | 7800.50 | 7848.28 | 7819.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 7829.50 | 7844.53 | 7820.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 7811.50 | 7844.53 | 7820.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 7797.50 | 7835.12 | 7818.54 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 12:15:00 | 7786.50 | 7806.47 | 7807.84 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 11:15:00 | 7838.00 | 7810.04 | 7808.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 7859.00 | 7826.80 | 7817.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 11:15:00 | 7879.50 | 7885.07 | 7861.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 11:15:00 | 7879.50 | 7885.07 | 7861.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 7879.50 | 7885.07 | 7861.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:45:00 | 7866.00 | 7885.07 | 7861.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 7869.50 | 7879.58 | 7867.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:45:00 | 7862.00 | 7879.58 | 7867.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 7842.50 | 7872.16 | 7864.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:00:00 | 7842.50 | 7872.16 | 7864.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 7870.50 | 7871.83 | 7865.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 12:15:00 | 7881.00 | 7871.83 | 7865.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 14:15:00 | 7880.00 | 7874.65 | 7867.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 7827.00 | 7863.54 | 7864.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 09:15:00 | 7827.00 | 7863.54 | 7864.43 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 7899.00 | 7847.54 | 7844.72 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 11:15:00 | 7834.00 | 7853.04 | 7853.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 7825.00 | 7843.07 | 7848.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 7702.50 | 7700.63 | 7734.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 10:00:00 | 7702.50 | 7700.63 | 7734.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 7450.50 | 7436.44 | 7463.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 7416.50 | 7436.44 | 7463.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 7422.00 | 7433.55 | 7459.73 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 7591.50 | 7473.80 | 7467.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 11:15:00 | 7740.00 | 7695.37 | 7661.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 14:15:00 | 7697.50 | 7701.58 | 7673.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-09 15:00:00 | 7697.50 | 7701.58 | 7673.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 7682.00 | 7697.05 | 7678.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:15:00 | 7698.50 | 7697.05 | 7678.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 7718.00 | 7701.24 | 7681.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 7721.50 | 7691.17 | 7682.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 11:15:00 | 7652.50 | 7682.19 | 7680.61 | SL hit (close<static) qty=1.00 sl=7670.50 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 7665.00 | 7678.75 | 7679.20 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 09:15:00 | 7718.50 | 7685.26 | 7681.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 7859.50 | 7813.34 | 7770.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 14:15:00 | 7902.00 | 7906.84 | 7859.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 14:30:00 | 7918.00 | 7906.84 | 7859.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 8071.00 | 8011.90 | 7958.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 10:15:00 | 8095.50 | 8011.90 | 7958.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 09:15:00 | 7900.00 | 7978.33 | 7969.79 | SL hit (close<static) qty=1.00 sl=7920.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 7872.00 | 7957.06 | 7960.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 7853.50 | 7911.95 | 7937.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 7883.50 | 7883.36 | 7915.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 09:45:00 | 7891.50 | 7883.36 | 7915.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 7848.50 | 7860.71 | 7886.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 7900.00 | 7860.71 | 7886.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 7841.50 | 7850.73 | 7875.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:45:00 | 7866.50 | 7850.73 | 7875.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 7878.00 | 7856.54 | 7873.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 7878.00 | 7856.54 | 7873.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 7861.50 | 7857.54 | 7872.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 7864.00 | 7857.54 | 7872.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 7863.50 | 7858.73 | 7871.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:15:00 | 7811.50 | 7859.68 | 7867.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 15:15:00 | 7835.00 | 7787.03 | 7782.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 15:15:00 | 7835.00 | 7787.03 | 7782.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 09:15:00 | 7870.50 | 7803.72 | 7790.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 13:15:00 | 7820.50 | 7825.12 | 7806.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 14:00:00 | 7820.50 | 7825.12 | 7806.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 7824.00 | 7824.89 | 7808.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:45:00 | 7784.00 | 7824.89 | 7808.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 7790.00 | 7817.91 | 7806.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:15:00 | 7733.50 | 7817.91 | 7806.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 7768.00 | 7807.93 | 7803.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 11:15:00 | 7799.50 | 7804.95 | 7802.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 14:00:00 | 7815.50 | 7812.19 | 7806.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 15:15:00 | 7783.00 | 7801.68 | 7802.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 15:15:00 | 7783.00 | 7801.68 | 7802.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 7732.50 | 7787.85 | 7796.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 15:15:00 | 7520.00 | 7505.05 | 7566.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 09:15:00 | 7467.00 | 7505.05 | 7566.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 7454.00 | 7431.52 | 7457.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:45:00 | 7488.00 | 7431.52 | 7457.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 7446.00 | 7434.42 | 7456.15 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 15:15:00 | 7492.50 | 7468.16 | 7466.17 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 7413.50 | 7457.23 | 7461.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 7382.00 | 7427.40 | 7443.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 10:15:00 | 7408.00 | 7404.32 | 7427.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 11:00:00 | 7408.00 | 7404.32 | 7427.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 7456.00 | 7417.88 | 7429.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:00:00 | 7456.00 | 7417.88 | 7429.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 7466.00 | 7427.51 | 7433.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:30:00 | 7462.50 | 7427.51 | 7433.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 15:15:00 | 7457.00 | 7438.76 | 7437.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 10:15:00 | 7471.00 | 7447.89 | 7442.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 14:15:00 | 7426.50 | 7451.31 | 7446.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 14:15:00 | 7426.50 | 7451.31 | 7446.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 7426.50 | 7451.31 | 7446.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 7426.50 | 7451.31 | 7446.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 7440.00 | 7449.05 | 7445.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 7417.00 | 7449.05 | 7445.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 7410.00 | 7441.24 | 7442.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 7385.00 | 7413.67 | 7426.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 7416.00 | 7414.13 | 7425.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 10:00:00 | 7416.00 | 7414.13 | 7425.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 7405.00 | 7382.61 | 7401.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 14:45:00 | 7348.00 | 7367.86 | 7386.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 10:00:00 | 7345.00 | 7354.91 | 7377.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 11:45:00 | 7350.50 | 7371.19 | 7376.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 09:30:00 | 7342.50 | 7337.97 | 7346.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 7347.00 | 7339.77 | 7346.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 12:00:00 | 7335.50 | 7338.92 | 7345.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 12:30:00 | 7335.50 | 7337.24 | 7344.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 12:15:00 | 6980.60 | 7023.49 | 7061.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 12:15:00 | 6977.75 | 7023.49 | 7061.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 12:15:00 | 6982.97 | 7023.49 | 7061.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 13:15:00 | 6975.38 | 7022.19 | 7057.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-12 09:15:00 | 7050.00 | 7024.13 | 7049.12 | SL hit (close>ema200) qty=0.50 sl=7024.13 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 7103.00 | 7061.99 | 7061.17 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 7021.00 | 7061.99 | 7062.28 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 12:15:00 | 7065.00 | 7062.47 | 7062.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 13:15:00 | 7076.50 | 7065.27 | 7063.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 12:15:00 | 7083.50 | 7090.90 | 7079.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 12:15:00 | 7083.50 | 7090.90 | 7079.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 7083.50 | 7090.90 | 7079.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 7083.50 | 7090.90 | 7079.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 7045.00 | 7081.72 | 7076.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 7045.00 | 7081.72 | 7076.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 7046.50 | 7074.68 | 7073.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:30:00 | 7039.50 | 7074.68 | 7073.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 7045.00 | 7068.74 | 7071.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 7010.00 | 7056.99 | 7065.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 6963.00 | 6937.61 | 6968.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 6963.00 | 6937.61 | 6968.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 6963.00 | 6937.61 | 6968.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 6963.00 | 6937.61 | 6968.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 6967.50 | 6943.58 | 6967.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 6967.50 | 6943.58 | 6967.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 6985.00 | 6951.87 | 6969.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:00:00 | 6985.00 | 6951.87 | 6969.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 6978.00 | 6957.09 | 6970.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:15:00 | 6998.50 | 6957.09 | 6970.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 6992.00 | 6964.08 | 6972.26 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 7022.00 | 6983.97 | 6980.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 7085.00 | 7004.17 | 6989.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 13:15:00 | 7164.50 | 7170.60 | 7137.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 13:45:00 | 7159.00 | 7170.60 | 7137.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 7156.50 | 7167.78 | 7139.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:30:00 | 7139.50 | 7167.78 | 7139.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 7135.50 | 7160.00 | 7140.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 7135.50 | 7160.00 | 7140.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 7130.00 | 7154.00 | 7139.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:00:00 | 7130.00 | 7154.00 | 7139.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 7130.00 | 7149.20 | 7138.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:30:00 | 7117.00 | 7149.20 | 7138.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 7096.00 | 7138.56 | 7135.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 13:00:00 | 7096.00 | 7138.56 | 7135.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 13:15:00 | 7080.00 | 7126.85 | 7130.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 7030.50 | 7094.73 | 7113.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 7011.00 | 7006.35 | 7045.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 10:30:00 | 7008.50 | 7006.35 | 7045.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 7040.00 | 7017.47 | 7041.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:45:00 | 7041.50 | 7017.47 | 7041.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 7045.50 | 7023.08 | 7041.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 7045.50 | 7023.08 | 7041.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 7032.50 | 7024.96 | 7040.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 7057.50 | 7024.96 | 7040.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 7047.50 | 7029.47 | 7041.45 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 12:15:00 | 7083.50 | 7054.51 | 7051.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 13:15:00 | 7087.00 | 7061.01 | 7054.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 13:15:00 | 7092.50 | 7096.71 | 7080.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 14:00:00 | 7092.50 | 7096.71 | 7080.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 7124.50 | 7109.86 | 7091.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 7069.00 | 7109.86 | 7091.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 7091.50 | 7109.35 | 7097.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 7091.50 | 7109.35 | 7097.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 7088.00 | 7105.08 | 7096.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:30:00 | 7079.50 | 7105.08 | 7096.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 7372.00 | 7383.17 | 7326.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 12:15:00 | 7396.50 | 7383.17 | 7326.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 13:30:00 | 7382.50 | 7383.95 | 7336.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 14:30:00 | 7383.50 | 7379.46 | 7338.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 11:15:00 | 7281.00 | 7342.06 | 7332.74 | SL hit (close<static) qty=1.00 sl=7325.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 7256.00 | 7313.24 | 7320.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 7183.50 | 7270.39 | 7297.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 7254.50 | 7246.68 | 7277.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 12:15:00 | 7254.50 | 7246.68 | 7277.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 7254.50 | 7246.68 | 7277.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:30:00 | 7272.50 | 7246.68 | 7277.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 7265.00 | 7252.24 | 7275.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 7284.00 | 7252.24 | 7275.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 7267.00 | 7255.19 | 7274.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 7268.50 | 7255.19 | 7274.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 7255.00 | 7255.15 | 7272.63 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 15:15:00 | 7308.00 | 7280.74 | 7278.76 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 09:15:00 | 7273.00 | 7279.89 | 7280.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 6909.00 | 7016.22 | 7105.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 6891.00 | 6868.01 | 6953.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:45:00 | 6895.50 | 6868.01 | 6953.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 6820.00 | 6793.39 | 6818.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 6861.50 | 6793.39 | 6818.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 6854.00 | 6805.51 | 6821.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 6843.50 | 6805.51 | 6821.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 6883.50 | 6821.11 | 6827.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 6883.50 | 6821.11 | 6827.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 6866.50 | 6836.73 | 6833.68 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 6807.00 | 6833.63 | 6834.67 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 6962.00 | 6843.44 | 6835.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 11:15:00 | 6989.00 | 6941.22 | 6903.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 14:15:00 | 6941.50 | 6951.97 | 6918.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 14:15:00 | 6941.50 | 6951.97 | 6918.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 6941.50 | 6951.97 | 6918.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:45:00 | 6956.00 | 6951.97 | 6918.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 6890.00 | 6939.57 | 6915.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 6890.00 | 6939.57 | 6915.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 6864.00 | 6924.46 | 6911.20 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 12:15:00 | 6876.00 | 6902.22 | 6903.09 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 6928.00 | 6906.70 | 6904.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 7034.50 | 6935.11 | 6918.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 7072.00 | 7088.44 | 7049.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 7072.00 | 7088.44 | 7049.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 7065.00 | 7079.60 | 7052.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:15:00 | 7086.00 | 7080.08 | 7054.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 7040.00 | 7085.53 | 7066.85 | SL hit (close<static) qty=1.00 sl=7043.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 7568.00 | 7595.58 | 7598.56 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 7622.50 | 7600.30 | 7599.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 7653.50 | 7612.20 | 7605.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 7755.00 | 7765.78 | 7733.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 11:15:00 | 7755.00 | 7765.78 | 7733.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 7755.00 | 7765.78 | 7733.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:00:00 | 7755.00 | 7765.78 | 7733.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 7807.00 | 7777.38 | 7751.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 11:45:00 | 7827.00 | 7787.32 | 7760.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 12:15:00 | 7829.50 | 7787.32 | 7760.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 7701.50 | 7792.65 | 7777.31 | SL hit (close<static) qty=1.00 sl=7731.50 alert=retest2 |

### Cycle 60 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 7720.50 | 7766.76 | 7767.49 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-03-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 14:15:00 | 7805.50 | 7768.37 | 7767.34 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 7556.50 | 7727.86 | 7749.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 7497.50 | 7656.61 | 7711.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 14:15:00 | 7645.00 | 7621.55 | 7678.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 14:15:00 | 7645.00 | 7621.55 | 7678.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 7645.00 | 7621.55 | 7678.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:45:00 | 7699.50 | 7621.55 | 7678.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 7715.00 | 7640.24 | 7682.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:15:00 | 7706.00 | 7640.24 | 7682.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 7720.00 | 7656.19 | 7685.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:30:00 | 7735.50 | 7656.19 | 7685.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 7664.00 | 7657.75 | 7683.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 11:45:00 | 7640.00 | 7655.40 | 7680.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 7761.50 | 7685.80 | 7688.65 | SL hit (close>static) qty=1.00 sl=7722.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 7792.50 | 7707.14 | 7698.09 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 7626.50 | 7702.97 | 7703.47 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-03-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 14:15:00 | 7776.00 | 7712.05 | 7704.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 09:15:00 | 7827.50 | 7747.77 | 7722.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 09:15:00 | 7722.50 | 7776.94 | 7755.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 09:15:00 | 7722.50 | 7776.94 | 7755.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 7722.50 | 7776.94 | 7755.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:00:00 | 7722.50 | 7776.94 | 7755.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 7712.50 | 7764.05 | 7751.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:00:00 | 7712.50 | 7764.05 | 7751.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 7698.00 | 7744.91 | 7744.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 7667.00 | 7723.03 | 7734.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 11:15:00 | 7598.00 | 7585.83 | 7634.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-13 11:45:00 | 7610.00 | 7585.83 | 7634.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 7548.50 | 7560.16 | 7601.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 10:15:00 | 7494.50 | 7560.16 | 7601.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 10:30:00 | 7536.50 | 7504.67 | 7539.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 12:15:00 | 7540.00 | 7531.78 | 7535.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 7119.77 | 7283.23 | 7349.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 7159.67 | 7283.23 | 7349.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 7163.00 | 7283.23 | 7349.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 7371.00 | 7217.59 | 7274.90 | SL hit (close>ema200) qty=0.50 sl=7217.59 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 7431.50 | 7304.06 | 7303.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 7541.00 | 7396.69 | 7351.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 7533.00 | 7536.52 | 7461.04 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 11:00:00 | 7600.50 | 7549.32 | 7473.71 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 11:30:00 | 7618.00 | 7556.45 | 7483.83 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 12:15:00 | 7602.00 | 7556.45 | 7483.83 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 7427.50 | 7529.07 | 7498.99 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 7427.50 | 7529.07 | 7498.99 | SL hit (close<ema400) qty=1.00 sl=7498.99 alert=retest1 |

### Cycle 68 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 7394.00 | 7464.66 | 7473.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 10:15:00 | 7306.50 | 7424.00 | 7449.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 7346.50 | 7295.50 | 7345.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 13:15:00 | 7346.50 | 7295.50 | 7345.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 7346.50 | 7295.50 | 7345.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:00:00 | 7346.50 | 7295.50 | 7345.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 7325.50 | 7301.50 | 7344.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:30:00 | 7330.00 | 7301.50 | 7344.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 7274.50 | 7295.70 | 7334.02 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 7381.00 | 7335.74 | 7332.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 10:15:00 | 7545.50 | 7480.84 | 7434.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 14:15:00 | 7506.50 | 7507.91 | 7464.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 7467.00 | 7501.58 | 7469.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 7467.00 | 7501.58 | 7469.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 7518.00 | 7509.37 | 7475.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 15:15:00 | 7740.00 | 7760.47 | 7761.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 15:15:00 | 7740.00 | 7760.47 | 7761.13 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 7777.00 | 7763.78 | 7762.57 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 10:15:00 | 7753.50 | 7761.72 | 7761.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 12:15:00 | 7722.00 | 7752.54 | 7757.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 7762.00 | 7663.53 | 7686.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 7762.00 | 7663.53 | 7686.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 7762.00 | 7663.53 | 7686.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 7762.00 | 7663.53 | 7686.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 7760.00 | 7682.82 | 7692.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:30:00 | 7776.00 | 7682.82 | 7692.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 7737.00 | 7704.33 | 7701.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 14:15:00 | 7774.50 | 7738.69 | 7723.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 15:15:00 | 7755.00 | 7759.20 | 7745.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 09:15:00 | 7765.50 | 7759.20 | 7745.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 7768.50 | 7761.06 | 7747.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:30:00 | 7760.00 | 7761.06 | 7747.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-27 11:15:00 | 7087.00 | 2025-05-28 09:15:00 | 7000.50 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-05-27 14:45:00 | 7073.50 | 2025-05-28 09:15:00 | 7000.50 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-05-30 13:00:00 | 6931.50 | 2025-06-02 09:15:00 | 7065.00 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-05-30 14:00:00 | 6932.00 | 2025-06-02 09:15:00 | 7065.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-05-30 14:30:00 | 6928.00 | 2025-06-02 09:15:00 | 7065.00 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-06-02 14:00:00 | 6920.50 | 2025-06-05 13:15:00 | 6880.00 | STOP_HIT | 1.00 | 0.59% |
| SELL | retest2 | 2025-06-03 10:45:00 | 6828.50 | 2025-06-05 13:15:00 | 6880.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-06-03 13:45:00 | 6825.50 | 2025-06-05 13:15:00 | 6880.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-06-03 14:45:00 | 6818.00 | 2025-06-05 13:15:00 | 6880.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-06-04 10:45:00 | 6820.50 | 2025-06-05 13:15:00 | 6880.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-06-05 11:15:00 | 6877.50 | 2025-06-05 13:15:00 | 6880.00 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-06-05 12:15:00 | 6884.00 | 2025-06-05 13:15:00 | 6880.00 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-06-09 14:15:00 | 6921.50 | 2025-06-11 09:15:00 | 6878.50 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-06-10 13:45:00 | 6924.50 | 2025-06-11 09:15:00 | 6878.50 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-06-11 12:15:00 | 6921.00 | 2025-06-11 13:15:00 | 6899.50 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-06-23 11:15:00 | 7042.00 | 2025-06-24 15:15:00 | 7000.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-06-24 10:00:00 | 7046.00 | 2025-06-24 15:15:00 | 7000.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-06-26 13:30:00 | 7058.00 | 2025-07-09 11:15:00 | 7498.00 | STOP_HIT | 1.00 | 6.23% |
| BUY | retest2 | 2025-07-17 09:30:00 | 7399.50 | 2025-07-18 09:15:00 | 7276.50 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-07-17 10:30:00 | 7397.00 | 2025-07-18 09:15:00 | 7276.50 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-07-17 11:00:00 | 7401.00 | 2025-07-18 09:15:00 | 7276.50 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-07-25 11:15:00 | 7435.00 | 2025-07-28 12:15:00 | 7340.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-07-25 12:30:00 | 7435.00 | 2025-07-28 12:15:00 | 7340.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-07-25 13:15:00 | 7448.00 | 2025-07-28 12:15:00 | 7340.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-07-28 09:45:00 | 7443.50 | 2025-07-28 12:15:00 | 7340.00 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-07-31 13:00:00 | 7445.00 | 2025-08-01 12:15:00 | 7389.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-08-08 10:30:00 | 7121.50 | 2025-08-11 13:15:00 | 7215.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-08-08 12:00:00 | 7118.50 | 2025-08-11 13:15:00 | 7215.00 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-08-22 11:15:00 | 7927.00 | 2025-08-25 10:15:00 | 7849.50 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-08-22 14:30:00 | 7931.00 | 2025-08-25 10:15:00 | 7849.50 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-08-22 15:15:00 | 7930.00 | 2025-08-25 10:15:00 | 7849.50 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-08-28 12:30:00 | 7785.50 | 2025-09-03 10:15:00 | 7739.50 | STOP_HIT | 1.00 | 0.59% |
| BUY | retest2 | 2025-09-12 12:15:00 | 7881.00 | 2025-09-15 09:15:00 | 7827.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-09-12 14:15:00 | 7880.00 | 2025-09-15 09:15:00 | 7827.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-10-13 09:15:00 | 7721.50 | 2025-10-13 11:15:00 | 7652.50 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-10-23 10:15:00 | 8095.50 | 2025-10-24 09:15:00 | 7900.00 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2025-10-30 09:15:00 | 7811.50 | 2025-11-03 15:15:00 | 7835.00 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-11-06 11:15:00 | 7799.50 | 2025-11-06 15:15:00 | 7783.00 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-11-06 14:00:00 | 7815.50 | 2025-11-06 15:15:00 | 7783.00 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-11-25 14:45:00 | 7348.00 | 2025-12-11 12:15:00 | 6980.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 10:00:00 | 7345.00 | 2025-12-11 12:15:00 | 6977.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 11:45:00 | 7350.50 | 2025-12-11 12:15:00 | 6982.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 09:30:00 | 7342.50 | 2025-12-11 13:15:00 | 6975.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-25 14:45:00 | 7348.00 | 2025-12-12 09:15:00 | 7050.00 | STOP_HIT | 0.50 | 4.06% |
| SELL | retest2 | 2025-11-26 10:00:00 | 7345.00 | 2025-12-12 09:15:00 | 7050.00 | STOP_HIT | 0.50 | 4.02% |
| SELL | retest2 | 2025-11-27 11:45:00 | 7350.50 | 2025-12-12 09:15:00 | 7050.00 | STOP_HIT | 0.50 | 4.09% |
| SELL | retest2 | 2025-12-01 09:30:00 | 7342.50 | 2025-12-12 09:15:00 | 7050.00 | STOP_HIT | 0.50 | 3.98% |
| SELL | retest2 | 2025-12-01 12:00:00 | 7335.50 | 2025-12-12 13:15:00 | 7103.00 | STOP_HIT | 1.00 | 3.17% |
| SELL | retest2 | 2025-12-01 12:30:00 | 7335.50 | 2025-12-12 13:15:00 | 7103.00 | STOP_HIT | 1.00 | 3.17% |
| BUY | retest2 | 2026-01-08 12:15:00 | 7396.50 | 2026-01-09 11:15:00 | 7281.00 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2026-01-08 13:30:00 | 7382.50 | 2026-01-09 11:15:00 | 7281.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-01-08 14:30:00 | 7383.50 | 2026-01-09 11:15:00 | 7281.00 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2026-02-05 13:15:00 | 7086.00 | 2026-02-06 09:15:00 | 7040.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2026-02-06 11:45:00 | 7109.00 | 2026-02-20 09:15:00 | 7568.00 | STOP_HIT | 1.00 | 6.46% |
| BUY | retest2 | 2026-02-27 11:45:00 | 7827.00 | 2026-03-02 09:15:00 | 7701.50 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2026-02-27 12:15:00 | 7829.50 | 2026-03-02 09:15:00 | 7701.50 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2026-03-05 11:45:00 | 7640.00 | 2026-03-05 14:15:00 | 7761.50 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-03-16 10:15:00 | 7494.50 | 2026-03-23 10:15:00 | 7119.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 10:30:00 | 7536.50 | 2026-03-23 10:15:00 | 7159.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 12:15:00 | 7540.00 | 2026-03-23 10:15:00 | 7163.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-16 10:15:00 | 7494.50 | 2026-03-24 09:15:00 | 7371.00 | STOP_HIT | 0.50 | 1.65% |
| SELL | retest2 | 2026-03-17 10:30:00 | 7536.50 | 2026-03-24 09:15:00 | 7371.00 | STOP_HIT | 0.50 | 2.20% |
| SELL | retest2 | 2026-03-18 12:15:00 | 7540.00 | 2026-03-24 09:15:00 | 7371.00 | STOP_HIT | 0.50 | 2.24% |
| BUY | retest1 | 2026-03-27 11:00:00 | 7600.50 | 2026-03-30 09:15:00 | 7427.50 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest1 | 2026-03-27 11:30:00 | 7618.00 | 2026-03-30 09:15:00 | 7427.50 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest1 | 2026-03-27 12:15:00 | 7602.00 | 2026-03-30 09:15:00 | 7427.50 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2026-04-13 10:45:00 | 7518.00 | 2026-04-28 15:15:00 | 7740.00 | STOP_HIT | 1.00 | 2.95% |
