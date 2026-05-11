# Bosch Ltd. (BOSCHLTD)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 38050.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 207 |
| ALERT1 | 141 |
| ALERT2 | 141 |
| ALERT2_SKIP | 74 |
| ALERT3 | 392 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 213 |
| PARTIAL | 11 |
| TARGET_HIT | 14 |
| STOP_HIT | 204 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 229 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 90 / 139
- **Target hits / Stop hits / Partials:** 14 / 204 / 11
- **Avg / median % per leg:** 0.77% / -0.48%
- **Sum % (uncompounded):** 176.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 109 | 58 | 53.2% | 14 | 95 | 0 | 1.71% | 185.9% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | 0.06% | 0.1% |
| BUY @ 3rd Alert (retest2) | 107 | 57 | 53.3% | 14 | 93 | 0 | 1.74% | 185.7% |
| SELL (all) | 120 | 32 | 26.7% | 0 | 109 | 11 | -0.08% | -9.7% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.85% | -8.5% |
| SELL @ 3rd Alert (retest2) | 117 | 32 | 27.4% | 0 | 106 | 11 | -0.01% | -1.1% |
| retest1 (combined) | 5 | 1 | 20.0% | 0 | 5 | 0 | -1.69% | -8.4% |
| retest2 (combined) | 224 | 89 | 39.7% | 14 | 199 | 11 | 0.82% | 184.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 10:15:00 | 19216.90 | 19127.74 | 19117.71 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 14:15:00 | 18987.30 | 19094.26 | 19106.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-19 09:15:00 | 18784.40 | 19034.79 | 19077.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-23 09:15:00 | 18981.20 | 18869.10 | 18918.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 09:15:00 | 18981.20 | 18869.10 | 18918.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 09:15:00 | 18981.20 | 18869.10 | 18918.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-23 10:00:00 | 18981.20 | 18869.10 | 18918.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 10:15:00 | 19000.00 | 18895.28 | 18926.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-23 11:45:00 | 18910.00 | 18904.55 | 18927.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-23 13:00:00 | 18946.90 | 18913.02 | 18929.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-23 13:45:00 | 18935.00 | 18930.41 | 18935.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-23 14:15:00 | 18921.10 | 18930.41 | 18935.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 14:15:00 | 18874.60 | 18919.25 | 18930.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-23 15:15:00 | 18822.10 | 18919.25 | 18930.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-24 09:45:00 | 18834.90 | 18896.66 | 18916.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-24 10:15:00 | 18841.00 | 18896.66 | 18916.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-24 12:45:00 | 18842.40 | 18869.43 | 18897.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 14:15:00 | 18785.20 | 18850.66 | 18884.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-24 14:30:00 | 18860.00 | 18850.66 | 18884.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 18794.90 | 18815.40 | 18860.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-25 11:30:00 | 18699.60 | 18784.37 | 18838.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-26 09:15:00 | 18940.40 | 18802.01 | 18822.75 | SL hit (close>static) qty=1.00 sl=18861.10 alert=retest2 |

### Cycle 3 — BUY (started 2023-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 11:15:00 | 18926.60 | 18845.17 | 18839.90 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-29 09:15:00 | 18430.00 | 18792.38 | 18822.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-29 14:15:00 | 18294.80 | 18527.42 | 18666.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-30 09:15:00 | 18530.00 | 18496.19 | 18625.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-30 10:00:00 | 18530.00 | 18496.19 | 18625.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 11:15:00 | 18570.00 | 18519.79 | 18614.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-30 11:30:00 | 18589.40 | 18519.79 | 18614.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 13:15:00 | 18624.30 | 18547.80 | 18576.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-31 13:45:00 | 18660.00 | 18547.80 | 18576.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 14:15:00 | 18626.40 | 18563.52 | 18580.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-31 14:45:00 | 18630.50 | 18563.52 | 18580.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 09:15:00 | 18511.20 | 18542.89 | 18567.74 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-06-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 12:15:00 | 18665.30 | 18583.17 | 18581.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-02 10:15:00 | 18745.40 | 18637.81 | 18609.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-06 11:15:00 | 18835.10 | 18838.59 | 18781.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-06 12:00:00 | 18835.10 | 18838.59 | 18781.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 18813.00 | 18923.18 | 18893.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:00:00 | 18813.00 | 18923.18 | 18893.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 12:15:00 | 18816.80 | 18901.90 | 18886.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:30:00 | 18827.20 | 18901.90 | 18886.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2023-06-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 14:15:00 | 18780.00 | 18860.61 | 18869.59 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 13:15:00 | 18990.00 | 18856.47 | 18843.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 09:15:00 | 19250.00 | 18973.96 | 18903.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-14 14:15:00 | 19063.70 | 19102.60 | 19044.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-14 15:00:00 | 19063.70 | 19102.60 | 19044.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 15:15:00 | 19041.80 | 19090.44 | 19044.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-15 09:45:00 | 19174.60 | 19114.61 | 19059.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-15 12:45:00 | 19106.20 | 19123.50 | 19078.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-15 13:30:00 | 19103.80 | 19115.02 | 19079.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-15 14:15:00 | 19015.20 | 19095.06 | 19073.29 | SL hit (close<static) qty=1.00 sl=19023.50 alert=retest2 |

### Cycle 8 — SELL (started 2023-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-16 13:15:00 | 18990.00 | 19067.42 | 19070.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-20 10:15:00 | 18837.10 | 18967.05 | 19011.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 13:15:00 | 19073.60 | 18972.59 | 19001.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 13:15:00 | 19073.60 | 18972.59 | 19001.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 13:15:00 | 19073.60 | 18972.59 | 19001.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-20 14:00:00 | 19073.60 | 18972.59 | 19001.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 14:15:00 | 19051.60 | 18988.39 | 19005.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-20 14:30:00 | 19025.00 | 18988.39 | 19005.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2023-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 09:15:00 | 19355.80 | 19070.93 | 19040.77 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-06-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 12:15:00 | 18806.60 | 19087.86 | 19108.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 18731.10 | 18933.96 | 19022.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 15:15:00 | 18630.00 | 18620.16 | 18732.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-27 09:15:00 | 18636.80 | 18620.16 | 18732.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 09:15:00 | 18589.20 | 18579.40 | 18647.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-28 09:30:00 | 18626.00 | 18579.40 | 18647.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 11:15:00 | 18665.10 | 18585.60 | 18638.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-28 12:00:00 | 18665.10 | 18585.60 | 18638.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 12:15:00 | 18721.90 | 18612.86 | 18645.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-28 12:30:00 | 18732.00 | 18612.86 | 18645.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 15:15:00 | 18689.00 | 18655.71 | 18659.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-30 09:15:00 | 18866.70 | 18655.71 | 18659.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2023-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 09:15:00 | 19038.30 | 18732.23 | 18693.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 10:15:00 | 19210.00 | 18827.78 | 18740.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 11:15:00 | 18963.40 | 19006.87 | 18908.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 11:15:00 | 18963.40 | 19006.87 | 18908.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 11:15:00 | 18963.40 | 19006.87 | 18908.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 12:00:00 | 18963.40 | 19006.87 | 18908.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 14:15:00 | 19020.70 | 19010.01 | 18933.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 14:45:00 | 18954.90 | 19010.01 | 18933.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 14:15:00 | 19103.40 | 19101.70 | 19028.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 15:00:00 | 19103.40 | 19101.70 | 19028.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 19134.00 | 19106.29 | 19043.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-05 10:30:00 | 19195.60 | 19128.01 | 19058.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-05 11:00:00 | 19214.90 | 19128.01 | 19058.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-05 15:00:00 | 19207.30 | 19135.67 | 19083.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-06 12:15:00 | 19194.40 | 19202.17 | 19136.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 19603.60 | 19663.47 | 19504.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 09:30:00 | 19577.80 | 19663.47 | 19504.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 10:15:00 | 19499.80 | 19630.74 | 19504.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 11:00:00 | 19499.80 | 19630.74 | 19504.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 11:15:00 | 19477.40 | 19600.07 | 19501.61 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-07-10 15:15:00 | 19215.90 | 19407.65 | 19432.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2023-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 15:15:00 | 19215.90 | 19407.65 | 19432.13 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-07-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 14:15:00 | 19520.00 | 19417.93 | 19405.26 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-07-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 13:15:00 | 19164.70 | 19378.21 | 19395.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-14 09:15:00 | 18796.30 | 19197.82 | 19303.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-17 09:15:00 | 18964.80 | 18963.23 | 19102.27 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-17 10:30:00 | 18886.60 | 18948.59 | 19082.97 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-17 12:15:00 | 18924.60 | 18948.73 | 19070.82 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-18 09:15:00 | 19639.40 | 19090.54 | 19087.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest1 |

### Cycle 15 — BUY (started 2023-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-18 09:15:00 | 19639.40 | 19090.54 | 19087.98 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-19 13:15:00 | 19020.00 | 19129.03 | 19130.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-20 09:15:00 | 18927.80 | 19078.54 | 19106.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-20 13:15:00 | 19048.30 | 19043.12 | 19076.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 13:15:00 | 19048.30 | 19043.12 | 19076.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 13:15:00 | 19048.30 | 19043.12 | 19076.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-21 09:15:00 | 18977.10 | 19046.79 | 19073.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-21 14:00:00 | 18910.90 | 18943.63 | 19005.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-25 10:15:00 | 18993.80 | 18982.68 | 18981.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2023-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 10:15:00 | 18993.80 | 18982.68 | 18981.58 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 11:15:00 | 18916.90 | 18969.53 | 18975.70 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 15:15:00 | 19021.00 | 18978.03 | 18976.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-26 10:15:00 | 19106.60 | 19017.87 | 18995.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-27 09:15:00 | 19044.90 | 19107.68 | 19061.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 09:15:00 | 19044.90 | 19107.68 | 19061.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 19044.90 | 19107.68 | 19061.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 10:00:00 | 19044.90 | 19107.68 | 19061.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 10:15:00 | 19010.80 | 19088.30 | 19057.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 10:30:00 | 18992.00 | 19088.30 | 19057.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2023-07-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 13:15:00 | 18965.00 | 19025.50 | 19033.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-27 15:15:00 | 18813.80 | 18966.30 | 19003.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-28 10:15:00 | 18995.10 | 18962.08 | 18994.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-28 10:15:00 | 18995.10 | 18962.08 | 18994.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 10:15:00 | 18995.10 | 18962.08 | 18994.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 11:00:00 | 18995.10 | 18962.08 | 18994.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 11:15:00 | 18942.90 | 18958.24 | 18990.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 11:30:00 | 18990.10 | 18958.24 | 18990.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 14:15:00 | 18970.00 | 18945.96 | 18975.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 15:00:00 | 18970.00 | 18945.96 | 18975.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 19059.20 | 18972.07 | 18982.21 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 10:15:00 | 19065.00 | 18990.65 | 18989.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 12:15:00 | 19080.00 | 19010.21 | 18998.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-31 13:15:00 | 19003.00 | 19008.77 | 18999.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-31 13:45:00 | 19005.00 | 19008.77 | 18999.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 14:15:00 | 18944.30 | 18995.87 | 18994.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-31 14:30:00 | 18934.30 | 18995.87 | 18994.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2023-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-31 15:15:00 | 18950.50 | 18986.80 | 18990.29 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-01 09:15:00 | 19080.60 | 19005.56 | 18998.50 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2023-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 09:15:00 | 18690.20 | 18942.45 | 18975.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 10:15:00 | 18482.10 | 18850.38 | 18930.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-04 12:15:00 | 18274.00 | 18231.93 | 18375.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-04 13:00:00 | 18274.00 | 18231.93 | 18375.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 09:15:00 | 18259.90 | 18237.33 | 18331.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-08 10:45:00 | 18139.60 | 18201.53 | 18263.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-08 14:30:00 | 18150.10 | 18189.76 | 18238.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-09 09:15:00 | 18139.30 | 18191.81 | 18235.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-09 13:15:00 | 18310.10 | 18258.54 | 18252.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2023-08-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 13:15:00 | 18310.10 | 18258.54 | 18252.84 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-08-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 14:15:00 | 18173.90 | 18246.79 | 18253.78 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-11 12:15:00 | 18305.10 | 18260.31 | 18255.16 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 14:15:00 | 18145.60 | 18233.19 | 18243.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 09:15:00 | 18068.40 | 18202.92 | 18228.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 10:15:00 | 18134.00 | 18125.26 | 18163.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-16 10:45:00 | 18130.80 | 18125.26 | 18163.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 11:15:00 | 18188.00 | 18137.81 | 18165.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 12:00:00 | 18188.00 | 18137.81 | 18165.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 12:15:00 | 18150.40 | 18140.33 | 18164.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-16 13:45:00 | 18075.00 | 18131.76 | 18158.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-17 09:15:00 | 18206.70 | 18156.48 | 18163.65 | SL hit (close>static) qty=1.00 sl=18193.90 alert=retest2 |

### Cycle 29 — BUY (started 2023-08-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 10:15:00 | 18305.00 | 18186.19 | 18176.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 09:15:00 | 18323.70 | 18273.95 | 18251.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 12:15:00 | 18355.50 | 18365.16 | 18326.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-23 12:30:00 | 18345.10 | 18365.16 | 18326.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 10:15:00 | 18299.90 | 18359.06 | 18339.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 11:00:00 | 18299.90 | 18359.06 | 18339.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 11:15:00 | 18394.90 | 18366.23 | 18344.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 12:15:00 | 18403.20 | 18366.23 | 18344.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 13:00:00 | 18455.00 | 18383.98 | 18354.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-25 12:00:00 | 18395.00 | 18358.89 | 18352.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-25 14:15:00 | 18284.90 | 18371.67 | 18362.09 | SL hit (close<static) qty=1.00 sl=18299.00 alert=retest2 |

### Cycle 30 — SELL (started 2023-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 12:15:00 | 19090.00 | 19292.67 | 19313.72 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-09-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 11:15:00 | 19432.00 | 19299.14 | 19298.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 09:15:00 | 19519.90 | 19375.66 | 19338.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 11:15:00 | 19492.90 | 19498.63 | 19438.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-15 11:45:00 | 19500.00 | 19498.63 | 19438.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 15:15:00 | 19590.00 | 19596.61 | 19550.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 09:15:00 | 19606.70 | 19596.61 | 19550.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 19531.80 | 19583.65 | 19548.43 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 11:15:00 | 19400.00 | 19515.95 | 19521.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 13:15:00 | 19362.70 | 19466.75 | 19497.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 12:15:00 | 19153.20 | 19152.31 | 19254.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-22 13:00:00 | 19153.20 | 19152.31 | 19254.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 13:15:00 | 19235.80 | 19169.01 | 19252.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 14:00:00 | 19235.80 | 19169.01 | 19252.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 14:15:00 | 19120.00 | 19159.21 | 19240.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 14:30:00 | 19250.80 | 19159.21 | 19240.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 19105.10 | 19099.06 | 19151.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 12:30:00 | 19025.00 | 19104.73 | 19126.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-27 14:15:00 | 19213.50 | 19127.33 | 19133.21 | SL hit (close>static) qty=1.00 sl=19170.00 alert=retest2 |

### Cycle 33 — BUY (started 2023-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 15:15:00 | 19242.00 | 19150.26 | 19143.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-28 15:15:00 | 19499.90 | 19221.66 | 19181.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-29 09:15:00 | 19197.10 | 19216.75 | 19182.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-29 09:15:00 | 19197.10 | 19216.75 | 19182.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 09:15:00 | 19197.10 | 19216.75 | 19182.62 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-29 11:15:00 | 19004.80 | 19152.35 | 19157.91 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-09-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 12:15:00 | 19203.90 | 19162.66 | 19162.09 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-29 13:15:00 | 19097.60 | 19149.65 | 19156.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-29 14:15:00 | 19000.00 | 19119.72 | 19142.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-04 15:15:00 | 18762.00 | 18756.87 | 18851.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-05 09:15:00 | 18881.60 | 18756.87 | 18851.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 18816.90 | 18768.88 | 18847.95 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 09:15:00 | 19088.70 | 18912.24 | 18889.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 10:15:00 | 19150.00 | 18959.79 | 18912.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 19150.00 | 19169.07 | 19056.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 19150.00 | 19169.07 | 19056.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 19150.00 | 19169.07 | 19056.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-10 09:15:00 | 19378.40 | 19213.87 | 19128.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-18 10:15:00 | 20265.90 | 20414.40 | 20421.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2023-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 10:15:00 | 20265.90 | 20414.40 | 20421.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 11:15:00 | 20181.20 | 20367.76 | 20399.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 14:15:00 | 20239.80 | 20157.78 | 20229.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 14:15:00 | 20239.80 | 20157.78 | 20229.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 14:15:00 | 20239.80 | 20157.78 | 20229.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 14:45:00 | 20201.20 | 20157.78 | 20229.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 15:15:00 | 20247.90 | 20175.81 | 20231.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 09:30:00 | 20085.90 | 20182.09 | 20229.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 10:15:00 | 20136.20 | 20182.09 | 20229.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-03 15:15:00 | 19540.00 | 19500.63 | 19496.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2023-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 15:15:00 | 19540.00 | 19500.63 | 19496.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 09:15:00 | 19608.80 | 19522.27 | 19506.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-06 12:15:00 | 19552.30 | 19557.99 | 19529.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-06 13:00:00 | 19552.30 | 19557.99 | 19529.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 13:15:00 | 19595.90 | 19565.58 | 19535.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-06 13:30:00 | 19558.00 | 19565.58 | 19535.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 14:15:00 | 19585.40 | 19569.54 | 19539.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-06 15:15:00 | 19540.00 | 19569.54 | 19539.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 15:15:00 | 19540.00 | 19563.63 | 19539.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 09:15:00 | 19569.90 | 19563.63 | 19539.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 19639.60 | 19578.83 | 19548.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-07 13:30:00 | 19718.40 | 19617.96 | 19577.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-08 12:15:00 | 19448.50 | 19611.80 | 19603.28 | SL hit (close<static) qty=1.00 sl=19511.00 alert=retest2 |

### Cycle 40 — SELL (started 2023-11-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 11:15:00 | 19537.10 | 19602.03 | 19604.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 12:15:00 | 19481.60 | 19577.94 | 19593.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-09 15:15:00 | 19580.00 | 19565.43 | 19582.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 15:15:00 | 19580.00 | 19565.43 | 19582.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 15:15:00 | 19580.00 | 19565.43 | 19582.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 09:15:00 | 19709.90 | 19565.43 | 19582.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 19563.70 | 19565.08 | 19580.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 09:30:00 | 19721.80 | 19565.08 | 19580.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 10:15:00 | 19600.00 | 19572.07 | 19582.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 10:45:00 | 19685.80 | 19572.07 | 19582.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 11:15:00 | 19590.00 | 19575.65 | 19583.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 11:45:00 | 19610.10 | 19575.65 | 19583.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 12:15:00 | 19575.00 | 19575.52 | 19582.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 12:30:00 | 19580.10 | 19575.52 | 19582.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 13:15:00 | 19570.00 | 19574.42 | 19581.36 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2023-11-12 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-12 18:15:00 | 19628.00 | 19585.49 | 19584.60 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2023-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 11:15:00 | 19555.00 | 19579.14 | 19582.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-13 14:15:00 | 19460.00 | 19538.10 | 19561.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 10:15:00 | 19562.00 | 19532.21 | 19551.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-15 10:15:00 | 19562.00 | 19532.21 | 19551.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 10:15:00 | 19562.00 | 19532.21 | 19551.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-15 10:45:00 | 19562.00 | 19532.21 | 19551.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 11:15:00 | 19659.90 | 19557.75 | 19561.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-15 11:45:00 | 19665.10 | 19557.75 | 19561.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2023-11-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 12:15:00 | 19652.00 | 19576.60 | 19569.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 09:15:00 | 19999.90 | 19709.70 | 19638.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 11:15:00 | 20563.00 | 20605.86 | 20403.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-20 12:00:00 | 20563.00 | 20605.86 | 20403.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 21127.30 | 20771.85 | 20627.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-23 12:45:00 | 21217.70 | 20997.88 | 20847.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-23 15:00:00 | 21201.30 | 21059.72 | 20902.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-24 10:30:00 | 21204.10 | 21121.31 | 20973.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-24 11:15:00 | 21240.30 | 21121.31 | 20973.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 13:15:00 | 21042.20 | 21109.84 | 21005.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 13:45:00 | 21040.90 | 21109.84 | 21005.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 15:15:00 | 21002.00 | 21077.77 | 21008.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 09:15:00 | 21175.00 | 21077.77 | 21008.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 21149.00 | 21092.01 | 21021.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-28 15:00:00 | 21312.20 | 21178.86 | 21093.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-06 09:15:00 | 21701.10 | 21782.41 | 21788.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2023-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 09:15:00 | 21701.10 | 21782.41 | 21788.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-06 11:15:00 | 21621.80 | 21740.86 | 21767.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-07 09:15:00 | 21725.00 | 21679.06 | 21720.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-07 09:15:00 | 21725.00 | 21679.06 | 21720.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 21725.00 | 21679.06 | 21720.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-07 10:00:00 | 21725.00 | 21679.06 | 21720.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 10:15:00 | 21754.00 | 21694.05 | 21723.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-07 10:30:00 | 21709.70 | 21694.05 | 21723.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 11:15:00 | 21640.00 | 21683.24 | 21715.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-07 14:30:00 | 21608.50 | 21687.06 | 21709.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-08 09:15:00 | 21634.40 | 21688.15 | 21707.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-08 10:15:00 | 21620.10 | 21681.94 | 21703.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-08 11:00:00 | 21595.90 | 21664.73 | 21693.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 21776.20 | 21656.74 | 21669.80 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-12-11 09:15:00 | 21776.20 | 21656.74 | 21669.80 | SL hit (close>static) qty=1.00 sl=21759.30 alert=retest2 |

### Cycle 45 — BUY (started 2023-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 11:15:00 | 21731.10 | 21683.33 | 21680.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 13:15:00 | 21839.20 | 21732.05 | 21704.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 12:15:00 | 21829.20 | 21854.40 | 21790.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 12:15:00 | 21829.20 | 21854.40 | 21790.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 12:15:00 | 21829.20 | 21854.40 | 21790.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 12:45:00 | 21830.00 | 21854.40 | 21790.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 14:15:00 | 21785.30 | 21844.98 | 21797.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 14:45:00 | 21775.80 | 21844.98 | 21797.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 15:15:00 | 21631.00 | 21802.18 | 21782.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-13 09:15:00 | 21831.20 | 21802.18 | 21782.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-13 10:00:00 | 21812.20 | 21804.19 | 21785.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-13 11:15:00 | 21632.70 | 21754.44 | 21765.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2023-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 11:15:00 | 21632.70 | 21754.44 | 21765.03 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2023-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 14:15:00 | 21918.90 | 21787.38 | 21776.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 15:15:00 | 22050.00 | 21839.91 | 21801.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 10:15:00 | 21940.00 | 21990.55 | 21918.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-15 10:15:00 | 21940.00 | 21990.55 | 21918.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 10:15:00 | 21940.00 | 21990.55 | 21918.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 10:45:00 | 21979.80 | 21990.55 | 21918.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 11:15:00 | 21936.10 | 21979.66 | 21920.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 11:45:00 | 21941.90 | 21979.66 | 21920.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 12:15:00 | 21828.20 | 21949.37 | 21912.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 13:00:00 | 21828.20 | 21949.37 | 21912.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 13:15:00 | 21744.30 | 21908.36 | 21896.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 14:00:00 | 21744.30 | 21908.36 | 21896.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2023-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 14:15:00 | 21693.20 | 21865.33 | 21878.25 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2023-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-18 10:15:00 | 21977.10 | 21886.64 | 21883.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-18 15:15:00 | 22018.00 | 21930.27 | 21907.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 13:15:00 | 21932.20 | 21948.79 | 21927.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-19 13:15:00 | 21932.20 | 21948.79 | 21927.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 13:15:00 | 21932.20 | 21948.79 | 21927.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 13:45:00 | 21917.90 | 21948.79 | 21927.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 14:15:00 | 21864.30 | 21931.89 | 21921.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 14:45:00 | 21835.30 | 21931.89 | 21921.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 15:15:00 | 21970.00 | 21939.52 | 21925.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-20 09:15:00 | 22001.00 | 21939.52 | 21925.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 12:15:00 | 21697.80 | 21880.56 | 21903.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2023-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 12:15:00 | 21697.80 | 21880.56 | 21903.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 13:15:00 | 21504.10 | 21805.27 | 21867.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 15:15:00 | 21697.00 | 21624.63 | 21704.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-22 09:15:00 | 21740.00 | 21624.63 | 21704.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 21837.20 | 21667.14 | 21716.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 10:00:00 | 21837.20 | 21667.14 | 21716.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 21702.80 | 21674.27 | 21715.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 12:00:00 | 21651.60 | 21669.74 | 21709.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-26 12:15:00 | 21761.70 | 21707.91 | 21703.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2023-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 12:15:00 | 21761.70 | 21707.91 | 21703.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 13:15:00 | 21816.00 | 21729.53 | 21713.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 11:15:00 | 21896.10 | 21932.92 | 21869.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 11:15:00 | 21896.10 | 21932.92 | 21869.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 11:15:00 | 21896.10 | 21932.92 | 21869.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 11:45:00 | 21884.90 | 21932.92 | 21869.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 14:15:00 | 22060.20 | 21960.17 | 21898.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 14:45:00 | 21912.20 | 21960.17 | 21898.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 22145.00 | 22312.79 | 22229.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:45:00 | 22140.10 | 22312.79 | 22229.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 11:15:00 | 22139.90 | 22278.21 | 22221.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 11:30:00 | 22144.80 | 22278.21 | 22221.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 09:15:00 | 22185.30 | 22289.41 | 22250.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-03 10:00:00 | 22185.30 | 22289.41 | 22250.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 10:15:00 | 22262.00 | 22283.93 | 22251.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 11:30:00 | 22301.00 | 22285.92 | 22255.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 13:30:00 | 22285.70 | 22278.48 | 22257.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 14:00:00 | 22279.20 | 22278.48 | 22257.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-04 11:30:00 | 22348.70 | 22320.62 | 22288.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 10:15:00 | 22520.00 | 22573.87 | 22493.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 11:00:00 | 22520.00 | 22573.87 | 22493.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 11:15:00 | 22487.00 | 22556.49 | 22492.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 11:45:00 | 22470.10 | 22556.49 | 22492.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 12:15:00 | 22502.90 | 22545.77 | 22493.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 09:15:00 | 22632.10 | 22512.72 | 22490.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 13:00:00 | 22621.70 | 22704.00 | 22653.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 13:45:00 | 22615.20 | 22689.92 | 22651.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-17 09:15:00 | 22950.10 | 23162.82 | 23176.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2024-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 09:15:00 | 22950.10 | 23162.82 | 23176.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 10:15:00 | 22740.10 | 23078.28 | 23136.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 15:15:00 | 22872.30 | 22835.93 | 22927.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-19 09:15:00 | 22801.20 | 22835.93 | 22927.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 22878.80 | 22844.50 | 22923.00 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2024-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 14:15:00 | 23171.10 | 22975.06 | 22961.82 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 10:15:00 | 22850.00 | 22985.18 | 22994.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 11:15:00 | 22525.00 | 22893.15 | 22951.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 10:15:00 | 22820.10 | 22770.15 | 22849.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 10:15:00 | 22820.10 | 22770.15 | 22849.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 10:15:00 | 22820.10 | 22770.15 | 22849.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 11:00:00 | 22820.10 | 22770.15 | 22849.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 11:15:00 | 22526.00 | 22721.32 | 22819.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-24 12:30:00 | 22510.00 | 22686.05 | 22794.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-24 13:30:00 | 22477.60 | 22641.84 | 22764.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 10:15:00 | 22513.40 | 22551.81 | 22687.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 11:45:00 | 22501.90 | 22524.25 | 22650.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 14:15:00 | 22797.20 | 22582.57 | 22645.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 15:00:00 | 22797.20 | 22582.57 | 22645.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 15:15:00 | 22750.00 | 22616.06 | 22655.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 09:15:00 | 22880.00 | 22616.06 | 22655.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-01-29 09:15:00 | 22945.30 | 22681.91 | 22681.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2024-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 09:15:00 | 22945.30 | 22681.91 | 22681.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 10:15:00 | 23167.90 | 22779.11 | 22725.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-31 13:15:00 | 23750.00 | 23814.44 | 23568.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-31 14:00:00 | 23750.00 | 23814.44 | 23568.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 14:15:00 | 23599.00 | 23771.35 | 23571.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 15:00:00 | 23599.00 | 23771.35 | 23571.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 15:15:00 | 23530.80 | 23723.24 | 23567.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 09:15:00 | 23737.20 | 23723.24 | 23567.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 12:00:00 | 23668.60 | 23718.92 | 23604.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 14:45:00 | 23756.30 | 23701.06 | 23624.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 09:15:00 | 23926.90 | 23680.85 | 23621.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 09:15:00 | 24197.00 | 23972.49 | 23837.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-05 11:15:00 | 24319.30 | 24027.57 | 23875.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-06 09:15:00 | 24335.40 | 24142.95 | 23998.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-06 10:30:00 | 24297.80 | 24214.42 | 24057.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-13 15:15:00 | 26110.92 | 25748.19 | 25486.48 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2024-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 14:15:00 | 28000.00 | 28511.80 | 28530.98 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-26 13:15:00 | 28796.80 | 28445.88 | 28405.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-27 09:15:00 | 29188.95 | 28685.99 | 28533.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-27 13:15:00 | 28720.10 | 28813.13 | 28656.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 13:15:00 | 28720.10 | 28813.13 | 28656.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 13:15:00 | 28720.10 | 28813.13 | 28656.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 14:00:00 | 28720.10 | 28813.13 | 28656.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 10:15:00 | 28633.10 | 28826.64 | 28717.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 11:00:00 | 28633.10 | 28826.64 | 28717.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 11:15:00 | 28507.00 | 28762.71 | 28698.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 12:00:00 | 28507.00 | 28762.71 | 28698.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 12:15:00 | 28416.10 | 28693.39 | 28672.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 13:00:00 | 28416.10 | 28693.39 | 28672.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2024-02-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 14:15:00 | 28370.95 | 28602.46 | 28633.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 15:15:00 | 28300.00 | 28541.96 | 28602.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 13:15:00 | 28507.85 | 28470.86 | 28537.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 14:00:00 | 28507.85 | 28470.86 | 28537.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 14:15:00 | 28580.00 | 28492.69 | 28541.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 14:45:00 | 28527.85 | 28492.69 | 28541.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 28750.00 | 28544.15 | 28560.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:15:00 | 28964.70 | 28544.15 | 28560.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2024-03-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 09:15:00 | 29050.85 | 28645.49 | 28604.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-02 09:15:00 | 29150.00 | 28941.00 | 28802.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 09:15:00 | 28677.00 | 28922.12 | 28831.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 09:15:00 | 28677.00 | 28922.12 | 28831.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 28677.00 | 28922.12 | 28831.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 10:00:00 | 28677.00 | 28922.12 | 28831.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 10:15:00 | 28715.30 | 28880.76 | 28821.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 11:00:00 | 28715.30 | 28880.76 | 28821.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 13:15:00 | 29574.05 | 29935.31 | 29782.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-07 14:00:00 | 29574.05 | 29935.31 | 29782.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 14:15:00 | 29497.50 | 29847.75 | 29756.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-07 14:45:00 | 29500.00 | 29847.75 | 29756.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2024-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 10:15:00 | 29648.85 | 29695.99 | 29700.45 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 11:15:00 | 29933.85 | 29743.56 | 29721.67 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 14:15:00 | 29376.75 | 29672.14 | 29695.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 11:15:00 | 28950.00 | 29407.60 | 29520.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 09:15:00 | 28974.55 | 28851.23 | 29155.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 09:15:00 | 28974.55 | 28851.23 | 29155.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 28974.55 | 28851.23 | 29155.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 10:00:00 | 28974.55 | 28851.23 | 29155.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 10:15:00 | 29462.95 | 28973.57 | 29183.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 10:45:00 | 29500.00 | 28973.57 | 29183.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 11:15:00 | 29530.95 | 29085.05 | 29215.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 12:00:00 | 29530.95 | 29085.05 | 29215.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 13:15:00 | 29212.95 | 29142.73 | 29220.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-14 14:15:00 | 29184.65 | 29142.73 | 29220.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-14 14:15:00 | 29387.75 | 29191.73 | 29235.99 | SL hit (close>static) qty=1.00 sl=29354.35 alert=retest2 |

### Cycle 63 — BUY (started 2024-03-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 13:15:00 | 29677.50 | 29264.15 | 29248.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-18 11:15:00 | 29782.05 | 29478.76 | 29370.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 09:15:00 | 29664.95 | 29672.95 | 29523.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-19 09:15:00 | 29664.95 | 29672.95 | 29523.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 09:15:00 | 29664.95 | 29672.95 | 29523.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 09:45:00 | 29650.95 | 29672.95 | 29523.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 10:15:00 | 29527.75 | 29643.91 | 29524.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 10:30:00 | 29463.60 | 29643.91 | 29524.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 11:15:00 | 29673.80 | 29649.89 | 29537.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-19 14:15:00 | 29829.25 | 29657.99 | 29561.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 10:45:00 | 29750.00 | 29708.02 | 29617.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-28 10:15:00 | 29890.00 | 30341.90 | 30365.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2024-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 10:15:00 | 29890.00 | 30341.90 | 30365.85 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-02 09:15:00 | 30449.95 | 30286.12 | 30281.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 10:15:00 | 30566.00 | 30342.09 | 30307.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-04 11:15:00 | 30654.45 | 30772.36 | 30671.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-04 11:15:00 | 30654.45 | 30772.36 | 30671.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 11:15:00 | 30654.45 | 30772.36 | 30671.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 12:00:00 | 30654.45 | 30772.36 | 30671.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 12:15:00 | 30785.00 | 30774.89 | 30681.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-05 09:15:00 | 30854.00 | 30758.59 | 30695.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-08 09:15:00 | 30980.00 | 30795.74 | 30751.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-09 11:15:00 | 30400.00 | 30865.55 | 30875.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2024-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 11:15:00 | 30400.00 | 30865.55 | 30875.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 12:15:00 | 30069.30 | 30706.30 | 30802.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-12 09:15:00 | 30075.05 | 29974.53 | 30222.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 09:15:00 | 30075.05 | 29974.53 | 30222.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 30075.05 | 29974.53 | 30222.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 10:00:00 | 30075.05 | 29974.53 | 30222.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 10:15:00 | 29950.95 | 29915.00 | 30054.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 10:45:00 | 30100.00 | 29915.00 | 30054.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 29951.95 | 29884.11 | 29975.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 09:30:00 | 29958.35 | 29884.11 | 29975.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 30025.10 | 29912.31 | 29980.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:00:00 | 30025.10 | 29912.31 | 29980.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 11:15:00 | 29910.00 | 29911.85 | 29973.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 12:30:00 | 29635.00 | 29855.89 | 29942.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-18 09:15:00 | 30138.75 | 29958.42 | 29967.03 | SL hit (close>static) qty=1.00 sl=30049.00 alert=retest2 |

### Cycle 67 — BUY (started 2024-04-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 10:15:00 | 30050.30 | 29976.80 | 29974.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 12:15:00 | 30320.00 | 30076.48 | 30022.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-18 14:15:00 | 29889.80 | 30060.51 | 30025.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 14:15:00 | 29889.80 | 30060.51 | 30025.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 14:15:00 | 29889.80 | 30060.51 | 30025.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 15:00:00 | 29889.80 | 30060.51 | 30025.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 15:15:00 | 29803.20 | 30009.04 | 30005.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-19 09:15:00 | 29687.95 | 30009.04 | 30005.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2024-04-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 09:15:00 | 29644.85 | 29936.21 | 29972.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-22 13:15:00 | 29274.00 | 29585.07 | 29730.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-23 13:15:00 | 29417.85 | 29364.53 | 29516.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-23 14:00:00 | 29417.85 | 29364.53 | 29516.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 09:15:00 | 29122.65 | 29287.20 | 29441.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 09:30:00 | 29521.00 | 29287.20 | 29441.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 15:15:00 | 28907.15 | 28917.06 | 29066.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-26 10:00:00 | 29175.00 | 28968.65 | 29076.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 10:15:00 | 28984.25 | 28971.77 | 29067.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-26 12:45:00 | 28907.90 | 28985.27 | 29058.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-26 13:15:00 | 28907.30 | 28985.27 | 29058.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-26 14:30:00 | 28932.45 | 28957.81 | 29033.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-29 11:15:00 | 29450.00 | 29084.11 | 29069.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2024-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-29 11:15:00 | 29450.00 | 29084.11 | 29069.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-29 12:15:00 | 29682.05 | 29203.70 | 29125.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-30 14:15:00 | 29358.85 | 29411.55 | 29301.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-30 15:00:00 | 29358.85 | 29411.55 | 29301.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 29251.00 | 29379.44 | 29297.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 09:15:00 | 29619.90 | 29379.44 | 29297.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-06 12:15:00 | 29580.00 | 29885.54 | 29908.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 12:15:00 | 29580.00 | 29885.54 | 29908.05 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 15:15:00 | 30084.00 | 29919.71 | 29916.14 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 09:15:00 | 29779.00 | 29891.56 | 29903.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 10:15:00 | 29355.00 | 29784.25 | 29853.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 30040.40 | 29698.71 | 29754.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 09:15:00 | 30040.40 | 29698.71 | 29754.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 30040.40 | 29698.71 | 29754.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 10:00:00 | 30040.40 | 29698.71 | 29754.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 30062.40 | 29771.45 | 29782.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 11:00:00 | 30062.40 | 29771.45 | 29782.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2024-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 11:15:00 | 30079.10 | 29832.98 | 29809.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 12:15:00 | 30329.50 | 29932.28 | 29857.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 12:15:00 | 29933.80 | 30007.38 | 29942.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-09 12:15:00 | 29933.80 | 30007.38 | 29942.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 12:15:00 | 29933.80 | 30007.38 | 29942.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 12:45:00 | 29873.70 | 30007.38 | 29942.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 13:15:00 | 30070.50 | 30020.01 | 29954.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 10:15:00 | 30125.40 | 30005.50 | 29960.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 11:00:00 | 30122.50 | 30028.90 | 29975.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 12:30:00 | 30109.40 | 30058.15 | 29998.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 13:00:00 | 30138.20 | 30058.15 | 29998.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 30205.45 | 30214.48 | 30101.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 09:30:00 | 30063.35 | 30214.48 | 30101.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 10:15:00 | 30257.15 | 30223.01 | 30115.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 10:30:00 | 30068.90 | 30223.01 | 30115.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 30970.15 | 30631.50 | 30387.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-14 10:45:00 | 31276.55 | 30751.73 | 30464.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 09:15:00 | 31032.40 | 30995.02 | 30924.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-16 12:15:00 | 30731.65 | 30891.69 | 30895.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 12:15:00 | 30731.65 | 30891.69 | 30895.79 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 14:15:00 | 31087.25 | 30895.36 | 30894.56 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 09:15:00 | 30766.40 | 30871.91 | 30884.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-17 11:15:00 | 30628.90 | 30803.99 | 30849.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-17 13:15:00 | 30817.50 | 30804.45 | 30842.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-17 14:00:00 | 30817.50 | 30804.45 | 30842.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 14:15:00 | 31103.95 | 30864.35 | 30865.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 15:00:00 | 31103.95 | 30864.35 | 30865.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2024-05-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 15:15:00 | 31000.00 | 30891.48 | 30878.07 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-18 11:15:00 | 30699.95 | 30854.54 | 30863.69 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 14:15:00 | 30894.05 | 30859.95 | 30858.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 09:15:00 | 31144.45 | 30918.46 | 30885.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 11:15:00 | 30750.00 | 30930.91 | 30899.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 11:15:00 | 30750.00 | 30930.91 | 30899.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 30750.00 | 30930.91 | 30899.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 12:00:00 | 30750.00 | 30930.91 | 30899.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 30781.75 | 30901.08 | 30888.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 15:00:00 | 30909.40 | 30892.84 | 30886.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 09:15:00 | 30760.10 | 30872.24 | 30878.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 09:15:00 | 30760.10 | 30872.24 | 30878.76 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2024-05-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 12:15:00 | 31000.00 | 30876.48 | 30870.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 13:15:00 | 31205.80 | 30942.34 | 30901.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 14:15:00 | 30753.15 | 30904.50 | 30887.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 14:15:00 | 30753.15 | 30904.50 | 30887.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 30753.15 | 30904.50 | 30887.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 30753.15 | 30904.50 | 30887.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 30795.20 | 30882.64 | 30879.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 09:15:00 | 31746.40 | 30882.64 | 30879.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 15:15:00 | 30975.00 | 31110.29 | 31113.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2024-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 15:15:00 | 30975.00 | 31110.29 | 31113.79 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 09:15:00 | 31169.35 | 31122.10 | 31118.84 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 11:15:00 | 31000.00 | 31119.31 | 31119.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 10:15:00 | 30903.65 | 31050.22 | 31083.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 14:15:00 | 29399.40 | 29313.54 | 29808.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-04 15:00:00 | 29399.40 | 29313.54 | 29808.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 29000.00 | 29250.83 | 29735.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 30304.10 | 29461.49 | 29786.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 30667.25 | 29702.64 | 29866.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 30667.25 | 29702.64 | 29866.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 30882.40 | 30126.71 | 30042.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 13:15:00 | 31011.90 | 30303.74 | 30130.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 14:15:00 | 30650.65 | 30659.13 | 30461.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 14:30:00 | 30613.35 | 30659.13 | 30461.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 13:15:00 | 30599.55 | 30706.03 | 30577.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-07 13:45:00 | 30619.95 | 30706.03 | 30577.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 14:15:00 | 30600.00 | 30684.83 | 30579.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-07 15:15:00 | 30575.00 | 30684.83 | 30579.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 15:15:00 | 30575.00 | 30662.86 | 30579.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 09:15:00 | 30872.90 | 30662.86 | 30579.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 14:30:00 | 30660.00 | 30716.99 | 30645.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-18 09:15:00 | 33726.00 | 32565.53 | 32137.55 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2024-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 13:15:00 | 32279.00 | 32716.72 | 32760.04 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2024-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 14:15:00 | 33033.65 | 32690.16 | 32652.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 15:15:00 | 33095.00 | 32771.12 | 32692.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 09:15:00 | 34300.00 | 34474.06 | 34095.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-28 10:15:00 | 34201.00 | 34474.06 | 34095.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 34077.35 | 34351.62 | 34129.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 13:00:00 | 34077.35 | 34351.62 | 34129.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 13:15:00 | 34260.00 | 34333.30 | 34141.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 10:00:00 | 34379.00 | 34253.57 | 34144.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 11:15:00 | 34332.60 | 34262.34 | 34158.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 15:15:00 | 33955.00 | 34176.22 | 34155.53 | SL hit (close<static) qty=1.00 sl=34070.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 12:15:00 | 34580.25 | 34948.71 | 34963.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 13:15:00 | 34399.00 | 34838.77 | 34911.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 09:15:00 | 34854.30 | 34733.42 | 34835.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 09:15:00 | 34854.30 | 34733.42 | 34835.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 34854.30 | 34733.42 | 34835.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:15:00 | 35000.00 | 34733.42 | 34835.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 35070.00 | 34800.74 | 34856.43 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 11:15:00 | 35447.45 | 34930.08 | 34910.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 13:15:00 | 35500.05 | 35122.07 | 35005.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 10:15:00 | 35097.35 | 35261.70 | 35126.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 10:15:00 | 35097.35 | 35261.70 | 35126.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 35097.35 | 35261.70 | 35126.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 35062.80 | 35261.70 | 35126.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 34935.60 | 35196.48 | 35109.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 12:00:00 | 34935.60 | 35196.48 | 35109.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 12:15:00 | 35265.00 | 35210.18 | 35123.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 15:00:00 | 35384.95 | 35259.48 | 35161.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 13:15:00 | 35407.55 | 35378.16 | 35267.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 12:15:00 | 34973.35 | 35276.38 | 35282.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2024-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 12:15:00 | 34973.35 | 35276.38 | 35282.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 12:15:00 | 34897.10 | 35080.43 | 35141.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 13:15:00 | 35104.95 | 35085.34 | 35137.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 13:15:00 | 35104.95 | 35085.34 | 35137.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 35104.95 | 35085.34 | 35137.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:00:00 | 35104.95 | 35085.34 | 35137.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 35110.00 | 35090.27 | 35135.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 15:00:00 | 35110.00 | 35090.27 | 35135.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 35199.95 | 35112.21 | 35141.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:15:00 | 34725.40 | 35112.21 | 35141.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 34391.25 | 34968.01 | 35073.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 10:30:00 | 34302.55 | 34799.41 | 34986.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 12:15:00 | 34344.95 | 34718.53 | 34933.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 13:15:00 | 34282.90 | 34404.75 | 34596.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 12:15:00 | 34254.60 | 34160.46 | 34253.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 12:15:00 | 34277.50 | 34183.87 | 34255.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 14:45:00 | 34141.10 | 34184.42 | 34243.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-25 13:15:00 | 34585.00 | 34191.36 | 34207.68 | SL hit (close>static) qty=1.00 sl=34379.55 alert=retest2 |

### Cycle 91 — BUY (started 2024-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 14:15:00 | 34665.00 | 34286.09 | 34249.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 15:15:00 | 34699.95 | 34368.86 | 34290.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 09:15:00 | 34872.80 | 34906.28 | 34770.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 10:00:00 | 34872.80 | 34906.28 | 34770.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 34873.00 | 34905.50 | 34837.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:30:00 | 34838.20 | 34905.50 | 34837.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 34760.00 | 34876.40 | 34830.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 10:45:00 | 34818.65 | 34876.40 | 34830.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 11:15:00 | 34564.55 | 34814.03 | 34806.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:45:00 | 34616.40 | 34814.03 | 34806.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2024-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 12:15:00 | 34710.00 | 34793.22 | 34797.52 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2024-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 13:15:00 | 34899.50 | 34814.48 | 34806.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 14:15:00 | 35000.00 | 34851.58 | 34824.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 15:15:00 | 34800.00 | 34841.27 | 34822.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 15:15:00 | 34800.00 | 34841.27 | 34822.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 34800.00 | 34841.27 | 34822.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:15:00 | 34795.00 | 34841.27 | 34822.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 34783.00 | 34829.61 | 34818.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:15:00 | 34621.05 | 34829.61 | 34818.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 10:15:00 | 34609.05 | 34785.50 | 34799.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 13:15:00 | 34339.95 | 34622.20 | 34714.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 10:15:00 | 34512.05 | 34451.62 | 34592.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-02 10:45:00 | 34469.80 | 34451.62 | 34592.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 11:15:00 | 34237.70 | 34408.83 | 34560.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 12:30:00 | 34121.95 | 34352.50 | 34520.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 14:15:00 | 34141.20 | 34321.82 | 34491.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 11:15:00 | 32415.85 | 33560.48 | 34031.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 11:15:00 | 32434.14 | 33560.48 | 34031.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-06 09:15:00 | 33000.00 | 32876.90 | 33456.86 | SL hit (close>ema200) qty=0.50 sl=32876.90 alert=retest2 |

### Cycle 95 — BUY (started 2024-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 14:15:00 | 31949.95 | 31552.41 | 31521.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 15:15:00 | 32000.00 | 31641.93 | 31564.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 10:15:00 | 31698.00 | 31701.19 | 31607.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 10:15:00 | 31698.00 | 31701.19 | 31607.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 31698.00 | 31701.19 | 31607.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:30:00 | 31636.70 | 31701.19 | 31607.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 13:15:00 | 31550.00 | 31664.34 | 31613.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 14:00:00 | 31550.00 | 31664.34 | 31613.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 14:15:00 | 31600.00 | 31651.47 | 31612.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 09:15:00 | 32312.85 | 31648.91 | 31615.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 15:15:00 | 32250.00 | 32429.01 | 32444.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-08-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 15:15:00 | 32250.00 | 32429.01 | 32444.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 10:15:00 | 32162.70 | 32364.35 | 32412.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 31888.50 | 31785.12 | 31956.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 31888.50 | 31785.12 | 31956.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 31888.50 | 31785.12 | 31956.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:30:00 | 32033.75 | 31785.12 | 31956.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 32058.45 | 31839.79 | 31965.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:00:00 | 32058.45 | 31839.79 | 31965.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 32056.85 | 31883.20 | 31974.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:30:00 | 32115.00 | 31883.20 | 31974.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 13:15:00 | 32083.65 | 31936.98 | 31983.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 14:00:00 | 32083.65 | 31936.98 | 31983.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 14:15:00 | 32382.85 | 32026.15 | 32019.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 14:15:00 | 32500.00 | 32152.77 | 32094.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 14:15:00 | 32351.25 | 32383.84 | 32263.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-03 15:00:00 | 32351.25 | 32383.84 | 32263.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 32233.35 | 32356.33 | 32271.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 10:00:00 | 32233.35 | 32356.33 | 32271.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 10:15:00 | 32095.80 | 32304.22 | 32255.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 10:45:00 | 32136.50 | 32304.22 | 32255.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2024-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 12:15:00 | 32089.50 | 32225.35 | 32226.18 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 14:15:00 | 32499.00 | 32259.30 | 32240.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 09:15:00 | 32615.95 | 32364.50 | 32293.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 33099.95 | 33195.55 | 32827.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-06 10:00:00 | 33099.95 | 33195.55 | 32827.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 32529.00 | 33062.24 | 32800.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:00:00 | 32529.00 | 33062.24 | 32800.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 32728.05 | 32995.40 | 32793.55 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2024-09-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 15:15:00 | 32350.00 | 32646.08 | 32673.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 32233.80 | 32563.63 | 32633.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 10:15:00 | 32623.35 | 32575.57 | 32632.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 10:15:00 | 32623.35 | 32575.57 | 32632.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 32623.35 | 32575.57 | 32632.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 10:30:00 | 32658.15 | 32575.57 | 32632.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 11:15:00 | 32577.65 | 32575.99 | 32627.23 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2024-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 13:15:00 | 33050.00 | 32680.24 | 32666.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 09:15:00 | 33562.45 | 32971.52 | 32813.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 09:15:00 | 33326.30 | 33395.42 | 33162.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 09:15:00 | 33326.30 | 33395.42 | 33162.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 33326.30 | 33395.42 | 33162.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 09:45:00 | 33118.00 | 33395.42 | 33162.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 33186.00 | 33400.07 | 33246.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 33186.00 | 33400.07 | 33246.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 33096.35 | 33339.33 | 33232.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 33096.35 | 33339.33 | 33232.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 33323.00 | 33336.06 | 33241.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:15:00 | 33701.00 | 33336.06 | 33241.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 12:15:00 | 34100.00 | 34309.84 | 34314.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2024-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 12:15:00 | 34100.00 | 34309.84 | 34314.64 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 15:15:00 | 34450.00 | 34337.20 | 34324.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 09:15:00 | 34900.75 | 34449.91 | 34376.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 11:15:00 | 36655.90 | 36682.34 | 36284.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 11:45:00 | 36646.60 | 36682.34 | 36284.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 37793.30 | 37787.99 | 37660.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 10:45:00 | 37990.00 | 37828.38 | 37690.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 13:15:00 | 37191.30 | 37594.71 | 37610.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2024-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 13:15:00 | 37191.30 | 37594.71 | 37610.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 09:15:00 | 36606.45 | 37265.99 | 37444.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 10:15:00 | 36895.70 | 36860.91 | 37093.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-07 11:00:00 | 36895.70 | 36860.91 | 37093.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 11:15:00 | 37016.60 | 36892.05 | 37086.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-07 12:00:00 | 37016.60 | 36892.05 | 37086.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 12:15:00 | 36721.80 | 36858.00 | 37053.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 14:00:00 | 36651.00 | 36816.60 | 37017.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 14:45:00 | 36643.80 | 36813.49 | 36997.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 10:15:00 | 37424.20 | 36975.04 | 37027.60 | SL hit (close>static) qty=1.00 sl=37125.00 alert=retest2 |

### Cycle 105 — BUY (started 2024-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 12:15:00 | 37706.25 | 37180.00 | 37115.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 13:15:00 | 37843.85 | 37312.77 | 37181.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 38195.45 | 38506.30 | 38177.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 12:15:00 | 38195.45 | 38506.30 | 38177.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 38195.45 | 38506.30 | 38177.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:00:00 | 38195.45 | 38506.30 | 38177.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 38068.35 | 38418.71 | 38167.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 15:15:00 | 38432.70 | 38388.15 | 38176.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 14:15:00 | 38421.45 | 38281.35 | 38213.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 13:15:00 | 38355.20 | 38535.53 | 38473.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 10:45:00 | 38321.30 | 38503.06 | 38486.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 11:15:00 | 38263.25 | 38455.10 | 38466.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 11:15:00 | 38263.25 | 38455.10 | 38466.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 37028.40 | 38073.07 | 38275.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 10:15:00 | 37149.95 | 36889.89 | 37148.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 10:15:00 | 37149.95 | 36889.89 | 37148.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 37149.95 | 36889.89 | 37148.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 11:00:00 | 37149.95 | 36889.89 | 37148.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 36935.05 | 36898.92 | 37128.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:15:00 | 36922.20 | 36898.92 | 37128.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:45:00 | 36784.05 | 36902.85 | 37073.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-29 14:15:00 | 36426.05 | 36088.62 | 36060.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 36426.05 | 36088.62 | 36060.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 11:15:00 | 36611.00 | 36255.99 | 36154.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 36300.00 | 36301.42 | 36194.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 13:30:00 | 36299.95 | 36301.42 | 36194.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 36157.70 | 36272.67 | 36191.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 15:00:00 | 36157.70 | 36272.67 | 36191.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 36055.00 | 36229.14 | 36178.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:15:00 | 35812.10 | 36229.14 | 36178.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2024-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 09:15:00 | 35408.70 | 36065.05 | 36108.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 11:15:00 | 35161.65 | 35780.54 | 35965.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 35430.45 | 35355.02 | 35644.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-01 18:00:00 | 35430.45 | 35355.02 | 35644.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 13:15:00 | 35391.60 | 35184.94 | 35408.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-04 14:00:00 | 35391.60 | 35184.94 | 35408.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 14:15:00 | 35227.75 | 35193.50 | 35392.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-04 15:15:00 | 35106.80 | 35193.50 | 35392.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 15:15:00 | 35106.80 | 35176.16 | 35366.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:45:00 | 35570.05 | 35251.58 | 35383.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 35771.05 | 35355.47 | 35418.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 11:00:00 | 35771.05 | 35355.47 | 35418.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 35239.80 | 35332.34 | 35402.13 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 15:15:00 | 35600.00 | 35452.19 | 35439.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 36262.50 | 35614.25 | 35514.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 35599.80 | 35959.06 | 35790.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 35599.80 | 35959.06 | 35790.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 35599.80 | 35959.06 | 35790.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 35599.80 | 35959.06 | 35790.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 35362.85 | 35839.81 | 35752.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:00:00 | 35362.85 | 35839.81 | 35752.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2024-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 12:15:00 | 35394.15 | 35669.44 | 35684.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 10:15:00 | 35218.65 | 35472.88 | 35572.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 35349.95 | 35271.50 | 35397.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 35349.95 | 35271.50 | 35397.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 35349.95 | 35271.50 | 35397.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 10:30:00 | 35438.50 | 35271.50 | 35397.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 35280.25 | 35273.25 | 35386.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:30:00 | 35410.10 | 35273.25 | 35386.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 34251.10 | 34953.61 | 35183.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 10:30:00 | 34150.00 | 34778.60 | 35082.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 12:15:00 | 34089.90 | 33887.14 | 33865.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 12:15:00 | 34089.90 | 33887.14 | 33865.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 10:15:00 | 34499.95 | 34069.61 | 33958.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 34229.75 | 34322.37 | 34137.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 15:00:00 | 34229.75 | 34322.37 | 34137.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 34114.70 | 34253.26 | 34136.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 13:45:00 | 34419.90 | 34275.77 | 34201.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 09:15:00 | 34734.40 | 34291.30 | 34221.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 14:15:00 | 34607.45 | 34797.38 | 34820.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2024-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 14:15:00 | 34607.45 | 34797.38 | 34820.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 11:15:00 | 34515.95 | 34674.37 | 34750.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 12:15:00 | 34715.00 | 34682.49 | 34747.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-29 13:00:00 | 34715.00 | 34682.49 | 34747.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 34977.00 | 34741.39 | 34768.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 14:00:00 | 34977.00 | 34741.39 | 34768.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 34955.10 | 34784.14 | 34785.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 15:15:00 | 34856.00 | 34784.14 | 34785.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2024-11-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 15:15:00 | 34856.00 | 34798.51 | 34791.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 09:15:00 | 35058.95 | 34905.29 | 34863.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 11:15:00 | 34778.75 | 34910.61 | 34875.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 11:15:00 | 34778.75 | 34910.61 | 34875.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 11:15:00 | 34778.75 | 34910.61 | 34875.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 12:00:00 | 34778.75 | 34910.61 | 34875.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 12:15:00 | 34657.00 | 34859.88 | 34855.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 13:00:00 | 34657.00 | 34859.88 | 34855.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2024-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 13:15:00 | 34730.00 | 34833.91 | 34844.18 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2024-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 14:15:00 | 34950.00 | 34857.13 | 34853.80 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2024-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 15:15:00 | 34783.75 | 34842.45 | 34847.43 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2024-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 10:15:00 | 35063.00 | 34886.36 | 34866.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 11:15:00 | 35149.00 | 34938.89 | 34892.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 14:15:00 | 36272.95 | 36405.09 | 35991.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-06 15:00:00 | 36272.95 | 36405.09 | 35991.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 36009.75 | 36289.00 | 36008.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:00:00 | 36009.75 | 36289.00 | 36008.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 36052.25 | 36241.65 | 36012.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 11:15:00 | 35992.60 | 36241.65 | 36012.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 11:15:00 | 35870.00 | 36167.32 | 35999.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 12:00:00 | 35870.00 | 36167.32 | 35999.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 12:15:00 | 35837.50 | 36101.36 | 35984.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 12:30:00 | 35831.00 | 36101.36 | 35984.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 35805.05 | 35969.00 | 35946.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:15:00 | 35724.35 | 35969.00 | 35946.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 09:15:00 | 35774.20 | 35930.04 | 35930.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 10:15:00 | 35625.15 | 35869.06 | 35902.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 14:15:00 | 35964.45 | 35777.33 | 35838.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 14:15:00 | 35964.45 | 35777.33 | 35838.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 35964.45 | 35777.33 | 35838.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 15:00:00 | 35964.45 | 35777.33 | 35838.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 36019.95 | 35825.86 | 35854.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:15:00 | 36100.15 | 35825.86 | 35854.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2024-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 09:15:00 | 36239.90 | 35908.67 | 35889.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 15:15:00 | 36330.00 | 36099.05 | 36002.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 09:15:00 | 36029.20 | 36085.08 | 36005.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 09:15:00 | 36029.20 | 36085.08 | 36005.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 36029.20 | 36085.08 | 36005.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:30:00 | 36044.85 | 36085.08 | 36005.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 36055.55 | 36079.17 | 36009.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:30:00 | 36010.00 | 36079.17 | 36009.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 35983.40 | 36131.92 | 36077.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:00:00 | 35983.40 | 36131.92 | 36077.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 36034.30 | 36112.39 | 36073.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:30:00 | 35885.00 | 36112.39 | 36073.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 36149.90 | 36119.90 | 36080.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 12:45:00 | 36275.00 | 36165.12 | 36104.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-17 12:15:00 | 36250.00 | 36372.25 | 36322.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 13:15:00 | 35947.35 | 36235.02 | 36265.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 13:15:00 | 35947.35 | 36235.02 | 36265.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 14:15:00 | 35740.35 | 36136.08 | 36218.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 09:15:00 | 34516.00 | 34481.31 | 34701.96 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 13:00:00 | 34397.35 | 34461.17 | 34638.12 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 34668.00 | 34395.67 | 34541.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-27 09:15:00 | 34668.00 | 34395.67 | 34541.13 | SL hit (close>ema400) qty=1.00 sl=34541.13 alert=retest1 |

### Cycle 121 — BUY (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 10:15:00 | 34068.05 | 34038.14 | 34036.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 11:15:00 | 34417.05 | 34113.92 | 34070.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 10:15:00 | 34303.95 | 34388.22 | 34246.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 11:00:00 | 34303.95 | 34388.22 | 34246.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 34239.70 | 34358.59 | 34278.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 34239.70 | 34358.59 | 34278.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 34280.00 | 34342.87 | 34278.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 34199.05 | 34342.87 | 34278.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 34188.60 | 34312.02 | 34270.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 34008.55 | 34312.02 | 34270.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 33759.85 | 34201.58 | 34223.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 33598.30 | 33953.24 | 34094.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 33130.50 | 32991.42 | 33290.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 14:00:00 | 33130.50 | 32991.42 | 33290.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 32999.95 | 33031.00 | 33217.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:30:00 | 33200.00 | 33031.00 | 33217.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 32405.95 | 32839.94 | 33038.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 12:30:00 | 32299.85 | 32653.79 | 32896.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 13:45:00 | 32185.00 | 32535.03 | 32820.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 14:15:00 | 30684.86 | 30937.50 | 31099.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 15:15:00 | 30575.75 | 30874.00 | 31056.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-22 14:15:00 | 30614.95 | 30599.06 | 30817.95 | SL hit (close>ema200) qty=0.50 sl=30599.06 alert=retest2 |

### Cycle 123 — BUY (started 2025-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 13:15:00 | 30950.10 | 30888.24 | 30887.60 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 09:15:00 | 30701.35 | 30875.90 | 30884.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 11:15:00 | 30535.65 | 30778.17 | 30836.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 29868.90 | 29828.60 | 30103.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 29868.90 | 29828.60 | 30103.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 29845.00 | 29829.53 | 30034.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 15:15:00 | 29614.60 | 29829.53 | 30034.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 09:15:00 | 28133.87 | 29564.30 | 29876.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-30 10:15:00 | 28590.95 | 28517.13 | 29027.76 | SL hit (close>ema200) qty=0.50 sl=28517.13 alert=retest2 |

### Cycle 125 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 27292.00 | 27027.04 | 27018.86 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 27005.20 | 27214.42 | 27236.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 11:15:00 | 26947.50 | 27161.04 | 27210.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 27109.90 | 27083.92 | 27146.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 27109.90 | 27083.92 | 27146.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 27109.90 | 27083.92 | 27146.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 09:30:00 | 26869.00 | 27089.64 | 27120.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 11:45:00 | 26835.00 | 27036.65 | 27090.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 13:30:00 | 26900.00 | 26993.60 | 27061.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 11:15:00 | 26911.50 | 26574.39 | 26568.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 26911.50 | 26574.39 | 26568.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 27080.00 | 26745.44 | 26653.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 27077.25 | 27099.06 | 26957.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 27077.25 | 27099.06 | 26957.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 27010.15 | 27081.28 | 26962.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:15:00 | 26888.05 | 27081.28 | 26962.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 26872.25 | 27039.47 | 26954.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:45:00 | 26859.95 | 27039.47 | 26954.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 26982.00 | 27027.98 | 26956.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:45:00 | 27190.30 | 27046.90 | 26976.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 13:15:00 | 26852.00 | 26976.09 | 26962.87 | SL hit (close<static) qty=1.00 sl=26871.35 alert=retest2 |

### Cycle 128 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 26839.75 | 26948.82 | 26951.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 26700.00 | 26899.06 | 26928.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 13:15:00 | 26814.65 | 26755.04 | 26833.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 13:15:00 | 26814.65 | 26755.04 | 26833.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 26814.65 | 26755.04 | 26833.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 14:00:00 | 26814.65 | 26755.04 | 26833.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 27002.65 | 26804.56 | 26848.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 15:00:00 | 27002.65 | 26804.56 | 26848.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 26999.00 | 26843.45 | 26862.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 26897.00 | 26843.45 | 26862.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 26838.65 | 26747.54 | 26792.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 14:30:00 | 26821.35 | 26747.54 | 26792.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 26800.00 | 26758.03 | 26792.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 09:15:00 | 26585.00 | 26758.03 | 26792.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 26906.00 | 26519.53 | 26558.49 | SL hit (close>static) qty=1.00 sl=26838.65 alert=retest2 |

### Cycle 129 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 26881.50 | 26591.93 | 26587.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 12:15:00 | 27049.10 | 26736.22 | 26657.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 14:15:00 | 27608.90 | 27625.88 | 27396.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 15:00:00 | 27608.90 | 27625.88 | 27396.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 27592.95 | 27661.74 | 27517.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 13:45:00 | 27546.50 | 27661.74 | 27517.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 27402.80 | 27609.95 | 27506.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 15:00:00 | 27402.80 | 27609.95 | 27506.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 27574.30 | 27602.82 | 27513.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:15:00 | 27671.00 | 27602.82 | 27513.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 15:15:00 | 27425.00 | 27714.62 | 27750.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 15:15:00 | 27425.00 | 27714.62 | 27750.76 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 10:15:00 | 28010.00 | 27791.12 | 27780.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-26 12:15:00 | 28053.40 | 27870.60 | 27820.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 09:15:00 | 28136.50 | 28188.67 | 28063.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-28 09:45:00 | 28186.05 | 28188.67 | 28063.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 28200.70 | 28277.74 | 28155.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:00:00 | 28200.70 | 28277.74 | 28155.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 28366.85 | 28295.56 | 28174.53 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 11:15:00 | 27923.25 | 28115.38 | 28121.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 09:15:00 | 27354.30 | 27807.85 | 27906.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 26769.90 | 26722.52 | 27090.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 26819.55 | 26722.52 | 27090.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 26579.00 | 26693.81 | 27044.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 26432.00 | 26671.89 | 27002.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:30:00 | 26506.10 | 26649.51 | 26962.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 13:45:00 | 26500.55 | 26643.35 | 26904.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 15:00:00 | 26499.90 | 26614.66 | 26867.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 26616.75 | 26341.82 | 26529.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 09:30:00 | 26681.05 | 26341.82 | 26529.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 10:15:00 | 26779.70 | 26429.40 | 26552.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 11:00:00 | 26779.70 | 26429.40 | 26552.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 13:15:00 | 26351.05 | 26458.01 | 26538.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 13:30:00 | 26505.05 | 26458.01 | 26538.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 26820.00 | 26508.98 | 26539.78 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-15 11:15:00 | 26760.00 | 26598.55 | 26577.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2025-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 11:15:00 | 26760.00 | 26598.55 | 26577.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 13:15:00 | 26825.00 | 26674.47 | 26617.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 27160.00 | 27241.38 | 27045.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 27160.00 | 27241.38 | 27045.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 27160.00 | 27241.38 | 27045.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:30:00 | 27180.00 | 27241.38 | 27045.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 28060.00 | 28277.38 | 28146.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 28060.00 | 28277.38 | 28146.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 28025.00 | 28226.90 | 28135.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 28025.00 | 28226.90 | 28135.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 28025.00 | 28186.52 | 28125.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 12:45:00 | 28115.00 | 28170.22 | 28123.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 13:30:00 | 28110.00 | 28144.17 | 28115.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 09:45:00 | 28150.00 | 28109.82 | 28103.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 11:15:00 | 28115.00 | 28100.85 | 28100.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 28220.00 | 28124.68 | 28111.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 12:15:00 | 28250.00 | 28124.68 | 28111.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 14:00:00 | 28255.00 | 28181.20 | 28141.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 14:45:00 | 28285.00 | 28200.96 | 28153.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 29715.00 | 30161.70 | 30197.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 29715.00 | 30161.70 | 30197.87 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 30885.00 | 30209.99 | 30182.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 30980.00 | 30532.80 | 30352.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 14:15:00 | 31380.00 | 31391.35 | 31121.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 14:45:00 | 31355.00 | 31391.35 | 31121.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 31550.00 | 31681.90 | 31591.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 13:00:00 | 31550.00 | 31681.90 | 31591.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 31505.00 | 31646.52 | 31583.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 31505.00 | 31646.52 | 31583.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 31625.00 | 31629.17 | 31585.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 31620.00 | 31624.34 | 31587.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 31815.00 | 31662.47 | 31607.99 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 31460.00 | 31589.07 | 31593.02 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 31760.00 | 31623.25 | 31608.20 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 31490.00 | 31610.99 | 31614.55 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 14:15:00 | 31740.00 | 31614.76 | 31610.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 15:15:00 | 31840.00 | 31659.80 | 31631.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 12:15:00 | 31975.00 | 32086.66 | 31941.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 12:15:00 | 31975.00 | 32086.66 | 31941.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 31975.00 | 32086.66 | 31941.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:45:00 | 31940.00 | 32086.66 | 31941.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 32490.00 | 32167.33 | 31991.03 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 10:15:00 | 31370.00 | 32067.85 | 32118.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 13:15:00 | 31190.00 | 31556.53 | 31755.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 14:15:00 | 31410.00 | 31315.89 | 31492.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-30 15:00:00 | 31410.00 | 31315.89 | 31492.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 31385.00 | 31329.71 | 31482.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 10:00:00 | 31175.00 | 31298.77 | 31454.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 11:15:00 | 31255.00 | 31295.01 | 31438.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 12:45:00 | 31260.00 | 31300.61 | 31416.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 15:15:00 | 31210.00 | 31314.39 | 31403.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 31375.00 | 31309.81 | 31384.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 11:30:00 | 31215.00 | 31289.48 | 31362.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 10:15:00 | 31605.00 | 31375.94 | 31372.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 31605.00 | 31375.94 | 31372.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 11:15:00 | 31780.00 | 31456.75 | 31409.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 14:15:00 | 31440.00 | 31527.30 | 31460.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 14:15:00 | 31440.00 | 31527.30 | 31460.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 31440.00 | 31527.30 | 31460.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 31440.00 | 31527.30 | 31460.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 31330.00 | 31487.84 | 31448.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:15:00 | 31285.00 | 31487.84 | 31448.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 31350.00 | 31460.27 | 31439.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 31745.00 | 31557.76 | 31515.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 11:00:00 | 31700.00 | 31623.36 | 31555.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 12:15:00 | 31700.00 | 31623.69 | 31561.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 13:30:00 | 31685.00 | 31637.16 | 31578.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 31755.00 | 31670.19 | 31609.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:45:00 | 31680.00 | 31670.19 | 31609.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 31655.00 | 31733.82 | 31659.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:30:00 | 31685.00 | 31733.82 | 31659.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 31655.00 | 31718.05 | 31658.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 14:15:00 | 31595.00 | 31718.05 | 31658.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 31560.00 | 31686.44 | 31649.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 31560.00 | 31686.44 | 31649.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 31490.00 | 31647.15 | 31635.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 31760.00 | 31647.15 | 31635.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 14:15:00 | 31610.00 | 31787.51 | 31754.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 15:15:00 | 31500.00 | 31687.21 | 31712.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 31500.00 | 31687.21 | 31712.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 31260.00 | 31601.76 | 31671.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 31605.00 | 31510.30 | 31587.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 14:15:00 | 31605.00 | 31510.30 | 31587.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 31605.00 | 31510.30 | 31587.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 31605.00 | 31510.30 | 31587.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 31600.00 | 31528.24 | 31588.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:30:00 | 31485.00 | 31548.59 | 31592.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 10:15:00 | 31970.00 | 31632.87 | 31626.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2025-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 10:15:00 | 31970.00 | 31632.87 | 31626.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 11:15:00 | 32020.00 | 31710.30 | 31662.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 12:15:00 | 32060.00 | 32138.44 | 31972.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 13:00:00 | 32060.00 | 32138.44 | 31972.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 32270.00 | 32416.97 | 32294.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 32270.00 | 32416.97 | 32294.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 32350.00 | 32403.58 | 32299.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 15:15:00 | 32400.00 | 32387.86 | 32301.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 31600.00 | 32170.92 | 32246.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 09:15:00 | 31600.00 | 32170.92 | 32246.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-23 15:15:00 | 31540.00 | 31772.94 | 31983.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 09:15:00 | 31750.00 | 31646.00 | 31790.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 09:15:00 | 31750.00 | 31646.00 | 31790.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 31750.00 | 31646.00 | 31790.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 09:45:00 | 31890.00 | 31646.00 | 31790.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 31785.00 | 31673.80 | 31790.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 11:00:00 | 31785.00 | 31673.80 | 31790.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 31790.00 | 31697.04 | 31790.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:00:00 | 31790.00 | 31697.04 | 31790.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 31750.00 | 31707.63 | 31786.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 14:45:00 | 31650.00 | 31717.88 | 31777.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-26 09:30:00 | 31600.00 | 31695.25 | 31757.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-26 12:45:00 | 31645.00 | 31719.69 | 31753.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 13:15:00 | 31840.00 | 31743.75 | 31761.39 | SL hit (close>static) qty=1.00 sl=31830.00 alert=retest2 |

### Cycle 145 — BUY (started 2025-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 14:15:00 | 32240.00 | 31843.00 | 31804.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 32335.00 | 31982.52 | 31877.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 32475.00 | 32562.18 | 32378.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 10:00:00 | 32475.00 | 32562.18 | 32378.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 32175.00 | 32484.74 | 32359.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 32175.00 | 32484.74 | 32359.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 32280.00 | 32443.80 | 32352.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 12:15:00 | 32305.00 | 32443.80 | 32352.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 13:30:00 | 32310.00 | 32390.43 | 32342.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 12:15:00 | 32305.00 | 32342.95 | 32334.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 14:00:00 | 32295.00 | 32333.69 | 32331.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 32455.00 | 32357.95 | 32342.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:15:00 | 32615.00 | 32381.36 | 32354.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-04 09:15:00 | 35535.50 | 34138.52 | 33421.50 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 09:15:00 | 38035.00 | 38254.65 | 38273.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 10:15:00 | 37765.00 | 38156.72 | 38226.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 37940.00 | 37793.71 | 37910.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 14:15:00 | 37940.00 | 37793.71 | 37910.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 37940.00 | 37793.71 | 37910.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:45:00 | 37975.00 | 37793.71 | 37910.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 37980.00 | 37830.97 | 37917.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 38090.00 | 37830.97 | 37917.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 38005.00 | 37942.93 | 37949.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 14:15:00 | 37910.00 | 37941.35 | 37948.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:30:00 | 37805.00 | 37881.65 | 37919.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 38170.00 | 37881.89 | 37884.84 | SL hit (close>static) qty=1.00 sl=38095.00 alert=retest2 |

### Cycle 147 — BUY (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 10:15:00 | 38245.00 | 37954.51 | 37917.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 12:15:00 | 39550.00 | 38416.88 | 38187.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 39760.00 | 39801.11 | 39307.00 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 10:30:00 | 40050.00 | 39843.88 | 39371.36 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 40155.00 | 40280.11 | 40048.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:00:00 | 40155.00 | 40280.11 | 40048.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 40600.00 | 40848.13 | 40525.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-05 10:15:00 | 40420.00 | 40762.50 | 40515.92 | SL hit (close<ema400) qty=1.00 sl=40515.92 alert=retest1 |

### Cycle 148 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 38685.00 | 40285.50 | 40395.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 11:15:00 | 38520.00 | 39662.72 | 40076.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 15:15:00 | 38690.00 | 38579.90 | 39015.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-08 09:15:00 | 38940.00 | 38579.90 | 39015.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 38850.00 | 38633.92 | 39000.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 11:15:00 | 38505.00 | 38625.14 | 38962.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 11:45:00 | 38540.00 | 38609.11 | 38924.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:30:00 | 38530.00 | 38554.34 | 38823.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 13:15:00 | 38550.00 | 38689.80 | 38727.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 38655.00 | 38682.84 | 38721.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:45:00 | 38760.00 | 38682.84 | 38721.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 38725.00 | 38691.27 | 38721.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 38725.00 | 38691.27 | 38721.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 38700.00 | 38693.02 | 38719.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 38790.00 | 38693.02 | 38719.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 38545.00 | 38663.41 | 38703.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 11:00:00 | 38470.00 | 38624.73 | 38682.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 12:00:00 | 38490.00 | 38597.78 | 38665.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 15:15:00 | 38490.00 | 38618.23 | 38660.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 39580.00 | 38790.06 | 38729.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 39580.00 | 38790.06 | 38729.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 39665.00 | 38965.05 | 38814.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 15:15:00 | 39890.00 | 39901.00 | 39692.84 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:15:00 | 40000.00 | 39901.00 | 39692.84 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 39680.00 | 39845.31 | 39717.22 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-21 11:15:00 | 39680.00 | 39845.31 | 39717.22 | SL hit (close<ema400) qty=1.00 sl=39717.22 alert=retest1 |

### Cycle 150 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 39500.00 | 39659.65 | 39660.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 12:15:00 | 39230.00 | 39518.98 | 39591.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 39510.00 | 39421.34 | 39505.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 10:15:00 | 39510.00 | 39421.34 | 39505.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 39510.00 | 39421.34 | 39505.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 39510.00 | 39421.34 | 39505.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 39485.00 | 39434.07 | 39503.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 14:15:00 | 39130.00 | 39435.81 | 39492.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 14:45:00 | 39340.00 | 39288.22 | 39343.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 09:15:00 | 39785.00 | 39407.06 | 39389.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 09:15:00 | 39785.00 | 39407.06 | 39389.15 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 15:15:00 | 39205.00 | 39399.95 | 39409.96 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 10:15:00 | 39825.00 | 39472.97 | 39440.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 11:15:00 | 40145.00 | 39607.37 | 39504.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 41225.00 | 41276.00 | 40904.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 12:45:00 | 41185.00 | 41276.00 | 40904.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 41115.00 | 41206.63 | 41006.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:45:00 | 41160.00 | 41206.63 | 41006.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 40970.00 | 41159.30 | 41002.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:45:00 | 40945.00 | 41159.30 | 41002.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 40785.00 | 41084.44 | 40983.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:00:00 | 40785.00 | 41084.44 | 40983.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 40605.00 | 40988.55 | 40948.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 40605.00 | 40988.55 | 40948.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 40800.00 | 40906.27 | 40915.47 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 10:15:00 | 41010.00 | 40929.22 | 40924.44 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 12:15:00 | 40895.00 | 40920.90 | 40921.39 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 13:15:00 | 40950.00 | 40926.72 | 40923.99 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 14:15:00 | 40890.00 | 40919.37 | 40920.90 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 41465.00 | 41025.40 | 40968.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 10:15:00 | 41625.00 | 41145.32 | 41028.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 41215.00 | 41294.92 | 41174.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 10:00:00 | 41215.00 | 41294.92 | 41174.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 41275.00 | 41290.94 | 41183.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 13:15:00 | 41380.00 | 41273.60 | 41192.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 10:45:00 | 41455.00 | 41356.56 | 41277.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 12:15:00 | 41010.00 | 41263.00 | 41247.45 | SL hit (close<static) qty=1.00 sl=41145.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 13:15:00 | 41045.00 | 41219.40 | 41229.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 09:15:00 | 40735.00 | 41068.93 | 41152.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 39805.00 | 39664.51 | 40005.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 10:15:00 | 39955.00 | 39722.61 | 40000.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 39955.00 | 39722.61 | 40000.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 39955.00 | 39722.61 | 40000.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 39860.00 | 39786.47 | 39983.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 14:30:00 | 39765.00 | 39885.78 | 39942.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:45:00 | 39775.00 | 39843.58 | 39886.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 11:15:00 | 39735.00 | 39777.73 | 39810.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 12:00:00 | 39750.00 | 39772.18 | 39804.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 39440.00 | 39610.63 | 39706.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 39160.00 | 39513.05 | 39606.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:15:00 | 37776.75 | 38126.80 | 38471.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:15:00 | 37786.25 | 38126.80 | 38471.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:15:00 | 37748.25 | 38126.80 | 38471.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:15:00 | 37762.50 | 38126.80 | 38471.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 13:15:00 | 38540.00 | 38193.15 | 38440.43 | SL hit (close>ema200) qty=0.50 sl=38193.15 alert=retest2 |

### Cycle 161 — BUY (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 13:15:00 | 38325.00 | 38261.99 | 38261.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 38650.00 | 38339.59 | 38296.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 12:15:00 | 38765.00 | 38795.70 | 38629.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 12:45:00 | 38820.00 | 38795.70 | 38629.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 38775.00 | 38801.27 | 38685.69 | EMA400 retest candle locked (from upside) |

### Cycle 162 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 38300.00 | 38571.41 | 38606.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 38245.00 | 38464.30 | 38549.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 38460.00 | 38397.15 | 38477.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 14:15:00 | 38460.00 | 38397.15 | 38477.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 38460.00 | 38397.15 | 38477.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 38460.00 | 38397.15 | 38477.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 38495.00 | 38416.72 | 38479.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 38565.00 | 38416.72 | 38479.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 38790.00 | 38491.38 | 38507.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 38795.00 | 38491.38 | 38507.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 38805.00 | 38554.10 | 38534.67 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 15:15:00 | 38450.00 | 38564.85 | 38576.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 37900.00 | 38431.88 | 38514.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 38325.00 | 38101.95 | 38246.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 38325.00 | 38101.95 | 38246.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 38325.00 | 38101.95 | 38246.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 38265.00 | 38101.95 | 38246.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 38265.00 | 38134.56 | 38248.28 | EMA400 retest candle locked (from downside) |

### Cycle 165 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 38425.00 | 38317.36 | 38309.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 38525.00 | 38367.31 | 38334.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 11:15:00 | 38390.00 | 38401.08 | 38357.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 12:00:00 | 38390.00 | 38401.08 | 38357.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 38550.00 | 38430.86 | 38374.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:30:00 | 38375.00 | 38430.86 | 38374.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 38370.00 | 38443.08 | 38396.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 38445.00 | 38443.08 | 38396.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 38490.00 | 38452.47 | 38405.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 10:45:00 | 38545.00 | 38475.97 | 38420.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 13:45:00 | 38600.00 | 38496.34 | 38443.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 10:15:00 | 38570.00 | 38934.80 | 38953.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 38570.00 | 38934.80 | 38953.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 12:15:00 | 38475.00 | 38783.67 | 38877.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 10:15:00 | 38650.00 | 38628.43 | 38752.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 11:00:00 | 38650.00 | 38628.43 | 38752.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 38755.00 | 38653.74 | 38752.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:45:00 | 38720.00 | 38653.74 | 38752.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 38790.00 | 38680.99 | 38755.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:45:00 | 38820.00 | 38680.99 | 38755.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 38845.00 | 38713.79 | 38763.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:45:00 | 38865.00 | 38713.79 | 38763.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 39025.00 | 38776.03 | 38787.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 15:00:00 | 39025.00 | 38776.03 | 38787.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — BUY (started 2025-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 15:15:00 | 39000.00 | 38820.83 | 38806.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 39150.00 | 38886.66 | 38838.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 11:15:00 | 38750.00 | 38887.86 | 38848.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 11:15:00 | 38750.00 | 38887.86 | 38848.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 38750.00 | 38887.86 | 38848.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 38750.00 | 38887.86 | 38848.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — SELL (started 2025-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 12:15:00 | 38550.00 | 38820.29 | 38821.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 13:15:00 | 38500.00 | 38756.23 | 38792.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 12:15:00 | 37260.00 | 37197.71 | 37468.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-31 13:00:00 | 37260.00 | 37197.71 | 37468.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 37315.00 | 37054.12 | 37135.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:00:00 | 37315.00 | 37054.12 | 37135.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 37650.00 | 37173.30 | 37182.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:00:00 | 37650.00 | 37173.30 | 37182.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 14:15:00 | 37865.00 | 37311.64 | 37244.19 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 36900.00 | 37171.65 | 37200.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 36670.00 | 37021.27 | 37113.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 09:15:00 | 36920.00 | 36856.14 | 36959.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 09:15:00 | 36920.00 | 36856.14 | 36959.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 36920.00 | 36856.14 | 36959.36 | EMA400 retest candle locked (from downside) |

### Cycle 171 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 37295.00 | 37033.67 | 37022.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 13:15:00 | 37470.00 | 37120.93 | 37063.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 10:15:00 | 36645.00 | 37081.54 | 37070.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 10:15:00 | 36645.00 | 37081.54 | 37070.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 36645.00 | 37081.54 | 37070.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 36645.00 | 37081.54 | 37070.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — SELL (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 11:15:00 | 36900.00 | 37045.23 | 37055.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 09:15:00 | 36235.00 | 36701.43 | 36867.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 13:15:00 | 37075.00 | 36648.74 | 36775.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 13:15:00 | 37075.00 | 36648.74 | 36775.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 37075.00 | 36648.74 | 36775.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:00:00 | 37075.00 | 36648.74 | 36775.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 37260.00 | 36770.99 | 36819.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 15:00:00 | 37260.00 | 36770.99 | 36819.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — BUY (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 09:15:00 | 37325.00 | 36944.03 | 36893.70 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 36800.00 | 36987.23 | 37001.10 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 37065.00 | 36995.97 | 36995.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 11:15:00 | 37190.00 | 37034.78 | 37013.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 36900.00 | 37176.60 | 37111.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 36900.00 | 37176.60 | 37111.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 36900.00 | 37176.60 | 37111.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 36900.00 | 37176.60 | 37111.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 36840.00 | 37109.28 | 37087.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:45:00 | 36865.00 | 37109.28 | 37087.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — SELL (started 2025-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 11:15:00 | 36905.00 | 37068.43 | 37070.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 10:15:00 | 36720.00 | 36959.75 | 37014.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 37090.00 | 36919.44 | 36958.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 37090.00 | 36919.44 | 36958.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 37090.00 | 36919.44 | 36958.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 36965.00 | 36919.44 | 36958.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 37090.00 | 36953.55 | 36970.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 37090.00 | 36953.55 | 36970.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 37060.00 | 36989.02 | 36983.64 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 36790.00 | 36970.78 | 36978.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 12:15:00 | 36660.00 | 36849.84 | 36916.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 36175.00 | 35967.32 | 36185.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 36175.00 | 35967.32 | 36185.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 36175.00 | 35967.32 | 36185.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 36175.00 | 35967.32 | 36185.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 36170.00 | 36007.85 | 36184.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 36170.00 | 36007.85 | 36184.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 36295.00 | 36065.28 | 36194.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:45:00 | 36315.00 | 36065.28 | 36194.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 36325.00 | 36117.23 | 36206.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:30:00 | 36345.00 | 36117.23 | 36206.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — BUY (started 2025-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 15:15:00 | 36295.00 | 36264.62 | 36261.33 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 36180.00 | 36250.96 | 36255.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 11:15:00 | 36135.00 | 36227.77 | 36244.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 13:15:00 | 36215.00 | 36197.57 | 36226.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 13:15:00 | 36215.00 | 36197.57 | 36226.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 36215.00 | 36197.57 | 36226.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:00:00 | 36215.00 | 36197.57 | 36226.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 36300.00 | 36218.06 | 36233.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 15:00:00 | 36300.00 | 36218.06 | 36233.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 36300.00 | 36234.44 | 36239.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:15:00 | 36240.00 | 36234.44 | 36239.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — BUY (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 10:15:00 | 36295.00 | 36251.44 | 36246.45 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 36135.00 | 36228.16 | 36236.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 15:15:00 | 36075.00 | 36177.44 | 36208.82 | Break + close below crossover candle low |

### Cycle 183 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 36445.00 | 36230.96 | 36230.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 09:15:00 | 36660.00 | 36368.36 | 36301.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 36350.00 | 36543.80 | 36451.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 36350.00 | 36543.80 | 36451.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 36350.00 | 36543.80 | 36451.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 36350.00 | 36543.80 | 36451.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 36215.00 | 36478.04 | 36430.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:00:00 | 36215.00 | 36478.04 | 36430.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 36525.00 | 36463.03 | 36432.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:45:00 | 36490.00 | 36463.03 | 36432.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 36615.00 | 36493.43 | 36448.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 14:45:00 | 36510.00 | 36493.43 | 36448.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 36880.00 | 36846.06 | 36710.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 10:15:00 | 37000.00 | 36846.06 | 36710.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 10:15:00 | 36565.00 | 36782.09 | 36757.86 | SL hit (close<static) qty=1.00 sl=36595.00 alert=retest2 |

### Cycle 184 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 36420.00 | 36685.39 | 36717.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 10:15:00 | 36385.00 | 36560.90 | 36643.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 36705.00 | 36589.72 | 36649.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 11:15:00 | 36705.00 | 36589.72 | 36649.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 36705.00 | 36589.72 | 36649.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 36705.00 | 36589.72 | 36649.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 36725.00 | 36616.78 | 36656.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:15:00 | 36670.00 | 36650.42 | 36668.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 15:15:00 | 36840.00 | 36709.87 | 36693.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 36840.00 | 36709.87 | 36693.36 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 10:15:00 | 36525.00 | 36680.92 | 36683.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 36405.00 | 36570.30 | 36625.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 36800.00 | 36592.99 | 36624.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 36800.00 | 36592.99 | 36624.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 36800.00 | 36592.99 | 36624.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:00:00 | 36800.00 | 36592.99 | 36624.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 36730.00 | 36620.40 | 36634.34 | EMA400 retest candle locked (from downside) |

### Cycle 187 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 36865.00 | 36669.32 | 36655.31 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 12:15:00 | 36400.00 | 36639.14 | 36670.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 36120.00 | 36409.53 | 36518.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 14:15:00 | 36245.00 | 36237.27 | 36374.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-16 15:00:00 | 36245.00 | 36237.27 | 36374.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 36270.00 | 36242.65 | 36353.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 12:00:00 | 36025.00 | 36189.90 | 36309.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 10:15:00 | 35940.00 | 35881.93 | 35905.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 14:15:00 | 36100.00 | 35935.38 | 35920.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — BUY (started 2025-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 14:15:00 | 36100.00 | 35935.38 | 35920.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 14:15:00 | 36300.00 | 36104.93 | 36021.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 14:15:00 | 36190.00 | 36251.27 | 36155.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 14:15:00 | 36190.00 | 36251.27 | 36155.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 36190.00 | 36251.27 | 36155.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:45:00 | 36205.00 | 36251.27 | 36155.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 36150.00 | 36231.02 | 36155.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 36215.00 | 36231.02 | 36155.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 36110.00 | 36206.81 | 36151.26 | EMA400 retest candle locked (from upside) |

### Cycle 190 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 35855.00 | 36077.82 | 36104.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 35690.00 | 35917.04 | 36017.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 14:15:00 | 35570.00 | 35496.00 | 35679.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 15:00:00 | 35570.00 | 35496.00 | 35679.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 35745.00 | 35545.80 | 35685.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 35880.00 | 35545.80 | 35685.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 35825.00 | 35601.64 | 35698.10 | EMA400 retest candle locked (from downside) |

### Cycle 191 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 36000.00 | 35778.65 | 35767.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 37215.00 | 36261.49 | 36050.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 13:15:00 | 38935.00 | 39078.72 | 38588.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 14:00:00 | 38935.00 | 39078.72 | 38588.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 38580.00 | 38943.18 | 38645.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:45:00 | 38220.00 | 38943.18 | 38645.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 38460.00 | 38846.55 | 38628.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 38460.00 | 38846.55 | 38628.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 38630.00 | 38803.24 | 38628.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 12:30:00 | 38900.00 | 38869.59 | 38674.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 13:00:00 | 38765.00 | 38787.56 | 38740.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 14:30:00 | 38770.00 | 38758.04 | 38734.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 15:15:00 | 38545.00 | 38715.43 | 38717.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 38545.00 | 38715.43 | 38717.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 38360.00 | 38630.51 | 38675.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 37610.00 | 37571.96 | 37969.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 37745.00 | 37571.96 | 37969.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 37420.00 | 37554.05 | 37893.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:30:00 | 37200.00 | 37614.01 | 37746.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 35340.00 | 35771.47 | 36288.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 35745.00 | 35324.44 | 35736.88 | SL hit (close>ema200) qty=0.50 sl=35324.44 alert=retest2 |

### Cycle 193 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 35710.00 | 35448.59 | 35417.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 36200.00 | 35598.87 | 35488.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 36230.00 | 36291.61 | 36099.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 12:00:00 | 36230.00 | 36291.61 | 36099.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 36090.00 | 36251.29 | 36099.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 35990.00 | 36251.29 | 36099.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 36100.00 | 36221.03 | 36099.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 14:15:00 | 36195.00 | 36221.03 | 36099.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 11:30:00 | 36185.00 | 36119.93 | 36088.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 12:30:00 | 36140.00 | 36164.94 | 36112.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 36310.00 | 36911.02 | 36939.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 36310.00 | 36911.02 | 36939.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 15:15:00 | 36100.00 | 36447.26 | 36659.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 36440.00 | 36174.27 | 36389.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 14:15:00 | 36440.00 | 36174.27 | 36389.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 36440.00 | 36174.27 | 36389.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 36440.00 | 36174.27 | 36389.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 36275.00 | 36194.41 | 36379.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 36385.00 | 36194.41 | 36379.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 36250.00 | 36205.53 | 36367.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 10:45:00 | 36105.00 | 36194.42 | 36347.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 11:15:00 | 36075.00 | 36194.42 | 36347.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 11:45:00 | 36085.00 | 36159.54 | 36317.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 13:30:00 | 35950.00 | 36080.31 | 36254.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 35695.00 | 35713.72 | 35901.46 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-11 13:15:00 | 36470.00 | 36060.61 | 36021.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — BUY (started 2026-02-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 13:15:00 | 36470.00 | 36060.61 | 36021.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 14:15:00 | 36635.00 | 36175.49 | 36077.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 35900.00 | 36364.31 | 36284.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 35900.00 | 36364.31 | 36284.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 35900.00 | 36364.31 | 36284.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 36000.00 | 36364.31 | 36284.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 35890.00 | 36269.45 | 36248.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 11:15:00 | 35850.00 | 36269.45 | 36248.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 11:15:00 | 35990.00 | 36213.56 | 36224.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 35715.00 | 36055.22 | 36145.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 13:15:00 | 35540.00 | 35390.56 | 35513.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 13:15:00 | 35540.00 | 35390.56 | 35513.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 35540.00 | 35390.56 | 35513.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:00:00 | 35540.00 | 35390.56 | 35513.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 35710.00 | 35454.45 | 35531.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 35710.00 | 35454.45 | 35531.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 35760.00 | 35515.56 | 35551.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 35835.00 | 35515.56 | 35551.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 35475.00 | 35532.57 | 35552.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:30:00 | 35615.00 | 35532.57 | 35552.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 35510.00 | 35528.05 | 35548.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:00:00 | 35510.00 | 35528.05 | 35548.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 35385.00 | 35229.78 | 35372.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:00:00 | 35385.00 | 35229.78 | 35372.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 35640.00 | 35311.83 | 35396.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:00:00 | 35640.00 | 35311.83 | 35396.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 35395.00 | 35328.46 | 35396.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 35605.00 | 35328.46 | 35396.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 35260.00 | 35314.77 | 35384.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:45:00 | 35170.00 | 35269.79 | 35338.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:45:00 | 35170.00 | 35252.83 | 35324.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:00:00 | 35170.00 | 35236.26 | 35310.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 35085.00 | 35264.77 | 35306.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 35080.00 | 35227.81 | 35285.81 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 36035.00 | 35363.00 | 35301.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 36035.00 | 35363.00 | 35301.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 10:15:00 | 37250.00 | 35740.40 | 35479.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 10:15:00 | 36315.00 | 36363.61 | 36003.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 10:45:00 | 36215.00 | 36363.61 | 36003.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 36160.00 | 36459.26 | 36222.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 36160.00 | 36459.26 | 36222.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 36055.00 | 36378.41 | 36207.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:00:00 | 36055.00 | 36378.41 | 36207.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 36125.00 | 36327.73 | 36199.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:30:00 | 36035.00 | 36327.73 | 36199.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 36200.00 | 36302.18 | 36199.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 13:45:00 | 36330.00 | 36299.75 | 36208.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 14:15:00 | 36300.00 | 36299.75 | 36208.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 14:45:00 | 36450.00 | 36328.80 | 36229.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 35840.00 | 36242.43 | 36208.29 | SL hit (close<static) qty=1.00 sl=36125.00 alert=retest2 |

### Cycle 198 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 35600.00 | 36113.94 | 36153.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 35295.00 | 35950.16 | 36075.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 11:15:00 | 33265.00 | 33260.88 | 33800.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 11:45:00 | 33315.00 | 33260.88 | 33800.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 32030.00 | 32018.70 | 32338.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 11:00:00 | 31935.00 | 32001.96 | 32302.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 14:15:00 | 30338.25 | 30711.81 | 31166.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 14:15:00 | 30370.00 | 30305.66 | 30688.98 | SL hit (close>ema200) qty=0.50 sl=30305.66 alert=retest2 |

### Cycle 199 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 31035.00 | 30736.84 | 30721.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 31125.00 | 30867.78 | 30787.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 30315.00 | 30821.00 | 30800.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 30315.00 | 30821.00 | 30800.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 30315.00 | 30821.00 | 30800.06 | EMA400 retest candle locked (from upside) |

### Cycle 200 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 30315.00 | 30719.80 | 30755.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 11:15:00 | 30240.00 | 30623.84 | 30709.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 30625.00 | 30422.66 | 30553.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 30625.00 | 30422.66 | 30553.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 30625.00 | 30422.66 | 30553.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 30625.00 | 30422.66 | 30553.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 30670.00 | 30472.12 | 30564.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 30670.00 | 30472.12 | 30564.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 30585.00 | 30494.70 | 30565.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 30680.00 | 30494.70 | 30565.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 30680.00 | 30531.76 | 30576.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:00:00 | 30680.00 | 30531.76 | 30576.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 30620.00 | 30549.41 | 30580.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:30:00 | 30640.00 | 30549.41 | 30580.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 30405.00 | 30520.53 | 30564.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 29790.00 | 30504.42 | 30553.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 30470.00 | 29979.33 | 29976.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 30470.00 | 29979.33 | 29976.65 | EMA200 above EMA400 |

### Cycle 202 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 29985.00 | 30105.86 | 30107.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 29600.00 | 29981.40 | 30048.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 30530.00 | 29358.09 | 29552.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 30530.00 | 29358.09 | 29552.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 30530.00 | 29358.09 | 29552.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 30530.00 | 29358.09 | 29552.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 30825.00 | 29651.47 | 29668.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:45:00 | 31030.00 | 29651.47 | 29668.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 30870.00 | 29895.18 | 29777.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 11:15:00 | 31515.00 | 30616.24 | 30259.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 32740.00 | 32849.90 | 32110.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 32740.00 | 32849.90 | 32110.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 37040.00 | 37168.98 | 36475.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 37240.00 | 37106.18 | 36510.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 10:30:00 | 37240.00 | 37056.61 | 36862.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 14:30:00 | 37265.00 | 37139.56 | 36962.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 37485.00 | 37904.28 | 37923.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 37485.00 | 37904.28 | 37923.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 37405.00 | 37673.44 | 37797.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 37330.00 | 36910.62 | 37195.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 37330.00 | 36910.62 | 37195.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 37330.00 | 36910.62 | 37195.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 37330.00 | 36910.62 | 37195.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 37190.00 | 36966.50 | 37195.32 | EMA400 retest candle locked (from downside) |

### Cycle 205 — BUY (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 12:15:00 | 37345.00 | 37251.30 | 37249.70 | EMA200 above EMA400 |

### Cycle 206 — SELL (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 15:15:00 | 37020.00 | 37216.35 | 37235.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 10:15:00 | 36830.00 | 37123.66 | 37189.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 15:15:00 | 36145.00 | 36115.50 | 36445.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 09:15:00 | 36440.00 | 36115.50 | 36445.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 36285.00 | 36149.40 | 36430.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:30:00 | 36020.00 | 36150.52 | 36405.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:15:00 | 36085.00 | 36150.52 | 36405.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 11:30:00 | 36110.00 | 36004.04 | 36148.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 12:15:00 | 36125.00 | 36004.04 | 36148.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 36175.00 | 36038.23 | 36150.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:30:00 | 36215.00 | 36038.23 | 36150.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 36035.00 | 36037.58 | 36140.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 14:15:00 | 35890.00 | 36037.58 | 36140.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 10:30:00 | 35985.00 | 35986.85 | 36079.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 36230.00 | 36035.48 | 36093.12 | SL hit (close>static) qty=1.00 sl=36200.00 alert=retest2 |

### Cycle 207 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 36690.00 | 36194.25 | 36153.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 37915.00 | 36610.52 | 36354.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 15:15:00 | 38050.00 | 38096.21 | 37639.57 | EMA200 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-17 11:30:00 | 19069.40 | 2023-05-18 10:15:00 | 19216.90 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2023-05-23 11:45:00 | 18910.00 | 2023-05-26 09:15:00 | 18940.40 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2023-05-23 13:00:00 | 18946.90 | 2023-05-26 11:15:00 | 18926.60 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2023-05-23 13:45:00 | 18935.00 | 2023-05-26 11:15:00 | 18926.60 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2023-05-23 14:15:00 | 18921.10 | 2023-05-26 11:15:00 | 18926.60 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2023-05-23 15:15:00 | 18822.10 | 2023-05-26 11:15:00 | 18926.60 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2023-05-24 09:45:00 | 18834.90 | 2023-05-26 11:15:00 | 18926.60 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2023-05-24 10:15:00 | 18841.00 | 2023-05-26 11:15:00 | 18926.60 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2023-05-24 12:45:00 | 18842.40 | 2023-05-26 11:15:00 | 18926.60 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2023-05-25 11:30:00 | 18699.60 | 2023-05-26 11:15:00 | 18926.60 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2023-06-15 09:45:00 | 19174.60 | 2023-06-15 14:15:00 | 19015.20 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2023-06-15 12:45:00 | 19106.20 | 2023-06-15 14:15:00 | 19015.20 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2023-06-15 13:30:00 | 19103.80 | 2023-06-15 14:15:00 | 19015.20 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2023-06-16 09:30:00 | 19135.10 | 2023-06-16 13:15:00 | 18990.00 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2023-07-05 10:30:00 | 19195.60 | 2023-07-10 15:15:00 | 19215.90 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2023-07-05 11:00:00 | 19214.90 | 2023-07-10 15:15:00 | 19215.90 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2023-07-05 15:00:00 | 19207.30 | 2023-07-10 15:15:00 | 19215.90 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2023-07-06 12:15:00 | 19194.40 | 2023-07-10 15:15:00 | 19215.90 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest1 | 2023-07-17 10:30:00 | 18886.60 | 2023-07-18 09:15:00 | 19639.40 | STOP_HIT | 1.00 | -3.99% |
| SELL | retest1 | 2023-07-17 12:15:00 | 18924.60 | 2023-07-18 09:15:00 | 19639.40 | STOP_HIT | 1.00 | -3.78% |
| SELL | retest2 | 2023-07-21 09:15:00 | 18977.10 | 2023-07-25 10:15:00 | 18993.80 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2023-07-21 14:00:00 | 18910.90 | 2023-07-25 10:15:00 | 18993.80 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2023-08-08 10:45:00 | 18139.60 | 2023-08-09 13:15:00 | 18310.10 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2023-08-08 14:30:00 | 18150.10 | 2023-08-09 13:15:00 | 18310.10 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2023-08-09 09:15:00 | 18139.30 | 2023-08-09 13:15:00 | 18310.10 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2023-08-16 13:45:00 | 18075.00 | 2023-08-17 09:15:00 | 18206.70 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2023-08-24 12:15:00 | 18403.20 | 2023-08-25 14:15:00 | 18284.90 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2023-08-24 13:00:00 | 18455.00 | 2023-08-25 14:15:00 | 18284.90 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2023-08-25 12:00:00 | 18395.00 | 2023-08-25 14:15:00 | 18284.90 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2023-08-28 10:00:00 | 18420.20 | 2023-09-12 12:15:00 | 19090.00 | STOP_HIT | 1.00 | 3.64% |
| BUY | retest2 | 2023-08-31 13:30:00 | 18654.50 | 2023-09-12 12:15:00 | 19090.00 | STOP_HIT | 1.00 | 2.33% |
| BUY | retest2 | 2023-09-01 09:15:00 | 18637.90 | 2023-09-12 12:15:00 | 19090.00 | STOP_HIT | 1.00 | 2.43% |
| SELL | retest2 | 2023-09-27 12:30:00 | 19025.00 | 2023-09-27 14:15:00 | 19213.50 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2023-10-10 09:15:00 | 19378.40 | 2023-10-18 10:15:00 | 20265.90 | STOP_HIT | 1.00 | 4.58% |
| SELL | retest2 | 2023-10-20 09:30:00 | 20085.90 | 2023-11-03 15:15:00 | 19540.00 | STOP_HIT | 1.00 | 2.72% |
| SELL | retest2 | 2023-10-20 10:15:00 | 20136.20 | 2023-11-03 15:15:00 | 19540.00 | STOP_HIT | 1.00 | 2.96% |
| BUY | retest2 | 2023-11-07 13:30:00 | 19718.40 | 2023-11-08 12:15:00 | 19448.50 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2023-11-23 12:45:00 | 21217.70 | 2023-12-06 09:15:00 | 21701.10 | STOP_HIT | 1.00 | 2.28% |
| BUY | retest2 | 2023-11-23 15:00:00 | 21201.30 | 2023-12-06 09:15:00 | 21701.10 | STOP_HIT | 1.00 | 2.36% |
| BUY | retest2 | 2023-11-24 10:30:00 | 21204.10 | 2023-12-06 09:15:00 | 21701.10 | STOP_HIT | 1.00 | 2.34% |
| BUY | retest2 | 2023-11-24 11:15:00 | 21240.30 | 2023-12-06 09:15:00 | 21701.10 | STOP_HIT | 1.00 | 2.17% |
| BUY | retest2 | 2023-11-28 15:00:00 | 21312.20 | 2023-12-06 09:15:00 | 21701.10 | STOP_HIT | 1.00 | 1.82% |
| SELL | retest2 | 2023-12-07 14:30:00 | 21608.50 | 2023-12-11 09:15:00 | 21776.20 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2023-12-08 09:15:00 | 21634.40 | 2023-12-11 09:15:00 | 21776.20 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2023-12-08 10:15:00 | 21620.10 | 2023-12-11 09:15:00 | 21776.20 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2023-12-08 11:00:00 | 21595.90 | 2023-12-11 09:15:00 | 21776.20 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2023-12-13 09:15:00 | 21831.20 | 2023-12-13 11:15:00 | 21632.70 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2023-12-13 10:00:00 | 21812.20 | 2023-12-13 11:15:00 | 21632.70 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2023-12-20 09:15:00 | 22001.00 | 2023-12-20 12:15:00 | 21697.80 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2023-12-22 12:00:00 | 21651.60 | 2023-12-26 12:15:00 | 21761.70 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-01-03 11:30:00 | 22301.00 | 2024-01-17 09:15:00 | 22950.10 | STOP_HIT | 1.00 | 2.91% |
| BUY | retest2 | 2024-01-03 13:30:00 | 22285.70 | 2024-01-17 09:15:00 | 22950.10 | STOP_HIT | 1.00 | 2.98% |
| BUY | retest2 | 2024-01-03 14:00:00 | 22279.20 | 2024-01-17 09:15:00 | 22950.10 | STOP_HIT | 1.00 | 3.01% |
| BUY | retest2 | 2024-01-04 11:30:00 | 22348.70 | 2024-01-17 09:15:00 | 22950.10 | STOP_HIT | 1.00 | 2.69% |
| BUY | retest2 | 2024-01-09 09:15:00 | 22632.10 | 2024-01-17 09:15:00 | 22950.10 | STOP_HIT | 1.00 | 1.41% |
| BUY | retest2 | 2024-01-10 13:00:00 | 22621.70 | 2024-01-17 09:15:00 | 22950.10 | STOP_HIT | 1.00 | 1.45% |
| BUY | retest2 | 2024-01-10 13:45:00 | 22615.20 | 2024-01-17 09:15:00 | 22950.10 | STOP_HIT | 1.00 | 1.48% |
| SELL | retest2 | 2024-01-24 12:30:00 | 22510.00 | 2024-01-29 09:15:00 | 22945.30 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2024-01-24 13:30:00 | 22477.60 | 2024-01-29 09:15:00 | 22945.30 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-01-25 10:15:00 | 22513.40 | 2024-01-29 09:15:00 | 22945.30 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-01-25 11:45:00 | 22501.90 | 2024-01-29 09:15:00 | 22945.30 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-02-01 09:15:00 | 23737.20 | 2024-02-13 15:15:00 | 26110.92 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-01 12:00:00 | 23668.60 | 2024-02-13 15:15:00 | 26035.46 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-01 14:45:00 | 23756.30 | 2024-02-13 15:15:00 | 26131.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-02 09:15:00 | 23926.90 | 2024-02-13 15:15:00 | 26319.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-05 11:15:00 | 24319.30 | 2024-02-13 15:15:00 | 26751.23 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-06 09:15:00 | 24335.40 | 2024-02-13 15:15:00 | 26768.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-06 10:30:00 | 24297.80 | 2024-02-13 15:15:00 | 26727.58 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-03-14 14:15:00 | 29184.65 | 2024-03-14 14:15:00 | 29387.75 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-03-15 09:30:00 | 29128.85 | 2024-03-15 13:15:00 | 29677.50 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-03-15 10:15:00 | 29083.95 | 2024-03-15 13:15:00 | 29677.50 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-03-19 14:15:00 | 29829.25 | 2024-03-28 10:15:00 | 29890.00 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2024-03-20 10:45:00 | 29750.00 | 2024-03-28 10:15:00 | 29890.00 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest2 | 2024-04-05 09:15:00 | 30854.00 | 2024-04-09 11:15:00 | 30400.00 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-04-08 09:15:00 | 30980.00 | 2024-04-09 11:15:00 | 30400.00 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2024-04-16 12:30:00 | 29635.00 | 2024-04-18 09:15:00 | 30138.75 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-04-26 12:45:00 | 28907.90 | 2024-04-29 11:15:00 | 29450.00 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-04-26 13:15:00 | 28907.30 | 2024-04-29 11:15:00 | 29450.00 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-04-26 14:30:00 | 28932.45 | 2024-04-29 11:15:00 | 29450.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-05-02 09:15:00 | 29619.90 | 2024-05-06 12:15:00 | 29580.00 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2024-05-10 10:15:00 | 30125.40 | 2024-05-16 12:15:00 | 30731.65 | STOP_HIT | 1.00 | 2.01% |
| BUY | retest2 | 2024-05-10 11:00:00 | 30122.50 | 2024-05-16 12:15:00 | 30731.65 | STOP_HIT | 1.00 | 2.02% |
| BUY | retest2 | 2024-05-10 12:30:00 | 30109.40 | 2024-05-16 12:15:00 | 30731.65 | STOP_HIT | 1.00 | 2.07% |
| BUY | retest2 | 2024-05-10 13:00:00 | 30138.20 | 2024-05-16 12:15:00 | 30731.65 | STOP_HIT | 1.00 | 1.97% |
| BUY | retest2 | 2024-05-14 10:45:00 | 31276.55 | 2024-05-16 12:15:00 | 30731.65 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-05-16 09:15:00 | 31032.40 | 2024-05-16 12:15:00 | 30731.65 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-05-22 15:00:00 | 30909.40 | 2024-05-23 09:15:00 | 30760.10 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-05-27 09:15:00 | 31746.40 | 2024-05-28 15:15:00 | 30975.00 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2024-06-10 09:15:00 | 30872.90 | 2024-06-18 09:15:00 | 33726.00 | TARGET_HIT | 1.00 | 9.24% |
| BUY | retest2 | 2024-06-10 14:30:00 | 30660.00 | 2024-06-18 10:15:00 | 33960.19 | TARGET_HIT | 1.00 | 10.76% |
| BUY | retest2 | 2024-07-01 10:00:00 | 34379.00 | 2024-07-01 15:15:00 | 33955.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-07-01 11:15:00 | 34332.60 | 2024-07-01 15:15:00 | 33955.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-07-02 09:15:00 | 34394.00 | 2024-07-08 12:15:00 | 34580.25 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2024-07-10 15:00:00 | 35384.95 | 2024-07-12 12:15:00 | 34973.35 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-07-11 13:15:00 | 35407.55 | 2024-07-12 12:15:00 | 34973.35 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-07-19 10:30:00 | 34302.55 | 2024-07-25 13:15:00 | 34585.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-07-19 12:15:00 | 34344.95 | 2024-07-25 14:15:00 | 34665.00 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-07-22 13:15:00 | 34282.90 | 2024-07-25 14:15:00 | 34665.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-07-24 12:15:00 | 34254.60 | 2024-07-25 14:15:00 | 34665.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-07-24 14:45:00 | 34141.10 | 2024-07-25 14:15:00 | 34665.00 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-08-02 12:30:00 | 34121.95 | 2024-08-05 11:15:00 | 32415.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-02 14:15:00 | 34141.20 | 2024-08-05 11:15:00 | 32434.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-02 12:30:00 | 34121.95 | 2024-08-06 09:15:00 | 33000.00 | STOP_HIT | 0.50 | 3.29% |
| SELL | retest2 | 2024-08-02 14:15:00 | 34141.20 | 2024-08-06 09:15:00 | 33000.00 | STOP_HIT | 0.50 | 3.34% |
| BUY | retest2 | 2024-08-20 09:15:00 | 32312.85 | 2024-08-27 15:15:00 | 32250.00 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2024-09-12 09:15:00 | 33701.00 | 2024-09-19 12:15:00 | 34100.00 | STOP_HIT | 1.00 | 1.18% |
| BUY | retest2 | 2024-10-03 10:45:00 | 37990.00 | 2024-10-03 13:15:00 | 37191.30 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2024-10-07 14:00:00 | 36651.00 | 2024-10-08 10:15:00 | 37424.20 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2024-10-07 14:45:00 | 36643.80 | 2024-10-08 10:15:00 | 37424.20 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-10-10 15:15:00 | 38432.70 | 2024-10-16 11:15:00 | 38263.25 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2024-10-11 14:15:00 | 38421.45 | 2024-10-16 11:15:00 | 38263.25 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-10-15 13:15:00 | 38355.20 | 2024-10-16 11:15:00 | 38263.25 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2024-10-16 10:45:00 | 38321.30 | 2024-10-16 11:15:00 | 38263.25 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2024-10-21 12:15:00 | 36922.20 | 2024-10-29 14:15:00 | 36426.05 | STOP_HIT | 1.00 | 1.34% |
| SELL | retest2 | 2024-10-21 14:45:00 | 36784.05 | 2024-10-29 14:15:00 | 36426.05 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2024-11-12 10:30:00 | 34150.00 | 2024-11-18 12:15:00 | 34089.90 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2024-11-22 13:45:00 | 34419.90 | 2024-11-28 14:15:00 | 34607.45 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2024-11-25 09:15:00 | 34734.40 | 2024-11-28 14:15:00 | 34607.45 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2024-12-13 12:45:00 | 36275.00 | 2024-12-17 13:15:00 | 35947.35 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-12-17 12:15:00 | 36250.00 | 2024-12-17 13:15:00 | 35947.35 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest1 | 2024-12-26 13:00:00 | 34397.35 | 2024-12-27 09:15:00 | 34668.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-12-27 11:45:00 | 34139.75 | 2025-01-02 10:15:00 | 34068.05 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2024-12-27 14:45:00 | 34040.05 | 2025-01-02 10:15:00 | 34068.05 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2024-12-31 15:00:00 | 34101.90 | 2025-01-02 10:15:00 | 34068.05 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-01-10 12:30:00 | 32299.85 | 2025-01-21 14:15:00 | 30684.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 13:45:00 | 32185.00 | 2025-01-21 15:15:00 | 30575.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 12:30:00 | 32299.85 | 2025-01-22 14:15:00 | 30614.95 | STOP_HIT | 0.50 | 5.22% |
| SELL | retest2 | 2025-01-10 13:45:00 | 32185.00 | 2025-01-22 14:15:00 | 30614.95 | STOP_HIT | 0.50 | 4.88% |
| SELL | retest2 | 2025-01-28 15:15:00 | 29614.60 | 2025-01-29 09:15:00 | 28133.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-28 15:15:00 | 29614.60 | 2025-01-30 10:15:00 | 28590.95 | STOP_HIT | 0.50 | 3.46% |
| SELL | retest2 | 2025-02-27 09:30:00 | 26869.00 | 2025-03-05 11:15:00 | 26911.50 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-02-27 11:45:00 | 26835.00 | 2025-03-05 11:15:00 | 26911.50 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-02-27 13:30:00 | 26900.00 | 2025-03-05 11:15:00 | 26911.50 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-03-10 09:45:00 | 27190.30 | 2025-03-10 13:15:00 | 26852.00 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-03-13 09:15:00 | 26585.00 | 2025-03-18 09:15:00 | 26906.00 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-03-24 09:15:00 | 27671.00 | 2025-03-25 15:15:00 | 27425.00 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-04-08 10:30:00 | 26432.00 | 2025-04-15 11:15:00 | 26760.00 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-04-08 11:30:00 | 26506.10 | 2025-04-15 11:15:00 | 26760.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-04-08 13:45:00 | 26500.55 | 2025-04-15 11:15:00 | 26760.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-04-08 15:00:00 | 26499.90 | 2025-04-15 11:15:00 | 26760.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-04-25 12:45:00 | 28115.00 | 2025-05-09 09:15:00 | 29715.00 | STOP_HIT | 1.00 | 5.69% |
| BUY | retest2 | 2025-04-25 13:30:00 | 28110.00 | 2025-05-09 09:15:00 | 29715.00 | STOP_HIT | 1.00 | 5.71% |
| BUY | retest2 | 2025-04-28 09:45:00 | 28150.00 | 2025-05-09 09:15:00 | 29715.00 | STOP_HIT | 1.00 | 5.56% |
| BUY | retest2 | 2025-04-28 11:15:00 | 28115.00 | 2025-05-09 09:15:00 | 29715.00 | STOP_HIT | 1.00 | 5.69% |
| BUY | retest2 | 2025-04-28 12:15:00 | 28250.00 | 2025-05-09 09:15:00 | 29715.00 | STOP_HIT | 1.00 | 5.19% |
| BUY | retest2 | 2025-04-28 14:00:00 | 28255.00 | 2025-05-09 09:15:00 | 29715.00 | STOP_HIT | 1.00 | 5.17% |
| BUY | retest2 | 2025-04-28 14:45:00 | 28285.00 | 2025-05-09 09:15:00 | 29715.00 | STOP_HIT | 1.00 | 5.06% |
| SELL | retest2 | 2025-06-02 10:00:00 | 31175.00 | 2025-06-04 10:15:00 | 31605.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-06-02 11:15:00 | 31255.00 | 2025-06-04 10:15:00 | 31605.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-06-02 12:45:00 | 31260.00 | 2025-06-04 10:15:00 | 31605.00 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-06-02 15:15:00 | 31210.00 | 2025-06-04 10:15:00 | 31605.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-06-03 11:30:00 | 31215.00 | 2025-06-04 10:15:00 | 31605.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-06-09 09:15:00 | 31745.00 | 2025-06-12 15:15:00 | 31500.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-06-09 11:00:00 | 31700.00 | 2025-06-12 15:15:00 | 31500.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-06-09 12:15:00 | 31700.00 | 2025-06-12 15:15:00 | 31500.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-06-09 13:30:00 | 31685.00 | 2025-06-12 15:15:00 | 31500.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-06-11 09:15:00 | 31760.00 | 2025-06-12 15:15:00 | 31500.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-06-12 14:15:00 | 31610.00 | 2025-06-12 15:15:00 | 31500.00 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-06-16 09:30:00 | 31485.00 | 2025-06-16 10:15:00 | 31970.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-06-19 15:15:00 | 32400.00 | 2025-06-23 09:15:00 | 31600.00 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-06-25 14:45:00 | 31650.00 | 2025-06-26 13:15:00 | 31840.00 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-06-26 09:30:00 | 31600.00 | 2025-06-26 13:15:00 | 31840.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-06-26 12:45:00 | 31645.00 | 2025-06-26 13:15:00 | 31840.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-07-01 12:15:00 | 32305.00 | 2025-07-04 09:15:00 | 35535.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-01 13:30:00 | 32310.00 | 2025-07-04 09:15:00 | 35541.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-02 12:15:00 | 32305.00 | 2025-07-04 09:15:00 | 35535.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-02 14:00:00 | 32295.00 | 2025-07-04 09:15:00 | 35524.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-03 09:15:00 | 32615.00 | 2025-07-04 14:15:00 | 35876.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-24 14:15:00 | 37910.00 | 2025-07-28 09:15:00 | 38170.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-07-25 09:30:00 | 37805.00 | 2025-07-28 09:15:00 | 38170.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest1 | 2025-07-31 10:30:00 | 40050.00 | 2025-08-05 10:15:00 | 40420.00 | STOP_HIT | 1.00 | 0.92% |
| SELL | retest2 | 2025-08-08 11:15:00 | 38505.00 | 2025-08-18 09:15:00 | 39580.00 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-08-08 11:45:00 | 38540.00 | 2025-08-18 09:15:00 | 39580.00 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-08-08 14:30:00 | 38530.00 | 2025-08-18 09:15:00 | 39580.00 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-08-13 13:15:00 | 38550.00 | 2025-08-18 09:15:00 | 39580.00 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-08-14 11:00:00 | 38470.00 | 2025-08-18 09:15:00 | 39580.00 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2025-08-14 12:00:00 | 38490.00 | 2025-08-18 09:15:00 | 39580.00 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-08-14 15:15:00 | 38490.00 | 2025-08-18 09:15:00 | 39580.00 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest1 | 2025-08-21 09:15:00 | 40000.00 | 2025-08-21 11:15:00 | 39680.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-08-25 14:15:00 | 39130.00 | 2025-08-28 09:15:00 | 39785.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-08-26 14:45:00 | 39340.00 | 2025-08-28 09:15:00 | 39785.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-09-09 13:15:00 | 41380.00 | 2025-09-10 12:15:00 | 41010.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-09-10 10:45:00 | 41455.00 | 2025-09-10 12:15:00 | 41010.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-09-17 14:30:00 | 39765.00 | 2025-09-29 11:15:00 | 37776.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 10:45:00 | 39775.00 | 2025-09-29 11:15:00 | 37786.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 11:15:00 | 39735.00 | 2025-09-29 11:15:00 | 37748.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 12:00:00 | 39750.00 | 2025-09-29 11:15:00 | 37762.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 14:30:00 | 39765.00 | 2025-09-29 13:15:00 | 38540.00 | STOP_HIT | 0.50 | 3.08% |
| SELL | retest2 | 2025-09-19 10:45:00 | 39775.00 | 2025-09-29 13:15:00 | 38540.00 | STOP_HIT | 0.50 | 3.10% |
| SELL | retest2 | 2025-09-22 11:15:00 | 39735.00 | 2025-09-29 13:15:00 | 38540.00 | STOP_HIT | 0.50 | 3.01% |
| SELL | retest2 | 2025-09-22 12:00:00 | 39750.00 | 2025-09-29 13:15:00 | 38540.00 | STOP_HIT | 0.50 | 3.04% |
| SELL | retest2 | 2025-09-24 09:15:00 | 39160.00 | 2025-10-03 13:15:00 | 38325.00 | STOP_HIT | 1.00 | 2.13% |
| BUY | retest2 | 2025-10-17 10:45:00 | 38545.00 | 2025-10-24 10:15:00 | 38570.00 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-10-17 13:45:00 | 38600.00 | 2025-10-24 10:15:00 | 38570.00 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-12-05 10:15:00 | 37000.00 | 2025-12-08 10:15:00 | 36565.00 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-12-09 14:15:00 | 36670.00 | 2025-12-09 15:15:00 | 36840.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-12-17 12:00:00 | 36025.00 | 2025-12-22 14:15:00 | 36100.00 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-12-22 10:15:00 | 35940.00 | 2025-12-22 14:15:00 | 36100.00 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2026-01-07 12:30:00 | 38900.00 | 2026-01-08 15:15:00 | 38545.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-01-08 13:00:00 | 38765.00 | 2026-01-08 15:15:00 | 38545.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2026-01-08 14:30:00 | 38770.00 | 2026-01-08 15:15:00 | 38545.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-01-16 09:30:00 | 37200.00 | 2026-01-21 09:15:00 | 35340.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:30:00 | 37200.00 | 2026-01-22 09:15:00 | 35745.00 | STOP_HIT | 0.50 | 3.91% |
| BUY | retest2 | 2026-02-01 14:15:00 | 36195.00 | 2026-02-05 09:15:00 | 36310.00 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2026-02-02 11:30:00 | 36185.00 | 2026-02-05 09:15:00 | 36310.00 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2026-02-02 12:30:00 | 36140.00 | 2026-02-05 09:15:00 | 36310.00 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2026-02-09 10:45:00 | 36105.00 | 2026-02-11 13:15:00 | 36470.00 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2026-02-09 11:15:00 | 36075.00 | 2026-02-11 13:15:00 | 36470.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-02-09 11:45:00 | 36085.00 | 2026-02-11 13:15:00 | 36470.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-02-09 13:30:00 | 35950.00 | 2026-02-11 13:15:00 | 36470.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-02-23 10:45:00 | 35170.00 | 2026-02-25 09:15:00 | 36035.00 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2026-02-23 11:45:00 | 35170.00 | 2026-02-25 09:15:00 | 36035.00 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2026-02-23 13:00:00 | 35170.00 | 2026-02-25 09:15:00 | 36035.00 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2026-02-24 09:15:00 | 35085.00 | 2026-02-25 09:15:00 | 36035.00 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2026-02-27 13:45:00 | 36330.00 | 2026-03-02 09:15:00 | 35840.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2026-02-27 14:15:00 | 36300.00 | 2026-03-02 09:15:00 | 35840.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2026-02-27 14:45:00 | 36450.00 | 2026-03-02 09:15:00 | 35840.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2026-03-11 11:00:00 | 31935.00 | 2026-03-13 14:15:00 | 30338.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 11:00:00 | 31935.00 | 2026-03-16 14:15:00 | 30370.00 | STOP_HIT | 0.50 | 4.90% |
| SELL | retest2 | 2026-03-23 09:15:00 | 29790.00 | 2026-03-25 09:15:00 | 30470.00 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2026-04-13 10:45:00 | 37240.00 | 2026-04-23 10:15:00 | 37485.00 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2026-04-16 10:30:00 | 37240.00 | 2026-04-23 10:15:00 | 37485.00 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2026-04-16 14:30:00 | 37265.00 | 2026-04-23 10:15:00 | 37485.00 | STOP_HIT | 1.00 | 0.59% |
| SELL | retest2 | 2026-05-04 10:30:00 | 36020.00 | 2026-05-06 11:15:00 | 36230.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-05-04 11:15:00 | 36085.00 | 2026-05-06 11:15:00 | 36230.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2026-05-05 11:30:00 | 36110.00 | 2026-05-06 14:15:00 | 36690.00 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-05-05 12:15:00 | 36125.00 | 2026-05-06 14:15:00 | 36690.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2026-05-05 14:15:00 | 35890.00 | 2026-05-06 14:15:00 | 36690.00 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2026-05-06 10:30:00 | 35985.00 | 2026-05-06 14:15:00 | 36690.00 | STOP_HIT | 1.00 | -1.96% |
