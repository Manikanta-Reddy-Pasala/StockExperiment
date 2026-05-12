# ABB India Ltd. (ABB)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 7010.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 87 |
| ALERT1 | 63 |
| ALERT2 | 62 |
| ALERT2_SKIP | 37 |
| ALERT3 | 157 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 53 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 51 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 15 / 40
- **Target hits / Stop hits / Partials:** 2 / 51 / 2
- **Avg / median % per leg:** 0.26% / -0.78%
- **Sum % (uncompounded):** 14.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 11 | 44.0% | 1 | 24 | 0 | 0.78% | 19.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 25 | 11 | 44.0% | 1 | 24 | 0 | 0.78% | 19.5% |
| SELL (all) | 30 | 4 | 13.3% | 1 | 27 | 2 | -0.18% | -5.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.21% | -0.2% |
| SELL @ 3rd Alert (retest2) | 29 | 4 | 13.8% | 1 | 26 | 2 | -0.18% | -5.3% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.21% | -0.2% |
| retest2 (combined) | 54 | 15 | 27.8% | 2 | 50 | 2 | 0.26% | 14.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 5750.00 | 5784.62 | 5784.89 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 5825.50 | 5792.79 | 5788.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 10:15:00 | 5845.00 | 5803.23 | 5793.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-21 11:15:00 | 5792.50 | 5801.09 | 5793.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 11:15:00 | 5792.50 | 5801.09 | 5793.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 5792.50 | 5801.09 | 5793.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:45:00 | 5815.00 | 5801.09 | 5793.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 5835.50 | 5807.97 | 5797.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:45:00 | 5842.50 | 5814.78 | 5801.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 5968.00 | 6018.71 | 6019.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 14:15:00 | 5968.00 | 6018.71 | 6019.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 15:15:00 | 5952.00 | 6005.37 | 6013.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 13:15:00 | 5965.00 | 5962.34 | 5985.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 14:00:00 | 5965.00 | 5962.34 | 5985.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 5977.00 | 5965.27 | 5984.34 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 13:15:00 | 6010.00 | 5995.41 | 5993.95 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 09:15:00 | 5985.50 | 5992.32 | 5992.89 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 13:15:00 | 6037.50 | 6000.00 | 5995.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 6058.00 | 6022.35 | 6008.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 13:15:00 | 6020.00 | 6034.50 | 6019.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 13:15:00 | 6020.00 | 6034.50 | 6019.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 6020.00 | 6034.50 | 6019.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:00:00 | 6020.00 | 6034.50 | 6019.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 6029.00 | 6033.40 | 6020.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:30:00 | 6033.00 | 6033.40 | 6020.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 13:15:00 | 6055.50 | 6061.76 | 6043.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 13:30:00 | 6051.00 | 6061.76 | 6043.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 6047.50 | 6058.91 | 6043.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 14:45:00 | 6060.50 | 6058.91 | 6043.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 6045.50 | 6056.22 | 6043.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 6083.00 | 6056.22 | 6043.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 12:15:00 | 6105.00 | 6127.42 | 6129.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-06-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 12:15:00 | 6105.00 | 6127.42 | 6129.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 13:15:00 | 6054.00 | 6112.73 | 6122.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 5997.50 | 5985.80 | 6023.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 5997.50 | 5985.80 | 6023.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 5962.50 | 5983.58 | 6016.15 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 6053.00 | 6023.19 | 6021.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 09:15:00 | 6084.50 | 6047.92 | 6036.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 10:15:00 | 6037.50 | 6045.84 | 6036.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 10:15:00 | 6037.50 | 6045.84 | 6036.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 6037.50 | 6045.84 | 6036.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:45:00 | 6041.00 | 6045.84 | 6036.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 6020.50 | 6040.77 | 6034.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:00:00 | 6020.50 | 6040.77 | 6034.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 6030.00 | 6038.61 | 6034.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:30:00 | 6015.00 | 6038.61 | 6034.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 6035.00 | 6037.89 | 6034.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:45:00 | 6036.50 | 6037.89 | 6034.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 6048.50 | 6040.01 | 6035.80 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 10:15:00 | 5976.00 | 6024.25 | 6029.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 11:15:00 | 5946.50 | 6008.70 | 6021.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 5962.50 | 5933.92 | 5969.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 10:15:00 | 5962.50 | 5933.92 | 5969.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 5962.50 | 5933.92 | 5969.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 5962.50 | 5933.92 | 5969.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 5987.50 | 5944.63 | 5971.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 5987.50 | 5944.63 | 5971.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 5967.00 | 5949.11 | 5970.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:45:00 | 5984.50 | 5949.11 | 5970.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 5962.00 | 5951.68 | 5970.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:45:00 | 5974.50 | 5951.68 | 5970.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 5960.00 | 5953.35 | 5969.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 5960.00 | 5953.35 | 5969.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 5949.00 | 5952.48 | 5967.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 5897.00 | 5952.48 | 5967.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 12:00:00 | 5930.50 | 5943.14 | 5958.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 13:45:00 | 5925.50 | 5942.17 | 5955.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 14:15:00 | 5930.00 | 5942.17 | 5955.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 6010.00 | 5950.02 | 5955.32 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 6010.00 | 5950.02 | 5955.32 | SL hit (close>static) qty=1.00 sl=5978.50 alert=retest2 |

### Cycle 10 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 6031.00 | 5966.22 | 5962.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 6057.50 | 5984.48 | 5970.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 6001.00 | 6003.04 | 5984.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 14:45:00 | 6007.00 | 6003.04 | 5984.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 5994.00 | 6001.23 | 5985.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 6051.00 | 6001.23 | 5985.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 11:15:00 | 5961.50 | 5996.03 | 5987.17 | SL hit (close<static) qty=1.00 sl=5981.00 alert=retest2 |

### Cycle 11 — SELL (started 2025-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 15:15:00 | 5964.50 | 5979.66 | 5981.42 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 11:15:00 | 6010.00 | 5985.76 | 5983.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 6136.00 | 6026.12 | 6004.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 6058.50 | 6064.05 | 6035.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 14:45:00 | 6062.00 | 6064.05 | 6035.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 6065.00 | 6075.95 | 6061.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:00:00 | 6065.00 | 6075.95 | 6061.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 6066.50 | 6074.06 | 6061.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 6066.50 | 6074.06 | 6061.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 5976.00 | 6054.45 | 6053.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:00:00 | 5976.00 | 6054.45 | 6053.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 12:15:00 | 5944.00 | 6032.36 | 6043.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 5900.00 | 5973.18 | 6009.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 5923.50 | 5917.27 | 5955.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 10:00:00 | 5923.50 | 5917.27 | 5955.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 5868.00 | 5846.58 | 5864.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 14:00:00 | 5838.00 | 5847.67 | 5859.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 5835.00 | 5845.87 | 5856.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 5886.50 | 5854.00 | 5859.65 | SL hit (close>static) qty=1.00 sl=5885.00 alert=retest2 |

### Cycle 14 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 11:15:00 | 5930.50 | 5874.74 | 5868.43 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 11:15:00 | 5842.50 | 5868.82 | 5870.27 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2025-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 14:15:00 | 5900.00 | 5875.20 | 5872.71 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 5819.50 | 5864.99 | 5868.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 5792.00 | 5850.39 | 5861.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 5695.00 | 5684.24 | 5739.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 09:30:00 | 5691.50 | 5684.24 | 5739.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 5580.50 | 5560.71 | 5609.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:45:00 | 5582.00 | 5560.71 | 5609.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 5603.50 | 5569.27 | 5609.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:30:00 | 5638.00 | 5569.27 | 5609.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 5622.50 | 5579.91 | 5610.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:45:00 | 5648.00 | 5579.91 | 5610.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 5614.00 | 5586.73 | 5610.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:30:00 | 5625.00 | 5586.73 | 5610.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 5645.00 | 5598.38 | 5613.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 15:00:00 | 5645.00 | 5598.38 | 5613.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 5634.00 | 5605.51 | 5615.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:15:00 | 5700.00 | 5605.51 | 5615.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 10:15:00 | 5680.00 | 5632.33 | 5626.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 09:15:00 | 5769.00 | 5674.45 | 5652.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 5757.50 | 5771.36 | 5726.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 11:00:00 | 5757.50 | 5771.36 | 5726.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 5754.00 | 5764.44 | 5737.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 5754.00 | 5764.44 | 5737.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 5695.50 | 5749.14 | 5735.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 5695.50 | 5749.14 | 5735.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 5691.50 | 5737.62 | 5731.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:30:00 | 5688.00 | 5737.62 | 5731.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 12:15:00 | 5703.50 | 5722.77 | 5724.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 5663.00 | 5704.79 | 5715.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 14:15:00 | 5697.00 | 5688.13 | 5701.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 14:15:00 | 5697.00 | 5688.13 | 5701.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 5697.00 | 5688.13 | 5701.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 15:00:00 | 5697.00 | 5688.13 | 5701.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 5688.00 | 5688.11 | 5699.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:15:00 | 5668.00 | 5688.11 | 5699.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 14:15:00 | 5384.60 | 5468.83 | 5504.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-04 10:15:00 | 5101.20 | 5340.79 | 5433.87 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 20 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 10:15:00 | 5085.00 | 5064.87 | 5064.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 15:15:00 | 5090.00 | 5076.49 | 5070.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 5053.50 | 5071.89 | 5069.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 5053.50 | 5071.89 | 5069.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 5053.50 | 5071.89 | 5069.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 5053.50 | 5071.89 | 5069.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 5054.50 | 5068.42 | 5067.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:45:00 | 5057.00 | 5068.42 | 5067.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 5055.00 | 5065.73 | 5066.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 12:15:00 | 5046.50 | 5061.89 | 5064.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 5054.00 | 5048.84 | 5056.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 5054.00 | 5048.84 | 5056.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 5054.00 | 5048.84 | 5056.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 11:30:00 | 5028.00 | 5039.96 | 5051.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 09:15:00 | 5023.50 | 5031.67 | 5042.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 12:15:00 | 5073.00 | 5051.92 | 5049.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2025-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 12:15:00 | 5073.00 | 5051.92 | 5049.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 5120.00 | 5070.12 | 5059.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 09:15:00 | 5134.50 | 5138.50 | 5119.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 09:15:00 | 5134.50 | 5138.50 | 5119.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 5134.50 | 5138.50 | 5119.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 5132.50 | 5138.50 | 5119.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 5107.50 | 5132.30 | 5118.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:30:00 | 5097.00 | 5132.30 | 5118.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 5104.00 | 5126.64 | 5116.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:15:00 | 5100.00 | 5126.64 | 5116.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 5087.50 | 5118.81 | 5114.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 13:00:00 | 5087.50 | 5118.81 | 5114.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 5060.00 | 5101.64 | 5106.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 5047.50 | 5088.38 | 5096.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 5038.00 | 5010.08 | 5031.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 5038.00 | 5010.08 | 5031.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 5038.00 | 5010.08 | 5031.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 5038.00 | 5010.08 | 5031.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 5055.00 | 5019.06 | 5033.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:45:00 | 5025.00 | 5019.15 | 5032.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:45:00 | 5030.00 | 5021.52 | 5032.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 14:15:00 | 5023.00 | 5021.52 | 5032.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 5085.10 | 5032.91 | 5033.36 | SL hit (close>static) qty=1.00 sl=5060.50 alert=retest2 |

### Cycle 24 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 5088.00 | 5043.93 | 5038.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 5125.50 | 5072.69 | 5053.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 12:15:00 | 5166.00 | 5176.39 | 5155.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 12:15:00 | 5166.00 | 5176.39 | 5155.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 5166.00 | 5176.39 | 5155.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 14:30:00 | 5181.50 | 5173.33 | 5157.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 09:15:00 | 5124.80 | 5160.69 | 5154.44 | SL hit (close<static) qty=1.00 sl=5145.00 alert=retest2 |

### Cycle 25 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 5107.00 | 5143.84 | 5147.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 5083.70 | 5131.81 | 5141.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 15:15:00 | 5129.00 | 5127.07 | 5136.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:15:00 | 5130.20 | 5127.07 | 5136.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 5183.50 | 5138.36 | 5140.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 5183.50 | 5138.36 | 5140.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 5146.70 | 5140.03 | 5141.51 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 12:15:00 | 5155.90 | 5144.49 | 5143.37 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 5129.20 | 5141.43 | 5142.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 14:15:00 | 5102.00 | 5133.55 | 5138.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 12:15:00 | 5136.00 | 5127.57 | 5132.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 12:15:00 | 5136.00 | 5127.57 | 5132.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 5136.00 | 5127.57 | 5132.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:00:00 | 5136.00 | 5127.57 | 5132.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 5129.30 | 5127.92 | 5132.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 14:45:00 | 5125.00 | 5127.57 | 5131.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 5176.20 | 5139.27 | 5136.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 5176.20 | 5139.27 | 5136.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 5184.00 | 5148.22 | 5140.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 13:15:00 | 5162.10 | 5162.22 | 5150.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 14:00:00 | 5162.10 | 5162.22 | 5150.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 5158.90 | 5161.55 | 5150.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:45:00 | 5153.90 | 5161.55 | 5150.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 5200.30 | 5171.45 | 5157.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 11:00:00 | 5233.30 | 5183.82 | 5164.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 11:45:00 | 5214.60 | 5191.74 | 5169.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 12:45:00 | 5215.00 | 5194.51 | 5172.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 13:30:00 | 5213.10 | 5197.41 | 5176.22 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 5390.90 | 5363.56 | 5336.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 5404.60 | 5367.89 | 5341.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 15:15:00 | 5360.00 | 5407.04 | 5409.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2025-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 15:15:00 | 5360.00 | 5407.04 | 5409.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 5340.30 | 5393.69 | 5403.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 5327.20 | 5325.80 | 5355.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 10:00:00 | 5327.20 | 5325.80 | 5355.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 5216.30 | 5185.45 | 5216.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 5216.30 | 5185.45 | 5216.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 5216.00 | 5191.56 | 5216.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:00:00 | 5216.00 | 5191.56 | 5216.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 5203.10 | 5193.87 | 5215.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:30:00 | 5184.90 | 5199.10 | 5210.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 10:15:00 | 5230.00 | 5205.28 | 5212.39 | SL hit (close>static) qty=1.00 sl=5219.90 alert=retest2 |

### Cycle 30 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 5208.50 | 5195.10 | 5193.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 13:15:00 | 5218.50 | 5199.37 | 5196.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 12:15:00 | 5244.50 | 5246.02 | 5224.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 12:15:00 | 5244.50 | 5246.02 | 5224.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 5244.50 | 5246.02 | 5224.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:45:00 | 5230.00 | 5246.02 | 5224.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 5212.50 | 5240.03 | 5225.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 5212.50 | 5240.03 | 5225.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 5230.00 | 5238.02 | 5225.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 5203.50 | 5238.02 | 5225.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 5214.00 | 5233.22 | 5224.88 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 5200.50 | 5217.36 | 5218.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 13:15:00 | 5141.00 | 5200.11 | 5210.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 5142.50 | 5135.81 | 5162.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 15:00:00 | 5142.50 | 5135.81 | 5162.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 5151.00 | 5138.85 | 5161.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 5172.50 | 5138.85 | 5161.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 5180.00 | 5147.08 | 5163.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 5206.00 | 5147.08 | 5163.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 5185.00 | 5154.66 | 5165.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:45:00 | 5176.00 | 5154.66 | 5165.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 5187.50 | 5170.99 | 5170.92 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 5142.00 | 5169.45 | 5170.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 5088.50 | 5128.91 | 5146.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 5219.50 | 5137.01 | 5140.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 5219.50 | 5137.01 | 5140.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 5219.50 | 5137.01 | 5140.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 5219.50 | 5137.01 | 5140.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 5212.50 | 5152.11 | 5147.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 5238.00 | 5209.17 | 5196.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 11:15:00 | 5217.00 | 5236.89 | 5223.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 11:15:00 | 5217.00 | 5236.89 | 5223.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 5217.00 | 5236.89 | 5223.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:00:00 | 5217.00 | 5236.89 | 5223.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 5197.50 | 5229.01 | 5221.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:00:00 | 5197.50 | 5229.01 | 5221.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 5194.00 | 5222.01 | 5218.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:30:00 | 5185.00 | 5222.01 | 5218.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 5178.00 | 5213.21 | 5215.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 15:15:00 | 5173.00 | 5205.17 | 5211.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 12:15:00 | 5202.00 | 5200.02 | 5206.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 12:15:00 | 5202.00 | 5200.02 | 5206.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 5202.00 | 5200.02 | 5206.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:45:00 | 5197.00 | 5200.02 | 5206.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 5191.00 | 5192.14 | 5200.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:15:00 | 5200.00 | 5192.14 | 5200.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 5197.50 | 5193.21 | 5199.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:00:00 | 5197.50 | 5193.21 | 5199.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 5213.50 | 5197.27 | 5201.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:00:00 | 5213.50 | 5197.27 | 5201.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 5212.00 | 5200.22 | 5202.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:00:00 | 5212.00 | 5200.22 | 5202.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 5230.50 | 5206.27 | 5204.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 14:15:00 | 5238.50 | 5212.72 | 5207.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 11:15:00 | 5184.00 | 5212.43 | 5210.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 11:15:00 | 5184.00 | 5212.43 | 5210.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 5184.00 | 5212.43 | 5210.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 5184.00 | 5212.43 | 5210.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2025-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 12:15:00 | 5163.00 | 5202.54 | 5205.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 13:15:00 | 5152.00 | 5192.43 | 5200.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 5246.50 | 5199.47 | 5201.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 5246.50 | 5199.47 | 5201.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 5246.50 | 5199.47 | 5201.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 5246.50 | 5199.47 | 5201.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 5269.50 | 5213.48 | 5207.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 5298.00 | 5261.53 | 5236.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 10:15:00 | 5262.00 | 5262.02 | 5241.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 10:45:00 | 5245.00 | 5262.02 | 5241.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 5243.50 | 5264.21 | 5253.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:30:00 | 5249.00 | 5264.21 | 5253.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 5223.00 | 5255.97 | 5251.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:45:00 | 5223.00 | 5255.97 | 5251.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 13:15:00 | 5235.50 | 5250.52 | 5249.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 14:00:00 | 5235.50 | 5250.52 | 5249.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 5223.00 | 5245.01 | 5246.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 11:15:00 | 5217.50 | 5238.25 | 5242.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 12:15:00 | 5240.00 | 5238.60 | 5242.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 13:00:00 | 5240.00 | 5238.60 | 5242.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 5240.50 | 5238.98 | 5242.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:30:00 | 5235.00 | 5238.98 | 5242.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 5255.50 | 5242.28 | 5243.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 5255.50 | 5242.28 | 5243.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 15:15:00 | 5256.50 | 5245.13 | 5244.80 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 5239.00 | 5244.44 | 5244.66 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 09:15:00 | 5258.00 | 5245.18 | 5244.50 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2025-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 15:15:00 | 5205.00 | 5237.71 | 5242.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 5072.00 | 5204.57 | 5226.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 4983.00 | 4978.13 | 5013.20 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 11:15:00 | 4946.50 | 4979.11 | 5010.46 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 4957.00 | 4937.50 | 4946.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-14 14:15:00 | 4957.00 | 4937.50 | 4946.76 | SL hit (close>ema400) qty=1.00 sl=4946.76 alert=retest1 |

### Cycle 44 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 4976.00 | 4952.64 | 4952.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 12:15:00 | 5005.50 | 4966.87 | 4958.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 09:15:00 | 5053.50 | 5055.94 | 5023.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 10:00:00 | 5053.50 | 5055.94 | 5023.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 5084.00 | 5064.71 | 5040.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 14:30:00 | 5059.50 | 5064.71 | 5040.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 5091.50 | 5123.83 | 5094.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 5091.50 | 5123.83 | 5094.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 5085.50 | 5116.16 | 5093.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:30:00 | 5082.00 | 5116.16 | 5093.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 5098.50 | 5112.63 | 5094.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:30:00 | 5085.00 | 5112.63 | 5094.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 5099.50 | 5110.00 | 5094.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:30:00 | 5095.00 | 5110.00 | 5094.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 5090.50 | 5104.74 | 5094.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:45:00 | 5077.50 | 5104.74 | 5094.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 5071.50 | 5098.09 | 5092.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:15:00 | 5085.50 | 5098.09 | 5092.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 5065.00 | 5091.48 | 5090.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:45:00 | 5059.00 | 5091.48 | 5090.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 10:15:00 | 5065.00 | 5086.18 | 5087.84 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 5137.00 | 5082.25 | 5079.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 11:15:00 | 5172.00 | 5110.16 | 5092.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 11:15:00 | 5216.00 | 5216.95 | 5184.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 12:15:00 | 5192.50 | 5216.95 | 5184.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 5185.00 | 5210.56 | 5184.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 13:00:00 | 5185.00 | 5210.56 | 5184.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 5179.00 | 5204.25 | 5183.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:15:00 | 5181.00 | 5204.25 | 5183.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 5174.00 | 5198.20 | 5182.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:30:00 | 5184.00 | 5191.35 | 5182.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 10:00:00 | 5182.50 | 5191.35 | 5182.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 11:15:00 | 5158.50 | 5182.16 | 5179.49 | SL hit (close<static) qty=1.00 sl=5170.00 alert=retest2 |

### Cycle 47 — SELL (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 15:15:00 | 5165.00 | 5179.73 | 5180.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 5120.00 | 5167.78 | 5174.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 5162.50 | 5136.48 | 5150.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 5162.50 | 5136.48 | 5150.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 5162.50 | 5136.48 | 5150.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:00:00 | 5162.50 | 5136.48 | 5150.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 5132.50 | 5135.69 | 5149.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 11:15:00 | 5127.00 | 5135.69 | 5149.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 12:15:00 | 5167.00 | 5143.12 | 5150.25 | SL hit (close>static) qty=1.00 sl=5165.00 alert=retest2 |

### Cycle 48 — BUY (started 2025-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 15:15:00 | 5166.00 | 5156.17 | 5155.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 15:15:00 | 5180.00 | 5165.70 | 5160.95 | Break + close above crossover candle high |

### Cycle 49 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 5121.00 | 5156.76 | 5157.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 5103.00 | 5137.02 | 5147.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 5085.00 | 5083.44 | 5110.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 5085.00 | 5083.44 | 5110.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 5116.00 | 5089.88 | 5108.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:45:00 | 5113.00 | 5089.88 | 5108.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 5120.50 | 5096.00 | 5109.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 5120.50 | 5096.00 | 5109.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 5108.00 | 5098.40 | 5109.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 5108.50 | 5098.40 | 5109.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 5127.50 | 5104.22 | 5111.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:15:00 | 5169.00 | 5104.22 | 5111.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 5150.00 | 5113.38 | 5114.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:30:00 | 5176.00 | 5113.38 | 5114.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 5156.00 | 5121.90 | 5118.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 09:15:00 | 5237.50 | 5161.08 | 5140.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 5216.00 | 5248.95 | 5220.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 5216.00 | 5248.95 | 5220.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 5216.00 | 5248.95 | 5220.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 5216.00 | 5248.95 | 5220.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 5246.00 | 5248.36 | 5223.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 11:15:00 | 5263.00 | 5248.36 | 5223.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 09:30:00 | 5260.00 | 5269.61 | 5246.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 12:45:00 | 5257.50 | 5268.15 | 5251.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 10:15:00 | 5219.00 | 5246.48 | 5246.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 5219.00 | 5246.48 | 5246.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 11:15:00 | 5204.00 | 5237.98 | 5242.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 10:15:00 | 5138.50 | 5125.46 | 5158.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 10:45:00 | 5143.50 | 5125.46 | 5158.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 5150.00 | 5132.76 | 5153.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 5150.00 | 5132.76 | 5153.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 5175.00 | 5141.21 | 5155.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 5175.00 | 5141.21 | 5155.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 5169.00 | 5146.77 | 5156.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 5228.00 | 5146.77 | 5156.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 5207.00 | 5167.65 | 5165.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 09:15:00 | 5237.00 | 5205.87 | 5190.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 14:15:00 | 5214.00 | 5224.69 | 5207.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 14:45:00 | 5220.00 | 5224.69 | 5207.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 5202.50 | 5220.25 | 5207.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 5229.50 | 5220.25 | 5207.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 5228.50 | 5221.90 | 5209.18 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 5192.50 | 5204.80 | 5205.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 14:15:00 | 5167.00 | 5184.63 | 5194.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 5154.00 | 5146.62 | 5163.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 15:15:00 | 5154.00 | 5146.62 | 5163.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 5154.00 | 5146.62 | 5163.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 5159.50 | 5146.62 | 5163.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 5161.50 | 5149.60 | 5163.47 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 12:15:00 | 5175.00 | 5167.71 | 5167.44 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 13:15:00 | 5164.00 | 5166.97 | 5167.13 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 14:15:00 | 5190.00 | 5171.57 | 5169.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 5205.50 | 5181.04 | 5174.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 11:15:00 | 5186.50 | 5196.09 | 5185.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 11:15:00 | 5186.50 | 5196.09 | 5185.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 5186.50 | 5196.09 | 5185.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:45:00 | 5182.50 | 5196.09 | 5185.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 5182.00 | 5193.27 | 5185.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:30:00 | 5178.00 | 5193.27 | 5185.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 5143.50 | 5183.32 | 5181.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 5143.50 | 5183.32 | 5181.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 5169.50 | 5180.55 | 5180.26 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 15:15:00 | 5165.50 | 5177.54 | 5178.92 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 5247.00 | 5191.43 | 5185.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 13:15:00 | 5258.00 | 5225.65 | 5211.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 5257.00 | 5263.34 | 5236.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 11:00:00 | 5257.00 | 5263.34 | 5236.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 5217.50 | 5254.17 | 5235.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 5217.50 | 5254.17 | 5235.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 5196.50 | 5242.63 | 5231.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:00:00 | 5196.50 | 5242.63 | 5231.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 14:15:00 | 5046.50 | 5192.19 | 5209.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 5011.00 | 5082.08 | 5128.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 5047.00 | 5035.96 | 5083.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 15:00:00 | 5047.00 | 5035.96 | 5083.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 5017.50 | 5033.80 | 5074.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:00:00 | 4980.50 | 5014.97 | 5055.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 4731.47 | 4825.19 | 4878.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 11:15:00 | 4738.00 | 4727.26 | 4786.09 | SL hit (close>ema200) qty=0.50 sl=4727.26 alert=retest2 |

### Cycle 60 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 4862.00 | 4740.37 | 4730.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 4928.50 | 4777.99 | 4748.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 5440.50 | 5496.99 | 5368.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 12:00:00 | 5440.50 | 5496.99 | 5368.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 5425.00 | 5482.59 | 5373.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 5417.50 | 5482.59 | 5373.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 5470.50 | 5480.17 | 5382.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 5413.50 | 5480.17 | 5382.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 5389.00 | 5446.00 | 5395.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 5389.00 | 5446.00 | 5395.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 5401.00 | 5437.00 | 5396.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 15:00:00 | 5477.50 | 5434.77 | 5404.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 11:15:00 | 5765.00 | 5803.85 | 5807.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 5765.00 | 5803.85 | 5807.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 5742.00 | 5782.54 | 5795.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 11:15:00 | 5851.00 | 5795.51 | 5798.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 11:15:00 | 5851.00 | 5795.51 | 5798.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 5851.00 | 5795.51 | 5798.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:00:00 | 5851.00 | 5795.51 | 5798.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 5816.00 | 5799.60 | 5800.33 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2026-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 13:15:00 | 5811.00 | 5801.88 | 5801.30 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 5784.50 | 5798.41 | 5799.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 15:15:00 | 5768.00 | 5792.33 | 5796.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 5812.50 | 5796.36 | 5798.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 5812.50 | 5796.36 | 5798.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 5812.50 | 5796.36 | 5798.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:00:00 | 5812.50 | 5796.36 | 5798.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 5823.00 | 5801.69 | 5800.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 11:15:00 | 5876.50 | 5816.65 | 5807.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 11:15:00 | 5835.00 | 5860.51 | 5840.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 11:15:00 | 5835.00 | 5860.51 | 5840.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 5835.00 | 5860.51 | 5840.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:00:00 | 5835.00 | 5860.51 | 5840.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 5845.50 | 5857.51 | 5841.33 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 09:15:00 | 5795.50 | 5834.39 | 5834.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 10:15:00 | 5783.00 | 5824.12 | 5829.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 12:15:00 | 5845.00 | 5827.07 | 5830.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 12:15:00 | 5845.00 | 5827.07 | 5830.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 12:15:00 | 5845.00 | 5827.07 | 5830.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 12:45:00 | 5848.00 | 5827.07 | 5830.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2026-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 13:15:00 | 5871.00 | 5835.86 | 5833.83 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 5775.00 | 5834.98 | 5836.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 5753.00 | 5818.58 | 5829.34 | Break + close below crossover candle low |

### Cycle 68 — BUY (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 09:15:00 | 6031.50 | 5830.69 | 5829.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 10:15:00 | 6137.00 | 5891.96 | 5857.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 14:15:00 | 5979.00 | 6032.07 | 5946.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-20 15:00:00 | 5979.00 | 6032.07 | 5946.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 5940.50 | 6009.42 | 5951.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:00:00 | 5940.50 | 6009.42 | 5951.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 5889.00 | 5985.34 | 5945.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 5889.00 | 5985.34 | 5945.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 5846.50 | 5957.57 | 5936.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:45:00 | 5850.50 | 5957.57 | 5936.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 5928.50 | 5923.88 | 5923.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:45:00 | 5900.00 | 5923.88 | 5923.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2026-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 15:15:00 | 5906.00 | 5920.30 | 5921.62 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 09:15:00 | 5934.00 | 5923.04 | 5922.75 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 10:15:00 | 5916.50 | 5921.73 | 5922.18 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 11:15:00 | 5928.00 | 5922.99 | 5922.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 12:15:00 | 5965.00 | 5931.39 | 5926.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 12:15:00 | 6130.00 | 6135.19 | 6082.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 12:45:00 | 6131.00 | 6135.19 | 6082.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 6061.00 | 6120.84 | 6092.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 6061.00 | 6120.84 | 6092.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 6097.50 | 6116.17 | 6093.05 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 6052.00 | 6084.18 | 6085.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 5827.00 | 5926.50 | 5988.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 10:15:00 | 5909.00 | 5870.23 | 5925.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 10:15:00 | 5909.00 | 5870.23 | 5925.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 5909.00 | 5870.23 | 5925.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 11:00:00 | 5909.00 | 5870.23 | 5925.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 5885.00 | 5873.19 | 5922.15 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 6016.50 | 5939.33 | 5935.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 11:15:00 | 6075.00 | 5966.47 | 5948.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 5956.00 | 6014.24 | 5985.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 5956.00 | 6014.24 | 5985.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 5956.00 | 6014.24 | 5985.08 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2026-03-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 12:15:00 | 5880.00 | 5953.91 | 5961.81 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 09:15:00 | 6150.50 | 6000.56 | 5980.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 6230.00 | 6144.92 | 6072.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 6192.50 | 6238.63 | 6174.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 6192.50 | 6238.63 | 6174.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 6192.50 | 6238.63 | 6174.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:45:00 | 6277.00 | 6244.91 | 6183.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 09:30:00 | 6302.50 | 6364.22 | 6326.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 6195.50 | 6303.32 | 6303.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2026-03-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 11:15:00 | 6195.50 | 6303.32 | 6303.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 12:15:00 | 6170.00 | 6276.66 | 6291.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 12:15:00 | 6256.50 | 6232.14 | 6255.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 12:15:00 | 6256.50 | 6232.14 | 6255.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 6256.50 | 6232.14 | 6255.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 13:00:00 | 6256.50 | 6232.14 | 6255.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 6281.50 | 6242.01 | 6257.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 13:30:00 | 6270.00 | 6242.01 | 6257.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 6319.00 | 6257.41 | 6263.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:45:00 | 6330.50 | 6257.41 | 6263.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 6310.00 | 6267.93 | 6267.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 6352.00 | 6284.74 | 6275.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 12:15:00 | 6304.00 | 6314.89 | 6293.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 12:15:00 | 6304.00 | 6314.89 | 6293.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 6304.00 | 6314.89 | 6293.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 12:45:00 | 6287.50 | 6314.89 | 6293.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 6260.50 | 6312.40 | 6300.61 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 6252.50 | 6292.04 | 6292.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 6220.00 | 6269.15 | 6281.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 6343.00 | 6269.76 | 6277.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 6343.00 | 6269.76 | 6277.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 6343.00 | 6269.76 | 6277.50 | EMA400 retest candle locked (from downside) |

### Cycle 80 — BUY (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 11:15:00 | 6304.00 | 6285.61 | 6283.92 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 6022.50 | 6246.07 | 6269.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 5997.00 | 6138.19 | 6209.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 6114.50 | 6094.88 | 6162.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 6114.50 | 6094.88 | 6162.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 6114.50 | 6094.88 | 6162.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 6076.00 | 6094.88 | 6162.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 6227.50 | 6122.76 | 6158.14 | SL hit (close>static) qty=1.00 sl=6200.00 alert=retest2 |

### Cycle 82 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 6277.50 | 6172.38 | 6172.33 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 6011.50 | 6171.11 | 6179.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 5981.00 | 6082.97 | 6125.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 6117.00 | 6034.46 | 6075.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 6117.00 | 6034.46 | 6075.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 6117.00 | 6034.46 | 6075.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:45:00 | 6043.00 | 6030.57 | 6070.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:45:00 | 6043.00 | 6062.17 | 6075.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 5963.00 | 6064.53 | 6075.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 6146.00 | 6073.48 | 6070.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 6146.00 | 6073.48 | 6070.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 09:15:00 | 6215.00 | 6114.82 | 6090.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 6148.50 | 6157.21 | 6127.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 6148.50 | 6157.21 | 6127.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 6148.50 | 6157.21 | 6127.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 11:15:00 | 6196.00 | 6160.77 | 6131.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-10 11:15:00 | 6815.60 | 6660.41 | 6535.43 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2026-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 14:15:00 | 7340.00 | 7423.77 | 7433.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 11:15:00 | 7307.50 | 7383.77 | 7403.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 12:15:00 | 7315.50 | 7293.48 | 7334.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 12:15:00 | 7315.50 | 7293.48 | 7334.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 7315.50 | 7293.48 | 7334.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 13:00:00 | 7315.50 | 7293.48 | 7334.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 7208.00 | 7214.61 | 7259.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 7287.50 | 7214.61 | 7259.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 7315.50 | 7241.89 | 7264.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 7200.50 | 7246.44 | 7261.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 14:15:00 | 7219.00 | 7244.25 | 7259.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 13:15:00 | 7281.00 | 7266.24 | 7264.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 13:15:00 | 7281.00 | 7266.24 | 7264.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 14:15:00 | 7328.00 | 7278.59 | 7270.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 09:15:00 | 7233.00 | 7278.42 | 7272.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 7233.00 | 7278.42 | 7272.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 7233.00 | 7278.42 | 7272.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:30:00 | 7219.00 | 7278.42 | 7272.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 10:15:00 | 7178.00 | 7258.34 | 7263.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-07 10:15:00 | 7086.00 | 7191.78 | 7223.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-07 13:15:00 | 7205.00 | 7182.33 | 7210.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 13:15:00 | 7205.00 | 7182.33 | 7210.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 7205.00 | 7182.33 | 7210.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 14:00:00 | 7205.00 | 7182.33 | 7210.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 7188.00 | 7183.33 | 7205.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:15:00 | 7134.50 | 7183.33 | 7205.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 7153.00 | 7177.26 | 7200.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 11:15:00 | 7068.00 | 7167.21 | 7194.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 10:45:00 | 5699.50 | 2025-05-20 15:15:00 | 5750.00 | STOP_HIT | 1.00 | 0.89% |
| BUY | retest2 | 2025-05-15 11:15:00 | 5694.50 | 2025-05-20 15:15:00 | 5750.00 | STOP_HIT | 1.00 | 0.97% |
| BUY | retest2 | 2025-05-15 13:15:00 | 5697.00 | 2025-05-20 15:15:00 | 5750.00 | STOP_HIT | 1.00 | 0.93% |
| BUY | retest2 | 2025-05-21 13:45:00 | 5842.50 | 2025-05-30 14:15:00 | 5968.00 | STOP_HIT | 1.00 | 2.15% |
| BUY | retest2 | 2025-06-09 09:15:00 | 6083.00 | 2025-06-11 12:15:00 | 6105.00 | STOP_HIT | 1.00 | 0.36% |
| SELL | retest2 | 2025-06-23 09:15:00 | 5897.00 | 2025-06-24 09:15:00 | 6010.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-06-23 12:00:00 | 5930.50 | 2025-06-24 09:15:00 | 6010.00 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-06-23 13:45:00 | 5925.50 | 2025-06-24 09:15:00 | 6010.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-06-23 14:15:00 | 5930.00 | 2025-06-24 09:15:00 | 6010.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-06-25 09:15:00 | 6051.00 | 2025-06-25 11:15:00 | 5961.50 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-07-08 14:00:00 | 5838.00 | 2025-07-09 09:15:00 | 5886.50 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-07-09 09:15:00 | 5835.00 | 2025-07-09 09:15:00 | 5886.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-07-25 09:15:00 | 5668.00 | 2025-08-01 14:15:00 | 5384.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 09:15:00 | 5668.00 | 2025-08-04 10:15:00 | 5101.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-18 11:30:00 | 5028.00 | 2025-08-19 12:15:00 | 5073.00 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-08-19 09:15:00 | 5023.50 | 2025-08-19 12:15:00 | 5073.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-08-29 12:45:00 | 5025.00 | 2025-09-01 10:15:00 | 5085.10 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-08-29 13:45:00 | 5030.00 | 2025-09-01 10:15:00 | 5085.10 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-08-29 14:15:00 | 5023.00 | 2025-09-01 10:15:00 | 5085.10 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-09-04 14:30:00 | 5181.50 | 2025-09-05 09:15:00 | 5124.80 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-09-09 14:45:00 | 5125.00 | 2025-09-10 09:15:00 | 5176.20 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-09-11 11:00:00 | 5233.30 | 2025-09-22 15:15:00 | 5360.00 | STOP_HIT | 1.00 | 2.42% |
| BUY | retest2 | 2025-09-11 11:45:00 | 5214.60 | 2025-09-22 15:15:00 | 5360.00 | STOP_HIT | 1.00 | 2.79% |
| BUY | retest2 | 2025-09-11 12:45:00 | 5215.00 | 2025-09-22 15:15:00 | 5360.00 | STOP_HIT | 1.00 | 2.78% |
| BUY | retest2 | 2025-09-11 13:30:00 | 5213.10 | 2025-09-22 15:15:00 | 5360.00 | STOP_HIT | 1.00 | 2.82% |
| BUY | retest2 | 2025-09-18 09:15:00 | 5404.60 | 2025-09-22 15:15:00 | 5360.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-09-30 09:30:00 | 5184.90 | 2025-09-30 10:15:00 | 5230.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-09-30 13:30:00 | 5184.50 | 2025-10-06 09:15:00 | 5207.50 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-09-30 14:00:00 | 5185.80 | 2025-10-06 09:15:00 | 5207.50 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-09-30 15:00:00 | 5186.20 | 2025-10-06 10:15:00 | 5208.50 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-10-01 11:15:00 | 5154.50 | 2025-10-06 10:15:00 | 5208.50 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-10-01 12:15:00 | 5153.00 | 2025-10-06 10:15:00 | 5208.50 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest1 | 2025-11-12 11:15:00 | 4946.50 | 2025-11-14 14:15:00 | 4957.00 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-12-01 09:30:00 | 5184.00 | 2025-12-01 11:15:00 | 5158.50 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-12-01 10:00:00 | 5182.50 | 2025-12-01 11:15:00 | 5158.50 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-12-01 12:45:00 | 5184.00 | 2025-12-02 11:15:00 | 5167.50 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-12-01 14:00:00 | 5185.00 | 2025-12-02 11:15:00 | 5167.50 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-12-02 10:45:00 | 5207.50 | 2025-12-02 11:15:00 | 5167.50 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-12-02 13:45:00 | 5201.00 | 2025-12-02 15:15:00 | 5165.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-12-04 11:15:00 | 5127.00 | 2025-12-04 12:15:00 | 5167.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-12-15 11:15:00 | 5263.00 | 2025-12-17 10:15:00 | 5219.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-12-16 09:30:00 | 5260.00 | 2025-12-17 10:15:00 | 5219.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-12-16 12:45:00 | 5257.50 | 2025-12-17 10:15:00 | 5219.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-01-13 13:00:00 | 4980.50 | 2026-01-20 10:15:00 | 4731.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 13:00:00 | 4980.50 | 2026-01-21 11:15:00 | 4738.00 | STOP_HIT | 0.50 | 4.87% |
| BUY | retest2 | 2026-02-02 15:00:00 | 5477.50 | 2026-02-12 11:15:00 | 5765.00 | STOP_HIT | 1.00 | 5.25% |
| BUY | retest2 | 2026-03-12 10:45:00 | 6277.00 | 2026-03-16 11:15:00 | 6195.50 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2026-03-16 09:30:00 | 6302.50 | 2026-03-16 11:15:00 | 6195.50 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-03-24 10:15:00 | 6076.00 | 2026-03-24 12:15:00 | 6227.50 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2026-04-01 10:45:00 | 6043.00 | 2026-04-02 14:15:00 | 6146.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-04-01 14:45:00 | 6043.00 | 2026-04-02 14:15:00 | 6146.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-04-02 09:15:00 | 5963.00 | 2026-04-02 14:15:00 | 6146.00 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2026-04-07 11:15:00 | 6196.00 | 2026-04-10 11:15:00 | 6815.60 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-05-04 13:15:00 | 7200.50 | 2026-05-05 13:15:00 | 7281.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-05-04 14:15:00 | 7219.00 | 2026-05-05 13:15:00 | 7281.00 | STOP_HIT | 1.00 | -0.86% |
