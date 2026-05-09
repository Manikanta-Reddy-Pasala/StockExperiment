# BRITANNIA (BRITANNIA)

## Backtest Summary

- **Window:** 2024-10-07 09:15:00 → 2026-05-08 15:15:00 (2741 bars)
- **Last close:** 5516.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 34 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 19 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 0 / 18
- **Target hits / Stop hits / Partials:** 0 / 18 / 0
- **Avg / median % per leg:** -1.54% / -1.36%
- **Sum % (uncompounded):** -27.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 0 | 0.0% | 0 | 13 | 0 | -1.70% | -22.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 0 | 0.0% | 0 | 13 | 0 | -1.70% | -22.0% |
| SELL (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.14% | -5.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.14% | -5.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 18 | 0 | 0.0% | 0 | 18 | 0 | -1.54% | -27.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 14:15:00 | 5816.50 | 5877.86 | 5878.01 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 5941.50 | 5878.11 | 5878.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 5960.00 | 5878.92 | 5878.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 11:15:00 | 5863.00 | 5880.03 | 5879.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 11:15:00 | 5863.00 | 5880.03 | 5879.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 5863.00 | 5880.03 | 5879.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 5863.00 | 5880.03 | 5879.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 5844.00 | 5879.67 | 5878.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 5844.00 | 5879.67 | 5878.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 10:15:00 | 5849.00 | 5878.11 | 5878.14 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 5955.00 | 5878.32 | 5878.23 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 5826.00 | 5877.86 | 5878.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 12:15:00 | 5820.50 | 5877.29 | 5877.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 5891.00 | 5875.44 | 5876.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 11:15:00 | 5891.00 | 5875.44 | 5876.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 5891.00 | 5875.44 | 5876.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 5891.00 | 5875.44 | 5876.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 5866.50 | 5875.35 | 5876.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:00:00 | 5844.50 | 5875.04 | 5876.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 09:45:00 | 5857.00 | 5874.03 | 5876.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 13:15:00 | 5900.00 | 5873.88 | 5875.90 | SL hit (close>static) qty=1.00 sl=5892.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-12 13:15:00 | 5900.00 | 5873.88 | 5875.90 | SL hit (close>static) qty=1.00 sl=5892.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 11:15:00 | 6010.00 | 5878.55 | 5878.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 13:15:00 | 6039.50 | 5881.66 | 5879.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 12:15:00 | 5955.00 | 5977.79 | 5940.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 12:45:00 | 5960.00 | 5977.79 | 5940.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 5992.00 | 5978.06 | 5941.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 10:15:00 | 5999.50 | 5978.06 | 5941.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 12:30:00 | 6007.00 | 5979.40 | 5942.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 11:30:00 | 6005.00 | 5981.83 | 5944.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 11:00:00 | 6002.00 | 6001.67 | 5957.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 5964.50 | 6003.49 | 5960.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:45:00 | 5965.00 | 6003.49 | 5960.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 5948.00 | 6002.94 | 5960.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:00:00 | 5948.00 | 6002.94 | 5960.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 5975.50 | 6002.67 | 5960.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 15:15:00 | 6000.00 | 6002.67 | 5960.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 11:15:00 | 5920.00 | 6000.96 | 5960.14 | SL hit (close<static) qty=1.00 sl=5930.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 11:15:00 | 5920.00 | 6000.96 | 5960.14 | SL hit (close<static) qty=1.00 sl=5930.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 11:15:00 | 5920.00 | 6000.96 | 5960.14 | SL hit (close<static) qty=1.00 sl=5930.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 11:15:00 | 5920.00 | 6000.96 | 5960.14 | SL hit (close<static) qty=1.00 sl=5930.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 11:15:00 | 5920.00 | 6000.96 | 5960.14 | SL hit (close<static) qty=1.00 sl=5930.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 10:30:00 | 5983.50 | 5977.68 | 5953.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 11:15:00 | 5984.00 | 5977.68 | 5953.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 12:15:00 | 5902.50 | 5976.84 | 5953.68 | SL hit (close<static) qty=1.00 sl=5930.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 12:15:00 | 5902.50 | 5976.84 | 5953.68 | SL hit (close<static) qty=1.00 sl=5930.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 5712.00 | 5934.85 | 5935.13 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 6071.00 | 5925.92 | 5925.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 13:15:00 | 6120.00 | 5929.32 | 5927.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 14:15:00 | 5992.00 | 6019.06 | 5979.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 15:00:00 | 5992.00 | 6019.06 | 5979.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 6000.00 | 6018.87 | 5980.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 5927.50 | 6018.87 | 5980.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 5980.50 | 6018.49 | 5980.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 11:30:00 | 5993.00 | 6017.65 | 5980.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 14:45:00 | 5993.50 | 6002.63 | 5975.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 13:00:00 | 5995.50 | 6001.67 | 5975.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 13:45:00 | 5994.00 | 6001.59 | 5975.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 5988.50 | 6001.45 | 5975.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 5988.50 | 6001.45 | 5975.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 5989.50 | 6001.34 | 5975.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 5856.50 | 6001.34 | 5975.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 5828.50 | 5999.62 | 5974.95 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 5828.50 | 5999.62 | 5974.95 | SL hit (close<static) qty=1.00 sl=5866.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 5828.50 | 5999.62 | 5974.95 | SL hit (close<static) qty=1.00 sl=5866.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 5828.50 | 5999.62 | 5974.95 | SL hit (close<static) qty=1.00 sl=5866.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 5828.50 | 5999.62 | 5974.95 | SL hit (close<static) qty=1.00 sl=5866.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-03-09 09:45:00 | 5840.00 | 5999.62 | 5974.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 5973.00 | 5992.33 | 5972.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 12:00:00 | 5973.00 | 5992.33 | 5972.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 5970.00 | 5992.11 | 5972.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 12:30:00 | 5977.00 | 5992.11 | 5972.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 5962.50 | 5991.81 | 5972.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:00:00 | 5962.50 | 5991.81 | 5972.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 5961.50 | 5991.51 | 5972.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:30:00 | 5954.50 | 5991.51 | 5972.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 5932.00 | 5990.81 | 5972.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:00:00 | 5932.00 | 5990.81 | 5972.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 5928.00 | 5990.18 | 5971.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:15:00 | 5910.00 | 5990.18 | 5971.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2026-03-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 13:15:00 | 5806.50 | 5955.73 | 5955.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 5707.00 | 5939.33 | 5947.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 09:15:00 | 5682.00 | 5669.91 | 5769.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-17 10:00:00 | 5682.00 | 5669.91 | 5769.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 5740.00 | 5676.24 | 5765.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:30:00 | 5760.00 | 5676.24 | 5765.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 5830.00 | 5679.28 | 5765.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 15:00:00 | 5830.00 | 5679.28 | 5765.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 5850.00 | 5680.98 | 5765.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 5785.50 | 5680.98 | 5765.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 5800.00 | 5683.00 | 5766.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:45:00 | 5816.00 | 5683.00 | 5766.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 5786.00 | 5684.03 | 5766.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 11:30:00 | 5810.00 | 5684.03 | 5766.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 5767.00 | 5685.75 | 5766.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 15:00:00 | 5722.50 | 5686.11 | 5765.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:00:00 | 5737.50 | 5689.15 | 5760.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 5818.00 | 5692.68 | 5753.39 | SL hit (close>static) qty=1.00 sl=5775.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 5818.00 | 5692.68 | 5753.39 | SL hit (close>static) qty=1.00 sl=5775.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 5738.00 | 5693.13 | 5753.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 13:15:00 | 5792.50 | 5694.83 | 5753.56 | SL hit (close>static) qty=1.00 sl=5775.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 5537.00 | 5717.10 | 5758.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-11-11 11:45:00 | 5902.00 | 2025-11-13 11:15:00 | 5858.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-11-11 12:15:00 | 5896.50 | 2025-11-13 11:15:00 | 5858.00 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-12-11 14:00:00 | 5844.50 | 2025-12-12 13:15:00 | 5900.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-12-12 09:45:00 | 5857.00 | 2025-12-12 13:15:00 | 5900.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2026-01-05 10:15:00 | 5999.50 | 2026-01-12 11:15:00 | 5920.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-01-05 12:30:00 | 6007.00 | 2026-01-12 11:15:00 | 5920.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2026-01-06 11:30:00 | 6005.00 | 2026-01-12 11:15:00 | 5920.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2026-01-08 11:00:00 | 6002.00 | 2026-01-12 11:15:00 | 5920.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-01-09 15:15:00 | 6000.00 | 2026-01-12 11:15:00 | 5920.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-01-20 10:30:00 | 5983.50 | 2026-01-20 12:15:00 | 5902.50 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2026-01-20 11:15:00 | 5984.00 | 2026-01-20 12:15:00 | 5902.50 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2026-03-02 11:30:00 | 5993.00 | 2026-03-09 09:15:00 | 5828.50 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2026-03-05 14:45:00 | 5993.50 | 2026-03-09 09:15:00 | 5828.50 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2026-03-06 13:00:00 | 5995.50 | 2026-03-09 09:15:00 | 5828.50 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2026-03-06 13:45:00 | 5994.00 | 2026-03-09 09:15:00 | 5828.50 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2026-04-22 15:00:00 | 5722.50 | 2026-05-04 10:15:00 | 5818.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2026-04-27 13:00:00 | 5737.50 | 2026-05-04 10:15:00 | 5818.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-05-04 12:00:00 | 5738.00 | 2026-05-04 13:15:00 | 5792.50 | STOP_HIT | 1.00 | -0.95% |
