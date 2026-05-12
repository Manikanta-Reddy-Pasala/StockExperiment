# BRITANNIA (BRITANNIA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 5516.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 2 |
| ALERT3 | 54 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 48 |
| PARTIAL | 14 |
| TARGET_HIT | 4 |
| STOP_HIT | 49 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 67 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 28 / 39
- **Target hits / Stop hits / Partials:** 4 / 49 / 14
- **Avg / median % per leg:** 1.15% / -0.72%
- **Sum % (uncompounded):** 76.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 42 | 12 | 28.6% | 4 | 32 | 6 | 0.71% | 29.6% |
| BUY @ 2nd Alert (retest1) | 12 | 12 | 100.0% | 4 | 2 | 6 | 6.32% | 75.8% |
| BUY @ 3rd Alert (retest2) | 30 | 0 | 0.0% | 0 | 30 | 0 | -1.54% | -46.2% |
| SELL (all) | 25 | 16 | 64.0% | 0 | 17 | 8 | 1.89% | 47.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 25 | 16 | 64.0% | 0 | 17 | 8 | 1.89% | 47.3% |
| retest1 (combined) | 12 | 12 | 100.0% | 4 | 2 | 6 | 6.32% | 75.8% |
| retest2 (combined) | 55 | 16 | 29.1% | 0 | 47 | 8 | 0.02% | 1.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-02 14:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-02 14:30:00 | 5775.85 | 5694.90 | 5517.71 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-02 15:15:00 | 5766.00 | 5694.90 | 5517.71 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-05 10:00:00 | 5780.55 | 5696.45 | 5520.25 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-05 11:30:00 | 5755.00 | 5697.39 | 5522.48 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 11:15:00 | 6042.75 | 5816.06 | 5701.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 09:15:00 | 6064.64 | 5826.03 | 5709.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 09:15:00 | 6054.30 | 5826.03 | 5709.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 09:15:00 | 6069.58 | 5826.03 | 5709.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-09-27 12:15:00 | 6330.50 | 6014.55 | 5858.85 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 2 — SELL (started 2024-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 13:15:00 | 5618.20 | 5885.26 | 5885.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 11:15:00 | 5593.65 | 5872.87 | 5879.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 4928.45 | 4878.91 | 5087.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-09 10:00:00 | 4928.45 | 4878.91 | 5087.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 5024.50 | 4890.78 | 5030.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 12:45:00 | 5031.95 | 4890.78 | 5030.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 5030.50 | 4892.17 | 5030.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:30:00 | 5030.60 | 4892.17 | 5030.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 5013.25 | 4893.37 | 5030.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 09:15:00 | 5002.00 | 4937.27 | 5039.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 13:00:00 | 5007.00 | 4940.36 | 5038.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 14:15:00 | 5043.20 | 4942.15 | 5038.83 | SL hit (close>static) qty=1.00 sl=5032.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 14:15:00 | 5343.60 | 4914.34 | 4913.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 5361.05 | 4931.18 | 4922.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 10:15:00 | 5520.00 | 5524.42 | 5407.43 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 11:45:00 | 5550.00 | 5524.65 | 5408.13 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 12:30:00 | 5544.00 | 5526.40 | 5413.60 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 14:15:00 | 5827.50 | 5558.37 | 5442.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 14:15:00 | 5821.20 | 5558.37 | 5442.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 5708.00 | 5726.85 | 5601.09 | SL hit (close<ema200) qty=0.50 sl=5726.85 alert=retest1 |

### Cycle 4 — SELL (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 09:15:00 | 5412.50 | 5574.88 | 5575.53 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 11:15:00 | 5662.50 | 5576.69 | 5576.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-26 09:15:00 | 5690.00 | 5577.90 | 5576.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 14:15:00 | 5931.00 | 5948.54 | 5817.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-23 15:00:00 | 5931.00 | 5948.54 | 5817.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 5844.00 | 5953.72 | 5856.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:15:00 | 5837.50 | 5953.72 | 5856.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 5811.50 | 5952.31 | 5856.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 5811.50 | 5952.31 | 5856.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 5840.00 | 5951.19 | 5856.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 14:00:00 | 5870.00 | 5940.97 | 5855.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 11:45:00 | 5880.00 | 5933.87 | 5856.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 11:15:00 | 5806.00 | 5928.05 | 5856.46 | SL hit (close<static) qty=1.00 sl=5810.50 alert=retest2 |

### Cycle 6 — SELL (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 14:15:00 | 5816.50 | 5877.86 | 5877.98 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 5941.50 | 5878.11 | 5878.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 5960.00 | 5878.92 | 5878.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 11:15:00 | 5863.00 | 5880.03 | 5879.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 11:15:00 | 5863.00 | 5880.03 | 5879.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 5863.00 | 5880.03 | 5879.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 5863.00 | 5880.03 | 5879.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 5844.00 | 5879.67 | 5878.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 5844.00 | 5879.67 | 5878.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 11:15:00 | 5833.00 | 5877.67 | 5877.88 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 5955.00 | 5878.32 | 5878.20 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 5826.00 | 5877.86 | 5877.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 12:15:00 | 5820.50 | 5877.29 | 5877.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 5891.00 | 5875.44 | 5876.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 11:15:00 | 5891.00 | 5875.44 | 5876.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 5891.00 | 5875.44 | 5876.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 5891.00 | 5875.44 | 5876.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 5866.50 | 5875.35 | 5876.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:00:00 | 5844.50 | 5875.04 | 5876.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 09:45:00 | 5857.00 | 5874.03 | 5875.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 13:15:00 | 5900.00 | 5873.88 | 5875.87 | SL hit (close>static) qty=1.00 sl=5892.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 11:15:00 | 6010.00 | 5878.55 | 5878.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 13:15:00 | 6039.50 | 5881.66 | 5879.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 12:15:00 | 5955.00 | 5977.79 | 5940.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 12:45:00 | 5960.00 | 5977.79 | 5940.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 5992.00 | 5978.06 | 5941.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 10:15:00 | 5999.50 | 5978.06 | 5941.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 12:30:00 | 6007.00 | 5979.40 | 5942.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 11:30:00 | 6005.00 | 5981.83 | 5944.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 11:00:00 | 6002.00 | 6001.67 | 5957.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 5964.50 | 6003.49 | 5960.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:45:00 | 5965.00 | 6003.49 | 5960.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 5948.00 | 6002.94 | 5960.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:00:00 | 5948.00 | 6002.94 | 5960.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 5975.50 | 6002.67 | 5960.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 15:15:00 | 6000.00 | 6002.67 | 5960.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 11:15:00 | 5920.00 | 6000.96 | 5960.13 | SL hit (close<static) qty=1.00 sl=5930.50 alert=retest2 |

### Cycle 12 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 5712.00 | 5934.85 | 5935.12 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 6071.00 | 5925.92 | 5925.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 13:15:00 | 6120.00 | 5929.32 | 5927.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 14:15:00 | 5992.00 | 6019.06 | 5979.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 15:00:00 | 5992.00 | 6019.06 | 5979.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 6000.00 | 6018.87 | 5980.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 5927.50 | 6018.87 | 5980.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 5980.50 | 6018.49 | 5980.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 11:30:00 | 5993.00 | 6017.65 | 5980.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 14:45:00 | 5993.50 | 6002.63 | 5975.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 13:00:00 | 5995.50 | 6001.67 | 5975.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 13:45:00 | 5994.00 | 6001.59 | 5975.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 5988.50 | 6001.45 | 5975.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 5988.50 | 6001.45 | 5975.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 5989.50 | 6001.34 | 5975.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 5856.50 | 6001.34 | 5975.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 5828.50 | 5999.62 | 5974.95 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 5828.50 | 5999.62 | 5974.95 | SL hit (close<static) qty=1.00 sl=5866.00 alert=retest2 |

### Cycle 14 — SELL (started 2026-03-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 13:15:00 | 5806.50 | 5955.73 | 5955.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 5707.00 | 5939.33 | 5947.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 09:15:00 | 5682.00 | 5669.91 | 5769.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-17 10:00:00 | 5682.00 | 5669.91 | 5769.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 5740.00 | 5676.24 | 5765.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:30:00 | 5760.00 | 5676.24 | 5765.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 5830.00 | 5679.28 | 5765.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 15:00:00 | 5830.00 | 5679.28 | 5765.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 5850.00 | 5680.98 | 5765.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 5785.50 | 5680.98 | 5765.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 5800.00 | 5683.00 | 5766.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:45:00 | 5816.00 | 5683.00 | 5766.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 5786.00 | 5684.03 | 5766.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 11:30:00 | 5810.00 | 5684.03 | 5766.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 5767.00 | 5685.75 | 5766.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 15:00:00 | 5722.50 | 5686.11 | 5765.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:00:00 | 5737.50 | 5689.15 | 5760.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 5818.00 | 5692.68 | 5753.38 | SL hit (close>static) qty=1.00 sl=5775.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-02 14:30:00 | 5775.85 | 2024-09-11 11:15:00 | 6042.75 | PARTIAL | 0.50 | 4.62% |
| BUY | retest1 | 2024-08-02 15:15:00 | 5766.00 | 2024-09-12 09:15:00 | 6064.64 | PARTIAL | 0.50 | 5.18% |
| BUY | retest1 | 2024-08-05 10:00:00 | 5780.55 | 2024-09-12 09:15:00 | 6054.30 | PARTIAL | 0.50 | 4.74% |
| BUY | retest1 | 2024-08-05 11:30:00 | 5755.00 | 2024-09-12 09:15:00 | 6069.58 | PARTIAL | 0.50 | 5.47% |
| BUY | retest1 | 2024-08-02 14:30:00 | 5775.85 | 2024-09-27 12:15:00 | 6330.50 | TARGET_HIT | 0.50 | 9.60% |
| BUY | retest1 | 2024-08-02 15:15:00 | 5766.00 | 2024-09-30 09:15:00 | 6353.44 | TARGET_HIT | 0.50 | 10.19% |
| BUY | retest1 | 2024-08-05 10:00:00 | 5780.55 | 2024-09-30 09:15:00 | 6342.60 | TARGET_HIT | 0.50 | 9.72% |
| BUY | retest1 | 2024-08-05 11:30:00 | 5755.00 | 2024-09-30 14:15:00 | 6358.61 | TARGET_HIT | 0.50 | 10.49% |
| BUY | retest2 | 2024-10-15 09:30:00 | 6028.00 | 2024-10-18 11:15:00 | 5905.95 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2024-10-15 11:00:00 | 6025.00 | 2024-10-18 11:15:00 | 5905.95 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-10-17 10:15:00 | 6052.85 | 2024-10-18 11:15:00 | 5905.95 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2024-10-17 11:15:00 | 6029.20 | 2024-10-18 11:15:00 | 5905.95 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-01-29 09:15:00 | 5002.00 | 2025-01-29 14:15:00 | 5043.20 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-01-29 13:00:00 | 5007.00 | 2025-01-29 14:15:00 | 5043.20 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-02-04 13:00:00 | 5002.60 | 2025-02-04 15:15:00 | 5040.30 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-02-05 09:15:00 | 4967.65 | 2025-02-24 09:15:00 | 4719.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-05 14:15:00 | 4927.15 | 2025-02-24 09:15:00 | 4712.00 | PARTIAL | 0.50 | 4.37% |
| SELL | retest2 | 2025-02-06 09:45:00 | 4942.35 | 2025-02-24 09:15:00 | 4704.97 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2025-02-06 10:15:00 | 4942.70 | 2025-02-28 10:15:00 | 4680.79 | PARTIAL | 0.50 | 5.30% |
| SELL | retest2 | 2025-02-06 11:00:00 | 4933.10 | 2025-02-28 10:15:00 | 4695.23 | PARTIAL | 0.50 | 4.82% |
| SELL | retest2 | 2025-02-07 10:15:00 | 4915.80 | 2025-02-28 10:15:00 | 4695.56 | PARTIAL | 0.50 | 4.48% |
| SELL | retest2 | 2025-02-10 11:45:00 | 4960.00 | 2025-02-28 10:15:00 | 4686.44 | PARTIAL | 0.50 | 5.52% |
| SELL | retest2 | 2025-02-17 15:15:00 | 4952.60 | 2025-02-28 12:15:00 | 4670.01 | PARTIAL | 0.50 | 5.71% |
| SELL | retest2 | 2025-02-05 09:15:00 | 4967.65 | 2025-03-13 09:15:00 | 4829.00 | STOP_HIT | 0.50 | 2.79% |
| SELL | retest2 | 2025-02-05 14:15:00 | 4927.15 | 2025-03-13 09:15:00 | 4829.00 | STOP_HIT | 0.50 | 1.99% |
| SELL | retest2 | 2025-02-06 09:45:00 | 4942.35 | 2025-03-13 09:15:00 | 4829.00 | STOP_HIT | 0.50 | 2.29% |
| SELL | retest2 | 2025-02-06 10:15:00 | 4942.70 | 2025-03-13 09:15:00 | 4829.00 | STOP_HIT | 0.50 | 2.30% |
| SELL | retest2 | 2025-02-06 11:00:00 | 4933.10 | 2025-03-13 09:15:00 | 4829.00 | STOP_HIT | 0.50 | 2.11% |
| SELL | retest2 | 2025-02-07 10:15:00 | 4915.80 | 2025-03-13 09:15:00 | 4829.00 | STOP_HIT | 0.50 | 1.77% |
| SELL | retest2 | 2025-02-10 11:45:00 | 4960.00 | 2025-03-13 09:15:00 | 4829.00 | STOP_HIT | 0.50 | 2.64% |
| SELL | retest2 | 2025-02-17 15:15:00 | 4952.60 | 2025-03-13 09:15:00 | 4829.00 | STOP_HIT | 0.50 | 2.50% |
| SELL | retest2 | 2025-03-28 11:15:00 | 4953.25 | 2025-04-04 09:15:00 | 5106.50 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest1 | 2025-06-20 11:45:00 | 5550.00 | 2025-06-26 14:15:00 | 5827.50 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-23 12:30:00 | 5544.00 | 2025-06-26 14:15:00 | 5821.20 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-20 11:45:00 | 5550.00 | 2025-07-21 09:15:00 | 5708.00 | STOP_HIT | 0.50 | 2.85% |
| BUY | retest1 | 2025-06-23 12:30:00 | 5544.00 | 2025-07-21 09:15:00 | 5708.00 | STOP_HIT | 0.50 | 2.96% |
| BUY | retest2 | 2025-07-28 09:15:00 | 5614.50 | 2025-08-06 09:15:00 | 5530.50 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-07-29 12:45:00 | 5616.50 | 2025-08-06 09:15:00 | 5530.50 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-07-29 14:00:00 | 5623.50 | 2025-08-06 09:15:00 | 5530.50 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-10-09 14:00:00 | 5870.00 | 2025-10-14 11:15:00 | 5806.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-10-13 11:45:00 | 5880.00 | 2025-10-14 11:15:00 | 5806.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-10-15 14:30:00 | 5860.50 | 2025-10-28 11:15:00 | 5830.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-10-15 15:15:00 | 5863.00 | 2025-10-28 11:15:00 | 5830.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-10-16 09:15:00 | 5920.00 | 2025-10-28 13:15:00 | 5808.50 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-10-27 14:00:00 | 5903.00 | 2025-10-28 13:15:00 | 5808.50 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-10-28 15:15:00 | 5885.00 | 2025-10-29 15:15:00 | 5850.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-10-29 10:15:00 | 5880.00 | 2025-10-29 15:15:00 | 5850.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-11-04 13:30:00 | 5902.00 | 2025-11-11 09:15:00 | 5815.00 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-11-04 15:00:00 | 5902.50 | 2025-11-11 09:15:00 | 5815.00 | STOP_HIT | 1.00 | -1.48% |
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
