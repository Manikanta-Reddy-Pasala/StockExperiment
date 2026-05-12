# Polycab India Ltd. (POLYCAB)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 9080.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 75 |
| ALERT1 | 58 |
| ALERT2 | 58 |
| ALERT2_SKIP | 30 |
| ALERT3 | 136 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 37 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 37 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 41 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 13 / 28
- **Target hits / Stop hits / Partials:** 2 / 37 / 2
- **Avg / median % per leg:** 0.16% / -0.89%
- **Sum % (uncompounded):** 6.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 6 | 25.0% | 2 | 22 | 0 | 0.04% | 1.0% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 3 | 0 | 0.02% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 4 | 19.0% | 2 | 19 | 0 | 0.05% | 1.0% |
| SELL (all) | 17 | 7 | 41.2% | 0 | 15 | 2 | 0.33% | 5.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 17 | 7 | 41.2% | 0 | 15 | 2 | 0.33% | 5.6% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 3 | 0 | 0.02% | 0.0% |
| retest2 (combined) | 38 | 11 | 28.9% | 2 | 34 | 2 | 0.17% | 6.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 5910.00 | 5817.54 | 5808.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 5956.00 | 5858.42 | 5829.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 5967.50 | 5989.12 | 5943.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 10:00:00 | 5967.50 | 5989.12 | 5943.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 5962.00 | 5979.44 | 5946.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:45:00 | 5945.50 | 5979.44 | 5946.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 5953.00 | 5974.15 | 5947.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 5934.00 | 5974.15 | 5947.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 5946.00 | 5968.52 | 5947.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 15:00:00 | 5979.50 | 5970.72 | 5950.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 14:15:00 | 6020.50 | 6070.57 | 6071.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 6020.50 | 6070.57 | 6071.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 10:15:00 | 5987.00 | 6030.94 | 6046.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 5935.00 | 5919.25 | 5957.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 5935.00 | 5919.25 | 5957.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 5935.00 | 5919.25 | 5957.49 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 10:15:00 | 5989.00 | 5963.71 | 5961.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 12:15:00 | 6022.50 | 5979.92 | 5969.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 09:15:00 | 6011.00 | 6014.70 | 5992.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 09:30:00 | 6029.50 | 6014.70 | 5992.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 5977.50 | 6007.26 | 5990.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:00:00 | 5977.50 | 6007.26 | 5990.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 6003.00 | 6006.41 | 5991.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 13:00:00 | 6016.00 | 6008.32 | 5994.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 15:15:00 | 6018.50 | 6001.19 | 5992.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 11:15:00 | 5969.50 | 5990.52 | 5990.50 | SL hit (close<static) qty=1.00 sl=5970.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 12:15:00 | 5973.50 | 5987.12 | 5988.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 10:15:00 | 5935.00 | 5971.27 | 5980.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 13:15:00 | 5890.00 | 5885.10 | 5906.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 13:45:00 | 5900.50 | 5885.10 | 5906.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 5923.50 | 5893.31 | 5905.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:00:00 | 5923.50 | 5893.31 | 5905.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 5934.50 | 5901.55 | 5907.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:30:00 | 5945.00 | 5901.55 | 5907.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 5973.00 | 5915.84 | 5913.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 12:15:00 | 5997.50 | 5932.17 | 5921.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 5965.50 | 5966.92 | 5945.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 11:00:00 | 5965.50 | 5966.92 | 5945.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 6150.50 | 6166.59 | 6136.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 6152.00 | 6166.59 | 6136.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 6129.50 | 6158.97 | 6140.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 6113.50 | 6158.97 | 6140.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 6121.00 | 6151.37 | 6138.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:45:00 | 6118.00 | 6151.37 | 6138.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 6081.00 | 6130.60 | 6130.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 6020.50 | 6108.58 | 6120.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 6083.00 | 6042.13 | 6063.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 10:15:00 | 6083.00 | 6042.13 | 6063.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 6083.00 | 6042.13 | 6063.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:45:00 | 6076.00 | 6042.13 | 6063.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 6070.00 | 6047.71 | 6064.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 6070.00 | 6047.71 | 6064.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 14:15:00 | 6108.00 | 6078.51 | 6075.64 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 11:15:00 | 6030.00 | 6069.10 | 6072.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 12:15:00 | 5987.50 | 6052.78 | 6065.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 6075.50 | 6049.91 | 6058.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 6075.50 | 6049.91 | 6058.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 6075.50 | 6049.91 | 6058.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:30:00 | 6038.00 | 6045.53 | 6055.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 6139.50 | 5965.99 | 5951.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 09:15:00 | 6139.50 | 5965.99 | 5951.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 10:15:00 | 6164.50 | 6005.69 | 5970.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 12:15:00 | 6411.00 | 6418.67 | 6351.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 12:45:00 | 6412.50 | 6418.67 | 6351.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 6764.00 | 6802.68 | 6753.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:00:00 | 6764.00 | 6802.68 | 6753.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 6721.50 | 6786.44 | 6750.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:00:00 | 6721.50 | 6786.44 | 6750.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 6694.00 | 6767.95 | 6745.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:30:00 | 6689.50 | 6767.95 | 6745.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 10:15:00 | 6717.00 | 6734.68 | 6735.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 12:15:00 | 6687.00 | 6718.31 | 6727.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 12:15:00 | 6693.00 | 6683.55 | 6701.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 12:45:00 | 6697.50 | 6683.55 | 6701.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 6710.50 | 6688.94 | 6702.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:45:00 | 6715.50 | 6688.94 | 6702.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 6756.00 | 6702.35 | 6707.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 6756.00 | 6702.35 | 6707.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 6740.00 | 6709.88 | 6710.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 6715.00 | 6709.88 | 6710.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 6738.00 | 6715.51 | 6712.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 15:15:00 | 6800.00 | 6741.31 | 6728.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 6728.00 | 6738.65 | 6728.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 6728.00 | 6738.65 | 6728.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 6728.00 | 6738.65 | 6728.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 6728.00 | 6738.65 | 6728.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 6690.00 | 6728.92 | 6725.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 6690.00 | 6728.92 | 6725.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 11:15:00 | 6688.00 | 6720.73 | 6721.87 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 09:15:00 | 6806.00 | 6723.16 | 6720.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 6892.50 | 6796.69 | 6763.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 6900.50 | 6936.15 | 6894.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 6900.50 | 6936.15 | 6894.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 6900.50 | 6936.15 | 6894.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 7014.00 | 6883.67 | 6880.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 13:15:00 | 7001.50 | 6949.54 | 6917.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 09:15:00 | 6980.50 | 6945.83 | 6924.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:00:00 | 6985.00 | 6953.66 | 6929.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 6975.00 | 7004.21 | 6973.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:00:00 | 6975.00 | 7004.21 | 6973.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 6996.50 | 7002.66 | 6976.04 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 6895.00 | 6966.84 | 6969.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 09:15:00 | 6895.00 | 6966.84 | 6969.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 10:15:00 | 6863.00 | 6946.07 | 6959.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 6814.50 | 6778.56 | 6820.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 6814.50 | 6778.56 | 6820.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 6814.50 | 6778.56 | 6820.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:45:00 | 6824.00 | 6778.56 | 6820.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 6806.00 | 6784.05 | 6819.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:30:00 | 6818.00 | 6784.05 | 6819.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 6799.50 | 6787.14 | 6817.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:30:00 | 6813.50 | 6787.14 | 6817.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 6767.50 | 6783.21 | 6813.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 12:30:00 | 6801.00 | 6783.21 | 6813.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 14:15:00 | 6840.00 | 6796.05 | 6813.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 14:45:00 | 6850.00 | 6796.05 | 6813.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 6833.00 | 6803.44 | 6815.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:15:00 | 6868.00 | 6803.44 | 6815.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 10:15:00 | 6866.00 | 6827.64 | 6825.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 12:15:00 | 6920.00 | 6853.21 | 6837.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 6771.00 | 6887.27 | 6880.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 6771.00 | 6887.27 | 6880.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 6771.00 | 6887.27 | 6880.16 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 6729.50 | 6855.72 | 6866.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 6692.50 | 6770.06 | 6808.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 6795.00 | 6761.44 | 6796.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 6795.00 | 6761.44 | 6796.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 6795.00 | 6761.44 | 6796.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:30:00 | 6785.00 | 6761.44 | 6796.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 6758.00 | 6760.75 | 6793.39 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 6918.00 | 6813.43 | 6808.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 10:15:00 | 6924.00 | 6864.81 | 6835.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 6784.00 | 6882.12 | 6862.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 6784.00 | 6882.12 | 6862.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 6784.00 | 6882.12 | 6862.78 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 12:15:00 | 6820.50 | 6851.81 | 6852.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 15:15:00 | 6795.00 | 6832.61 | 6842.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 6884.50 | 6823.16 | 6830.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 6884.50 | 6823.16 | 6830.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 6884.50 | 6823.16 | 6830.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 6884.50 | 6823.16 | 6830.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 6881.00 | 6834.73 | 6834.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 6872.00 | 6834.73 | 6834.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 10:15:00 | 6845.50 | 6836.20 | 6835.56 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 12:15:00 | 6814.00 | 6832.77 | 6834.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 13:15:00 | 6786.00 | 6823.42 | 6829.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 12:15:00 | 6730.50 | 6718.22 | 6753.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-12 13:00:00 | 6730.50 | 6718.22 | 6753.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 6799.50 | 6734.48 | 6757.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 13:45:00 | 6822.00 | 6734.48 | 6757.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 6801.00 | 6747.78 | 6761.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 6801.00 | 6747.78 | 6761.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 6927.00 | 6793.58 | 6780.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 10:15:00 | 6944.00 | 6823.66 | 6795.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 7050.00 | 7060.53 | 7003.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 09:45:00 | 7043.00 | 7060.53 | 7003.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 7106.00 | 7139.52 | 7110.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 7106.00 | 7139.52 | 7110.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 7090.50 | 7129.72 | 7108.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:15:00 | 7072.00 | 7129.72 | 7108.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 7108.50 | 7124.57 | 7115.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 15:15:00 | 7116.00 | 7124.57 | 7115.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 7116.00 | 7122.86 | 7115.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:15:00 | 7142.00 | 7122.86 | 7115.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 11:15:00 | 7086.00 | 7108.73 | 7110.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 11:15:00 | 7086.00 | 7108.73 | 7110.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 7072.00 | 7099.59 | 7105.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 7100.50 | 7078.43 | 7086.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 09:15:00 | 7100.50 | 7078.43 | 7086.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 7100.50 | 7078.43 | 7086.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:15:00 | 7140.50 | 7078.43 | 7086.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 7143.00 | 7091.35 | 7091.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:45:00 | 7150.50 | 7091.35 | 7091.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 11:15:00 | 7155.50 | 7104.18 | 7097.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 10:15:00 | 7180.00 | 7124.57 | 7111.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-29 12:15:00 | 7087.50 | 7119.63 | 7111.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 12:15:00 | 7087.50 | 7119.63 | 7111.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 7087.50 | 7119.63 | 7111.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:00:00 | 7087.50 | 7119.63 | 7111.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 7081.50 | 7112.00 | 7108.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 14:00:00 | 7081.50 | 7112.00 | 7108.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 14:15:00 | 7082.00 | 7106.00 | 7106.27 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 7141.50 | 7109.37 | 7107.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 7145.50 | 7116.60 | 7110.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 15:15:00 | 7250.00 | 7255.84 | 7222.16 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-05 09:15:00 | 7285.50 | 7255.84 | 7222.16 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 7250.00 | 7253.66 | 7231.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:30:00 | 7244.50 | 7253.66 | 7231.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 7219.50 | 7245.04 | 7231.62 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-05 14:15:00 | 7219.50 | 7245.04 | 7231.62 | SL hit (close<ema400) qty=1.00 sl=7231.62 alert=retest1 |

### Cycle 26 — SELL (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 13:15:00 | 7219.50 | 7262.79 | 7264.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 13:15:00 | 7185.00 | 7223.87 | 7235.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 15:15:00 | 7249.00 | 7223.32 | 7232.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 15:15:00 | 7249.00 | 7223.32 | 7232.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 7249.00 | 7223.32 | 7232.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 7262.50 | 7223.32 | 7232.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 7255.50 | 7229.75 | 7234.91 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2025-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 11:15:00 | 7259.50 | 7238.94 | 7238.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 12:15:00 | 7289.50 | 7249.05 | 7243.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 15:15:00 | 7250.00 | 7252.64 | 7246.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 15:15:00 | 7250.00 | 7252.64 | 7246.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 7250.00 | 7252.64 | 7246.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 7176.00 | 7252.64 | 7246.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 09:15:00 | 7183.50 | 7238.81 | 7240.79 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 7267.00 | 7241.10 | 7238.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 10:15:00 | 7322.00 | 7257.28 | 7246.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 10:15:00 | 7328.00 | 7336.47 | 7301.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 10:45:00 | 7339.00 | 7336.47 | 7301.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 7343.00 | 7337.77 | 7304.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:45:00 | 7335.00 | 7337.77 | 7304.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 7612.00 | 7641.40 | 7592.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:30:00 | 7578.50 | 7641.40 | 7592.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 7610.00 | 7637.22 | 7599.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:15:00 | 7565.00 | 7637.22 | 7599.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 7540.00 | 7617.77 | 7593.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:45:00 | 7558.50 | 7617.77 | 7593.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 7560.50 | 7606.32 | 7590.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:30:00 | 7563.50 | 7606.32 | 7590.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 14:15:00 | 7535.00 | 7581.78 | 7583.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 15:15:00 | 7527.00 | 7570.82 | 7578.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 09:15:00 | 7479.00 | 7467.57 | 7510.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-26 10:00:00 | 7479.00 | 7467.57 | 7510.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 7441.50 | 7462.36 | 7504.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 11:15:00 | 7417.50 | 7462.36 | 7504.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 10:15:00 | 7369.00 | 7327.91 | 7323.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 10:15:00 | 7369.00 | 7327.91 | 7323.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 11:15:00 | 7389.50 | 7340.23 | 7329.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 7573.00 | 7579.53 | 7522.11 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 10:45:00 | 7609.50 | 7582.02 | 7528.46 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 11:15:00 | 7628.00 | 7582.02 | 7528.46 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 7600.00 | 7597.20 | 7562.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:45:00 | 7628.50 | 7597.20 | 7562.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 7703.00 | 7710.61 | 7666.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:15:00 | 7660.00 | 7710.61 | 7666.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 7655.00 | 7699.48 | 7665.00 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-13 10:15:00 | 7655.00 | 7699.48 | 7665.00 | SL hit (close<ema400) qty=1.00 sl=7665.00 alert=retest1 |

### Cycle 32 — SELL (started 2025-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 14:15:00 | 7630.50 | 7657.14 | 7660.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 15:15:00 | 7572.50 | 7640.21 | 7652.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 7695.50 | 7651.27 | 7656.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 7695.50 | 7651.27 | 7656.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 7695.50 | 7651.27 | 7656.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 7695.50 | 7651.27 | 7656.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 7697.50 | 7660.52 | 7659.88 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 09:15:00 | 7601.50 | 7653.00 | 7657.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 13:15:00 | 7328.00 | 7542.61 | 7591.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 09:15:00 | 7630.00 | 7532.14 | 7572.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 7630.00 | 7532.14 | 7572.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 7630.00 | 7532.14 | 7572.04 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 13:15:00 | 7700.00 | 7611.83 | 7601.26 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 13:15:00 | 7576.50 | 7612.61 | 7612.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 14:15:00 | 7504.00 | 7590.89 | 7602.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 09:15:00 | 7585.50 | 7578.39 | 7594.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 09:15:00 | 7585.50 | 7578.39 | 7594.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 7585.50 | 7578.39 | 7594.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:30:00 | 7589.50 | 7578.39 | 7594.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 7592.50 | 7581.21 | 7594.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:30:00 | 7599.00 | 7581.21 | 7594.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 7577.00 | 7580.37 | 7592.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 12:15:00 | 7498.00 | 7580.37 | 7592.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 7620.00 | 7555.92 | 7571.95 | SL hit (close>static) qty=1.00 sl=7595.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 11:15:00 | 7690.00 | 7601.15 | 7590.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 11:15:00 | 7703.00 | 7661.48 | 7631.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 12:15:00 | 7658.50 | 7660.88 | 7634.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 13:00:00 | 7658.50 | 7660.88 | 7634.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 7620.00 | 7652.71 | 7632.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:45:00 | 7620.00 | 7652.71 | 7632.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 7502.00 | 7622.57 | 7620.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:45:00 | 7514.50 | 7622.57 | 7620.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 15:15:00 | 7500.00 | 7598.05 | 7609.94 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 7659.00 | 7622.00 | 7618.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 13:15:00 | 7679.00 | 7639.32 | 7627.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 7795.50 | 7808.48 | 7749.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 11:00:00 | 7795.50 | 7808.48 | 7749.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 7747.00 | 7796.19 | 7749.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:00:00 | 7747.00 | 7796.19 | 7749.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 7760.00 | 7788.95 | 7750.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 13:45:00 | 7773.00 | 7786.16 | 7752.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 7692.00 | 7767.33 | 7747.02 | SL hit (close<static) qty=1.00 sl=7736.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 7650.50 | 7726.44 | 7731.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 11:15:00 | 7596.00 | 7700.35 | 7719.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 12:15:00 | 7657.50 | 7651.64 | 7676.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-04 12:45:00 | 7658.50 | 7651.64 | 7676.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 7630.00 | 7617.36 | 7650.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:30:00 | 7660.00 | 7617.36 | 7650.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 7642.00 | 7622.29 | 7649.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:00:00 | 7642.00 | 7622.29 | 7649.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 7603.50 | 7618.53 | 7645.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 15:00:00 | 7592.50 | 7613.12 | 7635.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 15:15:00 | 7597.50 | 7581.32 | 7587.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 7654.00 | 7598.45 | 7594.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 09:15:00 | 7654.00 | 7598.45 | 7594.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 15:15:00 | 7662.00 | 7637.97 | 7618.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 15:15:00 | 7655.00 | 7677.18 | 7654.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 15:15:00 | 7655.00 | 7677.18 | 7654.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 7655.00 | 7677.18 | 7654.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 7714.50 | 7677.18 | 7654.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 12:15:00 | 7642.00 | 7687.68 | 7688.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 7642.00 | 7687.68 | 7688.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 7636.50 | 7677.45 | 7683.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 12:15:00 | 7672.50 | 7646.06 | 7661.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 12:15:00 | 7672.50 | 7646.06 | 7661.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 7672.50 | 7646.06 | 7661.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:00:00 | 7672.50 | 7646.06 | 7661.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 7697.00 | 7656.25 | 7664.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:45:00 | 7695.50 | 7656.25 | 7664.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 7678.50 | 7666.99 | 7667.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:30:00 | 7693.00 | 7666.99 | 7667.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 10:15:00 | 7685.00 | 7670.59 | 7669.37 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 15:15:00 | 7649.50 | 7674.22 | 7676.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 7559.50 | 7651.28 | 7665.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 7479.00 | 7466.14 | 7523.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 10:00:00 | 7479.00 | 7466.14 | 7523.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 7505.00 | 7481.23 | 7516.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:30:00 | 7513.00 | 7481.23 | 7516.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 7546.00 | 7481.54 | 7504.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 7546.00 | 7481.54 | 7504.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 7535.00 | 7492.23 | 7507.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 7535.00 | 7492.23 | 7507.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 7539.50 | 7516.43 | 7515.99 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 7453.50 | 7506.47 | 7512.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 7423.00 | 7460.96 | 7475.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 10:15:00 | 7445.00 | 7426.22 | 7448.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 10:15:00 | 7445.00 | 7426.22 | 7448.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 7445.00 | 7426.22 | 7448.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:45:00 | 7450.00 | 7426.22 | 7448.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 7314.50 | 7403.87 | 7436.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 12:15:00 | 7303.00 | 7403.87 | 7436.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:30:00 | 7296.00 | 7349.98 | 7394.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 7276.50 | 7321.50 | 7332.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 6937.85 | 7080.13 | 7165.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 10:15:00 | 7097.00 | 7083.50 | 7158.89 | SL hit (close>ema200) qty=0.50 sl=7083.50 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 7239.00 | 7178.16 | 7175.64 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 14:15:00 | 7145.50 | 7176.82 | 7177.15 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 7243.00 | 7180.97 | 7178.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 10:15:00 | 7328.00 | 7270.27 | 7236.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 10:15:00 | 7311.50 | 7311.94 | 7279.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:30:00 | 7297.50 | 7311.94 | 7279.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 7270.00 | 7301.70 | 7282.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 7270.00 | 7301.70 | 7282.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 7342.50 | 7309.86 | 7287.92 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 09:15:00 | 7093.50 | 7266.69 | 7272.11 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 7273.50 | 7188.31 | 7178.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 12:15:00 | 7319.50 | 7214.55 | 7191.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 14:15:00 | 7632.00 | 7637.52 | 7579.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 14:45:00 | 7624.00 | 7637.52 | 7579.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 7594.50 | 7629.15 | 7586.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 7594.50 | 7629.15 | 7586.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 7588.00 | 7620.92 | 7586.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 7588.00 | 7620.92 | 7586.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 7584.00 | 7613.54 | 7586.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:30:00 | 7573.50 | 7613.54 | 7586.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 7586.00 | 7608.03 | 7586.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:30:00 | 7572.50 | 7608.03 | 7586.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 7598.00 | 7606.02 | 7587.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:15:00 | 7582.50 | 7606.02 | 7587.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 7582.50 | 7601.32 | 7586.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:45:00 | 7599.00 | 7601.32 | 7586.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 7538.00 | 7588.65 | 7582.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 7579.00 | 7588.65 | 7582.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 7591.50 | 7589.22 | 7583.15 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 7537.00 | 7572.66 | 7576.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 12:15:00 | 7483.00 | 7554.73 | 7567.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 09:15:00 | 7559.50 | 7531.44 | 7549.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 09:15:00 | 7559.50 | 7531.44 | 7549.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 7559.50 | 7531.44 | 7549.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:00:00 | 7559.50 | 7531.44 | 7549.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 7572.50 | 7539.65 | 7551.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:15:00 | 7590.50 | 7539.65 | 7551.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 7544.00 | 7548.35 | 7553.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 14:30:00 | 7499.50 | 7538.58 | 7548.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 10:15:00 | 7590.50 | 7547.95 | 7549.89 | SL hit (close>static) qty=1.00 sl=7573.50 alert=retest2 |

### Cycle 53 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 7639.50 | 7566.26 | 7558.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 12:15:00 | 7655.50 | 7584.11 | 7566.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 15:15:00 | 7596.00 | 7601.24 | 7580.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 09:15:00 | 7582.50 | 7601.24 | 7580.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 7545.00 | 7589.99 | 7577.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:00:00 | 7545.00 | 7589.99 | 7577.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 7558.50 | 7583.69 | 7575.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 11:15:00 | 7586.00 | 7583.69 | 7575.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 09:15:00 | 7709.00 | 7795.18 | 7806.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 7709.00 | 7795.18 | 7806.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 7613.50 | 7690.86 | 7738.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 7552.00 | 7542.26 | 7610.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 15:00:00 | 7552.00 | 7542.26 | 7610.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 7382.00 | 7275.41 | 7352.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:00:00 | 7382.00 | 7275.41 | 7352.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 7343.00 | 7288.93 | 7352.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 7281.00 | 7360.42 | 7370.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 6916.95 | 7080.93 | 7199.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 7045.50 | 6998.66 | 7096.03 | SL hit (close>ema200) qty=0.50 sl=6998.66 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 09:15:00 | 6885.50 | 6864.31 | 6862.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 12:15:00 | 6965.00 | 6894.30 | 6877.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 12:15:00 | 6932.50 | 6960.17 | 6929.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 12:15:00 | 6932.50 | 6960.17 | 6929.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 6932.50 | 6960.17 | 6929.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 6932.50 | 6960.17 | 6929.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 6999.50 | 6968.04 | 6935.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 14:15:00 | 7003.00 | 6968.04 | 6935.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 14:45:00 | 7012.00 | 6975.13 | 6941.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 10:30:00 | 7008.00 | 6990.80 | 6957.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 13:15:00 | 6913.00 | 6964.13 | 6952.62 | SL hit (close<static) qty=1.00 sl=6915.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 6845.50 | 6940.40 | 6942.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 6810.00 | 6914.32 | 6930.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 6910.50 | 6886.62 | 6909.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 12:15:00 | 6910.50 | 6886.62 | 6909.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 6910.50 | 6886.62 | 6909.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:00:00 | 6910.50 | 6886.62 | 6909.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 6934.00 | 6896.09 | 6911.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:45:00 | 6946.00 | 6896.09 | 6911.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 7037.00 | 6924.27 | 6923.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 7437.50 | 7042.24 | 6977.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 7535.50 | 7567.99 | 7422.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:45:00 | 7540.50 | 7567.99 | 7422.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 7759.50 | 7779.33 | 7749.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:00:00 | 7759.50 | 7779.33 | 7749.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 7747.50 | 7772.96 | 7749.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:30:00 | 7735.50 | 7772.96 | 7749.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 7760.00 | 7770.37 | 7750.41 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 7652.50 | 7744.87 | 7744.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 10:15:00 | 7632.50 | 7722.40 | 7734.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 7656.00 | 7648.31 | 7686.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 10:00:00 | 7656.00 | 7648.31 | 7686.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 7673.50 | 7653.30 | 7670.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:30:00 | 7685.50 | 7653.30 | 7670.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 7653.50 | 7653.34 | 7669.28 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 7716.50 | 7681.17 | 7679.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 14:15:00 | 7745.00 | 7693.94 | 7685.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 7753.50 | 7788.15 | 7753.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 7753.50 | 7788.15 | 7753.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 7753.50 | 7788.15 | 7753.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:45:00 | 7760.00 | 7788.15 | 7753.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 7787.00 | 7787.92 | 7756.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 11:15:00 | 7800.00 | 7787.92 | 7756.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 09:15:00 | 7727.50 | 7788.28 | 7774.82 | SL hit (close<static) qty=1.00 sl=7742.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 11:15:00 | 8332.00 | 8470.14 | 8481.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 14:15:00 | 8279.50 | 8397.81 | 8442.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 10:15:00 | 8395.50 | 8370.17 | 8415.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 10:15:00 | 8395.50 | 8370.17 | 8415.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 8395.50 | 8370.17 | 8415.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 11:00:00 | 8395.50 | 8370.17 | 8415.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 8390.50 | 8376.53 | 8410.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:30:00 | 8407.00 | 8376.53 | 8410.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 8422.00 | 8385.62 | 8411.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:30:00 | 8430.00 | 8385.62 | 8411.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 8551.00 | 8418.70 | 8424.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 8551.00 | 8418.70 | 8424.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 8570.50 | 8449.06 | 8437.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 8591.50 | 8477.55 | 8451.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 8223.50 | 8484.78 | 8481.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 8223.50 | 8484.78 | 8481.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 8223.50 | 8484.78 | 8481.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:00:00 | 8223.50 | 8484.78 | 8481.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 8234.50 | 8434.73 | 8458.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-10 09:15:00 | 8158.50 | 8269.32 | 8352.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 7523.00 | 7490.44 | 7704.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 12:00:00 | 7523.00 | 7490.44 | 7704.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 7244.00 | 7155.40 | 7200.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 7290.00 | 7155.40 | 7200.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 7350.50 | 7194.42 | 7213.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 7350.50 | 7194.42 | 7213.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 7362.50 | 7247.81 | 7236.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 7377.50 | 7273.75 | 7248.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 7153.50 | 7279.21 | 7260.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 7153.50 | 7279.21 | 7260.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 7153.50 | 7279.21 | 7260.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 7169.50 | 7279.21 | 7260.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 7144.50 | 7252.27 | 7249.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:15:00 | 7120.50 | 7252.27 | 7249.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 7185.00 | 7238.81 | 7243.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 6831.50 | 7113.32 | 7170.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 6910.00 | 6869.44 | 6970.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:45:00 | 6931.00 | 6869.44 | 6970.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 7070.00 | 6909.55 | 6979.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 7057.00 | 6909.55 | 6979.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 7097.50 | 6947.14 | 6989.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 7103.00 | 6947.14 | 6989.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 7239.50 | 7038.50 | 7023.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 7271.00 | 7085.00 | 7046.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 7154.00 | 7161.03 | 7100.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 15:00:00 | 7154.00 | 7161.03 | 7100.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 6985.00 | 7126.22 | 7095.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 6985.00 | 7126.22 | 7095.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 6955.50 | 7092.07 | 7082.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 6955.50 | 7092.07 | 7082.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 7005.50 | 7074.76 | 7075.73 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 12:15:00 | 7095.00 | 7078.81 | 7077.49 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 7057.00 | 7074.45 | 7075.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 6973.00 | 7052.03 | 7064.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 7112.00 | 6946.26 | 6986.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 7112.00 | 6946.26 | 6986.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 7112.00 | 6946.26 | 6986.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:15:00 | 7154.00 | 6946.26 | 6986.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 6977.50 | 6889.38 | 6915.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:30:00 | 6952.00 | 6889.38 | 6915.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 6917.00 | 6894.91 | 6916.02 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 7009.50 | 6930.40 | 6929.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 7035.50 | 6951.42 | 6938.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 15:15:00 | 7695.00 | 7704.68 | 7590.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-13 09:15:00 | 7595.00 | 7704.68 | 7590.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 7461.00 | 7655.94 | 7578.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:45:00 | 7490.00 | 7655.94 | 7578.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 7507.50 | 7626.25 | 7572.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:15:00 | 7535.00 | 7626.25 | 7572.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-21 09:15:00 | 8288.50 | 8211.04 | 8105.46 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 15:15:00 | 7945.00 | 8061.01 | 8071.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 7922.00 | 8033.21 | 8057.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 10:15:00 | 8044.00 | 8035.36 | 8056.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-22 11:00:00 | 8044.00 | 8035.36 | 8056.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 8042.00 | 8038.47 | 8054.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 14:00:00 | 8027.00 | 8036.18 | 8051.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 15:00:00 | 8028.00 | 8034.54 | 8049.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 8022.00 | 8035.63 | 8048.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 8098.50 | 8048.21 | 8053.20 | SL hit (close>static) qty=1.00 sl=8075.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 11:15:00 | 8106.00 | 8066.05 | 8060.83 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 13:15:00 | 8000.00 | 8056.91 | 8057.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 7953.00 | 8036.13 | 8048.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 09:15:00 | 8053.00 | 8027.16 | 8041.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 09:15:00 | 8053.00 | 8027.16 | 8041.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 8053.00 | 8027.16 | 8041.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 10:45:00 | 7931.50 | 8002.93 | 8029.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 12:15:00 | 7935.00 | 7995.04 | 8023.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 10:15:00 | 8074.50 | 8023.54 | 8023.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 8074.50 | 8023.54 | 8023.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 13:15:00 | 8110.00 | 8058.34 | 8040.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-27 15:15:00 | 8055.00 | 8058.42 | 8043.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 09:15:00 | 8141.50 | 8058.42 | 8043.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 8142.00 | 8075.13 | 8052.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 13:00:00 | 8190.50 | 8123.11 | 8082.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 14:15:00 | 8213.50 | 8135.49 | 8091.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 8020.00 | 8147.84 | 8141.37 | SL hit (close<static) qty=1.00 sl=8040.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 7991.50 | 8116.57 | 8127.75 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 8214.50 | 8135.53 | 8125.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 12:15:00 | 8261.00 | 8160.62 | 8137.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 8219.00 | 8243.56 | 8191.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 8219.00 | 8243.56 | 8191.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 8219.00 | 8243.56 | 8191.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 15:15:00 | 8350.00 | 8276.07 | 8226.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 15:00:00 | 5979.50 | 2025-05-20 14:15:00 | 6020.50 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2025-05-29 13:00:00 | 6016.00 | 2025-05-30 11:15:00 | 5969.50 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-05-29 15:15:00 | 6018.50 | 2025-05-30 11:15:00 | 5969.50 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-06-18 10:30:00 | 6038.00 | 2025-06-23 09:15:00 | 6139.50 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-07-18 09:15:00 | 7014.00 | 2025-07-23 09:15:00 | 6895.00 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-07-18 13:15:00 | 7001.50 | 2025-07-23 09:15:00 | 6895.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-07-21 09:15:00 | 6980.50 | 2025-07-23 09:15:00 | 6895.00 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-07-21 10:00:00 | 6985.00 | 2025-07-23 09:15:00 | 6895.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-08-25 09:15:00 | 7142.00 | 2025-08-25 11:15:00 | 7086.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest1 | 2025-09-05 09:15:00 | 7285.50 | 2025-09-05 14:15:00 | 7219.50 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-09-08 09:15:00 | 7290.50 | 2025-09-09 13:15:00 | 7219.50 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-09-26 11:15:00 | 7417.50 | 2025-10-03 10:15:00 | 7369.00 | STOP_HIT | 1.00 | 0.65% |
| BUY | retest1 | 2025-10-08 10:45:00 | 7609.50 | 2025-10-13 10:15:00 | 7655.00 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest1 | 2025-10-08 11:15:00 | 7628.00 | 2025-10-13 10:15:00 | 7655.00 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2025-10-24 12:15:00 | 7498.00 | 2025-10-27 09:15:00 | 7620.00 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-10-31 13:45:00 | 7773.00 | 2025-10-31 14:15:00 | 7692.00 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-11-03 09:15:00 | 7770.00 | 2025-11-03 09:15:00 | 7704.50 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-11-06 15:00:00 | 7592.50 | 2025-11-11 09:15:00 | 7654.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-11-10 15:15:00 | 7597.50 | 2025-11-11 09:15:00 | 7654.00 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-11-13 09:15:00 | 7714.50 | 2025-11-14 12:15:00 | 7642.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-12-02 12:15:00 | 7303.00 | 2025-12-09 09:15:00 | 6937.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 12:15:00 | 7303.00 | 2025-12-09 10:15:00 | 7097.00 | STOP_HIT | 0.50 | 2.82% |
| SELL | retest2 | 2025-12-03 09:30:00 | 7296.00 | 2025-12-10 10:15:00 | 7239.00 | STOP_HIT | 1.00 | 0.78% |
| SELL | retest2 | 2025-12-05 09:15:00 | 7276.50 | 2025-12-10 10:15:00 | 7239.00 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2025-12-30 14:30:00 | 7499.50 | 2025-12-31 10:15:00 | 7590.50 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2026-01-01 11:15:00 | 7586.00 | 2026-01-09 09:15:00 | 7709.00 | STOP_HIT | 1.00 | 1.62% |
| SELL | retest2 | 2026-01-20 09:15:00 | 7281.00 | 2026-01-21 10:15:00 | 6916.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 09:15:00 | 7281.00 | 2026-01-22 09:15:00 | 7045.50 | STOP_HIT | 0.50 | 3.23% |
| BUY | retest2 | 2026-01-30 14:15:00 | 7003.00 | 2026-02-01 13:15:00 | 6913.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2026-01-30 14:45:00 | 7012.00 | 2026-02-01 13:15:00 | 6913.00 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2026-02-01 10:30:00 | 7008.00 | 2026-02-01 13:15:00 | 6913.00 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2026-02-19 11:15:00 | 7800.00 | 2026-02-20 09:15:00 | 7727.50 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2026-02-20 11:45:00 | 7805.00 | 2026-02-27 10:15:00 | 8585.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 11:15:00 | 7535.00 | 2026-04-21 09:15:00 | 8288.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-22 14:00:00 | 8027.00 | 2026-04-23 09:15:00 | 8098.50 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2026-04-22 15:00:00 | 8028.00 | 2026-04-23 09:15:00 | 8098.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2026-04-23 09:15:00 | 8022.00 | 2026-04-23 09:15:00 | 8098.50 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-04-24 10:45:00 | 7931.50 | 2026-04-27 10:15:00 | 8074.50 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2026-04-24 12:15:00 | 7935.00 | 2026-04-27 10:15:00 | 8074.50 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-04-28 13:00:00 | 8190.50 | 2026-04-30 09:15:00 | 8020.00 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2026-04-28 14:15:00 | 8213.50 | 2026-04-30 09:15:00 | 8020.00 | STOP_HIT | 1.00 | -2.36% |
