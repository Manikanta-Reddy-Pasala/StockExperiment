# Apar Industries Ltd. (APARINDS)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 12760.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 155 |
| ALERT1 | 105 |
| ALERT2 | 103 |
| ALERT2_SKIP | 53 |
| ALERT3 | 269 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 106 |
| PARTIAL | 16 |
| TARGET_HIT | 11 |
| STOP_HIT | 101 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 128 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 48 / 80
- **Target hits / Stop hits / Partials:** 11 / 101 / 16
- **Avg / median % per leg:** 0.65% / -1.13%
- **Sum % (uncompounded):** 82.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 36 | 15 | 41.7% | 7 | 27 | 2 | 1.39% | 49.9% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 1 | 1 | 2 | 6.21% | 24.8% |
| BUY @ 3rd Alert (retest2) | 32 | 11 | 34.4% | 6 | 26 | 0 | 0.78% | 25.1% |
| SELL (all) | 92 | 33 | 35.9% | 4 | 74 | 14 | 0.36% | 32.9% |
| SELL @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 0 | 5 | 0 | -1.60% | -8.0% |
| SELL @ 3rd Alert (retest2) | 87 | 30 | 34.5% | 4 | 69 | 14 | 0.47% | 40.9% |
| retest1 (combined) | 9 | 7 | 77.8% | 1 | 6 | 2 | 1.87% | 16.8% |
| retest2 (combined) | 119 | 41 | 34.5% | 10 | 95 | 14 | 0.55% | 66.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 14:15:00 | 7725.00 | 7659.33 | 7655.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 09:15:00 | 7832.50 | 7711.35 | 7680.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 14:15:00 | 8119.70 | 8140.16 | 8008.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 14:45:00 | 8132.95 | 8140.16 | 8008.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 7976.40 | 8103.69 | 8014.33 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2024-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 10:15:00 | 7892.45 | 7984.87 | 7996.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 09:15:00 | 7876.40 | 7929.09 | 7959.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 13:15:00 | 7924.90 | 7893.74 | 7929.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 13:15:00 | 7924.90 | 7893.74 | 7929.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 7924.90 | 7893.74 | 7929.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 13:30:00 | 7964.55 | 7893.74 | 7929.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 7855.90 | 7886.17 | 7922.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:30:00 | 7920.30 | 7886.17 | 7922.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 7880.15 | 7875.02 | 7910.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:00:00 | 7880.15 | 7875.02 | 7910.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 7902.00 | 7880.41 | 7910.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 11:00:00 | 7902.00 | 7880.41 | 7910.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 7923.85 | 7889.10 | 7911.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 12:00:00 | 7923.85 | 7889.10 | 7911.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 12:15:00 | 7926.90 | 7896.66 | 7912.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 14:15:00 | 7893.00 | 7899.33 | 7912.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 15:00:00 | 7840.10 | 7887.48 | 7905.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-24 09:15:00 | 8019.70 | 7910.95 | 7913.16 | SL hit (close>static) qty=1.00 sl=7975.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 10:15:00 | 8058.15 | 7940.39 | 7926.34 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 11:15:00 | 7823.20 | 7951.82 | 7966.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 14:15:00 | 7801.20 | 7886.74 | 7930.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 12:15:00 | 7826.35 | 7824.56 | 7877.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 12:30:00 | 7847.95 | 7824.56 | 7877.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 7924.80 | 7834.64 | 7863.62 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 13:15:00 | 7962.25 | 7882.44 | 7877.62 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 15:15:00 | 7785.00 | 7863.83 | 7870.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 09:15:00 | 7690.00 | 7829.07 | 7853.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 11:15:00 | 7825.00 | 7821.71 | 7845.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 11:15:00 | 7825.00 | 7821.71 | 7845.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 7825.00 | 7821.71 | 7845.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 12:00:00 | 7825.00 | 7821.71 | 7845.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 7819.80 | 7821.33 | 7843.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 12:45:00 | 7815.00 | 7821.33 | 7843.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 7939.50 | 7844.96 | 7852.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 7939.50 | 7844.96 | 7852.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 14:15:00 | 7968.05 | 7869.58 | 7862.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 14:15:00 | 8001.10 | 7903.70 | 7885.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 7832.20 | 7901.61 | 7888.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 7832.20 | 7901.61 | 7888.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 7832.20 | 7901.61 | 7888.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 7773.95 | 7901.61 | 7888.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 7458.10 | 7812.90 | 7849.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 7065.55 | 7663.43 | 7777.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 15:15:00 | 7500.00 | 7497.95 | 7650.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 15:15:00 | 7500.00 | 7497.95 | 7650.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 7500.00 | 7497.95 | 7650.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-05 09:30:00 | 7108.00 | 7452.65 | 7616.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-06 10:15:00 | 7803.00 | 7631.92 | 7628.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 7803.00 | 7631.92 | 7628.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 12:15:00 | 7911.65 | 7773.76 | 7717.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 11:15:00 | 8160.60 | 8170.98 | 8046.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 11:30:00 | 8156.05 | 8170.98 | 8046.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 8119.00 | 8157.13 | 8070.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 14:45:00 | 8172.95 | 8157.13 | 8070.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 8157.00 | 8157.10 | 8078.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 09:30:00 | 8099.10 | 8137.68 | 8076.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 10:15:00 | 8037.70 | 8117.68 | 8073.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 10:45:00 | 8034.35 | 8117.68 | 8073.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 8409.95 | 8457.97 | 8361.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 14:30:00 | 8318.00 | 8457.97 | 8361.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 15:15:00 | 8354.00 | 8437.17 | 8360.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 09:15:00 | 8517.15 | 8437.17 | 8360.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 13:45:00 | 8434.05 | 8441.63 | 8424.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 09:15:00 | 8490.65 | 8417.86 | 8415.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 11:15:00 | 8344.45 | 8420.76 | 8424.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 11:15:00 | 8344.45 | 8420.76 | 8424.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 12:15:00 | 8294.55 | 8395.52 | 8412.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 09:15:00 | 8381.95 | 8370.34 | 8393.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 8381.95 | 8370.34 | 8393.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 8381.95 | 8370.34 | 8393.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:00:00 | 8381.95 | 8370.34 | 8393.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 8425.70 | 8381.41 | 8396.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:00:00 | 8425.70 | 8381.41 | 8396.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 8478.10 | 8400.75 | 8403.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 12:00:00 | 8478.10 | 8400.75 | 8403.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 12:15:00 | 8544.60 | 8429.52 | 8416.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 09:15:00 | 8613.00 | 8497.92 | 8456.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 14:15:00 | 8508.85 | 8585.99 | 8526.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 14:15:00 | 8508.85 | 8585.99 | 8526.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 8508.85 | 8585.99 | 8526.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 15:00:00 | 8508.85 | 8585.99 | 8526.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 8510.00 | 8570.79 | 8524.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 09:15:00 | 8558.00 | 8570.79 | 8524.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 09:45:00 | 8535.00 | 8574.41 | 8530.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 14:15:00 | 8386.40 | 8526.96 | 8527.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 14:15:00 | 8386.40 | 8526.96 | 8527.93 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 11:15:00 | 8547.20 | 8525.72 | 8524.86 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 8506.90 | 8522.97 | 8523.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 14:15:00 | 8466.90 | 8511.75 | 8518.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 11:15:00 | 8533.15 | 8510.36 | 8515.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 11:15:00 | 8533.15 | 8510.36 | 8515.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 8533.15 | 8510.36 | 8515.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 12:00:00 | 8533.15 | 8510.36 | 8515.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 8465.40 | 8501.37 | 8510.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 13:45:00 | 8444.05 | 8487.12 | 8498.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 09:30:00 | 8401.00 | 8470.19 | 8487.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-03 09:15:00 | 8576.95 | 8491.69 | 8488.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 09:15:00 | 8576.95 | 8491.69 | 8488.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 10:15:00 | 8603.55 | 8514.06 | 8499.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 12:15:00 | 8680.00 | 8681.59 | 8617.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 12:45:00 | 8675.00 | 8681.59 | 8617.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 8628.30 | 8662.68 | 8619.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 15:00:00 | 8628.30 | 8662.68 | 8619.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 8567.95 | 8643.73 | 8614.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 09:15:00 | 8651.10 | 8643.73 | 8614.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 10:00:00 | 8674.00 | 8649.79 | 8620.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 11:15:00 | 8703.00 | 8749.44 | 8753.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 11:15:00 | 8703.00 | 8749.44 | 8753.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-11 10:15:00 | 8619.60 | 8689.68 | 8718.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-12 09:15:00 | 8664.00 | 8640.74 | 8675.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 09:15:00 | 8664.00 | 8640.74 | 8675.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 8664.00 | 8640.74 | 8675.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 10:30:00 | 8621.80 | 8626.79 | 8665.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 15:15:00 | 8630.00 | 8614.03 | 8644.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-15 09:15:00 | 9012.70 | 8696.32 | 8677.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 09:15:00 | 9012.70 | 8696.32 | 8677.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 14:15:00 | 9036.45 | 8900.65 | 8833.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 09:15:00 | 8826.00 | 8900.97 | 8846.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 09:15:00 | 8826.00 | 8900.97 | 8846.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 8826.00 | 8900.97 | 8846.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 8857.80 | 8900.97 | 8846.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 8657.95 | 8852.37 | 8828.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 8657.95 | 8852.37 | 8828.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 8679.35 | 8817.77 | 8815.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:30:00 | 8670.00 | 8817.77 | 8815.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 12:15:00 | 8633.70 | 8780.95 | 8798.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 13:15:00 | 8597.35 | 8744.23 | 8780.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 11:15:00 | 8213.10 | 8182.73 | 8357.10 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-22 13:00:00 | 8130.15 | 8172.21 | 8336.47 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-22 14:00:00 | 8133.00 | 8164.37 | 8317.97 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 09:15:00 | 8094.40 | 8164.43 | 8291.50 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 8078.05 | 8147.15 | 8272.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 7878.70 | 8131.24 | 8242.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:00:00 | 7978.50 | 8100.69 | 8218.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:30:00 | 7965.00 | 8085.35 | 8201.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 14:15:00 | 7969.65 | 8085.35 | 8201.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 8092.00 | 8086.61 | 8172.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:30:00 | 8107.45 | 8086.61 | 8172.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 8080.95 | 8074.17 | 8119.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 11:30:00 | 8115.40 | 8074.17 | 8119.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 8073.80 | 8038.78 | 8082.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:45:00 | 8097.00 | 8038.78 | 8082.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 8033.00 | 8037.62 | 8078.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:30:00 | 8068.30 | 8037.62 | 8078.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 11:15:00 | 8037.30 | 8037.56 | 8074.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-26 13:15:00 | 8084.40 | 8050.52 | 8074.06 | SL hit (close>ema400) qty=1.00 sl=8074.06 alert=retest1 |

### Cycle 19 — BUY (started 2024-07-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 15:15:00 | 8200.00 | 8101.29 | 8094.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 09:15:00 | 8386.50 | 8158.33 | 8120.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 09:15:00 | 8840.00 | 8897.25 | 8677.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 10:15:00 | 8847.00 | 8897.25 | 8677.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 8989.05 | 9127.43 | 8967.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:00:00 | 8989.05 | 9127.43 | 8967.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 8951.00 | 9092.14 | 8966.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 13:00:00 | 8951.00 | 9092.14 | 8966.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 8918.35 | 9057.38 | 8961.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:00:00 | 8918.35 | 9057.38 | 8961.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 8830.00 | 9011.91 | 8949.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 15:00:00 | 8830.00 | 9011.91 | 8949.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 10:15:00 | 8717.35 | 8899.36 | 8909.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 11:15:00 | 8689.65 | 8857.42 | 8889.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 8316.85 | 8314.21 | 8497.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 8316.85 | 8314.21 | 8497.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 8316.85 | 8314.21 | 8497.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 11:30:00 | 8224.10 | 8286.40 | 8452.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:30:00 | 8219.20 | 8270.15 | 8429.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:15:00 | 8227.00 | 8261.23 | 8294.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 10:15:00 | 8222.20 | 8274.05 | 8285.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 8317.35 | 8282.71 | 8288.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 11:00:00 | 8317.35 | 8282.71 | 8288.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-12 11:15:00 | 8346.00 | 8295.37 | 8293.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 11:15:00 | 8346.00 | 8295.37 | 8293.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 12:15:00 | 8378.25 | 8311.94 | 8301.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 09:15:00 | 8424.80 | 8474.70 | 8422.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 09:15:00 | 8424.80 | 8474.70 | 8422.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 8424.80 | 8474.70 | 8422.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 8467.45 | 8474.70 | 8422.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 8343.00 | 8448.36 | 8415.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:00:00 | 8343.00 | 8448.36 | 8415.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 8412.95 | 8441.28 | 8415.16 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2024-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 14:15:00 | 8323.55 | 8396.78 | 8399.22 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 12:15:00 | 8470.15 | 8410.02 | 8402.18 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 11:15:00 | 8362.00 | 8402.83 | 8404.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-19 12:15:00 | 8328.15 | 8387.90 | 8397.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 12:15:00 | 8377.05 | 8347.78 | 8367.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 12:15:00 | 8377.05 | 8347.78 | 8367.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 8377.05 | 8347.78 | 8367.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 12:45:00 | 8354.70 | 8347.78 | 8367.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 13:15:00 | 8379.05 | 8354.03 | 8368.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 13:45:00 | 8417.90 | 8354.03 | 8368.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 8374.50 | 8358.12 | 8369.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-21 13:00:00 | 8280.95 | 8328.57 | 8349.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-21 13:45:00 | 8291.35 | 8321.60 | 8344.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-21 14:30:00 | 8290.10 | 8319.06 | 8341.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-22 10:15:00 | 8384.95 | 8339.07 | 8345.38 | SL hit (close>static) qty=1.00 sl=8384.40 alert=retest2 |

### Cycle 25 — BUY (started 2024-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 11:15:00 | 8494.95 | 8370.24 | 8358.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-23 10:15:00 | 8850.00 | 8548.88 | 8460.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 10:15:00 | 8922.05 | 8933.20 | 8819.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-27 11:00:00 | 8922.05 | 8933.20 | 8819.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 8831.40 | 8905.13 | 8825.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 12:45:00 | 8846.90 | 8905.13 | 8825.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 13:15:00 | 8891.05 | 8902.31 | 8831.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 13:45:00 | 8805.20 | 8902.31 | 8831.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 9042.05 | 9036.48 | 8952.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 15:00:00 | 9042.05 | 9036.48 | 8952.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 9188.05 | 9060.16 | 8977.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 11:00:00 | 9407.30 | 9116.78 | 9048.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-02 10:15:00 | 8929.25 | 9113.24 | 9094.98 | SL hit (close<static) qty=1.00 sl=8959.15 alert=retest2 |

### Cycle 26 — SELL (started 2024-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 11:15:00 | 8869.85 | 9064.56 | 9074.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 12:15:00 | 8795.00 | 9010.65 | 9049.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 15:15:00 | 8900.00 | 8834.62 | 8903.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 15:15:00 | 8900.00 | 8834.62 | 8903.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 8900.00 | 8834.62 | 8903.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 10:00:00 | 8945.45 | 8856.78 | 8907.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 10:15:00 | 8896.70 | 8864.77 | 8906.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 12:00:00 | 8874.00 | 8866.61 | 8903.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 12:30:00 | 8880.55 | 8861.03 | 8897.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 13:00:00 | 8838.70 | 8861.03 | 8897.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 14:15:00 | 9003.75 | 8881.92 | 8900.25 | SL hit (close>static) qty=1.00 sl=8956.45 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 09:15:00 | 9072.00 | 8938.83 | 8924.11 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 10:15:00 | 8908.00 | 8996.45 | 8998.03 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 15:15:00 | 9078.00 | 8998.63 | 8993.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 09:15:00 | 9242.30 | 9047.37 | 9016.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 10280.00 | 10436.06 | 10270.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 09:15:00 | 10280.00 | 10436.06 | 10270.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 10280.00 | 10436.06 | 10270.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 10280.00 | 10436.06 | 10270.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 10224.80 | 10393.81 | 10266.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:45:00 | 10206.95 | 10393.81 | 10266.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 10151.20 | 10345.29 | 10255.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 12:00:00 | 10151.20 | 10345.29 | 10255.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 09:15:00 | 9998.90 | 10183.66 | 10202.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 9785.50 | 9988.38 | 10079.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 09:15:00 | 9641.00 | 9548.78 | 9687.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-23 09:15:00 | 9641.00 | 9548.78 | 9687.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 9641.00 | 9548.78 | 9687.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:45:00 | 9703.95 | 9548.78 | 9687.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 9754.15 | 9589.86 | 9693.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 10:45:00 | 9797.85 | 9589.86 | 9693.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 11:15:00 | 9820.00 | 9635.88 | 9705.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 11:45:00 | 9819.65 | 9635.88 | 9705.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 15:15:00 | 9785.00 | 9721.60 | 9728.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-24 09:15:00 | 9840.10 | 9721.60 | 9728.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 09:15:00 | 9870.15 | 9751.31 | 9741.69 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 12:15:00 | 9709.95 | 9783.52 | 9783.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 14:15:00 | 9681.00 | 9752.06 | 9768.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 13:15:00 | 9560.00 | 9516.22 | 9581.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-27 14:00:00 | 9560.00 | 9516.22 | 9581.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 9500.05 | 9512.98 | 9574.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 09:30:00 | 9435.35 | 9495.36 | 9555.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 11:00:00 | 9420.00 | 9480.29 | 9542.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 12:15:00 | 9436.65 | 9475.25 | 9534.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 14:00:00 | 9394.45 | 9440.65 | 9507.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 9527.80 | 9458.08 | 9509.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 15:00:00 | 9527.80 | 9458.08 | 9509.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 9510.00 | 9468.46 | 9509.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:30:00 | 9527.95 | 9472.45 | 9507.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 9689.00 | 9515.76 | 9524.08 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-01 10:15:00 | 9689.00 | 9515.76 | 9524.08 | SL hit (close>static) qty=1.00 sl=9613.15 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 11:15:00 | 9590.00 | 9530.61 | 9530.07 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 12:15:00 | 9504.70 | 9549.13 | 9553.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 13:15:00 | 9442.20 | 9527.74 | 9543.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-03 14:15:00 | 9547.35 | 9531.66 | 9543.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 14:15:00 | 9547.35 | 9531.66 | 9543.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 9547.35 | 9531.66 | 9543.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-03 14:30:00 | 9606.50 | 9531.66 | 9543.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 15:15:00 | 9521.15 | 9529.56 | 9541.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 09:15:00 | 9412.80 | 9529.56 | 9541.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 12:30:00 | 9501.00 | 9529.16 | 9537.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 09:15:00 | 9408.00 | 9520.20 | 9531.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 10:30:00 | 9496.40 | 9432.71 | 9450.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 9476.95 | 9441.56 | 9453.21 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-08 13:15:00 | 9521.95 | 9465.69 | 9462.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 13:15:00 | 9521.95 | 9465.69 | 9462.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 14:15:00 | 9560.00 | 9484.55 | 9471.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 10:15:00 | 10421.40 | 10423.48 | 10171.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 11:00:00 | 10421.40 | 10423.48 | 10171.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 10758.00 | 10579.76 | 10452.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 13:30:00 | 10829.65 | 10682.89 | 10545.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 10:45:00 | 10826.25 | 10760.14 | 10634.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 10426.10 | 10601.85 | 10603.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 10426.10 | 10601.85 | 10603.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 12:15:00 | 10307.35 | 10463.93 | 10532.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-22 14:15:00 | 9800.05 | 9713.68 | 9890.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-22 15:00:00 | 9800.05 | 9713.68 | 9890.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 9970.00 | 9780.20 | 9891.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:00:00 | 9970.00 | 9780.20 | 9891.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 9880.00 | 9800.16 | 9890.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 12:15:00 | 9823.05 | 9815.23 | 9889.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 13:00:00 | 9821.00 | 9816.38 | 9882.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 14:00:00 | 9857.85 | 9824.67 | 9880.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 09:30:00 | 9829.60 | 9789.02 | 9846.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 9780.00 | 9787.22 | 9840.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 14:15:00 | 9711.50 | 9778.26 | 9822.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:15:00 | 9641.85 | 9767.86 | 9809.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 09:15:00 | 9331.90 | 9493.54 | 9617.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 09:15:00 | 9329.95 | 9493.54 | 9617.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 09:15:00 | 9364.96 | 9493.54 | 9617.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 09:15:00 | 9338.12 | 9493.54 | 9617.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-29 12:15:00 | 9348.90 | 9341.96 | 9440.11 | SL hit (close>ema200) qty=0.50 sl=9341.96 alert=retest2 |

### Cycle 37 — BUY (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 10:15:00 | 9633.55 | 9489.07 | 9469.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 11:15:00 | 9773.35 | 9545.92 | 9497.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 9689.85 | 9857.48 | 9720.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 9689.85 | 9857.48 | 9720.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 9689.85 | 9857.48 | 9720.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:15:00 | 9719.45 | 9857.48 | 9720.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 9667.60 | 9819.51 | 9715.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:30:00 | 9655.85 | 9819.51 | 9715.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 9684.10 | 9792.42 | 9712.55 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 09:15:00 | 9525.25 | 9670.96 | 9678.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 10:15:00 | 9451.30 | 9627.03 | 9657.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 8928.70 | 8847.13 | 9012.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-12 10:00:00 | 8928.70 | 8847.13 | 9012.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 9027.20 | 8883.14 | 9014.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:00:00 | 9027.20 | 8883.14 | 9014.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 9062.80 | 8919.07 | 9018.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:45:00 | 9060.00 | 8919.07 | 9018.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 12:15:00 | 8990.00 | 8933.26 | 9015.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 12:30:00 | 9060.00 | 8933.26 | 9015.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 14:15:00 | 8939.50 | 8936.56 | 9003.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 14:45:00 | 8999.30 | 8936.56 | 9003.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 8869.80 | 8792.06 | 8858.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:45:00 | 8876.50 | 8792.06 | 8858.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 8861.25 | 8805.90 | 8858.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 12:00:00 | 8861.25 | 8805.90 | 8858.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 8922.75 | 8829.27 | 8864.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:00:00 | 8922.75 | 8829.27 | 8864.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 8902.45 | 8843.91 | 8868.05 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2024-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 15:15:00 | 8946.95 | 8891.48 | 8887.16 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 09:15:00 | 8794.70 | 8872.13 | 8878.75 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 9249.90 | 8912.57 | 8884.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 10:15:00 | 9339.90 | 8998.03 | 8926.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 12:15:00 | 9380.15 | 9380.53 | 9228.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-21 12:45:00 | 9383.00 | 9380.53 | 9228.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 9348.15 | 9390.09 | 9321.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 13:45:00 | 9299.65 | 9390.09 | 9321.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 9312.00 | 9374.47 | 9320.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 15:00:00 | 9312.00 | 9374.47 | 9320.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 15:15:00 | 9315.00 | 9362.58 | 9319.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 09:15:00 | 9504.75 | 9362.58 | 9319.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-03 09:15:00 | 10455.23 | 10158.29 | 10035.53 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 10:15:00 | 9972.20 | 10091.19 | 10106.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 11:15:00 | 9936.00 | 10060.15 | 10090.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 14:15:00 | 10053.15 | 10029.28 | 10066.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-05 15:00:00 | 10053.15 | 10029.28 | 10066.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 15:15:00 | 10080.00 | 10039.43 | 10067.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:15:00 | 10073.60 | 10039.43 | 10067.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 9982.85 | 10028.11 | 10059.99 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2024-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 11:15:00 | 10185.40 | 10070.05 | 10059.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 15:15:00 | 10234.00 | 10147.86 | 10103.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 14:15:00 | 9801.35 | 10167.37 | 10147.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 14:15:00 | 9801.35 | 10167.37 | 10147.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 9801.35 | 10167.37 | 10147.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 15:00:00 | 9801.35 | 10167.37 | 10147.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 10010.00 | 10135.90 | 10134.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 09:15:00 | 10185.25 | 10135.90 | 10134.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 09:15:00 | 10063.00 | 10121.32 | 10128.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 09:15:00 | 10063.00 | 10121.32 | 10128.20 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 14:15:00 | 10263.45 | 10136.20 | 10129.80 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 13:15:00 | 9999.05 | 10108.21 | 10122.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 9917.00 | 10037.91 | 10083.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-17 15:15:00 | 9925.00 | 9892.66 | 9922.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 15:15:00 | 9925.00 | 9892.66 | 9922.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 15:15:00 | 9925.00 | 9892.66 | 9922.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:15:00 | 10007.90 | 9892.66 | 9922.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 9955.00 | 9905.13 | 9925.75 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2024-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 11:15:00 | 10038.45 | 9946.65 | 9941.96 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 12:15:00 | 9887.05 | 9946.79 | 9950.08 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 09:15:00 | 10000.95 | 9950.84 | 9949.71 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 15:15:00 | 9940.30 | 9955.75 | 9957.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 09:15:00 | 9877.45 | 9940.09 | 9950.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 12:15:00 | 9910.80 | 9910.44 | 9932.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-23 12:15:00 | 9910.80 | 9910.44 | 9932.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 12:15:00 | 9910.80 | 9910.44 | 9932.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 13:00:00 | 9910.80 | 9910.44 | 9932.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 13:15:00 | 10010.35 | 9930.42 | 9939.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 13:45:00 | 9995.60 | 9930.42 | 9939.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2024-12-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 14:15:00 | 10182.00 | 9980.74 | 9961.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 10213.15 | 10113.16 | 10057.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 10:15:00 | 10091.95 | 10108.92 | 10060.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-27 10:45:00 | 10095.65 | 10108.92 | 10060.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 10124.25 | 10111.98 | 10066.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:30:00 | 10085.20 | 10111.98 | 10066.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 10100.00 | 10113.13 | 10075.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:30:00 | 10094.60 | 10113.13 | 10075.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 10043.60 | 10099.23 | 10072.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 15:00:00 | 10043.60 | 10099.23 | 10072.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 10094.00 | 10098.18 | 10074.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:15:00 | 9990.75 | 10098.18 | 10074.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 9985.85 | 10075.71 | 10066.30 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 11:15:00 | 10000.00 | 10048.47 | 10054.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 12:15:00 | 9978.00 | 10034.37 | 10047.82 | Break + close below crossover candle low |

### Cycle 53 — BUY (started 2024-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 15:15:00 | 10200.00 | 10054.84 | 10052.68 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 09:15:00 | 9942.25 | 10032.32 | 10042.64 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2024-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 10:15:00 | 10161.00 | 10058.06 | 10053.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 12:15:00 | 10194.10 | 10100.46 | 10074.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 13:15:00 | 10875.25 | 10879.88 | 10737.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 13:30:00 | 10815.40 | 10879.88 | 10737.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 13:15:00 | 11198.70 | 11348.08 | 11220.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 13:45:00 | 11209.40 | 11348.08 | 11220.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 10800.00 | 11238.47 | 11182.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 15:00:00 | 10800.00 | 11238.47 | 11182.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 15:15:00 | 10733.00 | 11137.37 | 11141.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 10:15:00 | 10670.65 | 10992.35 | 11071.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 12:15:00 | 10616.55 | 10616.38 | 10775.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-10 13:15:00 | 10664.95 | 10616.38 | 10775.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 14:15:00 | 10210.60 | 10205.67 | 10437.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-13 15:00:00 | 10210.60 | 10205.67 | 10437.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 10189.80 | 10193.59 | 10391.52 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 10640.60 | 10432.94 | 10419.26 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-01-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 15:15:00 | 10269.90 | 10403.47 | 10413.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-16 13:15:00 | 10163.95 | 10276.58 | 10341.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-17 14:15:00 | 10079.40 | 10064.05 | 10169.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-17 15:00:00 | 10079.40 | 10064.05 | 10169.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 10066.95 | 10061.58 | 10150.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 09:15:00 | 9956.50 | 10099.51 | 10109.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 10:00:00 | 9953.45 | 10070.30 | 10095.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 12:15:00 | 9458.67 | 9590.81 | 9747.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 12:15:00 | 9455.78 | 9590.81 | 9747.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-27 09:15:00 | 8960.85 | 9315.51 | 9559.43 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 59 — BUY (started 2025-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 11:15:00 | 7691.40 | 7584.29 | 7577.27 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 13:15:00 | 7420.00 | 7543.28 | 7559.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 10:15:00 | 7167.80 | 7413.94 | 7488.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 14:15:00 | 7137.00 | 7120.66 | 7240.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 14:45:00 | 7155.10 | 7120.66 | 7240.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 7263.70 | 7149.16 | 7232.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:30:00 | 7259.95 | 7149.16 | 7232.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 7328.70 | 7185.07 | 7241.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 11:00:00 | 7328.70 | 7185.07 | 7241.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 7315.00 | 7211.06 | 7248.26 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2025-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 15:15:00 | 7302.90 | 7266.01 | 7265.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 09:15:00 | 7347.85 | 7282.38 | 7273.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 10:15:00 | 7273.60 | 7280.62 | 7273.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 10:15:00 | 7273.60 | 7280.62 | 7273.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 7273.60 | 7280.62 | 7273.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:45:00 | 7278.15 | 7280.62 | 7273.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 7242.85 | 7273.07 | 7270.67 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2025-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 13:15:00 | 7231.75 | 7262.51 | 7266.14 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 15:15:00 | 7290.00 | 7272.33 | 7270.24 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 09:15:00 | 7235.45 | 7264.96 | 7267.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 14:15:00 | 7100.00 | 7209.10 | 7238.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 15:15:00 | 6769.00 | 6700.86 | 6822.43 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-12 09:15:00 | 6409.30 | 6700.86 | 6822.43 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 6832.75 | 6651.31 | 6727.20 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-12 14:15:00 | 6832.75 | 6651.31 | 6727.20 | SL hit (close>ema400) qty=1.00 sl=6727.20 alert=retest1 |

### Cycle 65 — BUY (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 11:15:00 | 6830.30 | 6762.15 | 6761.41 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 15:15:00 | 6730.00 | 6762.42 | 6763.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 6578.25 | 6725.59 | 6746.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 12:15:00 | 6437.95 | 6410.09 | 6518.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 13:00:00 | 6437.95 | 6410.09 | 6518.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 6448.75 | 6417.82 | 6512.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 6518.05 | 6417.82 | 6512.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 6472.10 | 6428.68 | 6508.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 14:45:00 | 6517.40 | 6428.68 | 6508.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 6258.85 | 6403.08 | 6483.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 10:15:00 | 6229.20 | 6403.08 | 6483.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 12:30:00 | 6243.45 | 6320.81 | 6344.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 11:15:00 | 6407.30 | 6348.63 | 6347.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 11:15:00 | 6407.30 | 6348.63 | 6347.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-21 12:15:00 | 6442.00 | 6367.30 | 6356.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 6288.25 | 6382.25 | 6370.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 09:15:00 | 6288.25 | 6382.25 | 6370.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 6288.25 | 6382.25 | 6370.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:45:00 | 6298.75 | 6382.25 | 6370.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 6262.95 | 6358.39 | 6360.97 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 11:15:00 | 6465.25 | 6379.76 | 6370.45 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 13:15:00 | 6299.50 | 6374.87 | 6383.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 6062.50 | 6294.74 | 6343.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 13:15:00 | 5710.40 | 5692.51 | 5836.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 14:00:00 | 5710.40 | 5692.51 | 5836.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 5845.20 | 5723.05 | 5837.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 15:00:00 | 5845.20 | 5723.05 | 5837.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 5766.05 | 5731.65 | 5831.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:15:00 | 5851.65 | 5731.65 | 5831.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 5851.00 | 5755.52 | 5832.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:30:00 | 5932.50 | 5755.52 | 5832.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 5800.00 | 5764.42 | 5829.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:30:00 | 5780.00 | 5751.54 | 5818.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 14:30:00 | 5778.70 | 5736.05 | 5792.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 10:00:00 | 5755.95 | 5747.21 | 5788.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 11:45:00 | 5769.60 | 5763.72 | 5789.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 12:15:00 | 5770.10 | 5764.99 | 5787.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 12:30:00 | 5770.00 | 5764.99 | 5787.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 14:15:00 | 5798.15 | 5769.23 | 5785.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 14:45:00 | 5789.00 | 5769.23 | 5785.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 15:15:00 | 5800.45 | 5775.47 | 5786.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-06 09:15:00 | 5829.90 | 5775.47 | 5786.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 10:15:00 | 5772.95 | 5780.17 | 5787.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-06 10:30:00 | 5807.00 | 5780.17 | 5787.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 5929.00 | 5768.88 | 5770.69 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-07 09:15:00 | 5929.00 | 5768.88 | 5770.69 | SL hit (close>static) qty=1.00 sl=5877.50 alert=retest2 |

### Cycle 71 — BUY (started 2025-03-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 10:15:00 | 6059.70 | 5827.04 | 5796.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 11:15:00 | 6153.90 | 5892.41 | 5829.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 5910.20 | 5998.45 | 5917.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 5910.20 | 5998.45 | 5917.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 5910.20 | 5998.45 | 5917.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 5910.20 | 5998.45 | 5917.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 5841.20 | 5967.00 | 5911.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:45:00 | 5836.80 | 5967.00 | 5911.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 5797.45 | 5933.09 | 5900.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 5797.45 | 5933.09 | 5900.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 5748.80 | 5896.23 | 5886.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 13:00:00 | 5748.80 | 5896.23 | 5886.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2025-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 13:15:00 | 5750.85 | 5867.16 | 5874.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 5713.05 | 5836.33 | 5859.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 14:15:00 | 5540.00 | 5493.41 | 5585.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 14:45:00 | 5555.00 | 5493.41 | 5585.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 5600.00 | 5518.66 | 5581.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:00:00 | 5600.00 | 5518.66 | 5581.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 5571.20 | 5529.17 | 5580.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 14:15:00 | 5510.70 | 5543.74 | 5575.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 11:15:00 | 5658.90 | 5510.05 | 5505.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 5658.90 | 5510.05 | 5505.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 5754.50 | 5591.00 | 5545.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 5687.60 | 5753.60 | 5683.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 09:15:00 | 5687.60 | 5753.60 | 5683.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 5687.60 | 5753.60 | 5683.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:00:00 | 5687.60 | 5753.60 | 5683.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 5690.00 | 5740.88 | 5684.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:45:00 | 5666.50 | 5740.88 | 5684.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 11:15:00 | 5676.00 | 5727.90 | 5683.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 11:30:00 | 5680.05 | 5727.90 | 5683.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 12:15:00 | 5669.10 | 5716.14 | 5681.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 14:00:00 | 5725.05 | 5717.92 | 5685.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 14:30:00 | 5702.40 | 5719.22 | 5689.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 10:15:00 | 5859.40 | 5924.86 | 5933.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 10:15:00 | 5859.40 | 5924.86 | 5933.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 13:15:00 | 5840.85 | 5901.10 | 5920.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 12:15:00 | 5469.70 | 5466.75 | 5535.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 12:45:00 | 5472.25 | 5466.75 | 5535.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 5473.25 | 5467.63 | 5524.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 14:45:00 | 5476.20 | 5467.63 | 5524.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 4946.40 | 4974.71 | 5091.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 4918.30 | 4964.51 | 5076.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 13:45:00 | 4912.05 | 4951.14 | 5041.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 15:15:00 | 4932.85 | 4952.31 | 5034.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 12:15:00 | 5040.00 | 4954.07 | 4952.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-04-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 12:15:00 | 5040.00 | 4954.07 | 4952.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 14:15:00 | 5059.50 | 5025.45 | 4999.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 09:15:00 | 5002.00 | 5024.85 | 5003.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-16 09:15:00 | 5002.00 | 5024.85 | 5003.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 09:15:00 | 5002.00 | 5024.85 | 5003.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 10:00:00 | 5002.00 | 5024.85 | 5003.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 10:15:00 | 5000.00 | 5019.88 | 5003.55 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2025-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-17 09:15:00 | 4923.00 | 4985.63 | 4992.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-17 14:15:00 | 4903.00 | 4938.26 | 4963.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-21 13:15:00 | 4942.00 | 4914.82 | 4937.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-21 13:15:00 | 4942.00 | 4914.82 | 4937.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 13:15:00 | 4942.00 | 4914.82 | 4937.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 13:30:00 | 4961.00 | 4914.82 | 4937.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 14:15:00 | 4958.00 | 4923.46 | 4939.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 14:45:00 | 5010.00 | 4923.46 | 4939.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 15:15:00 | 4979.00 | 4934.57 | 4942.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-22 09:15:00 | 4939.50 | 4934.57 | 4942.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 09:15:00 | 5051.00 | 4957.85 | 4952.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 10:15:00 | 5256.50 | 5017.58 | 4980.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 09:15:00 | 5414.50 | 5467.91 | 5369.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 09:15:00 | 5414.50 | 5467.91 | 5369.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 5414.50 | 5467.91 | 5369.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 09:45:00 | 5380.00 | 5467.91 | 5369.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 5379.00 | 5450.13 | 5370.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 5379.00 | 5450.13 | 5370.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 5394.50 | 5439.00 | 5372.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:45:00 | 5349.50 | 5439.00 | 5372.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 5600.00 | 5588.30 | 5544.19 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2025-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 13:15:00 | 5525.00 | 5556.86 | 5559.09 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 5563.50 | 5558.86 | 5558.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 11:15:00 | 5654.50 | 5577.99 | 5567.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 10:15:00 | 5670.00 | 5684.19 | 5637.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 11:00:00 | 5670.00 | 5684.19 | 5637.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 5598.00 | 5666.95 | 5633.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:45:00 | 5600.00 | 5666.95 | 5633.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 5632.50 | 5660.06 | 5633.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 12:30:00 | 5596.00 | 5660.06 | 5633.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 5597.00 | 5647.45 | 5630.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:00:00 | 5597.00 | 5647.45 | 5630.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 5599.50 | 5637.86 | 5627.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:30:00 | 5573.00 | 5637.86 | 5627.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 5582.50 | 5615.13 | 5618.12 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 11:15:00 | 5632.50 | 5622.26 | 5621.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 13:15:00 | 5663.50 | 5631.75 | 5625.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 14:15:00 | 6027.00 | 6047.57 | 5887.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 15:00:00 | 6027.00 | 6047.57 | 5887.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 6078.00 | 6046.04 | 5914.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 11:15:00 | 6151.00 | 6061.94 | 5933.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 09:15:00 | 6324.50 | 6090.72 | 5998.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-14 13:15:00 | 6766.10 | 6598.65 | 6440.40 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 7500.00 | 7658.05 | 7679.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 7445.00 | 7615.44 | 7657.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 7629.50 | 7583.78 | 7634.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 7629.50 | 7583.78 | 7634.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 7629.50 | 7583.78 | 7634.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:45:00 | 7600.00 | 7583.78 | 7634.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 7580.00 | 7583.02 | 7629.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 11:15:00 | 7535.00 | 7583.02 | 7629.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 11:45:00 | 7532.00 | 7560.92 | 7614.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 15:15:00 | 7651.00 | 7603.40 | 7597.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 15:15:00 | 7651.00 | 7603.40 | 7597.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 7741.00 | 7630.92 | 7610.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 14:15:00 | 7745.00 | 7754.42 | 7690.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-23 15:00:00 | 7745.00 | 7754.42 | 7690.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 7696.50 | 7743.87 | 7701.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:45:00 | 7697.50 | 7743.87 | 7701.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 7650.00 | 7725.09 | 7696.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:45:00 | 7629.00 | 7725.09 | 7696.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 7644.50 | 7708.98 | 7691.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:30:00 | 7646.00 | 7708.98 | 7691.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 14:15:00 | 7600.00 | 7673.74 | 7677.97 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 11:15:00 | 7720.00 | 7680.74 | 7678.89 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 14:15:00 | 7663.00 | 7677.44 | 7677.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 15:15:00 | 7620.00 | 7665.95 | 7672.69 | Break + close below crossover candle low |

### Cycle 87 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 7771.50 | 7687.06 | 7681.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 10:15:00 | 7820.00 | 7713.65 | 7694.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 10:15:00 | 8071.00 | 8082.73 | 8001.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 10:30:00 | 8045.50 | 8082.73 | 8001.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 8034.00 | 8065.52 | 8034.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 12:00:00 | 8034.00 | 8065.52 | 8034.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 7925.00 | 8037.42 | 8024.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:00:00 | 7925.00 | 8037.42 | 8024.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 7934.00 | 8016.74 | 8016.34 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 7980.50 | 8009.49 | 8013.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 12:15:00 | 7887.00 | 7955.30 | 7982.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 14:15:00 | 7951.50 | 7946.18 | 7972.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 14:15:00 | 7951.50 | 7946.18 | 7972.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 7951.50 | 7946.18 | 7972.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:45:00 | 7939.50 | 7946.18 | 7972.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 7975.50 | 7952.57 | 7971.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 10:30:00 | 7948.00 | 7960.76 | 7973.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 12:15:00 | 8029.00 | 7983.00 | 7981.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 8029.00 | 7983.00 | 7981.73 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 11:15:00 | 7909.00 | 7972.72 | 7981.26 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 13:15:00 | 8039.00 | 7992.34 | 7989.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 15:15:00 | 8141.00 | 8035.22 | 8009.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 8146.50 | 8158.27 | 8105.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 11:15:00 | 8063.50 | 8132.23 | 8102.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 8063.50 | 8132.23 | 8102.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:30:00 | 8052.50 | 8132.23 | 8102.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 8035.00 | 8112.79 | 8096.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:30:00 | 8011.00 | 8112.79 | 8096.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 8078.50 | 8110.75 | 8099.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:00:00 | 8078.50 | 8110.75 | 8099.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 8084.50 | 8105.50 | 8098.54 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2025-06-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 12:15:00 | 8052.50 | 8090.34 | 8092.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 13:15:00 | 8000.00 | 8072.27 | 8084.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 14:15:00 | 8149.50 | 8087.72 | 8090.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 14:15:00 | 8149.50 | 8087.72 | 8090.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 8149.50 | 8087.72 | 8090.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:00:00 | 8149.50 | 8087.72 | 8090.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 15:15:00 | 8110.00 | 8092.17 | 8091.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 11:15:00 | 8162.00 | 8109.59 | 8100.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 8035.50 | 8107.64 | 8101.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 13:15:00 | 8035.50 | 8107.64 | 8101.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 8035.50 | 8107.64 | 8101.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 8035.50 | 8107.64 | 8101.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 8005.50 | 8087.21 | 8092.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 15:15:00 | 7967.50 | 8063.27 | 8081.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 7890.00 | 7884.03 | 7961.36 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:15:00 | 7815.50 | 7884.03 | 7961.36 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 8025.00 | 7825.22 | 7873.38 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-17 09:15:00 | 8025.00 | 7825.22 | 7873.38 | SL hit (close>ema400) qty=1.00 sl=7873.38 alert=retest1 |

### Cycle 95 — BUY (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 12:15:00 | 7979.50 | 7911.36 | 7905.83 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 10:15:00 | 7819.00 | 7904.54 | 7909.08 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 15:15:00 | 7987.50 | 7915.34 | 7909.73 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 10:15:00 | 7862.00 | 7899.18 | 7903.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 11:15:00 | 7840.50 | 7887.44 | 7897.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 13:15:00 | 7776.00 | 7773.86 | 7816.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 13:15:00 | 7776.00 | 7773.86 | 7816.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 7776.00 | 7773.86 | 7816.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:30:00 | 7797.50 | 7773.86 | 7816.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 7952.00 | 7809.49 | 7828.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 7952.00 | 7809.49 | 7828.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 7950.00 | 7837.59 | 7839.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 7870.00 | 7837.59 | 7839.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 7865.00 | 7843.07 | 7842.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 09:15:00 | 7865.00 | 7843.07 | 7842.21 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 12:15:00 | 7835.50 | 7841.02 | 7841.42 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 7872.50 | 7847.32 | 7844.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 14:15:00 | 7895.00 | 7856.86 | 7848.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 15:15:00 | 7823.00 | 7850.08 | 7846.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 15:15:00 | 7823.00 | 7850.08 | 7846.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 7823.00 | 7850.08 | 7846.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 7985.00 | 7850.08 | 7846.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 7999.50 | 7906.56 | 7888.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-26 11:15:00 | 8783.50 | 8370.47 | 8159.32 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 12:15:00 | 8560.50 | 8785.46 | 8786.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 14:15:00 | 8541.00 | 8618.91 | 8680.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 12:15:00 | 8602.00 | 8585.37 | 8636.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 12:15:00 | 8602.00 | 8585.37 | 8636.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 8602.00 | 8585.37 | 8636.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:45:00 | 8609.00 | 8585.37 | 8636.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 8632.50 | 8593.78 | 8631.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 8632.50 | 8593.78 | 8631.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 8600.00 | 8595.02 | 8628.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 8774.50 | 8595.02 | 8628.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 8780.50 | 8632.12 | 8642.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:30:00 | 8778.00 | 8632.12 | 8642.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 8765.00 | 8658.70 | 8653.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 09:15:00 | 8919.50 | 8776.76 | 8720.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 8831.50 | 8857.63 | 8799.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 10:00:00 | 8831.50 | 8857.63 | 8799.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 8676.00 | 8821.30 | 8788.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 8676.00 | 8821.30 | 8788.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 8629.00 | 8782.84 | 8774.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:30:00 | 8653.50 | 8782.84 | 8774.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 8633.00 | 8752.87 | 8761.29 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 8845.00 | 8752.63 | 8745.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 8878.50 | 8788.34 | 8763.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 14:15:00 | 9026.50 | 9027.37 | 8939.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 15:00:00 | 9026.50 | 9027.37 | 8939.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 8878.50 | 8992.90 | 8938.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:30:00 | 8884.50 | 8992.90 | 8938.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 8843.00 | 8962.92 | 8930.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:00:00 | 8843.00 | 8962.92 | 8930.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 8850.00 | 8906.85 | 8909.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 12:15:00 | 8810.00 | 8870.66 | 8888.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 14:15:00 | 8916.00 | 8871.94 | 8885.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 14:15:00 | 8916.00 | 8871.94 | 8885.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 8916.00 | 8871.94 | 8885.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:45:00 | 8942.00 | 8871.94 | 8885.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 8872.00 | 8871.95 | 8884.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 8972.00 | 8871.95 | 8884.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 09:15:00 | 9054.00 | 8908.36 | 8899.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 10:15:00 | 9193.50 | 8965.39 | 8926.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 14:15:00 | 9185.00 | 9217.02 | 9130.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 14:15:00 | 9185.00 | 9217.02 | 9130.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 9185.00 | 9217.02 | 9130.34 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 9080.00 | 9121.19 | 9122.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 8924.00 | 9081.75 | 9104.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 9070.00 | 9022.20 | 9056.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 9070.00 | 9022.20 | 9056.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 9070.00 | 9022.20 | 9056.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:45:00 | 9073.00 | 9022.20 | 9056.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 8952.00 | 9008.16 | 9046.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:15:00 | 8940.00 | 9008.16 | 9046.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 9330.00 | 8838.78 | 8894.56 | SL hit (close>static) qty=1.00 sl=9082.50 alert=retest2 |

### Cycle 109 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 9750.50 | 9021.12 | 8972.37 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 13:15:00 | 9030.00 | 9236.07 | 9251.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 8947.00 | 9178.25 | 9223.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 9016.00 | 8894.15 | 8980.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 11:15:00 | 9016.00 | 8894.15 | 8980.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 9016.00 | 8894.15 | 8980.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:45:00 | 9028.00 | 8894.15 | 8980.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 9025.00 | 8920.32 | 8984.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 14:15:00 | 8974.00 | 8944.96 | 8989.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 09:30:00 | 8993.00 | 8959.98 | 8986.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 12:15:00 | 8543.35 | 8758.46 | 8852.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 13:15:00 | 8525.30 | 8722.67 | 8827.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-08 13:15:00 | 8724.50 | 8660.15 | 8738.36 | SL hit (close>ema200) qty=0.50 sl=8660.15 alert=retest2 |

### Cycle 111 — BUY (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 15:15:00 | 8824.00 | 8723.15 | 8719.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 13:15:00 | 8843.50 | 8791.33 | 8758.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 14:15:00 | 8744.00 | 8781.87 | 8757.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 14:15:00 | 8744.00 | 8781.87 | 8757.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 8744.00 | 8781.87 | 8757.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 8744.00 | 8781.87 | 8757.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 8739.50 | 8773.39 | 8755.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:30:00 | 8744.00 | 8755.91 | 8749.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 8668.00 | 8738.33 | 8741.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 11:15:00 | 8649.00 | 8720.47 | 8733.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 8496.50 | 8468.09 | 8515.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 10:15:00 | 8471.50 | 8468.09 | 8515.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 8438.00 | 8462.07 | 8508.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 11:30:00 | 8408.00 | 8457.66 | 8502.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 12:15:00 | 8419.00 | 8457.66 | 8502.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 11:30:00 | 8422.00 | 8434.41 | 8463.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:15:00 | 7987.60 | 8097.02 | 8188.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:15:00 | 7998.05 | 8097.02 | 8188.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:15:00 | 8000.90 | 8097.02 | 8188.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 10:15:00 | 7816.00 | 7795.73 | 7898.19 | SL hit (close>ema200) qty=0.50 sl=7795.73 alert=retest2 |

### Cycle 113 — BUY (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 12:15:00 | 7899.50 | 7830.04 | 7826.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 7907.00 | 7860.54 | 7843.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 11:15:00 | 7861.00 | 7861.26 | 7846.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 12:00:00 | 7861.00 | 7861.26 | 7846.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 7866.00 | 7862.25 | 7850.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:45:00 | 7861.50 | 7862.25 | 7850.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 7888.50 | 7884.57 | 7868.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 15:00:00 | 7905.50 | 7888.76 | 7872.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 7801.50 | 7866.13 | 7866.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 7801.50 | 7866.13 | 7866.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 13:15:00 | 7797.00 | 7845.72 | 7856.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 11:15:00 | 7802.00 | 7799.23 | 7825.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 11:15:00 | 7802.00 | 7799.23 | 7825.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 7802.00 | 7799.23 | 7825.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:30:00 | 7815.00 | 7799.23 | 7825.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 7869.00 | 7813.19 | 7829.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:00:00 | 7869.00 | 7813.19 | 7829.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 7875.00 | 7825.55 | 7833.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:00:00 | 7875.00 | 7825.55 | 7833.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 7843.50 | 7829.14 | 7834.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:15:00 | 7803.00 | 7829.14 | 7834.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 09:15:00 | 7899.00 | 7838.93 | 7838.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 09:15:00 | 7899.00 | 7838.93 | 7838.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 14:15:00 | 7961.00 | 7894.79 | 7868.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 14:15:00 | 8463.50 | 8463.57 | 8301.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 15:00:00 | 8463.50 | 8463.57 | 8301.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 8832.00 | 8902.32 | 8852.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:00:00 | 8832.00 | 8902.32 | 8852.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 8726.50 | 8867.15 | 8841.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:00:00 | 8726.50 | 8867.15 | 8841.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 8857.50 | 8864.83 | 8846.46 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 8725.50 | 8816.91 | 8826.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 15:15:00 | 8700.00 | 8766.86 | 8797.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 8708.00 | 8699.94 | 8738.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 8708.00 | 8699.94 | 8738.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 8708.00 | 8699.94 | 8738.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 8708.00 | 8699.94 | 8738.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 8774.00 | 8719.27 | 8735.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:30:00 | 8762.50 | 8719.27 | 8735.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 8756.00 | 8726.62 | 8737.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 15:15:00 | 8721.00 | 8726.62 | 8737.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:30:00 | 8743.50 | 8735.29 | 8739.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 10:15:00 | 8796.00 | 8747.44 | 8744.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 10:15:00 | 8796.00 | 8747.44 | 8744.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 13:15:00 | 8804.50 | 8769.87 | 8756.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 14:15:00 | 8716.00 | 8759.10 | 8752.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 14:15:00 | 8716.00 | 8759.10 | 8752.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 8716.00 | 8759.10 | 8752.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 8716.00 | 8759.10 | 8752.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 8690.00 | 8745.28 | 8746.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 8651.00 | 8726.42 | 8738.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 14:15:00 | 8240.50 | 8216.81 | 8344.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 15:00:00 | 8240.50 | 8216.81 | 8344.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 8348.00 | 8240.65 | 8279.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:45:00 | 8373.50 | 8240.65 | 8279.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 8343.50 | 8261.22 | 8285.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 11:00:00 | 8343.50 | 8261.22 | 8285.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 12:15:00 | 8424.00 | 8308.14 | 8303.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 09:15:00 | 8450.00 | 8376.41 | 8340.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 12:15:00 | 8367.50 | 8385.90 | 8355.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 13:00:00 | 8367.50 | 8385.90 | 8355.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 8340.50 | 8376.82 | 8353.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:45:00 | 8331.50 | 8376.82 | 8353.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 8474.00 | 8396.26 | 8364.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:45:00 | 8519.00 | 8427.91 | 8385.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:00:00 | 8493.00 | 8485.94 | 8430.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 8575.00 | 8494.00 | 8444.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 14:45:00 | 8548.50 | 8467.93 | 8448.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 8422.00 | 8462.44 | 8449.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:45:00 | 8455.00 | 8462.44 | 8449.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 8394.00 | 8448.75 | 8444.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:45:00 | 8375.00 | 8448.75 | 8444.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-09 11:15:00 | 8349.00 | 8428.80 | 8435.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 8349.00 | 8428.80 | 8435.79 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 8554.50 | 8450.34 | 8440.97 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 14:15:00 | 8376.50 | 8426.85 | 8433.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 8285.50 | 8391.40 | 8415.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 12:15:00 | 8369.50 | 8359.74 | 8392.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-13 13:00:00 | 8369.50 | 8359.74 | 8392.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 8375.50 | 8362.89 | 8390.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:00:00 | 8375.50 | 8362.89 | 8390.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 8534.50 | 8397.21 | 8404.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 15:00:00 | 8534.50 | 8397.21 | 8404.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 15:15:00 | 8475.00 | 8412.77 | 8410.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 09:15:00 | 8561.00 | 8442.42 | 8424.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 11:15:00 | 8400.00 | 8435.87 | 8424.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 11:15:00 | 8400.00 | 8435.87 | 8424.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 8400.00 | 8435.87 | 8424.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:00:00 | 8400.00 | 8435.87 | 8424.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 8363.00 | 8421.29 | 8418.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:00:00 | 8363.00 | 8421.29 | 8418.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 13:15:00 | 8375.00 | 8412.03 | 8414.90 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 14:15:00 | 8531.00 | 8435.83 | 8425.46 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 11:15:00 | 8359.00 | 8410.85 | 8417.09 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 8655.50 | 8433.28 | 8420.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 13:15:00 | 8883.00 | 8667.22 | 8548.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 8700.50 | 8718.48 | 8633.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 13:00:00 | 8700.50 | 8718.48 | 8633.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 8621.00 | 8686.28 | 8644.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:00:00 | 8621.00 | 8686.28 | 8644.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 8559.50 | 8660.92 | 8636.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:00:00 | 8559.50 | 8660.92 | 8636.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 8573.00 | 8643.34 | 8631.06 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 13:15:00 | 8565.00 | 8614.66 | 8619.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-21 14:15:00 | 8550.00 | 8594.93 | 8607.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 09:15:00 | 8636.00 | 8603.14 | 8609.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 8636.00 | 8603.14 | 8609.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 8636.00 | 8603.14 | 8609.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:15:00 | 8746.00 | 8603.14 | 8609.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 8800.00 | 8642.51 | 8627.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 10:15:00 | 8856.00 | 8767.07 | 8719.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 9308.50 | 9356.53 | 9204.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 10:00:00 | 9308.50 | 9356.53 | 9204.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 9268.00 | 9328.98 | 9217.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 12:00:00 | 9268.00 | 9328.98 | 9217.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 9227.00 | 9289.84 | 9225.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:30:00 | 9216.00 | 9289.84 | 9225.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 9230.00 | 9277.87 | 9226.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 9227.50 | 9277.87 | 9226.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 8880.00 | 9198.30 | 9194.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 8880.00 | 9198.30 | 9194.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 8792.50 | 9117.14 | 9158.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 13:15:00 | 8647.50 | 8928.26 | 9053.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 09:15:00 | 8425.00 | 8409.63 | 8552.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 09:30:00 | 8460.50 | 8409.63 | 8552.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 8220.00 | 8280.51 | 8338.07 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 8567.50 | 8396.35 | 8373.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 8927.00 | 8529.31 | 8439.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 10:15:00 | 8830.50 | 8863.99 | 8708.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 11:00:00 | 8830.50 | 8863.99 | 8708.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 9179.50 | 9291.76 | 9249.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 9179.50 | 9291.76 | 9249.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 9252.50 | 9283.90 | 9249.49 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 9113.00 | 9221.66 | 9230.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 9051.00 | 9187.53 | 9214.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 14:15:00 | 9159.50 | 9144.67 | 9181.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 14:15:00 | 9159.50 | 9144.67 | 9181.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 9159.50 | 9144.67 | 9181.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 9159.50 | 9144.67 | 9181.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 9122.00 | 9140.14 | 9176.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 9110.00 | 9140.14 | 9176.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 11:00:00 | 9102.00 | 9123.85 | 9162.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 11:15:00 | 9110.00 | 9025.20 | 9044.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 12:15:00 | 9129.50 | 9062.03 | 9058.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 9129.50 | 9062.03 | 9058.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 15:15:00 | 9174.00 | 9103.21 | 9080.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 14:15:00 | 9127.50 | 9163.28 | 9127.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 14:15:00 | 9127.50 | 9163.28 | 9127.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 9127.50 | 9163.28 | 9127.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 15:00:00 | 9127.50 | 9163.28 | 9127.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 9163.00 | 9163.22 | 9131.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 9210.00 | 9163.22 | 9131.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 9174.50 | 9206.46 | 9177.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 10:15:00 | 9038.50 | 9164.63 | 9163.34 | SL hit (close<static) qty=1.00 sl=9115.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 11:15:00 | 8995.00 | 9130.71 | 9148.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 12:15:00 | 8971.50 | 9098.87 | 9131.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 8817.50 | 8770.97 | 8855.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 09:30:00 | 8787.00 | 8770.97 | 8855.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 8845.00 | 8790.02 | 8832.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 8845.00 | 8790.02 | 8832.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 8924.00 | 8816.82 | 8840.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 8786.00 | 8816.82 | 8840.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:45:00 | 8820.00 | 8818.32 | 8826.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 15:15:00 | 8821.00 | 8753.58 | 8752.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 15:15:00 | 8821.00 | 8753.58 | 8752.28 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 09:15:00 | 8727.00 | 8748.27 | 8749.98 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 8989.50 | 8795.82 | 8770.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 9077.00 | 8876.49 | 8813.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 8969.50 | 9030.38 | 8981.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 8969.50 | 9030.38 | 8981.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 8969.50 | 9030.38 | 8981.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 8969.50 | 9030.38 | 8981.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 8952.00 | 9014.71 | 8978.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:15:00 | 8930.00 | 9014.71 | 8978.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 8923.50 | 8969.73 | 8964.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:30:00 | 8925.00 | 8969.73 | 8964.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 8924.50 | 8955.21 | 8958.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 8842.50 | 8932.67 | 8948.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 8778.50 | 8690.36 | 8764.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 8778.50 | 8690.36 | 8764.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 8778.50 | 8690.36 | 8764.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 8778.50 | 8690.36 | 8764.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 8653.50 | 8682.99 | 8754.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 8621.50 | 8682.99 | 8754.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 8942.50 | 8750.54 | 8752.55 | SL hit (close>static) qty=1.00 sl=8825.00 alert=retest2 |

### Cycle 139 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 8928.50 | 8786.13 | 8768.54 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 11:15:00 | 8740.00 | 8826.69 | 8830.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 13:15:00 | 8687.50 | 8784.02 | 8809.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 8454.00 | 8444.01 | 8522.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 10:00:00 | 8454.00 | 8444.01 | 8522.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 8390.00 | 8405.19 | 8452.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:15:00 | 8380.50 | 8405.19 | 8452.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 7961.47 | 8103.70 | 8144.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-14 10:15:00 | 7542.45 | 7670.58 | 7808.73 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 141 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 7218.00 | 7143.74 | 7138.68 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 7031.50 | 7130.86 | 7137.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 7029.00 | 7094.99 | 7119.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 7245.50 | 7102.84 | 7114.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 7245.50 | 7102.84 | 7114.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 7245.50 | 7102.84 | 7114.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:00:00 | 7245.50 | 7102.84 | 7114.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 7174.00 | 7117.07 | 7120.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:30:00 | 7230.00 | 7117.07 | 7120.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 11:15:00 | 7217.50 | 7137.16 | 7129.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 12:15:00 | 7308.00 | 7192.61 | 7165.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 13:15:00 | 7874.50 | 7913.66 | 7779.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 13:30:00 | 7907.50 | 7913.66 | 7779.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 8162.00 | 7990.60 | 7898.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 14:30:00 | 7990.00 | 7990.60 | 7898.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 9484.50 | 9517.28 | 9408.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:30:00 | 9486.00 | 9517.28 | 9408.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 9533.00 | 9509.40 | 9431.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 11:00:00 | 9570.00 | 9521.52 | 9443.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 13:30:00 | 9560.00 | 9532.99 | 9468.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 12:45:00 | 9552.50 | 9541.50 | 9503.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 09:45:00 | 9575.00 | 9523.24 | 9504.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 9539.00 | 9548.74 | 9525.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:45:00 | 9538.00 | 9548.74 | 9525.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 9550.00 | 9549.00 | 9527.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 14:30:00 | 9532.50 | 9549.00 | 9527.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 9512.00 | 9540.96 | 9527.46 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-12 11:15:00 | 9487.50 | 9515.79 | 9517.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 9487.50 | 9515.79 | 9517.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 9359.00 | 9479.56 | 9498.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 11:15:00 | 9518.00 | 9480.92 | 9495.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 11:15:00 | 9518.00 | 9480.92 | 9495.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 9518.00 | 9480.92 | 9495.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:00:00 | 9518.00 | 9480.92 | 9495.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 9446.50 | 9474.04 | 9491.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 13:45:00 | 9419.00 | 9466.53 | 9486.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 14:30:00 | 9412.00 | 9476.82 | 9489.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:15:00 | 9420.00 | 9476.82 | 9489.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 10:30:00 | 9415.00 | 9456.49 | 9475.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 9525.00 | 9459.31 | 9469.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 9525.00 | 9459.31 | 9469.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 9515.00 | 9470.45 | 9473.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 9490.00 | 9470.45 | 9473.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 9536.00 | 9483.56 | 9479.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 9536.00 | 9483.56 | 9479.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 9859.00 | 9586.82 | 9533.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 13:15:00 | 10055.00 | 10058.94 | 9910.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 14:00:00 | 10055.00 | 10058.94 | 9910.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 10109.00 | 10064.27 | 9950.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 11:15:00 | 10299.50 | 10073.81 | 9964.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-27 12:15:00 | 11329.45 | 10980.96 | 10790.07 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2026-03-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 15:15:00 | 10688.00 | 10813.95 | 10829.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 10167.00 | 10684.56 | 10769.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 10380.00 | 10272.48 | 10465.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:00:00 | 10380.00 | 10272.48 | 10465.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 10432.00 | 10304.38 | 10462.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 11:00:00 | 10432.00 | 10304.38 | 10462.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 10305.00 | 10304.50 | 10447.96 | EMA400 retest candle locked (from downside) |

### Cycle 147 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 10630.00 | 10484.78 | 10474.54 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 15:15:00 | 10395.00 | 10463.31 | 10468.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 10021.00 | 10374.85 | 10428.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 15:15:00 | 10145.00 | 10142.05 | 10261.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-10 09:15:00 | 10345.00 | 10142.05 | 10261.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 10329.00 | 10179.44 | 10267.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 13:15:00 | 10187.00 | 10227.54 | 10270.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:15:00 | 9677.65 | 9953.30 | 10104.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-12 09:15:00 | 9168.30 | 9492.44 | 9796.27 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 149 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 9483.00 | 9282.44 | 9266.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 13:15:00 | 9537.00 | 9333.35 | 9291.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 13:15:00 | 9533.00 | 9588.75 | 9514.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 14:00:00 | 9533.00 | 9588.75 | 9514.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 9592.00 | 9589.40 | 9521.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 14:45:00 | 9614.00 | 9589.40 | 9521.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 9412.00 | 9612.40 | 9586.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 9412.00 | 9612.40 | 9586.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 9402.00 | 9570.32 | 9569.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:45:00 | 9403.00 | 9570.32 | 9569.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — SELL (started 2026-03-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 11:15:00 | 9425.00 | 9541.25 | 9556.38 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2026-03-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-23 14:15:00 | 9822.00 | 9581.36 | 9568.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 12:15:00 | 9850.00 | 9713.15 | 9644.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 9960.00 | 9966.84 | 9854.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:15:00 | 10011.00 | 9966.84 | 9854.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 9891.00 | 9951.67 | 9857.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 11:15:00 | 10142.00 | 9973.54 | 9876.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 15:15:00 | 9782.00 | 10002.59 | 10020.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2026-03-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 15:15:00 | 9782.00 | 10002.59 | 10020.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 15:15:00 | 9738.00 | 9880.82 | 9948.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 9752.00 | 9631.36 | 9772.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 14:00:00 | 9752.00 | 9631.36 | 9772.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 9779.50 | 9660.99 | 9772.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 15:00:00 | 9779.50 | 9660.99 | 9772.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 9817.00 | 9692.19 | 9776.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:15:00 | 9759.50 | 9692.19 | 9776.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 9825.00 | 9718.75 | 9781.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:30:00 | 9929.50 | 9718.75 | 9781.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 9824.00 | 9739.80 | 9785.17 | EMA400 retest candle locked (from downside) |

### Cycle 153 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 9949.50 | 9837.35 | 9823.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 09:15:00 | 10065.00 | 9912.96 | 9864.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 10:15:00 | 11104.50 | 11144.49 | 10925.60 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 12:15:00 | 11203.50 | 11151.89 | 10948.86 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:15:00 | 11328.00 | 11156.15 | 11017.81 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 11295.00 | 11228.13 | 11134.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:45:00 | 11239.50 | 11228.13 | 11134.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 11188.50 | 11249.98 | 11171.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 13:00:00 | 11188.50 | 11249.98 | 11171.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 11252.00 | 11250.38 | 11178.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 11600.00 | 11258.62 | 11194.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 09:15:00 | 11763.68 | 11343.30 | 11239.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-20 10:15:00 | 11894.40 | 11587.97 | 11430.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-04-21 09:15:00 | 12323.85 | 11889.85 | 11672.90 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 154 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 11765.00 | 11844.44 | 11850.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 11615.00 | 11798.55 | 11829.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 11701.00 | 11690.55 | 11760.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 11701.00 | 11690.55 | 11760.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 11701.00 | 11690.55 | 11760.02 | EMA400 retest candle locked (from downside) |

### Cycle 155 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 11998.00 | 11784.39 | 11782.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 14:15:00 | 12116.00 | 11850.71 | 11812.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 10:15:00 | 11860.00 | 11900.41 | 11850.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 10:15:00 | 11860.00 | 11900.41 | 11850.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 11860.00 | 11900.41 | 11850.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:45:00 | 11915.50 | 11900.41 | 11850.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 11780.00 | 11876.33 | 11843.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:00:00 | 11780.00 | 11876.33 | 11843.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 11773.00 | 11855.66 | 11837.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:00:00 | 11773.00 | 11855.66 | 11837.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 11870.00 | 11851.38 | 11838.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 11990.00 | 11857.11 | 11841.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-23 14:15:00 | 7893.00 | 2024-05-24 09:15:00 | 8019.70 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-05-23 15:00:00 | 7840.10 | 2024-05-24 09:15:00 | 8019.70 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2024-06-05 09:30:00 | 7108.00 | 2024-06-06 10:15:00 | 7803.00 | STOP_HIT | 1.00 | -9.78% |
| BUY | retest2 | 2024-06-18 09:15:00 | 8517.15 | 2024-06-21 11:15:00 | 8344.45 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-06-19 13:45:00 | 8434.05 | 2024-06-21 11:15:00 | 8344.45 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-06-20 09:15:00 | 8490.65 | 2024-06-21 11:15:00 | 8344.45 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-06-26 09:15:00 | 8558.00 | 2024-06-26 14:15:00 | 8386.40 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2024-06-26 09:45:00 | 8535.00 | 2024-06-26 14:15:00 | 8386.40 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-07-01 13:45:00 | 8444.05 | 2024-07-03 09:15:00 | 8576.95 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-07-02 09:30:00 | 8401.00 | 2024-07-03 09:15:00 | 8576.95 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2024-07-05 09:15:00 | 8651.10 | 2024-07-10 11:15:00 | 8703.00 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest2 | 2024-07-05 10:00:00 | 8674.00 | 2024-07-10 11:15:00 | 8703.00 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2024-07-12 10:30:00 | 8621.80 | 2024-07-15 09:15:00 | 9012.70 | STOP_HIT | 1.00 | -4.53% |
| SELL | retest2 | 2024-07-12 15:15:00 | 8630.00 | 2024-07-15 09:15:00 | 9012.70 | STOP_HIT | 1.00 | -4.43% |
| SELL | retest1 | 2024-07-22 13:00:00 | 8130.15 | 2024-07-26 13:15:00 | 8084.40 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest1 | 2024-07-22 14:00:00 | 8133.00 | 2024-07-26 13:15:00 | 8084.40 | STOP_HIT | 1.00 | 0.60% |
| SELL | retest1 | 2024-07-23 09:15:00 | 8094.40 | 2024-07-26 13:15:00 | 8084.40 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2024-07-23 12:15:00 | 7878.70 | 2024-07-26 15:15:00 | 8200.00 | STOP_HIT | 1.00 | -4.08% |
| SELL | retest2 | 2024-07-23 13:00:00 | 7978.50 | 2024-07-26 15:15:00 | 8200.00 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2024-07-23 13:30:00 | 7965.00 | 2024-07-26 15:15:00 | 8200.00 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2024-07-23 14:15:00 | 7969.65 | 2024-07-26 15:15:00 | 8200.00 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2024-08-06 11:30:00 | 8224.10 | 2024-08-12 11:15:00 | 8346.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-08-06 12:30:00 | 8219.20 | 2024-08-12 11:15:00 | 8346.00 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-08-08 13:15:00 | 8227.00 | 2024-08-12 11:15:00 | 8346.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-08-12 10:15:00 | 8222.20 | 2024-08-12 11:15:00 | 8346.00 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-08-21 13:00:00 | 8280.95 | 2024-08-22 10:15:00 | 8384.95 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-08-21 13:45:00 | 8291.35 | 2024-08-22 10:15:00 | 8384.95 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-08-21 14:30:00 | 8290.10 | 2024-08-22 10:15:00 | 8384.95 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-08-30 11:00:00 | 9407.30 | 2024-09-02 10:15:00 | 8929.25 | STOP_HIT | 1.00 | -5.08% |
| SELL | retest2 | 2024-09-04 12:00:00 | 8874.00 | 2024-09-04 14:15:00 | 9003.75 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-09-04 12:30:00 | 8880.55 | 2024-09-04 14:15:00 | 9003.75 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-09-04 13:00:00 | 8838.70 | 2024-09-04 14:15:00 | 9003.75 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2024-09-30 09:30:00 | 9435.35 | 2024-10-01 10:15:00 | 9689.00 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2024-09-30 11:00:00 | 9420.00 | 2024-10-01 10:15:00 | 9689.00 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2024-09-30 12:15:00 | 9436.65 | 2024-10-01 10:15:00 | 9689.00 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2024-09-30 14:00:00 | 9394.45 | 2024-10-01 10:15:00 | 9689.00 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2024-10-04 09:15:00 | 9412.80 | 2024-10-08 13:15:00 | 9521.95 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-10-04 12:30:00 | 9501.00 | 2024-10-08 13:15:00 | 9521.95 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-10-07 09:15:00 | 9408.00 | 2024-10-08 13:15:00 | 9521.95 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-10-08 10:30:00 | 9496.40 | 2024-10-08 13:15:00 | 9521.95 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2024-10-15 13:30:00 | 10829.65 | 2024-10-17 09:15:00 | 10426.10 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2024-10-16 10:45:00 | 10826.25 | 2024-10-17 09:15:00 | 10426.10 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2024-10-23 12:15:00 | 9823.05 | 2024-10-28 09:15:00 | 9331.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 13:00:00 | 9821.00 | 2024-10-28 09:15:00 | 9329.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 14:00:00 | 9857.85 | 2024-10-28 09:15:00 | 9364.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 09:30:00 | 9829.60 | 2024-10-28 09:15:00 | 9338.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 12:15:00 | 9823.05 | 2024-10-29 12:15:00 | 9348.90 | STOP_HIT | 0.50 | 4.83% |
| SELL | retest2 | 2024-10-23 13:00:00 | 9821.00 | 2024-10-29 12:15:00 | 9348.90 | STOP_HIT | 0.50 | 4.81% |
| SELL | retest2 | 2024-10-23 14:00:00 | 9857.85 | 2024-10-29 12:15:00 | 9348.90 | STOP_HIT | 0.50 | 5.16% |
| SELL | retest2 | 2024-10-24 09:30:00 | 9829.60 | 2024-10-29 12:15:00 | 9348.90 | STOP_HIT | 0.50 | 4.89% |
| SELL | retest2 | 2024-10-24 14:15:00 | 9711.50 | 2024-10-29 13:15:00 | 9225.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 14:15:00 | 9711.50 | 2024-10-29 13:15:00 | 9521.25 | STOP_HIT | 0.50 | 1.96% |
| SELL | retest2 | 2024-10-25 09:15:00 | 9641.85 | 2024-10-31 10:15:00 | 9633.55 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2024-11-25 09:15:00 | 9504.75 | 2024-12-03 09:15:00 | 10455.23 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-11 09:15:00 | 10185.25 | 2024-12-11 09:15:00 | 10063.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-01-22 09:15:00 | 9956.50 | 2025-01-24 12:15:00 | 9458.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-22 10:00:00 | 9953.45 | 2025-01-24 12:15:00 | 9455.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-22 09:15:00 | 9956.50 | 2025-01-27 09:15:00 | 8960.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-22 10:00:00 | 9953.45 | 2025-01-27 09:15:00 | 8958.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2025-02-12 09:15:00 | 6409.30 | 2025-02-12 14:15:00 | 6832.75 | STOP_HIT | 1.00 | -6.61% |
| SELL | retest2 | 2025-02-13 09:15:00 | 6671.05 | 2025-02-13 09:15:00 | 6880.00 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2025-02-18 10:15:00 | 6229.20 | 2025-02-21 11:15:00 | 6407.30 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2025-02-20 12:30:00 | 6243.45 | 2025-02-21 11:15:00 | 6407.30 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-03-04 11:30:00 | 5780.00 | 2025-03-07 09:15:00 | 5929.00 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2025-03-04 14:30:00 | 5778.70 | 2025-03-07 09:15:00 | 5929.00 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-03-05 10:00:00 | 5755.95 | 2025-03-07 09:15:00 | 5929.00 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2025-03-05 11:45:00 | 5769.60 | 2025-03-07 09:15:00 | 5929.00 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-03-13 14:15:00 | 5510.70 | 2025-03-18 11:15:00 | 5658.90 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2025-03-20 14:00:00 | 5725.05 | 2025-03-26 10:15:00 | 5859.40 | STOP_HIT | 1.00 | 2.35% |
| BUY | retest2 | 2025-03-20 14:30:00 | 5702.40 | 2025-03-26 10:15:00 | 5859.40 | STOP_HIT | 1.00 | 2.75% |
| SELL | retest2 | 2025-04-08 10:30:00 | 4918.30 | 2025-04-11 12:15:00 | 5040.00 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-04-08 13:45:00 | 4912.05 | 2025-04-11 12:15:00 | 5040.00 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-04-08 15:15:00 | 4932.85 | 2025-04-11 12:15:00 | 5040.00 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-05-09 11:15:00 | 6151.00 | 2025-05-14 13:15:00 | 6766.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-12 09:15:00 | 6324.50 | 2025-05-14 13:15:00 | 6956.95 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-21 11:15:00 | 7535.00 | 2025-05-22 15:15:00 | 7651.00 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-05-21 11:45:00 | 7532.00 | 2025-05-22 15:15:00 | 7651.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-06-05 10:30:00 | 7948.00 | 2025-06-05 12:15:00 | 8029.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest1 | 2025-06-16 09:15:00 | 7815.50 | 2025-06-17 09:15:00 | 8025.00 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-06-23 09:15:00 | 7870.00 | 2025-06-23 09:15:00 | 7865.00 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-06-24 09:15:00 | 7985.00 | 2025-06-26 11:15:00 | 8783.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-25 09:15:00 | 7999.50 | 2025-06-26 11:15:00 | 8799.45 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-28 11:15:00 | 8940.00 | 2025-07-29 12:15:00 | 9330.00 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest2 | 2025-08-04 14:15:00 | 8974.00 | 2025-08-07 12:15:00 | 8543.35 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2025-08-05 09:30:00 | 8993.00 | 2025-08-07 13:15:00 | 8525.30 | PARTIAL | 0.50 | 5.20% |
| SELL | retest2 | 2025-08-04 14:15:00 | 8974.00 | 2025-08-08 13:15:00 | 8724.50 | STOP_HIT | 0.50 | 2.78% |
| SELL | retest2 | 2025-08-05 09:30:00 | 8993.00 | 2025-08-08 13:15:00 | 8724.50 | STOP_HIT | 0.50 | 2.99% |
| SELL | retest2 | 2025-08-20 11:30:00 | 8408.00 | 2025-08-26 09:15:00 | 7987.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-20 12:15:00 | 8419.00 | 2025-08-26 09:15:00 | 7998.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 11:30:00 | 8422.00 | 2025-08-26 09:15:00 | 8000.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-20 11:30:00 | 8408.00 | 2025-08-29 10:15:00 | 7816.00 | STOP_HIT | 0.50 | 7.04% |
| SELL | retest2 | 2025-08-20 12:15:00 | 8419.00 | 2025-08-29 10:15:00 | 7816.00 | STOP_HIT | 0.50 | 7.16% |
| SELL | retest2 | 2025-08-21 11:30:00 | 8422.00 | 2025-08-29 10:15:00 | 7816.00 | STOP_HIT | 0.50 | 7.20% |
| BUY | retest2 | 2025-09-04 15:00:00 | 7905.50 | 2025-09-05 11:15:00 | 7801.50 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-09-08 15:15:00 | 7803.00 | 2025-09-09 09:15:00 | 7899.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-09-24 15:15:00 | 8721.00 | 2025-09-25 10:15:00 | 8796.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-09-25 09:30:00 | 8743.50 | 2025-09-25 10:15:00 | 8796.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-10-07 09:45:00 | 8519.00 | 2025-10-09 11:15:00 | 8349.00 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-10-07 14:00:00 | 8493.00 | 2025-10-09 11:15:00 | 8349.00 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-10-08 09:15:00 | 8575.00 | 2025-10-09 11:15:00 | 8349.00 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2025-10-08 14:45:00 | 8548.50 | 2025-10-09 11:15:00 | 8349.00 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-11-24 09:15:00 | 9110.00 | 2025-11-26 12:15:00 | 9129.50 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-11-24 11:00:00 | 9102.00 | 2025-11-26 12:15:00 | 9129.50 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-11-26 11:15:00 | 9110.00 | 2025-11-26 12:15:00 | 9129.50 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-11-28 09:15:00 | 9210.00 | 2025-12-01 10:15:00 | 9038.50 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-12-01 09:15:00 | 9174.50 | 2025-12-01 10:15:00 | 9038.50 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-12-05 09:15:00 | 8786.00 | 2025-12-10 15:15:00 | 8821.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-12-08 09:45:00 | 8820.00 | 2025-12-10 15:15:00 | 8821.00 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-12-19 11:15:00 | 8621.50 | 2025-12-22 09:15:00 | 8942.50 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest2 | 2026-01-01 12:15:00 | 8380.50 | 2026-01-12 09:15:00 | 7961.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 12:15:00 | 8380.50 | 2026-01-14 10:15:00 | 7542.45 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-09 11:00:00 | 9570.00 | 2026-02-12 11:15:00 | 9487.50 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-02-09 13:30:00 | 9560.00 | 2026-02-12 11:15:00 | 9487.50 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2026-02-10 12:45:00 | 9552.50 | 2026-02-12 11:15:00 | 9487.50 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2026-02-11 09:45:00 | 9575.00 | 2026-02-12 11:15:00 | 9487.50 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-02-13 13:45:00 | 9419.00 | 2026-02-17 09:15:00 | 9536.00 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-02-13 14:30:00 | 9412.00 | 2026-02-17 09:15:00 | 9536.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-02-13 15:15:00 | 9420.00 | 2026-02-17 09:15:00 | 9536.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-02-16 10:30:00 | 9415.00 | 2026-02-17 09:15:00 | 9536.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2026-02-20 11:15:00 | 10299.50 | 2026-02-27 12:15:00 | 11329.45 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-10 13:15:00 | 10187.00 | 2026-03-11 11:15:00 | 9677.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-10 13:15:00 | 10187.00 | 2026-03-12 09:15:00 | 9168.30 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-03-27 11:15:00 | 10142.00 | 2026-03-30 15:15:00 | 9782.00 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest1 | 2026-04-13 12:15:00 | 11203.50 | 2026-04-17 09:15:00 | 11763.68 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-04-15 09:15:00 | 11328.00 | 2026-04-20 10:15:00 | 11894.40 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-04-13 12:15:00 | 11203.50 | 2026-04-21 09:15:00 | 12323.85 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2026-04-15 09:15:00 | 11328.00 | 2026-04-21 13:15:00 | 11875.50 | STOP_HIT | 0.50 | 4.83% |
| BUY | retest2 | 2026-04-17 09:15:00 | 11600.00 | 2026-04-24 11:15:00 | 11765.00 | STOP_HIT | 1.00 | 1.42% |
