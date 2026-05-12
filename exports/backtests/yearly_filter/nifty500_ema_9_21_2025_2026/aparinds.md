# Apar Industries Ltd. (APARINDS)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 12760.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 74 |
| ALERT1 | 52 |
| ALERT2 | 51 |
| ALERT2_SKIP | 26 |
| ALERT3 | 118 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 43 |
| PARTIAL | 9 |
| TARGET_HIT | 8 |
| STOP_HIT | 38 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 54 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 24 / 30
- **Target hits / Stop hits / Partials:** 7 / 38 / 9
- **Avg / median % per leg:** 1.88% / -0.21%
- **Sum % (uncompounded):** 101.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 9 | 42.9% | 5 | 14 | 2 | 2.05% | 43.2% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 1 | 1 | 2 | 6.21% | 24.8% |
| BUY @ 3rd Alert (retest2) | 17 | 5 | 29.4% | 4 | 13 | 0 | 1.08% | 18.3% |
| SELL (all) | 33 | 15 | 45.5% | 2 | 24 | 7 | 1.77% | 58.4% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.68% | -2.7% |
| SELL @ 3rd Alert (retest2) | 32 | 15 | 46.9% | 2 | 23 | 7 | 1.91% | 61.1% |
| retest1 (combined) | 5 | 4 | 80.0% | 1 | 2 | 2 | 4.43% | 22.2% |
| retest2 (combined) | 49 | 20 | 40.8% | 6 | 36 | 7 | 1.62% | 79.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-20 13:15:00)

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

### Cycle 2 — BUY (started 2025-05-22 15:15:00)

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

### Cycle 3 — SELL (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 14:15:00 | 7600.00 | 7673.74 | 7677.97 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 11:15:00 | 7720.00 | 7680.74 | 7678.89 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 14:15:00 | 7663.00 | 7677.44 | 7677.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 15:15:00 | 7620.00 | 7665.95 | 7672.69 | Break + close below crossover candle low |

### Cycle 6 — BUY (started 2025-05-28 09:15:00)

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

### Cycle 7 — SELL (started 2025-06-03 14:15:00)

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

### Cycle 8 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 8029.00 | 7983.00 | 7981.73 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 11:15:00 | 7909.00 | 7972.72 | 7981.26 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-06-06 13:15:00)

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

### Cycle 11 — SELL (started 2025-06-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 12:15:00 | 8052.50 | 8090.34 | 8092.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 13:15:00 | 8000.00 | 8072.27 | 8084.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 14:15:00 | 8149.50 | 8087.72 | 8090.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 14:15:00 | 8149.50 | 8087.72 | 8090.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 8149.50 | 8087.72 | 8090.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:00:00 | 8149.50 | 8087.72 | 8090.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 15:15:00 | 8110.00 | 8092.17 | 8091.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 11:15:00 | 8162.00 | 8109.59 | 8100.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 8035.50 | 8107.64 | 8101.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 13:15:00 | 8035.50 | 8107.64 | 8101.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 8035.50 | 8107.64 | 8101.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 8035.50 | 8107.64 | 8101.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 8005.50 | 8087.21 | 8092.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 15:15:00 | 7967.50 | 8063.27 | 8081.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 7890.00 | 7884.03 | 7961.36 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:15:00 | 7815.50 | 7884.03 | 7961.36 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 8025.00 | 7825.22 | 7873.38 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-17 09:15:00 | 8025.00 | 7825.22 | 7873.38 | SL hit (close>ema400) qty=1.00 sl=7873.38 alert=retest1 |

### Cycle 14 — BUY (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 12:15:00 | 7979.50 | 7911.36 | 7905.83 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 10:15:00 | 7819.00 | 7904.54 | 7909.08 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2025-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 15:15:00 | 7987.50 | 7915.34 | 7909.73 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-06-19 10:15:00)

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

### Cycle 18 — BUY (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 09:15:00 | 7865.00 | 7843.07 | 7842.21 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 12:15:00 | 7835.50 | 7841.02 | 7841.42 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-06-23 13:15:00)

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

### Cycle 21 — SELL (started 2025-07-04 12:15:00)

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

### Cycle 22 — BUY (started 2025-07-09 10:15:00)

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

### Cycle 23 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 8633.00 | 8752.87 | 8761.29 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2025-07-15 09:15:00)

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

### Cycle 25 — SELL (started 2025-07-17 13:15:00)

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

### Cycle 26 — BUY (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 09:15:00 | 9054.00 | 8908.36 | 8899.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 10:15:00 | 9193.50 | 8965.39 | 8926.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 14:15:00 | 9185.00 | 9217.02 | 9130.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 14:15:00 | 9185.00 | 9217.02 | 9130.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 9185.00 | 9217.02 | 9130.34 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2025-07-25 09:15:00)

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

### Cycle 28 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 9750.50 | 9021.12 | 8972.37 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-07-31 13:15:00)

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

### Cycle 30 — BUY (started 2025-08-12 15:15:00)

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

### Cycle 31 — SELL (started 2025-08-14 10:15:00)

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

### Cycle 32 — BUY (started 2025-09-02 12:15:00)

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

### Cycle 33 — SELL (started 2025-09-05 11:15:00)

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

### Cycle 34 — BUY (started 2025-09-09 09:15:00)

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

### Cycle 35 — SELL (started 2025-09-22 11:15:00)

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

### Cycle 36 — BUY (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 10:15:00 | 8796.00 | 8747.44 | 8744.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 13:15:00 | 8804.50 | 8769.87 | 8756.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 14:15:00 | 8716.00 | 8759.10 | 8752.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 14:15:00 | 8716.00 | 8759.10 | 8752.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 8716.00 | 8759.10 | 8752.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 8716.00 | 8759.10 | 8752.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2025-09-25 15:15:00)

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

### Cycle 38 — BUY (started 2025-10-03 12:15:00)

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

### Cycle 39 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 8349.00 | 8428.80 | 8435.79 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 8554.50 | 8450.34 | 8440.97 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-10-10 14:15:00)

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

### Cycle 42 — BUY (started 2025-10-13 15:15:00)

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

### Cycle 43 — SELL (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 13:15:00 | 8375.00 | 8412.03 | 8414.90 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2025-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 14:15:00 | 8531.00 | 8435.83 | 8425.46 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 11:15:00 | 8359.00 | 8410.85 | 8417.09 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2025-10-16 09:15:00)

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

### Cycle 47 — SELL (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 13:15:00 | 8565.00 | 8614.66 | 8619.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-21 14:15:00 | 8550.00 | 8594.93 | 8607.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 09:15:00 | 8636.00 | 8603.14 | 8609.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 8636.00 | 8603.14 | 8609.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 8636.00 | 8603.14 | 8609.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:15:00 | 8746.00 | 8603.14 | 8609.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2025-10-23 10:15:00)

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

### Cycle 49 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 8792.50 | 9117.14 | 9158.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 13:15:00 | 8647.50 | 8928.26 | 9053.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 09:15:00 | 8425.00 | 8409.63 | 8552.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 09:30:00 | 8460.50 | 8409.63 | 8552.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 8220.00 | 8280.51 | 8338.07 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 8567.50 | 8396.35 | 8373.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 8927.00 | 8529.31 | 8439.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 10:15:00 | 8830.50 | 8863.99 | 8708.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 11:00:00 | 8830.50 | 8863.99 | 8708.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 9179.50 | 9291.76 | 9249.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 9179.50 | 9291.76 | 9249.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 9252.50 | 9283.90 | 9249.49 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2025-11-21 09:15:00)

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

### Cycle 52 — BUY (started 2025-11-26 12:15:00)

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

### Cycle 53 — SELL (started 2025-12-01 11:15:00)

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

### Cycle 54 — BUY (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 15:15:00 | 8821.00 | 8753.58 | 8752.28 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 09:15:00 | 8727.00 | 8748.27 | 8749.98 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-12-11 14:15:00)

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

### Cycle 57 — SELL (started 2025-12-16 15:15:00)

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

### Cycle 58 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 8928.50 | 8786.13 | 8768.54 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2025-12-24 11:15:00)

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

### Cycle 60 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 7218.00 | 7143.74 | 7138.68 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2026-01-23 11:15:00)

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

### Cycle 62 — BUY (started 2026-01-27 11:15:00)

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

### Cycle 63 — SELL (started 2026-02-12 11:15:00)

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

### Cycle 64 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 9536.00 | 9483.56 | 9479.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 9859.00 | 9586.82 | 9533.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 13:15:00 | 10055.00 | 10058.94 | 9910.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 14:00:00 | 10055.00 | 10058.94 | 9910.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 10109.00 | 10064.27 | 9950.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 11:15:00 | 10299.50 | 10073.81 | 9964.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-27 12:15:00 | 11329.45 | 10980.96 | 10790.07 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2026-03-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 15:15:00 | 10688.00 | 10813.95 | 10829.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 10167.00 | 10684.56 | 10769.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 10380.00 | 10272.48 | 10465.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:00:00 | 10380.00 | 10272.48 | 10465.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 10432.00 | 10304.38 | 10462.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 11:00:00 | 10432.00 | 10304.38 | 10462.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 10305.00 | 10304.50 | 10447.96 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 10630.00 | 10484.78 | 10474.54 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2026-03-06 15:15:00)

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

### Cycle 68 — BUY (started 2026-03-17 12:15:00)

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

### Cycle 69 — SELL (started 2026-03-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 11:15:00 | 9425.00 | 9541.25 | 9556.38 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2026-03-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-23 14:15:00 | 9822.00 | 9581.36 | 9568.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 12:15:00 | 9850.00 | 9713.15 | 9644.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 9960.00 | 9966.84 | 9854.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:15:00 | 10011.00 | 9966.84 | 9854.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 9891.00 | 9951.67 | 9857.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 11:15:00 | 10142.00 | 9973.54 | 9876.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 15:15:00 | 9782.00 | 10002.59 | 10020.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2026-03-30 15:15:00)

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

### Cycle 72 — BUY (started 2026-04-06 13:15:00)

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

### Cycle 73 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 11765.00 | 11844.44 | 11850.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 11615.00 | 11798.55 | 11829.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 11701.00 | 11690.55 | 11760.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 11701.00 | 11690.55 | 11760.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 11701.00 | 11690.55 | 11760.02 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2026-04-27 13:15:00)

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
| BUY | retest2 | 2025-05-12 09:15:00 | 6324.50 | 2025-05-14 13:15:00 | 6766.10 | TARGET_HIT | 1.00 | 6.98% |
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
