# Atul Ltd. (ATUL)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 7090.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 169 |
| ALERT1 | 103 |
| ALERT2 | 100 |
| ALERT2_SKIP | 45 |
| ALERT3 | 278 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 131 |
| PARTIAL | 4 |
| TARGET_HIT | 6 |
| STOP_HIT | 129 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 139 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 49 / 90
- **Target hits / Stop hits / Partials:** 6 / 129 / 4
- **Avg / median % per leg:** -0.10% / -0.86%
- **Sum % (uncompounded):** -13.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 71 | 29 | 40.8% | 3 | 68 | 0 | 0.36% | 25.4% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 5 | 0 | -1.06% | -5.3% |
| BUY @ 3rd Alert (retest2) | 66 | 28 | 42.4% | 3 | 63 | 0 | 0.47% | 30.7% |
| SELL (all) | 68 | 20 | 29.4% | 3 | 61 | 4 | -0.57% | -39.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 68 | 20 | 29.4% | 3 | 61 | 4 | -0.57% | -39.0% |
| retest1 (combined) | 5 | 1 | 20.0% | 0 | 5 | 0 | -1.06% | -5.3% |
| retest2 (combined) | 134 | 48 | 35.8% | 6 | 124 | 4 | -0.06% | -8.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 12:15:00 | 5964.80 | 5916.23 | 5909.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 13:15:00 | 5970.55 | 5927.10 | 5915.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-18 12:15:00 | 5965.00 | 5968.82 | 5944.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 09:15:00 | 5950.70 | 5968.82 | 5944.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 5924.90 | 5960.04 | 5942.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:45:00 | 5915.45 | 5960.04 | 5942.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 5930.00 | 5954.03 | 5941.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:30:00 | 5910.70 | 5954.03 | 5941.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 5960.00 | 5963.16 | 5953.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:00:00 | 5960.00 | 5963.16 | 5953.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 5959.95 | 5962.52 | 5954.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:45:00 | 5941.20 | 5962.52 | 5954.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 5963.00 | 5962.62 | 5955.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:45:00 | 5941.00 | 5956.29 | 5952.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 10:15:00 | 5925.25 | 5950.08 | 5950.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 14:15:00 | 5881.65 | 5914.57 | 5928.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 09:15:00 | 5910.10 | 5907.99 | 5922.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 09:15:00 | 5910.10 | 5907.99 | 5922.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 5910.10 | 5907.99 | 5922.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:00:00 | 5910.10 | 5907.99 | 5922.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 5905.45 | 5907.48 | 5921.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 11:45:00 | 5898.55 | 5905.97 | 5919.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 14:30:00 | 5900.90 | 5902.75 | 5914.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 09:45:00 | 5901.40 | 5896.52 | 5909.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 5603.62 | 5718.16 | 5727.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 5605.85 | 5718.16 | 5727.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 5606.33 | 5718.16 | 5727.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 12:15:00 | 5308.70 | 5608.32 | 5669.62 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 5792.80 | 5675.43 | 5667.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 14:15:00 | 5816.95 | 5721.13 | 5690.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-18 09:15:00 | 6224.05 | 6224.29 | 6183.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-18 10:00:00 | 6224.05 | 6224.29 | 6183.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 6200.00 | 6243.02 | 6217.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:00:00 | 6200.00 | 6243.02 | 6217.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 6246.75 | 6243.76 | 6219.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 11:30:00 | 6285.30 | 6256.39 | 6227.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 09:15:00 | 6263.45 | 6251.37 | 6235.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 12:15:00 | 6348.05 | 6406.83 | 6413.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 12:15:00 | 6348.05 | 6406.83 | 6413.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 6304.80 | 6362.41 | 6381.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 09:15:00 | 6440.15 | 6370.75 | 6380.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 09:15:00 | 6440.15 | 6370.75 | 6380.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 6440.15 | 6370.75 | 6380.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:45:00 | 6428.30 | 6370.75 | 6380.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 10:15:00 | 6524.70 | 6401.54 | 6393.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 6573.60 | 6470.52 | 6433.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 6577.95 | 6590.57 | 6532.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 12:00:00 | 6577.95 | 6590.57 | 6532.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 6558.30 | 6584.11 | 6534.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:45:00 | 6545.95 | 6584.11 | 6534.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 6528.70 | 6578.60 | 6565.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 11:15:00 | 6612.95 | 6572.72 | 6563.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 13:15:00 | 6993.30 | 7027.84 | 7029.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 13:15:00 | 6993.30 | 7027.84 | 7029.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 14:15:00 | 6953.40 | 7012.95 | 7022.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 7032.45 | 6994.75 | 7011.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 7032.45 | 6994.75 | 7011.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 7032.45 | 6994.75 | 7011.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 7032.45 | 6994.75 | 7011.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 7043.55 | 7004.51 | 7014.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 7039.95 | 7004.51 | 7014.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 11:15:00 | 7185.65 | 7040.74 | 7029.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 12:15:00 | 7399.20 | 7112.43 | 7063.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 09:15:00 | 7177.70 | 7188.59 | 7121.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-23 10:00:00 | 7177.70 | 7188.59 | 7121.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 7226.65 | 7202.03 | 7139.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:15:00 | 7042.30 | 7202.03 | 7139.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 7238.15 | 7209.25 | 7148.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 7205.00 | 7209.25 | 7148.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 7283.95 | 7279.71 | 7237.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 14:15:00 | 7309.90 | 7270.16 | 7244.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 14:45:00 | 7309.95 | 7280.16 | 7251.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 7308.80 | 7278.53 | 7253.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 15:15:00 | 7793.85 | 7828.56 | 7832.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-08-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 15:15:00 | 7793.85 | 7828.56 | 7832.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 10:15:00 | 7684.65 | 7794.45 | 7816.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 7896.25 | 7772.12 | 7788.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 7896.25 | 7772.12 | 7788.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 7896.25 | 7772.12 | 7788.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 7896.25 | 7772.12 | 7788.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 7814.75 | 7780.64 | 7790.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:30:00 | 7796.85 | 7784.87 | 7790.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 10:15:00 | 7871.35 | 7808.63 | 7800.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2024-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 10:15:00 | 7871.35 | 7808.63 | 7800.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 12:15:00 | 7947.05 | 7844.71 | 7818.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 8000.00 | 8013.17 | 7946.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 15:00:00 | 8000.00 | 8013.17 | 7946.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 12:15:00 | 7959.95 | 8006.26 | 7969.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 13:00:00 | 7959.95 | 8006.26 | 7969.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 7965.00 | 7998.01 | 7969.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 14:00:00 | 7965.00 | 7998.01 | 7969.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 8006.10 | 7998.83 | 7974.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:15:00 | 8052.95 | 7998.83 | 7974.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 8024.50 | 8003.97 | 7978.96 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2024-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 12:15:00 | 7907.90 | 7962.99 | 7964.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 14:15:00 | 7855.50 | 7932.69 | 7950.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 11:15:00 | 7710.20 | 7700.04 | 7775.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 12:00:00 | 7710.20 | 7700.04 | 7775.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 7774.55 | 7714.94 | 7775.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 13:00:00 | 7774.55 | 7714.94 | 7775.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 13:15:00 | 7744.55 | 7720.86 | 7772.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 14:15:00 | 7730.50 | 7720.86 | 7772.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 09:30:00 | 7716.10 | 7725.25 | 7761.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 10:00:00 | 7729.95 | 7725.25 | 7761.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 12:15:00 | 7854.60 | 7762.41 | 7770.63 | SL hit (close>static) qty=1.00 sl=7782.40 alert=retest2 |

### Cycle 11 — BUY (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 13:15:00 | 7888.55 | 7787.64 | 7781.35 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 10:15:00 | 7772.95 | 7800.70 | 7801.02 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 12:15:00 | 7830.10 | 7805.84 | 7803.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 13:15:00 | 7860.55 | 7816.78 | 7808.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 11:15:00 | 7945.00 | 7971.51 | 7921.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 11:15:00 | 7945.00 | 7971.51 | 7921.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 11:15:00 | 7945.00 | 7971.51 | 7921.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 11:45:00 | 7930.00 | 7971.51 | 7921.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 7937.80 | 7958.12 | 7923.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:45:00 | 7936.35 | 7958.12 | 7923.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 7923.55 | 7951.21 | 7923.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 15:00:00 | 7923.55 | 7951.21 | 7923.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 7902.85 | 7941.54 | 7921.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 7855.00 | 7941.54 | 7921.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 7781.05 | 7909.44 | 7908.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:30:00 | 7768.60 | 7909.44 | 7908.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2024-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 10:15:00 | 7762.15 | 7879.98 | 7895.45 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 15:15:00 | 7925.00 | 7875.00 | 7869.82 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 09:15:00 | 7838.20 | 7885.95 | 7886.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 10:15:00 | 7815.25 | 7871.81 | 7879.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 15:15:00 | 7848.00 | 7835.22 | 7854.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 15:15:00 | 7848.00 | 7835.22 | 7854.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 7848.00 | 7835.22 | 7854.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 7890.65 | 7835.22 | 7854.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 7968.70 | 7861.92 | 7865.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:00:00 | 7968.70 | 7861.92 | 7865.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 10:15:00 | 7937.60 | 7877.06 | 7871.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 11:15:00 | 7970.90 | 7895.82 | 7880.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 10:15:00 | 7930.00 | 7938.91 | 7913.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 11:15:00 | 7935.00 | 7938.91 | 7913.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 7883.00 | 7927.73 | 7910.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:30:00 | 7883.00 | 7927.73 | 7910.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 7861.45 | 7914.47 | 7905.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 13:00:00 | 7861.45 | 7914.47 | 7905.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 15:15:00 | 7895.00 | 7900.33 | 7900.63 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 09:15:00 | 7965.80 | 7913.42 | 7906.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 10:15:00 | 7994.20 | 7929.58 | 7914.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 13:15:00 | 7943.75 | 7945.80 | 7927.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 13:15:00 | 7943.75 | 7945.80 | 7927.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 13:15:00 | 7943.75 | 7945.80 | 7927.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 14:00:00 | 7943.75 | 7945.80 | 7927.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 7909.00 | 7938.44 | 7925.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 7909.00 | 7938.44 | 7925.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 7935.00 | 7937.75 | 7926.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 7964.25 | 7937.75 | 7926.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 7983.70 | 7946.94 | 7931.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 09:30:00 | 8028.05 | 7944.69 | 7937.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 14:45:00 | 8017.80 | 7949.40 | 7941.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 09:15:00 | 7817.85 | 7925.68 | 7932.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 7817.85 | 7925.68 | 7932.42 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 11:15:00 | 7997.30 | 7922.12 | 7919.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 09:15:00 | 8070.00 | 7973.31 | 7946.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 13:15:00 | 7999.95 | 8004.33 | 7972.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-10 14:15:00 | 7992.10 | 8004.33 | 7972.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 7999.20 | 8003.31 | 7975.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 15:00:00 | 7999.20 | 8003.31 | 7975.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 8019.80 | 8006.08 | 7981.42 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 7866.10 | 7957.79 | 7967.72 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 15:15:00 | 7992.85 | 7966.90 | 7964.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 11:15:00 | 8016.00 | 7979.89 | 7971.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 14:15:00 | 7965.00 | 7980.35 | 7974.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 14:15:00 | 7965.00 | 7980.35 | 7974.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 7965.00 | 7980.35 | 7974.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 15:00:00 | 7965.00 | 7980.35 | 7974.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 7967.95 | 7977.87 | 7973.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 09:15:00 | 7910.05 | 7977.87 | 7973.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 09:15:00 | 7896.00 | 7961.50 | 7966.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 11:15:00 | 7893.45 | 7938.70 | 7954.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 7742.10 | 7710.33 | 7776.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 7742.10 | 7710.33 | 7776.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 7742.10 | 7710.33 | 7776.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:30:00 | 7844.00 | 7710.33 | 7776.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 7608.00 | 7659.01 | 7712.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:45:00 | 7597.85 | 7653.78 | 7693.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 14:30:00 | 7599.65 | 7658.01 | 7691.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 11:15:00 | 7598.35 | 7642.27 | 7663.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 11:15:00 | 7587.45 | 7549.75 | 7547.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 11:15:00 | 7587.45 | 7549.75 | 7547.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 13:15:00 | 7621.45 | 7569.70 | 7557.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 15:15:00 | 7650.00 | 7674.81 | 7636.91 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 09:15:00 | 7709.45 | 7674.81 | 7636.91 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 12:15:00 | 7847.50 | 7906.21 | 7838.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:00:00 | 7847.50 | 7906.21 | 7838.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 7839.00 | 7892.77 | 7838.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:45:00 | 7820.05 | 7892.77 | 7838.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 7822.95 | 7878.80 | 7836.76 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-03 14:15:00 | 7822.95 | 7878.80 | 7836.76 | SL hit (close<ema400) qty=1.00 sl=7836.76 alert=retest1 |

### Cycle 26 — SELL (started 2024-10-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 14:15:00 | 7683.45 | 7811.83 | 7824.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 7634.05 | 7755.46 | 7794.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 7593.55 | 7581.54 | 7666.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 09:45:00 | 7612.00 | 7581.54 | 7666.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 7607.05 | 7596.64 | 7659.13 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 7815.00 | 7710.39 | 7698.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 7963.65 | 7761.04 | 7722.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 10:15:00 | 7907.70 | 7913.13 | 7838.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 11:00:00 | 7907.70 | 7913.13 | 7838.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 7910.55 | 7935.40 | 7904.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 15:00:00 | 7910.55 | 7935.40 | 7904.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 7866.10 | 7921.54 | 7900.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:15:00 | 7867.50 | 7921.54 | 7900.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 7882.15 | 7913.66 | 7899.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 09:15:00 | 7960.45 | 7914.74 | 7903.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 11:15:00 | 7706.50 | 7884.98 | 7894.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 11:15:00 | 7706.50 | 7884.98 | 7894.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 7585.00 | 7703.73 | 7753.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 7542.95 | 7516.63 | 7597.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 09:45:00 | 7498.80 | 7516.63 | 7597.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 7621.35 | 7537.57 | 7599.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:00:00 | 7621.35 | 7537.57 | 7599.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 7659.25 | 7561.91 | 7604.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:00:00 | 7659.25 | 7561.91 | 7604.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 7646.10 | 7606.51 | 7614.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:00:00 | 7646.10 | 7606.51 | 7614.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 7652.20 | 7615.65 | 7617.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:30:00 | 7720.00 | 7615.65 | 7617.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2024-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 12:15:00 | 7626.70 | 7620.03 | 7619.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-24 15:15:00 | 7680.00 | 7636.18 | 7627.20 | Break + close above crossover candle high |

### Cycle 30 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 7525.00 | 7613.94 | 7617.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 10:15:00 | 7498.00 | 7590.75 | 7607.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 09:15:00 | 7556.70 | 7489.30 | 7537.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-28 09:15:00 | 7556.70 | 7489.30 | 7537.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 7556.70 | 7489.30 | 7537.59 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 13:15:00 | 7637.95 | 7567.33 | 7563.31 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 11:15:00 | 7493.00 | 7554.86 | 7562.41 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 7649.90 | 7562.72 | 7560.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 12:15:00 | 7679.95 | 7609.10 | 7584.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 7760.75 | 7797.59 | 7734.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 10:00:00 | 7760.75 | 7797.59 | 7734.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 7717.15 | 7781.50 | 7732.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 7717.15 | 7781.50 | 7732.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 7734.90 | 7772.18 | 7732.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 15:15:00 | 7786.15 | 7768.91 | 7740.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 09:15:00 | 7725.90 | 7906.80 | 7926.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 7725.90 | 7906.80 | 7926.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 11:15:00 | 7674.85 | 7843.32 | 7893.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 7366.95 | 7330.19 | 7455.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 7366.95 | 7330.19 | 7455.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 15:15:00 | 7214.60 | 7243.99 | 7313.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 7361.05 | 7267.40 | 7317.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 7348.95 | 7283.71 | 7320.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 7320.40 | 7283.71 | 7320.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 7340.60 | 7295.09 | 7322.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:45:00 | 7346.20 | 7295.09 | 7322.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 12:15:00 | 7359.45 | 7307.96 | 7325.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 13:00:00 | 7359.45 | 7307.96 | 7325.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 7330.40 | 7312.45 | 7326.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:15:00 | 7324.15 | 7312.45 | 7326.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:45:00 | 7282.85 | 7305.41 | 7321.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 10:15:00 | 7411.30 | 7303.76 | 7292.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 7411.30 | 7303.76 | 7292.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 14:15:00 | 7463.25 | 7379.95 | 7336.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 12:15:00 | 7396.55 | 7414.17 | 7373.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 13:00:00 | 7396.55 | 7414.17 | 7373.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 13:15:00 | 7384.95 | 7408.32 | 7374.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 13:45:00 | 7394.05 | 7408.32 | 7374.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 7377.15 | 7402.09 | 7374.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 14:30:00 | 7390.00 | 7402.09 | 7374.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 7432.00 | 7408.07 | 7379.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:15:00 | 7348.70 | 7408.07 | 7379.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 7395.95 | 7405.65 | 7381.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:30:00 | 7375.00 | 7405.65 | 7381.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 7362.00 | 7396.92 | 7379.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:45:00 | 7373.00 | 7396.92 | 7379.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 7355.60 | 7388.65 | 7377.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:45:00 | 7341.95 | 7388.65 | 7377.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 15:15:00 | 7395.00 | 7376.93 | 7373.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 09:30:00 | 7402.35 | 7377.44 | 7374.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 10:15:00 | 7303.70 | 7362.69 | 7368.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 10:15:00 | 7303.70 | 7362.69 | 7368.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 11:15:00 | 7300.00 | 7350.16 | 7361.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 13:15:00 | 7290.65 | 7280.40 | 7306.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-29 14:00:00 | 7290.65 | 7280.40 | 7306.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 7291.60 | 7282.64 | 7304.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 15:00:00 | 7291.60 | 7282.64 | 7304.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 7315.05 | 7289.12 | 7305.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:15:00 | 7257.55 | 7289.12 | 7305.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 7308.00 | 7292.90 | 7305.89 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 11:15:00 | 7378.00 | 7314.48 | 7313.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 10:15:00 | 7400.00 | 7355.20 | 7337.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 11:15:00 | 7378.55 | 7388.07 | 7368.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 11:15:00 | 7378.55 | 7388.07 | 7368.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 7378.55 | 7388.07 | 7368.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:45:00 | 7350.40 | 7388.07 | 7368.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 7406.50 | 7391.76 | 7371.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 09:15:00 | 7415.70 | 7393.23 | 7377.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 09:15:00 | 7345.95 | 7383.77 | 7374.50 | SL hit (close<static) qty=1.00 sl=7368.80 alert=retest2 |

### Cycle 38 — SELL (started 2024-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 14:15:00 | 7355.75 | 7374.16 | 7375.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 15:15:00 | 7335.10 | 7366.34 | 7371.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 09:15:00 | 7398.60 | 7320.73 | 7333.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 09:15:00 | 7398.60 | 7320.73 | 7333.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 7398.60 | 7320.73 | 7333.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:45:00 | 7416.95 | 7320.73 | 7333.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2024-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 10:15:00 | 7431.30 | 7342.84 | 7342.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 11:15:00 | 7457.95 | 7365.86 | 7352.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 14:15:00 | 7485.05 | 7494.24 | 7449.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-11 14:30:00 | 7485.95 | 7494.24 | 7449.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 7307.60 | 7452.44 | 7438.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:00:00 | 7307.60 | 7452.44 | 7438.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 10:15:00 | 7245.00 | 7410.95 | 7420.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 7065.00 | 7261.94 | 7333.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 15:15:00 | 7230.00 | 7210.89 | 7269.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-16 09:15:00 | 7261.20 | 7210.89 | 7269.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 7316.85 | 7232.08 | 7273.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 7303.80 | 7232.08 | 7273.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 7305.85 | 7246.83 | 7276.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:00:00 | 7305.85 | 7246.83 | 7276.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2024-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 14:15:00 | 7335.85 | 7292.94 | 7291.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 10:15:00 | 7365.90 | 7325.05 | 7307.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 14:15:00 | 7326.35 | 7329.95 | 7316.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 15:00:00 | 7326.35 | 7329.95 | 7316.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 15:15:00 | 7358.45 | 7335.65 | 7319.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 09:30:00 | 7363.55 | 7340.11 | 7323.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 10:15:00 | 7360.00 | 7340.11 | 7323.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 09:30:00 | 7387.65 | 7371.74 | 7351.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 10:15:00 | 7377.70 | 7371.74 | 7351.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 10:15:00 | 7374.90 | 7372.37 | 7354.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 11:30:00 | 7417.05 | 7378.96 | 7358.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 15:00:00 | 7412.05 | 7387.02 | 7367.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 09:15:00 | 7297.00 | 7372.38 | 7364.56 | SL hit (close<static) qty=1.00 sl=7346.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 11:15:00 | 7277.30 | 7343.54 | 7352.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 12:15:00 | 7182.75 | 7311.38 | 7336.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 11:15:00 | 7073.60 | 7073.02 | 7136.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 12:00:00 | 7073.60 | 7073.02 | 7136.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 15:15:00 | 7100.00 | 7090.83 | 7126.24 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 15:15:00 | 6990.00 | 6944.28 | 6942.21 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 12:15:00 | 6903.60 | 6937.53 | 6940.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 13:15:00 | 6890.80 | 6928.18 | 6935.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-06 09:15:00 | 6917.45 | 6915.41 | 6927.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 6917.45 | 6915.41 | 6927.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 6917.45 | 6915.41 | 6927.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 10:30:00 | 6869.45 | 6902.48 | 6920.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-07 14:15:00 | 6920.00 | 6901.06 | 6898.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 14:15:00 | 6920.00 | 6901.06 | 6898.86 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 10:15:00 | 6869.00 | 6893.22 | 6895.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 11:15:00 | 6826.05 | 6879.79 | 6889.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 15:15:00 | 6874.00 | 6852.77 | 6870.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 15:15:00 | 6874.00 | 6852.77 | 6870.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 6874.00 | 6852.77 | 6870.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:15:00 | 6962.40 | 6852.77 | 6870.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 6950.10 | 6872.23 | 6877.71 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 11:15:00 | 6920.00 | 6881.69 | 6881.06 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 6787.20 | 6881.62 | 6884.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 6711.10 | 6806.56 | 6840.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 11:15:00 | 6701.40 | 6684.65 | 6738.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:30:00 | 6691.40 | 6684.65 | 6738.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 6754.90 | 6699.73 | 6736.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:00:00 | 6754.90 | 6699.73 | 6736.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 6737.60 | 6707.31 | 6736.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 15:15:00 | 6756.00 | 6707.31 | 6736.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 6756.00 | 6717.05 | 6738.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 6711.80 | 6717.05 | 6738.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 6703.20 | 6714.28 | 6735.08 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 6783.75 | 6742.81 | 6741.74 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 10:15:00 | 6708.15 | 6744.90 | 6747.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 12:15:00 | 6680.35 | 6725.95 | 6737.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 10:15:00 | 6699.00 | 6696.58 | 6716.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-20 11:00:00 | 6699.00 | 6696.58 | 6716.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 11:15:00 | 6707.90 | 6698.84 | 6716.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 11:45:00 | 6713.00 | 6698.84 | 6716.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 12:15:00 | 6731.85 | 6705.44 | 6717.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 13:00:00 | 6731.85 | 6705.44 | 6717.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 6713.90 | 6707.13 | 6717.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 13:45:00 | 6713.45 | 6707.13 | 6717.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 6730.60 | 6711.83 | 6718.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:45:00 | 6749.95 | 6711.83 | 6718.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 15:15:00 | 6732.50 | 6715.96 | 6719.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:15:00 | 6829.65 | 6715.96 | 6719.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 09:15:00 | 6784.70 | 6729.71 | 6725.63 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 6671.25 | 6726.94 | 6729.53 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 14:15:00 | 6793.60 | 6734.75 | 6730.69 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 09:15:00 | 6694.70 | 6733.48 | 6734.65 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 10:15:00 | 6769.90 | 6740.77 | 6737.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-24 12:15:00 | 6783.30 | 6747.07 | 6741.09 | Break + close above crossover candle high |

### Cycle 56 — SELL (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 14:15:00 | 6519.05 | 6708.66 | 6725.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 6274.00 | 6597.94 | 6670.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 12:15:00 | 6250.50 | 6250.32 | 6334.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 12:30:00 | 6255.65 | 6250.32 | 6334.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 6368.20 | 6278.99 | 6321.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 10:00:00 | 6368.20 | 6278.99 | 6321.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 6317.25 | 6286.65 | 6321.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 11:15:00 | 6301.45 | 6286.65 | 6321.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 09:45:00 | 6272.00 | 6304.18 | 6318.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 10:00:00 | 6305.30 | 6291.29 | 6301.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 11:45:00 | 6197.05 | 6284.04 | 6297.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 6289.95 | 6251.10 | 6271.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:30:00 | 6295.40 | 6251.10 | 6271.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 6268.10 | 6254.50 | 6271.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 10:45:00 | 6290.90 | 6254.50 | 6271.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 11:15:00 | 6231.00 | 6249.80 | 6267.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 12:15:00 | 6201.25 | 6249.80 | 6267.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 09:15:00 | 6189.00 | 6202.58 | 6216.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-05 12:15:00 | 6240.25 | 6224.25 | 6223.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 12:15:00 | 6240.25 | 6224.25 | 6223.30 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 13:15:00 | 6215.50 | 6222.50 | 6222.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 10:15:00 | 6181.10 | 6207.30 | 6214.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 14:15:00 | 6185.20 | 6183.94 | 6199.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 14:15:00 | 6185.20 | 6183.94 | 6199.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 6185.20 | 6183.94 | 6199.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:00:00 | 6185.20 | 6183.94 | 6199.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 6179.80 | 6172.37 | 6189.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:00:00 | 6179.80 | 6172.37 | 6189.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 6166.35 | 6171.16 | 6187.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:45:00 | 6185.30 | 6171.16 | 6187.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 5515.75 | 5407.59 | 5439.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 5515.75 | 5407.59 | 5439.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 5541.60 | 5434.39 | 5448.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 5562.75 | 5434.39 | 5448.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 5583.00 | 5464.11 | 5460.98 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 15:15:00 | 5500.00 | 5515.06 | 5515.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 5310.00 | 5474.05 | 5496.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 14:15:00 | 5450.00 | 5443.52 | 5469.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-24 14:45:00 | 5448.25 | 5443.52 | 5469.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 5347.00 | 5422.05 | 5455.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 11:30:00 | 5293.80 | 5378.99 | 5429.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 10:00:00 | 5298.50 | 5308.20 | 5369.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 15:15:00 | 5270.00 | 5324.46 | 5330.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:45:00 | 5296.80 | 5308.35 | 5321.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 10:15:00 | 5300.00 | 5306.68 | 5319.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 10:30:00 | 5307.70 | 5306.68 | 5319.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-03 12:15:00 | 5597.95 | 5364.51 | 5343.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-03-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 12:15:00 | 5597.95 | 5364.51 | 5343.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 13:15:00 | 5610.00 | 5541.00 | 5487.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 14:15:00 | 5637.00 | 5639.89 | 5580.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 15:00:00 | 5637.00 | 5639.89 | 5580.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 5617.15 | 5635.35 | 5583.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:15:00 | 5635.00 | 5635.35 | 5583.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 5706.45 | 5649.57 | 5595.03 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2025-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 10:15:00 | 5554.85 | 5619.51 | 5627.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 11:15:00 | 5510.05 | 5569.12 | 5594.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 15:15:00 | 5548.00 | 5542.19 | 5571.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-13 09:15:00 | 5502.00 | 5542.19 | 5571.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 5546.00 | 5542.95 | 5568.95 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 10:15:00 | 5670.45 | 5563.83 | 5561.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 11:15:00 | 5777.50 | 5606.57 | 5581.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-18 09:15:00 | 5665.70 | 5710.09 | 5651.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-18 09:30:00 | 5664.30 | 5710.09 | 5651.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 5677.70 | 5703.61 | 5653.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:30:00 | 5662.65 | 5703.61 | 5653.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 5703.75 | 5692.37 | 5667.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 10:15:00 | 5744.55 | 5672.64 | 5668.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 10:45:00 | 5793.20 | 5697.33 | 5679.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 10:15:00 | 5770.75 | 5819.94 | 5826.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 10:15:00 | 5770.75 | 5819.94 | 5826.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 15:15:00 | 5760.00 | 5792.12 | 5809.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 5790.00 | 5787.73 | 5803.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 10:30:00 | 5787.30 | 5787.73 | 5803.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 5854.80 | 5801.14 | 5808.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:45:00 | 5829.70 | 5801.14 | 5808.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 5860.00 | 5812.91 | 5813.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:00:00 | 5860.00 | 5812.91 | 5813.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 13:15:00 | 5852.00 | 5820.73 | 5816.74 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 15:15:00 | 5755.50 | 5811.49 | 5813.47 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 5844.00 | 5817.99 | 5816.24 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 11:15:00 | 5776.25 | 5810.76 | 5813.33 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 13:15:00 | 5950.00 | 5834.79 | 5823.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 14:15:00 | 6051.00 | 5878.03 | 5844.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 5911.45 | 5917.03 | 5869.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-01 10:00:00 | 5911.45 | 5917.03 | 5869.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 5880.40 | 5907.80 | 5873.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:30:00 | 5871.65 | 5907.80 | 5873.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 12:15:00 | 5889.85 | 5904.21 | 5875.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 12:45:00 | 5875.00 | 5904.21 | 5875.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 5805.05 | 5884.38 | 5868.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 13:45:00 | 5810.90 | 5884.38 | 5868.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 14:15:00 | 5817.65 | 5871.03 | 5864.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 14:45:00 | 5809.00 | 5871.03 | 5864.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 15:15:00 | 5800.00 | 5856.83 | 5858.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-03 09:15:00 | 5731.00 | 5812.06 | 5831.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 5218.95 | 5209.36 | 5348.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 5218.95 | 5209.36 | 5348.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 5218.95 | 5209.36 | 5348.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 5193.90 | 5206.98 | 5334.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:00:00 | 5197.45 | 5206.98 | 5334.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 13:15:00 | 5185.75 | 5210.10 | 5314.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 11:00:00 | 5195.75 | 5206.70 | 5271.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 5359.20 | 5237.61 | 5257.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 10:00:00 | 5359.20 | 5237.61 | 5257.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 10:15:00 | 5365.90 | 5263.27 | 5267.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 11:00:00 | 5365.90 | 5263.27 | 5267.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-11 11:15:00 | 5425.40 | 5295.70 | 5282.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 5425.40 | 5295.70 | 5282.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 12:15:00 | 5660.30 | 5368.62 | 5316.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 14:15:00 | 5621.00 | 5658.83 | 5546.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-15 15:00:00 | 5621.00 | 5658.83 | 5546.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 5681.50 | 5697.76 | 5643.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:15:00 | 5685.00 | 5697.76 | 5643.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 14:30:00 | 5688.00 | 5684.84 | 5653.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:15:00 | 5740.00 | 5684.48 | 5656.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-23 09:15:00 | 6253.50 | 6025.43 | 5925.29 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 6814.50 | 6863.92 | 6863.93 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 10:15:00 | 6868.00 | 6864.74 | 6864.30 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 11:15:00 | 6840.50 | 6859.89 | 6862.14 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 15:15:00 | 6892.00 | 6866.70 | 6864.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 6908.00 | 6874.96 | 6868.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 10:15:00 | 6845.50 | 6869.07 | 6866.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 10:15:00 | 6845.50 | 6869.07 | 6866.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 6845.50 | 6869.07 | 6866.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:45:00 | 6835.00 | 6869.07 | 6866.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 11:15:00 | 6845.00 | 6864.25 | 6864.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 11:30:00 | 6822.00 | 6864.25 | 6864.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 12:15:00 | 6813.00 | 6854.00 | 6859.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 6772.00 | 6837.60 | 6851.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-08 14:15:00 | 6845.50 | 6839.18 | 6851.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 14:15:00 | 6845.50 | 6839.18 | 6851.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 6845.50 | 6839.18 | 6851.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 6845.50 | 6839.18 | 6851.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 6761.00 | 6823.55 | 6842.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 09:15:00 | 6685.50 | 6823.55 | 6842.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 10:15:00 | 6759.00 | 6817.24 | 6838.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 12:15:00 | 6760.00 | 6800.29 | 6826.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 13:00:00 | 6760.00 | 6792.23 | 6820.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 6810.00 | 6785.33 | 6806.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 11:15:00 | 6786.00 | 6789.37 | 6806.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-13 09:15:00 | 6771.50 | 6800.86 | 6807.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-13 14:15:00 | 6875.00 | 6818.23 | 6812.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2025-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 14:15:00 | 6875.00 | 6818.23 | 6812.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 15:15:00 | 6885.00 | 6831.58 | 6819.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 11:15:00 | 6832.00 | 6840.57 | 6827.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 11:15:00 | 6832.00 | 6840.57 | 6827.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 6832.00 | 6840.57 | 6827.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:45:00 | 6834.00 | 6840.57 | 6827.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 6814.00 | 6835.26 | 6826.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 6807.50 | 6835.26 | 6826.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 6808.50 | 6829.91 | 6824.53 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2025-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 15:15:00 | 6781.00 | 6815.34 | 6818.55 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 11:15:00 | 6835.00 | 6820.12 | 6820.00 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-05-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 15:15:00 | 6801.00 | 6818.49 | 6819.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 09:15:00 | 6776.00 | 6809.99 | 6815.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 12:15:00 | 6805.00 | 6804.02 | 6811.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 12:15:00 | 6805.00 | 6804.02 | 6811.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 12:15:00 | 6805.00 | 6804.02 | 6811.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 13:00:00 | 6805.00 | 6804.02 | 6811.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 13:15:00 | 6809.00 | 6805.02 | 6810.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 14:00:00 | 6809.00 | 6805.02 | 6810.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 6782.50 | 6800.51 | 6808.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-19 09:15:00 | 6760.00 | 6798.41 | 6806.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 09:15:00 | 6831.00 | 6804.93 | 6808.81 | SL hit (close>static) qty=1.00 sl=6812.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 6851.00 | 6814.14 | 6812.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 6915.50 | 6845.73 | 6830.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-21 13:15:00 | 6940.00 | 6941.71 | 6904.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-21 14:00:00 | 6940.00 | 6941.71 | 6904.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 6947.50 | 6944.74 | 6915.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 11:45:00 | 6980.00 | 6957.44 | 6926.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 12:15:00 | 7074.50 | 7104.18 | 7105.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 12:15:00 | 7074.50 | 7104.18 | 7105.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 13:15:00 | 6983.00 | 7079.94 | 7094.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 09:15:00 | 7100.00 | 7065.95 | 7082.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 09:15:00 | 7100.00 | 7065.95 | 7082.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 7100.00 | 7065.95 | 7082.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:30:00 | 7104.00 | 7065.95 | 7082.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 7095.00 | 7071.76 | 7083.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:30:00 | 7114.00 | 7071.76 | 7083.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 7112.50 | 7079.91 | 7086.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:45:00 | 7123.00 | 7079.91 | 7086.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 7100.50 | 7084.03 | 7087.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 13:30:00 | 7078.50 | 7088.32 | 7089.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 14:15:00 | 7137.50 | 7098.16 | 7093.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 14:15:00 | 7137.50 | 7098.16 | 7093.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 09:15:00 | 7147.00 | 7112.70 | 7101.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 14:15:00 | 7089.50 | 7124.23 | 7113.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 14:15:00 | 7089.50 | 7124.23 | 7113.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 7089.50 | 7124.23 | 7113.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 7089.50 | 7124.23 | 7113.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 7071.00 | 7113.58 | 7109.84 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 7061.50 | 7103.17 | 7105.45 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 13:15:00 | 7135.00 | 7107.54 | 7106.05 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 10:15:00 | 7099.50 | 7105.71 | 7106.18 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 11:15:00 | 7114.50 | 7107.47 | 7106.94 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 13:15:00 | 7091.00 | 7106.74 | 7106.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 14:15:00 | 7070.00 | 7099.39 | 7103.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 09:15:00 | 7099.00 | 7092.21 | 7099.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 09:15:00 | 7099.00 | 7092.21 | 7099.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 7099.00 | 7092.21 | 7099.09 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 09:15:00 | 7140.50 | 7100.34 | 7098.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 10:15:00 | 7194.00 | 7119.07 | 7107.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 7227.00 | 7238.47 | 7205.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 11:00:00 | 7227.00 | 7238.47 | 7205.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 7205.00 | 7231.78 | 7205.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:00:00 | 7205.00 | 7231.78 | 7205.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 7211.50 | 7227.72 | 7205.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:30:00 | 7203.50 | 7227.72 | 7205.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 13:15:00 | 7332.50 | 7248.68 | 7217.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 14:30:00 | 7361.50 | 7270.44 | 7229.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 7364.00 | 7278.75 | 7237.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 10:15:00 | 7352.50 | 7291.30 | 7246.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 12:15:00 | 7380.00 | 7302.87 | 7259.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 7347.00 | 7388.80 | 7342.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:45:00 | 7359.50 | 7388.80 | 7342.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 7349.00 | 7380.84 | 7343.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:30:00 | 7345.50 | 7380.84 | 7343.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 7333.00 | 7371.27 | 7342.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 7333.00 | 7371.27 | 7342.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 7326.00 | 7362.22 | 7340.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:30:00 | 7314.00 | 7347.37 | 7336.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-11 10:15:00 | 7233.50 | 7324.60 | 7326.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 10:15:00 | 7233.50 | 7324.60 | 7326.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 11:15:00 | 7204.00 | 7300.48 | 7315.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 7033.50 | 7032.86 | 7093.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 7033.50 | 7032.86 | 7093.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 7033.50 | 7032.86 | 7093.15 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 12:15:00 | 7263.00 | 7128.75 | 7125.51 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 09:15:00 | 7106.00 | 7164.34 | 7164.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 10:15:00 | 7063.00 | 7144.07 | 7155.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 15:15:00 | 7028.00 | 7022.22 | 7060.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 09:15:00 | 6991.50 | 7022.22 | 7060.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 7014.00 | 7020.58 | 7056.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 11:30:00 | 6969.50 | 7009.85 | 7045.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 13:15:00 | 6977.00 | 6943.86 | 6957.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 09:15:00 | 7182.00 | 7001.89 | 6980.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 7182.00 | 7001.89 | 6980.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 10:15:00 | 7226.00 | 7046.71 | 7002.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 10:15:00 | 7443.00 | 7446.81 | 7391.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 11:00:00 | 7443.00 | 7446.81 | 7391.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 7378.00 | 7433.05 | 7389.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:45:00 | 7374.00 | 7433.05 | 7389.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 7409.00 | 7428.24 | 7391.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 13:15:00 | 7410.00 | 7428.24 | 7391.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 14:00:00 | 7411.50 | 7424.89 | 7393.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 10:00:00 | 7411.00 | 7417.94 | 7397.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 10:45:00 | 7443.50 | 7424.85 | 7402.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 7664.00 | 7690.85 | 7640.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 09:15:00 | 7710.00 | 7690.85 | 7640.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 11:45:00 | 7736.50 | 7702.57 | 7658.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 7553.00 | 7647.92 | 7647.90 | SL hit (close<static) qty=1.00 sl=7602.50 alert=retest2 |

### Cycle 94 — SELL (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 10:15:00 | 7517.50 | 7621.83 | 7636.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 10:15:00 | 7479.50 | 7547.96 | 7586.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 13:15:00 | 7530.00 | 7527.31 | 7565.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 14:00:00 | 7530.00 | 7527.31 | 7565.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 7543.00 | 7520.18 | 7552.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 11:15:00 | 7410.50 | 7507.55 | 7543.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 15:15:00 | 7575.00 | 7510.70 | 7503.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 7575.00 | 7510.70 | 7503.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 7589.50 | 7526.46 | 7511.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 14:15:00 | 7473.00 | 7552.13 | 7534.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 14:15:00 | 7473.00 | 7552.13 | 7534.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 7473.00 | 7552.13 | 7534.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 15:00:00 | 7473.00 | 7552.13 | 7534.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 7423.50 | 7526.40 | 7524.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 7552.50 | 7526.40 | 7524.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 10:15:00 | 7473.50 | 7513.52 | 7518.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 10:15:00 | 7473.50 | 7513.52 | 7518.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 12:15:00 | 7458.00 | 7494.65 | 7508.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 11:15:00 | 7423.50 | 7341.95 | 7387.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 11:15:00 | 7423.50 | 7341.95 | 7387.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 7423.50 | 7341.95 | 7387.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:00:00 | 7423.50 | 7341.95 | 7387.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 7339.00 | 7341.36 | 7383.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:15:00 | 7171.00 | 7341.36 | 7383.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 6986.50 | 7270.39 | 7347.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 15:15:00 | 6914.00 | 7206.11 | 7310.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 15:15:00 | 6568.30 | 6646.27 | 6729.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 12:15:00 | 6585.00 | 6583.84 | 6668.76 | SL hit (close>ema200) qty=0.50 sl=6583.84 alert=retest2 |

### Cycle 97 — BUY (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 09:15:00 | 6716.50 | 6667.18 | 6666.77 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 12:15:00 | 6685.00 | 6696.72 | 6697.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 6638.50 | 6679.68 | 6689.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 10:15:00 | 6555.00 | 6533.06 | 6583.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 11:00:00 | 6555.00 | 6533.06 | 6583.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 6570.50 | 6540.55 | 6582.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:30:00 | 6571.50 | 6540.55 | 6582.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 6584.50 | 6552.68 | 6577.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 6584.50 | 6552.68 | 6577.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 6571.50 | 6556.45 | 6576.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 6574.00 | 6556.45 | 6576.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 6528.50 | 6550.86 | 6572.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:15:00 | 6513.50 | 6550.86 | 6572.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 12:15:00 | 6634.50 | 6574.73 | 6578.41 | SL hit (close>static) qty=1.00 sl=6614.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 13:15:00 | 6673.00 | 6594.38 | 6587.01 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 13:15:00 | 6607.00 | 6610.56 | 6610.99 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 6634.50 | 6615.35 | 6613.12 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 15:15:00 | 6579.00 | 6619.51 | 6621.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 09:15:00 | 6505.00 | 6596.61 | 6611.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 6509.50 | 6494.70 | 6540.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-13 10:00:00 | 6509.50 | 6494.70 | 6540.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 6350.00 | 6344.79 | 6387.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:15:00 | 6416.00 | 6344.79 | 6387.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 6468.00 | 6369.43 | 6394.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 6468.00 | 6369.43 | 6394.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 6401.00 | 6375.74 | 6395.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:30:00 | 6448.00 | 6375.74 | 6395.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 6415.00 | 6388.36 | 6397.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:45:00 | 6420.00 | 6388.36 | 6397.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 6421.50 | 6394.37 | 6398.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 6421.50 | 6394.37 | 6398.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 6419.00 | 6399.29 | 6400.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:15:00 | 6395.00 | 6399.29 | 6400.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 6384.00 | 6393.63 | 6397.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 11:30:00 | 6372.50 | 6388.90 | 6395.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 14:15:00 | 6416.50 | 6388.71 | 6393.04 | SL hit (close>static) qty=1.00 sl=6402.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 6425.50 | 6398.99 | 6397.15 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 11:15:00 | 6385.00 | 6396.19 | 6396.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 13:15:00 | 6375.00 | 6389.76 | 6393.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 09:15:00 | 6447.50 | 6389.03 | 6390.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 09:15:00 | 6447.50 | 6389.03 | 6390.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 6447.50 | 6389.03 | 6390.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:00:00 | 6447.50 | 6389.03 | 6390.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 10:15:00 | 6493.00 | 6409.82 | 6400.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 12:15:00 | 6538.50 | 6448.15 | 6420.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 11:15:00 | 6481.50 | 6488.41 | 6456.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-25 11:45:00 | 6477.50 | 6488.41 | 6456.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 6483.00 | 6487.33 | 6458.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 13:45:00 | 6506.50 | 6490.97 | 6462.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 14:30:00 | 6506.50 | 6493.77 | 6466.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 6345.00 | 6465.01 | 6458.43 | SL hit (close<static) qty=1.00 sl=6450.50 alert=retest2 |

### Cycle 106 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 6324.50 | 6436.91 | 6446.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 6260.00 | 6350.57 | 6397.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 13:15:00 | 6331.00 | 6323.03 | 6359.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 13:30:00 | 6326.50 | 6323.03 | 6359.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 6278.50 | 6288.22 | 6319.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 15:00:00 | 6278.50 | 6288.22 | 6319.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 6315.50 | 6299.63 | 6317.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:45:00 | 6315.00 | 6299.63 | 6317.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 6323.00 | 6304.30 | 6317.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:45:00 | 6319.00 | 6304.30 | 6317.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 6325.50 | 6308.54 | 6318.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:45:00 | 6325.50 | 6308.54 | 6318.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 6327.50 | 6312.33 | 6319.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:45:00 | 6330.50 | 6312.33 | 6319.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 6285.50 | 6310.59 | 6317.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 6284.00 | 6310.59 | 6317.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 6304.50 | 6309.37 | 6316.44 | EMA400 retest candle locked (from downside) |

### Cycle 107 — BUY (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 12:15:00 | 6344.00 | 6324.48 | 6322.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 14:15:00 | 6377.00 | 6338.27 | 6329.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 6374.00 | 6387.24 | 6367.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 6374.00 | 6387.24 | 6367.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 6374.00 | 6387.24 | 6367.83 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 6297.00 | 6364.86 | 6369.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 6272.00 | 6346.29 | 6360.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 11:15:00 | 6310.00 | 6300.74 | 6325.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 12:00:00 | 6310.00 | 6300.74 | 6325.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 6326.50 | 6305.90 | 6325.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:00:00 | 6326.50 | 6305.90 | 6325.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 6323.00 | 6309.32 | 6325.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:45:00 | 6323.50 | 6309.32 | 6325.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 6333.00 | 6314.05 | 6326.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 15:15:00 | 6331.00 | 6314.05 | 6326.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 6331.00 | 6317.44 | 6326.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:15:00 | 6296.50 | 6317.44 | 6326.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 6335.00 | 6323.76 | 6328.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:15:00 | 6337.50 | 6323.76 | 6328.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 6342.50 | 6327.51 | 6329.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:30:00 | 6337.00 | 6327.51 | 6329.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 6326.50 | 6328.83 | 6329.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:15:00 | 6335.50 | 6328.83 | 6329.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 14:15:00 | 6353.50 | 6333.76 | 6331.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 15:15:00 | 6360.00 | 6339.01 | 6334.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 15:15:00 | 6410.00 | 6417.22 | 6398.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 09:15:00 | 6391.50 | 6417.22 | 6398.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 6377.50 | 6409.27 | 6396.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:45:00 | 6374.50 | 6409.27 | 6396.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 6326.00 | 6392.62 | 6389.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:00:00 | 6326.00 | 6392.62 | 6389.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2025-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 11:15:00 | 6361.00 | 6386.30 | 6387.19 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 12:15:00 | 6394.00 | 6387.84 | 6387.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 6468.00 | 6405.30 | 6395.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 12:15:00 | 6420.00 | 6431.35 | 6412.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 13:00:00 | 6420.00 | 6431.35 | 6412.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 6435.00 | 6436.60 | 6421.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 10:15:00 | 6463.00 | 6436.60 | 6421.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 10:45:00 | 6455.00 | 6439.88 | 6424.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 15:00:00 | 6448.50 | 6445.72 | 6432.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 12:15:00 | 6448.00 | 6489.81 | 6495.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 12:15:00 | 6448.00 | 6489.81 | 6495.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 13:15:00 | 6423.50 | 6476.55 | 6488.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 11:15:00 | 6283.50 | 6269.49 | 6306.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 12:00:00 | 6283.50 | 6269.49 | 6306.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 6049.00 | 6019.97 | 6052.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 6049.00 | 6019.97 | 6052.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 6050.00 | 6025.97 | 6052.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 6015.50 | 6025.97 | 6052.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 6100.00 | 6045.12 | 6051.60 | SL hit (close>static) qty=1.00 sl=6080.00 alert=retest2 |

### Cycle 113 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 6194.00 | 6074.89 | 6064.54 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 12:15:00 | 6050.00 | 6081.19 | 6083.69 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 12:15:00 | 6100.50 | 6081.72 | 6080.88 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 6049.50 | 6077.60 | 6079.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 6034.50 | 6068.98 | 6075.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 6061.00 | 6008.81 | 6024.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 6061.00 | 6008.81 | 6024.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 6061.00 | 6008.81 | 6024.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 6061.00 | 6008.81 | 6024.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 6000.50 | 6007.15 | 6022.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 11:15:00 | 5995.00 | 6007.15 | 6022.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:45:00 | 5992.50 | 6013.63 | 6020.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 11:15:00 | 5991.00 | 6010.90 | 6018.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 11:45:00 | 5987.00 | 6006.82 | 6015.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 5995.00 | 6000.71 | 6009.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 5987.50 | 6000.71 | 6009.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 5943.00 | 5989.16 | 6003.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:30:00 | 5917.00 | 5974.53 | 5995.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:15:00 | 5859.00 | 5842.01 | 5876.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 5939.50 | 5878.79 | 5872.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 11:15:00 | 5939.50 | 5878.79 | 5872.32 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 12:15:00 | 5853.50 | 5881.32 | 5882.83 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 13:15:00 | 5913.00 | 5887.65 | 5885.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 15:15:00 | 5922.50 | 5898.84 | 5891.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 11:15:00 | 5917.00 | 5922.95 | 5908.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 11:45:00 | 5916.00 | 5922.95 | 5908.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 5905.00 | 5917.77 | 5908.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:00:00 | 5905.00 | 5917.77 | 5908.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 5940.00 | 5922.21 | 5911.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 10:15:00 | 5967.00 | 5925.92 | 5915.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 13:15:00 | 5874.00 | 5910.95 | 5911.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 13:15:00 | 5874.00 | 5910.95 | 5911.36 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 5925.00 | 5904.02 | 5903.13 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 5887.00 | 5904.19 | 5904.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 09:15:00 | 5835.50 | 5880.91 | 5891.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 11:15:00 | 5831.50 | 5799.37 | 5819.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 11:15:00 | 5831.50 | 5799.37 | 5819.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 5831.50 | 5799.37 | 5819.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:00:00 | 5831.50 | 5799.37 | 5819.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 5865.00 | 5812.49 | 5823.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:45:00 | 5868.00 | 5812.49 | 5823.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 5894.00 | 5828.79 | 5829.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:00:00 | 5894.00 | 5828.79 | 5829.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 5911.00 | 5845.24 | 5837.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 09:15:00 | 5940.00 | 5872.95 | 5851.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 5829.50 | 5878.17 | 5867.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 5829.50 | 5878.17 | 5867.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 5829.50 | 5878.17 | 5867.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 5829.50 | 5878.17 | 5867.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 5825.00 | 5867.54 | 5863.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 5820.00 | 5867.54 | 5863.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 5802.00 | 5854.43 | 5858.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 13:15:00 | 5764.00 | 5825.72 | 5843.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 10:15:00 | 5720.00 | 5715.48 | 5756.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 10:30:00 | 5725.50 | 5715.48 | 5756.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 5758.50 | 5720.78 | 5740.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:15:00 | 5788.00 | 5720.78 | 5740.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 5777.50 | 5732.13 | 5743.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 13:45:00 | 5750.00 | 5745.65 | 5747.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 15:15:00 | 5744.50 | 5748.72 | 5749.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 5785.50 | 5755.40 | 5752.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 5785.50 | 5755.40 | 5752.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 10:15:00 | 5827.00 | 5769.72 | 5758.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 09:15:00 | 6100.00 | 6120.48 | 6051.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 09:30:00 | 6061.00 | 6120.48 | 6051.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 6080.00 | 6121.61 | 6087.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:45:00 | 6087.00 | 6121.61 | 6087.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 6069.50 | 6111.19 | 6086.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 6069.50 | 6111.19 | 6086.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 6094.00 | 6107.75 | 6086.91 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 5976.50 | 6061.35 | 6071.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 5976.00 | 6044.28 | 6062.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 10:15:00 | 5943.50 | 5934.31 | 5968.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 14:15:00 | 5970.00 | 5941.60 | 5961.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 5970.00 | 5941.60 | 5961.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 5970.00 | 5941.60 | 5961.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 5980.00 | 5949.28 | 5962.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 5979.00 | 5949.28 | 5962.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 5927.00 | 5945.42 | 5958.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:15:00 | 5893.50 | 5945.82 | 5954.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 5882.50 | 5926.27 | 5936.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 15:15:00 | 5856.00 | 5785.45 | 5782.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 15:15:00 | 5856.00 | 5785.45 | 5782.48 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 5759.00 | 5779.26 | 5780.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 5710.00 | 5765.41 | 5774.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 5740.00 | 5732.19 | 5749.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 13:00:00 | 5740.00 | 5732.19 | 5749.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 5745.00 | 5734.75 | 5749.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 5752.00 | 5734.75 | 5749.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 5855.00 | 5758.80 | 5759.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 5855.00 | 5758.80 | 5759.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 5858.00 | 5778.64 | 5768.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 5895.50 | 5858.52 | 5839.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 5868.00 | 5890.18 | 5865.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 5868.00 | 5890.18 | 5865.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 5868.00 | 5890.18 | 5865.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 5868.00 | 5890.18 | 5865.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 5845.00 | 5881.14 | 5863.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:30:00 | 5840.50 | 5881.14 | 5863.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 5851.00 | 5875.11 | 5862.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 15:15:00 | 5855.00 | 5861.23 | 5858.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 5803.50 | 5848.69 | 5853.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 5803.50 | 5848.69 | 5853.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 10:15:00 | 5785.00 | 5835.95 | 5847.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 14:15:00 | 5878.00 | 5764.11 | 5785.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 14:15:00 | 5878.00 | 5764.11 | 5785.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 5878.00 | 5764.11 | 5785.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 5878.00 | 5764.11 | 5785.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 5850.00 | 5781.29 | 5791.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:30:00 | 5764.00 | 5781.93 | 5790.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 5835.00 | 5799.50 | 5796.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2025-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 15:15:00 | 5835.00 | 5799.50 | 5796.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 09:15:00 | 6104.00 | 5860.40 | 5824.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 5965.00 | 5990.31 | 5947.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:15:00 | 5967.00 | 5990.31 | 5947.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 6040.00 | 6000.24 | 5956.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 5975.00 | 6000.24 | 5956.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 5963.50 | 5997.66 | 5962.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:30:00 | 5961.00 | 5997.66 | 5962.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 5924.00 | 5982.93 | 5959.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 5924.00 | 5982.93 | 5959.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 5923.50 | 5971.04 | 5956.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:30:00 | 5918.00 | 5971.04 | 5956.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 6047.50 | 6043.62 | 6014.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:30:00 | 6066.00 | 6045.49 | 6017.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 6066.00 | 6045.49 | 6017.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 11:00:00 | 6078.00 | 6051.99 | 6023.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 12:30:00 | 6065.00 | 6111.33 | 6093.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 6039.00 | 6096.65 | 6089.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 6039.00 | 6096.65 | 6089.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 6052.00 | 6087.72 | 6086.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 09:15:00 | 6084.50 | 6087.72 | 6086.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 6071.50 | 6084.48 | 6084.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 09:15:00 | 6071.50 | 6084.48 | 6084.83 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 6111.00 | 6089.78 | 6087.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 11:15:00 | 6130.50 | 6097.92 | 6091.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 6107.00 | 6123.13 | 6108.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 6107.00 | 6123.13 | 6108.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 6107.00 | 6123.13 | 6108.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:30:00 | 6096.00 | 6123.13 | 6108.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 6094.00 | 6117.30 | 6107.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:00:00 | 6094.00 | 6117.30 | 6107.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 6111.50 | 6115.77 | 6108.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 13:30:00 | 6141.00 | 6123.82 | 6112.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 10:30:00 | 6125.50 | 6139.55 | 6125.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 14:00:00 | 6126.00 | 6130.44 | 6124.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 6155.50 | 6121.30 | 6120.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 6127.50 | 6153.00 | 6142.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 6115.00 | 6153.00 | 6142.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 6088.00 | 6140.00 | 6137.66 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-06 10:15:00 | 6088.00 | 6140.00 | 6137.66 | SL hit (close<static) qty=1.00 sl=6094.50 alert=retest2 |

### Cycle 134 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 6092.00 | 6130.40 | 6133.51 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 6154.00 | 6132.97 | 6132.27 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 15:15:00 | 6077.00 | 6126.22 | 6130.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 6072.00 | 6117.50 | 6126.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 13:15:00 | 6179.00 | 6116.60 | 6122.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 13:15:00 | 6179.00 | 6116.60 | 6122.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 6179.00 | 6116.60 | 6122.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 14:00:00 | 6179.00 | 6116.60 | 6122.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 14:15:00 | 6186.50 | 6130.58 | 6128.13 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 6079.00 | 6123.29 | 6125.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 10:15:00 | 6009.50 | 6062.61 | 6087.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 6078.00 | 6053.51 | 6077.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 12:15:00 | 6078.00 | 6053.51 | 6077.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 6078.00 | 6053.51 | 6077.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:00:00 | 6078.00 | 6053.51 | 6077.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 6184.50 | 6079.71 | 6087.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 6184.50 | 6079.71 | 6087.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2026-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 14:15:00 | 6197.50 | 6103.27 | 6097.64 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 12:15:00 | 6056.00 | 6115.10 | 6116.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 09:15:00 | 6000.00 | 6058.04 | 6086.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 15:15:00 | 6015.50 | 6010.46 | 6044.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-19 09:15:00 | 5955.00 | 6010.46 | 6044.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 5801.00 | 5731.38 | 5795.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 15:00:00 | 5801.00 | 5731.38 | 5795.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 5829.50 | 5751.01 | 5798.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 5841.00 | 5751.01 | 5798.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 5869.50 | 5774.70 | 5805.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:30:00 | 5775.00 | 5793.41 | 5806.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 14:00:00 | 5771.00 | 5793.41 | 5806.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 15:00:00 | 5763.00 | 5787.33 | 5802.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 12:00:00 | 5755.00 | 5792.28 | 5801.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 5809.00 | 5795.62 | 5801.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:00:00 | 5809.00 | 5795.62 | 5801.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 5780.00 | 5792.50 | 5799.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:15:00 | 5865.50 | 5792.50 | 5799.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 5814.00 | 5796.80 | 5801.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-23 15:15:00 | 5856.50 | 5808.74 | 5806.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 15:15:00 | 5856.50 | 5808.74 | 5806.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 09:15:00 | 5872.50 | 5821.49 | 5812.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-27 13:15:00 | 5828.50 | 5828.80 | 5819.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 13:15:00 | 5828.50 | 5828.80 | 5819.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 13:15:00 | 5828.50 | 5828.80 | 5819.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 13:30:00 | 5809.50 | 5828.80 | 5819.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 5836.00 | 5830.24 | 5820.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 15:15:00 | 5845.50 | 5830.24 | 5820.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 10:00:00 | 5851.50 | 5836.93 | 5825.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 5990.00 | 6109.42 | 6113.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 5990.00 | 6109.42 | 6113.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 5886.00 | 6031.06 | 6073.86 | Break + close below crossover candle low |

### Cycle 143 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 6403.50 | 6058.92 | 6057.39 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 12:15:00 | 6555.00 | 6581.38 | 6582.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 15:15:00 | 6539.50 | 6567.49 | 6575.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 6580.50 | 6570.09 | 6576.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 6580.50 | 6570.09 | 6576.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 6580.50 | 6570.09 | 6576.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 6599.50 | 6570.09 | 6576.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 6594.50 | 6574.97 | 6577.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 6597.00 | 6574.97 | 6577.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 6686.50 | 6597.28 | 6587.78 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 6552.00 | 6605.10 | 6611.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 15:15:00 | 6505.00 | 6564.85 | 6582.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 6514.00 | 6498.87 | 6530.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 10:00:00 | 6514.00 | 6498.87 | 6530.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 6515.00 | 6502.09 | 6529.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:45:00 | 6524.00 | 6502.09 | 6529.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 6509.50 | 6504.76 | 6523.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:45:00 | 6520.00 | 6504.76 | 6523.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 6546.50 | 6513.11 | 6525.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 15:00:00 | 6546.50 | 6513.11 | 6525.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 6544.00 | 6519.29 | 6527.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:15:00 | 6559.50 | 6519.29 | 6527.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 6574.00 | 6530.23 | 6531.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:30:00 | 6582.00 | 6530.23 | 6531.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 6561.00 | 6536.38 | 6534.45 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 13:15:00 | 6522.00 | 6531.41 | 6532.58 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2026-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 15:15:00 | 6553.50 | 6533.36 | 6533.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 09:15:00 | 6587.50 | 6544.19 | 6538.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 13:15:00 | 6562.00 | 6562.14 | 6549.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 13:15:00 | 6562.00 | 6562.14 | 6549.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 6562.00 | 6562.14 | 6549.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:45:00 | 6572.50 | 6562.14 | 6549.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 6699.00 | 6589.51 | 6563.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:30:00 | 6567.50 | 6589.51 | 6563.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 6575.00 | 6593.41 | 6570.06 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2026-03-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 15:15:00 | 6526.00 | 6554.58 | 6557.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 6383.50 | 6520.36 | 6541.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 15:15:00 | 6700.00 | 6461.20 | 6486.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 15:15:00 | 6700.00 | 6461.20 | 6486.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 6700.00 | 6461.20 | 6486.53 | EMA400 retest candle locked (from downside) |

### Cycle 151 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 6578.00 | 6500.21 | 6490.58 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 12:15:00 | 6467.50 | 6483.61 | 6484.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 14:15:00 | 6423.00 | 6469.55 | 6477.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-11 10:15:00 | 6124.00 | 6107.83 | 6196.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-11 11:00:00 | 6124.00 | 6107.83 | 6196.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 6179.00 | 6135.89 | 6188.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:30:00 | 6180.00 | 6135.89 | 6188.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 6207.50 | 6150.21 | 6190.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:30:00 | 6185.00 | 6150.21 | 6190.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 6200.00 | 6160.17 | 6191.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 09:15:00 | 6101.00 | 6160.17 | 6191.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 11:15:00 | 6290.00 | 6178.17 | 6190.79 | SL hit (close>static) qty=1.00 sl=6227.50 alert=retest2 |

### Cycle 153 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 6303.00 | 6203.13 | 6200.99 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 6108.00 | 6205.26 | 6208.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 11:15:00 | 6085.00 | 6181.21 | 6197.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 14:15:00 | 6428.00 | 6188.20 | 6192.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 14:15:00 | 6428.00 | 6188.20 | 6192.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 6428.00 | 6188.20 | 6192.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 15:00:00 | 6428.00 | 6188.20 | 6192.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2026-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 15:15:00 | 6448.00 | 6240.16 | 6215.64 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 12:15:00 | 6173.50 | 6218.21 | 6224.31 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 6342.50 | 6247.93 | 6236.54 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 15:15:00 | 6228.00 | 6261.53 | 6262.32 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 6291.50 | 6267.52 | 6264.98 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 6186.50 | 6253.91 | 6262.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 6068.00 | 6189.25 | 6228.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 6143.00 | 6134.30 | 6177.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:45:00 | 6150.00 | 6134.30 | 6177.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 6155.00 | 6138.44 | 6175.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 6196.00 | 6138.44 | 6175.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 6182.50 | 6147.25 | 6176.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 6182.50 | 6147.25 | 6176.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 6131.50 | 6144.10 | 6172.26 | EMA400 retest candle locked (from downside) |

### Cycle 161 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 6245.00 | 6190.02 | 6186.28 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2026-03-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 11:15:00 | 6175.00 | 6198.91 | 6200.41 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 13:15:00 | 6286.50 | 6218.04 | 6208.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-30 14:15:00 | 6370.50 | 6248.53 | 6223.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-01 15:15:00 | 6366.00 | 6394.50 | 6330.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-02 09:15:00 | 6316.00 | 6394.50 | 6330.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 6318.50 | 6379.30 | 6329.64 | EMA400 retest candle locked (from upside) |

### Cycle 164 — SELL (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 09:15:00 | 6160.50 | 6292.18 | 6305.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-06 10:15:00 | 6139.00 | 6261.55 | 6290.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 09:15:00 | 6209.50 | 6193.44 | 6236.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 6209.50 | 6193.44 | 6236.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 6209.50 | 6193.44 | 6236.21 | EMA400 retest candle locked (from downside) |

### Cycle 165 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 6374.00 | 6265.61 | 6252.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 15:15:00 | 6400.00 | 6340.31 | 6298.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 6377.00 | 6398.46 | 6353.18 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 6501.00 | 6398.46 | 6353.18 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:15:00 | 6479.50 | 6411.57 | 6363.25 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 12:30:00 | 6485.00 | 6438.90 | 6389.09 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 14:00:00 | 6500.00 | 6451.12 | 6399.17 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 6381.50 | 6440.14 | 6407.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 6381.50 | 6440.14 | 6407.28 | SL hit (close<ema400) qty=1.00 sl=6407.28 alert=retest1 |

### Cycle 166 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 6333.00 | 6392.65 | 6393.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 15:15:00 | 6299.00 | 6363.50 | 6379.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 6429.00 | 6376.60 | 6383.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 6429.00 | 6376.60 | 6383.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 6429.00 | 6376.60 | 6383.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:00:00 | 6429.00 | 6376.60 | 6383.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 6486.00 | 6398.48 | 6393.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 12:15:00 | 6565.00 | 6444.43 | 6415.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 09:15:00 | 6566.50 | 6573.45 | 6524.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-17 10:00:00 | 6566.50 | 6573.45 | 6524.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 6575.00 | 6581.28 | 6552.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:30:00 | 6550.50 | 6581.28 | 6552.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 6543.50 | 6574.96 | 6558.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 13:30:00 | 6540.00 | 6574.96 | 6558.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 6549.00 | 6569.77 | 6558.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 15:00:00 | 6549.00 | 6569.77 | 6558.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 6532.00 | 6562.21 | 6555.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 6562.00 | 6562.21 | 6555.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:30:00 | 6567.00 | 6558.30 | 6554.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 12:30:00 | 6567.00 | 6568.61 | 6560.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 6596.50 | 6568.50 | 6562.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 6591.50 | 6573.10 | 6565.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 15:00:00 | 6659.50 | 6610.55 | 6587.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 10:00:00 | 6650.00 | 6624.75 | 6598.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 10:15:00 | 6668.50 | 6700.84 | 6692.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 12:00:00 | 6716.50 | 6694.64 | 6690.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 6700.00 | 6703.34 | 6696.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-27 15:00:00 | 6700.00 | 6703.34 | 6696.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 6813.00 | 6724.73 | 6707.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 10:15:00 | 6879.00 | 6724.73 | 6707.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:30:00 | 6849.50 | 6857.39 | 6801.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 15:00:00 | 6830.00 | 6820.47 | 6800.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 10:15:00 | 6719.00 | 6783.34 | 6787.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 6719.00 | 6783.34 | 6787.39 | EMA200 below EMA400 |

### Cycle 169 — BUY (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 15:15:00 | 6825.00 | 6788.10 | 6786.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 6964.50 | 6823.38 | 6802.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 11:15:00 | 6911.50 | 6934.57 | 6888.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 11:15:00 | 6911.50 | 6934.57 | 6888.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 6911.50 | 6934.57 | 6888.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:45:00 | 6888.00 | 6934.57 | 6888.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 6893.00 | 6920.96 | 6890.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:00:00 | 6893.00 | 6920.96 | 6890.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 6895.50 | 6915.87 | 6890.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 15:00:00 | 6895.50 | 6915.87 | 6890.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 7005.00 | 7053.17 | 7016.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 7093.50 | 7053.17 | 7016.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 10:30:00 | 5863.85 | 2024-05-17 12:15:00 | 5964.80 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-05-14 10:00:00 | 5871.00 | 2024-05-17 12:15:00 | 5964.80 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-05-15 14:15:00 | 5873.00 | 2024-05-17 12:15:00 | 5964.80 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-05-15 15:00:00 | 5875.55 | 2024-05-17 12:15:00 | 5964.80 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-05-16 13:15:00 | 5867.55 | 2024-05-17 12:15:00 | 5964.80 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-05-16 14:00:00 | 5864.30 | 2024-05-17 12:15:00 | 5964.80 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-05-27 11:45:00 | 5898.55 | 2024-06-04 09:15:00 | 5603.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-27 14:30:00 | 5900.90 | 2024-06-04 09:15:00 | 5605.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-28 09:45:00 | 5901.40 | 2024-06-04 09:15:00 | 5606.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-27 11:45:00 | 5898.55 | 2024-06-04 12:15:00 | 5308.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-27 14:30:00 | 5900.90 | 2024-06-04 12:15:00 | 5310.81 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-28 09:45:00 | 5901.40 | 2024-06-04 12:15:00 | 5311.26 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-06-19 11:30:00 | 6285.30 | 2024-06-25 12:15:00 | 6348.05 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest2 | 2024-06-20 09:15:00 | 6263.45 | 2024-06-25 12:15:00 | 6348.05 | STOP_HIT | 1.00 | 1.35% |
| BUY | retest2 | 2024-07-04 11:15:00 | 6612.95 | 2024-07-19 13:15:00 | 6993.30 | STOP_HIT | 1.00 | 5.75% |
| BUY | retest2 | 2024-07-25 14:15:00 | 7309.90 | 2024-08-02 15:15:00 | 7793.85 | STOP_HIT | 1.00 | 6.62% |
| BUY | retest2 | 2024-07-25 14:45:00 | 7309.95 | 2024-08-02 15:15:00 | 7793.85 | STOP_HIT | 1.00 | 6.62% |
| BUY | retest2 | 2024-07-26 09:15:00 | 7308.80 | 2024-08-02 15:15:00 | 7793.85 | STOP_HIT | 1.00 | 6.64% |
| SELL | retest2 | 2024-08-06 14:30:00 | 7796.85 | 2024-08-07 10:15:00 | 7871.35 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-08-14 14:15:00 | 7730.50 | 2024-08-16 12:15:00 | 7854.60 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-08-16 09:30:00 | 7716.10 | 2024-08-16 12:15:00 | 7854.60 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-08-16 10:00:00 | 7729.95 | 2024-08-16 12:15:00 | 7854.60 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-09-05 09:30:00 | 8028.05 | 2024-09-06 09:15:00 | 7817.85 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2024-09-05 14:45:00 | 8017.80 | 2024-09-06 09:15:00 | 7817.85 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2024-09-20 13:45:00 | 7597.85 | 2024-09-27 11:15:00 | 7587.45 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2024-09-20 14:30:00 | 7599.65 | 2024-09-27 11:15:00 | 7587.45 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2024-09-24 11:15:00 | 7598.35 | 2024-09-27 11:15:00 | 7587.45 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest1 | 2024-10-01 09:15:00 | 7709.45 | 2024-10-03 14:15:00 | 7822.95 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest2 | 2024-10-15 09:15:00 | 7960.45 | 2024-10-15 11:15:00 | 7706.50 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2024-11-04 15:15:00 | 7786.15 | 2024-11-11 09:15:00 | 7725.90 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-11-19 14:15:00 | 7324.15 | 2024-11-25 10:15:00 | 7411.30 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-11-19 14:45:00 | 7282.85 | 2024-11-25 10:15:00 | 7411.30 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-11-28 09:30:00 | 7402.35 | 2024-11-28 10:15:00 | 7303.70 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-12-05 09:15:00 | 7415.70 | 2024-12-05 09:15:00 | 7345.95 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-12-06 09:30:00 | 7421.80 | 2024-12-06 13:15:00 | 7366.05 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-12-06 10:15:00 | 7415.00 | 2024-12-06 13:15:00 | 7366.05 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-12-18 09:30:00 | 7363.55 | 2024-12-20 09:15:00 | 7297.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-12-18 10:15:00 | 7360.00 | 2024-12-20 09:15:00 | 7297.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-12-19 09:30:00 | 7387.65 | 2024-12-20 11:15:00 | 7277.30 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-12-19 10:15:00 | 7377.70 | 2024-12-20 11:15:00 | 7277.30 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-12-19 11:30:00 | 7417.05 | 2024-12-20 11:15:00 | 7277.30 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2024-12-19 15:00:00 | 7412.05 | 2024-12-20 11:15:00 | 7277.30 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-01-06 10:30:00 | 6869.45 | 2025-01-07 14:15:00 | 6920.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-01-30 11:15:00 | 6301.45 | 2025-02-05 12:15:00 | 6240.25 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2025-01-31 09:45:00 | 6272.00 | 2025-02-05 12:15:00 | 6240.25 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest2 | 2025-02-01 10:00:00 | 6305.30 | 2025-02-05 12:15:00 | 6240.25 | STOP_HIT | 1.00 | 1.03% |
| SELL | retest2 | 2025-02-01 11:45:00 | 6197.05 | 2025-02-05 12:15:00 | 6240.25 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-02-03 12:15:00 | 6201.25 | 2025-02-05 12:15:00 | 6240.25 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-02-05 09:15:00 | 6189.00 | 2025-02-05 12:15:00 | 6240.25 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-02-25 11:30:00 | 5293.80 | 2025-03-03 12:15:00 | 5597.95 | STOP_HIT | 1.00 | -5.75% |
| SELL | retest2 | 2025-02-27 10:00:00 | 5298.50 | 2025-03-03 12:15:00 | 5597.95 | STOP_HIT | 1.00 | -5.65% |
| SELL | retest2 | 2025-02-28 15:15:00 | 5270.00 | 2025-03-03 12:15:00 | 5597.95 | STOP_HIT | 1.00 | -6.22% |
| SELL | retest2 | 2025-03-03 09:45:00 | 5296.80 | 2025-03-03 12:15:00 | 5597.95 | STOP_HIT | 1.00 | -5.69% |
| BUY | retest2 | 2025-03-21 10:15:00 | 5744.55 | 2025-03-26 10:15:00 | 5770.75 | STOP_HIT | 1.00 | 0.46% |
| BUY | retest2 | 2025-03-21 10:45:00 | 5793.20 | 2025-03-26 10:15:00 | 5770.75 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-04-08 10:30:00 | 5193.90 | 2025-04-11 11:15:00 | 5425.40 | STOP_HIT | 1.00 | -4.46% |
| SELL | retest2 | 2025-04-08 11:00:00 | 5197.45 | 2025-04-11 11:15:00 | 5425.40 | STOP_HIT | 1.00 | -4.39% |
| SELL | retest2 | 2025-04-08 13:15:00 | 5185.75 | 2025-04-11 11:15:00 | 5425.40 | STOP_HIT | 1.00 | -4.62% |
| SELL | retest2 | 2025-04-09 11:00:00 | 5195.75 | 2025-04-11 11:15:00 | 5425.40 | STOP_HIT | 1.00 | -4.42% |
| BUY | retest2 | 2025-04-17 11:15:00 | 5685.00 | 2025-04-23 09:15:00 | 6253.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-17 14:30:00 | 5688.00 | 2025-04-23 09:15:00 | 6256.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-21 09:15:00 | 5740.00 | 2025-04-25 09:15:00 | 6314.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-09 09:15:00 | 6685.50 | 2025-05-13 14:15:00 | 6875.00 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-05-09 10:15:00 | 6759.00 | 2025-05-13 14:15:00 | 6875.00 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-05-09 12:15:00 | 6760.00 | 2025-05-13 14:15:00 | 6875.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-05-09 13:00:00 | 6760.00 | 2025-05-13 14:15:00 | 6875.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-05-12 11:15:00 | 6786.00 | 2025-05-13 14:15:00 | 6875.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-05-13 09:15:00 | 6771.50 | 2025-05-13 14:15:00 | 6875.00 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-05-19 09:15:00 | 6760.00 | 2025-05-19 09:15:00 | 6831.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-05-22 11:45:00 | 6980.00 | 2025-05-27 12:15:00 | 7074.50 | STOP_HIT | 1.00 | 1.35% |
| SELL | retest2 | 2025-05-28 13:30:00 | 7078.50 | 2025-05-28 14:15:00 | 7137.50 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-06-06 14:30:00 | 7361.50 | 2025-06-11 10:15:00 | 7233.50 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-06-09 09:15:00 | 7364.00 | 2025-06-11 10:15:00 | 7233.50 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-06-09 10:15:00 | 7352.50 | 2025-06-11 10:15:00 | 7233.50 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-06-09 12:15:00 | 7380.00 | 2025-06-11 10:15:00 | 7233.50 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-06-20 11:30:00 | 6969.50 | 2025-06-25 09:15:00 | 7182.00 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-06-24 13:15:00 | 6977.00 | 2025-06-25 09:15:00 | 7182.00 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2025-07-01 13:15:00 | 7410.00 | 2025-07-09 09:15:00 | 7553.00 | STOP_HIT | 1.00 | 1.93% |
| BUY | retest2 | 2025-07-01 14:00:00 | 7411.50 | 2025-07-09 09:15:00 | 7553.00 | STOP_HIT | 1.00 | 1.91% |
| BUY | retest2 | 2025-07-02 10:00:00 | 7411.00 | 2025-07-09 10:15:00 | 7517.50 | STOP_HIT | 1.00 | 1.44% |
| BUY | retest2 | 2025-07-02 10:45:00 | 7443.50 | 2025-07-09 10:15:00 | 7517.50 | STOP_HIT | 1.00 | 0.99% |
| BUY | retest2 | 2025-07-08 09:15:00 | 7710.00 | 2025-07-09 10:15:00 | 7517.50 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-07-08 11:45:00 | 7736.50 | 2025-07-09 10:15:00 | 7517.50 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-07-11 11:15:00 | 7410.50 | 2025-07-14 15:15:00 | 7575.00 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-07-16 09:15:00 | 7552.50 | 2025-07-16 10:15:00 | 7473.50 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-07-18 15:15:00 | 6914.00 | 2025-07-24 15:15:00 | 6568.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 15:15:00 | 6914.00 | 2025-07-25 12:15:00 | 6585.00 | STOP_HIT | 0.50 | 4.76% |
| SELL | retest2 | 2025-08-05 10:15:00 | 6513.50 | 2025-08-05 12:15:00 | 6634.50 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-08-20 11:30:00 | 6372.50 | 2025-08-20 14:15:00 | 6416.50 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-08-25 13:45:00 | 6506.50 | 2025-08-26 09:15:00 | 6345.00 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-08-25 14:30:00 | 6506.50 | 2025-08-26 09:15:00 | 6345.00 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-09-16 10:15:00 | 6463.00 | 2025-09-19 12:15:00 | 6448.00 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-09-16 10:45:00 | 6455.00 | 2025-09-19 12:15:00 | 6448.00 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-09-16 15:00:00 | 6448.50 | 2025-09-19 12:15:00 | 6448.00 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-10-01 09:15:00 | 6015.50 | 2025-10-01 13:15:00 | 6100.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-10-10 11:15:00 | 5995.00 | 2025-10-17 11:15:00 | 5939.50 | STOP_HIT | 1.00 | 0.93% |
| SELL | retest2 | 2025-10-13 09:45:00 | 5992.50 | 2025-10-17 11:15:00 | 5939.50 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2025-10-13 11:15:00 | 5991.00 | 2025-10-17 11:15:00 | 5939.50 | STOP_HIT | 1.00 | 0.86% |
| SELL | retest2 | 2025-10-13 11:45:00 | 5987.00 | 2025-10-17 11:15:00 | 5939.50 | STOP_HIT | 1.00 | 0.79% |
| SELL | retest2 | 2025-10-14 10:30:00 | 5917.00 | 2025-10-17 11:15:00 | 5939.50 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-10-16 10:15:00 | 5859.00 | 2025-10-17 11:15:00 | 5939.50 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-10-24 10:15:00 | 5967.00 | 2025-10-24 13:15:00 | 5874.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-11-11 13:45:00 | 5750.00 | 2025-11-12 09:15:00 | 5785.50 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-11-11 15:15:00 | 5744.50 | 2025-11-12 09:15:00 | 5785.50 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-11-27 09:15:00 | 5893.50 | 2025-12-05 15:15:00 | 5856.00 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2025-11-28 09:15:00 | 5882.50 | 2025-12-05 15:15:00 | 5856.00 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest2 | 2025-12-15 15:15:00 | 5855.00 | 2025-12-16 09:15:00 | 5803.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-12-18 09:30:00 | 5764.00 | 2025-12-18 15:15:00 | 5835.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-12-26 09:30:00 | 6066.00 | 2025-12-31 09:15:00 | 6071.50 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-12-26 10:15:00 | 6066.00 | 2025-12-31 09:15:00 | 6071.50 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-12-26 11:00:00 | 6078.00 | 2025-12-31 09:15:00 | 6071.50 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-12-30 12:30:00 | 6065.00 | 2025-12-31 09:15:00 | 6071.50 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2025-12-31 09:15:00 | 6084.50 | 2025-12-31 09:15:00 | 6071.50 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2026-01-01 13:30:00 | 6141.00 | 2026-01-06 10:15:00 | 6088.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-01-02 10:30:00 | 6125.50 | 2026-01-06 10:15:00 | 6088.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2026-01-02 14:00:00 | 6126.00 | 2026-01-06 10:15:00 | 6088.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-01-05 09:15:00 | 6155.50 | 2026-01-06 10:15:00 | 6088.00 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-01-22 13:30:00 | 5775.00 | 2026-01-23 15:15:00 | 5856.50 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2026-01-22 14:00:00 | 5771.00 | 2026-01-23 15:15:00 | 5856.50 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-01-22 15:00:00 | 5763.00 | 2026-01-23 15:15:00 | 5856.50 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-01-23 12:00:00 | 5755.00 | 2026-01-23 15:15:00 | 5856.50 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-01-27 15:15:00 | 5845.50 | 2026-02-01 14:15:00 | 5990.00 | STOP_HIT | 1.00 | 2.47% |
| BUY | retest2 | 2026-01-28 10:00:00 | 5851.50 | 2026-02-01 14:15:00 | 5990.00 | STOP_HIT | 1.00 | 2.37% |
| SELL | retest2 | 2026-03-12 09:15:00 | 6101.00 | 2026-03-12 11:15:00 | 6290.00 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest1 | 2026-04-10 09:15:00 | 6501.00 | 2026-04-13 09:15:00 | 6381.50 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest1 | 2026-04-10 10:15:00 | 6479.50 | 2026-04-13 09:15:00 | 6381.50 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest1 | 2026-04-10 12:30:00 | 6485.00 | 2026-04-13 09:15:00 | 6381.50 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest1 | 2026-04-10 14:00:00 | 6500.00 | 2026-04-13 09:15:00 | 6381.50 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2026-04-21 09:15:00 | 6562.00 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | 2.39% |
| BUY | retest2 | 2026-04-21 10:30:00 | 6567.00 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | 2.31% |
| BUY | retest2 | 2026-04-21 12:30:00 | 6567.00 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | 2.31% |
| BUY | retest2 | 2026-04-22 09:15:00 | 6596.50 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | 1.86% |
| BUY | retest2 | 2026-04-22 15:00:00 | 6659.50 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | 0.89% |
| BUY | retest2 | 2026-04-23 10:00:00 | 6650.00 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | 1.04% |
| BUY | retest2 | 2026-04-27 10:15:00 | 6668.50 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2026-04-27 12:00:00 | 6716.50 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2026-04-28 10:15:00 | 6879.00 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-04-29 09:30:00 | 6849.50 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2026-04-29 15:00:00 | 6830.00 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | -1.63% |
