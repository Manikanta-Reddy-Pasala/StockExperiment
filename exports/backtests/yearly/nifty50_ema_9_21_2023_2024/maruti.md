# MARUTI (MARUTI)

## Backtest Summary

- **Window:** 2023-03-13 09:15:00 → 2026-05-08 15:15:00 (5443 bars)
- **Last close:** 13733.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 220 |
| ALERT1 | 156 |
| ALERT2 | 155 |
| ALERT2_SKIP | 76 |
| ALERT3 | 437 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 209 |
| PARTIAL | 9 |
| TARGET_HIT | 3 |
| STOP_HIT | 208 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 220 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 61 / 159
- **Target hits / Stop hits / Partials:** 3 / 208 / 9
- **Avg / median % per leg:** 0.07% / -0.70%
- **Sum % (uncompounded):** 14.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 106 | 32 | 30.2% | 2 | 104 | 0 | 0.20% | 21.3% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.92% | -1.8% |
| BUY @ 3rd Alert (retest2) | 104 | 32 | 30.8% | 2 | 102 | 0 | 0.22% | 23.1% |
| SELL (all) | 114 | 29 | 25.4% | 1 | 104 | 9 | -0.06% | -6.6% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 2 | 0 | 0.22% | 0.4% |
| SELL @ 3rd Alert (retest2) | 112 | 27 | 24.1% | 1 | 102 | 9 | -0.06% | -7.1% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 4 | 0 | -0.35% | -1.4% |
| retest2 (combined) | 216 | 59 | 27.3% | 3 | 204 | 9 | 0.07% | 16.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 10:15:00 | 9117.10 | 9193.94 | 9203.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-16 14:15:00 | 9062.05 | 9135.22 | 9170.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-17 14:15:00 | 9120.00 | 9099.55 | 9129.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-17 15:00:00 | 9120.00 | 9099.55 | 9129.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 15:15:00 | 9132.00 | 9106.04 | 9130.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-18 09:15:00 | 9130.85 | 9106.04 | 9130.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 09:15:00 | 9111.00 | 9107.03 | 9128.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-18 11:00:00 | 9088.75 | 9103.37 | 9124.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-22 10:15:00 | 9158.50 | 9107.48 | 9102.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 10:15:00 | 9158.50 | 9107.48 | 9102.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 09:15:00 | 9198.55 | 9149.86 | 9128.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-25 10:15:00 | 9251.60 | 9262.22 | 9228.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-25 10:45:00 | 9252.20 | 9262.22 | 9228.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 10:15:00 | 9359.80 | 9361.92 | 9322.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-29 10:30:00 | 9336.00 | 9361.92 | 9322.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 14:15:00 | 9334.00 | 9349.70 | 9328.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 09:30:00 | 9350.90 | 9349.17 | 9331.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 11:15:00 | 9354.70 | 9348.33 | 9332.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-31 11:15:00 | 9299.45 | 9349.85 | 9346.51 | SL hit (close<static) qty=1.00 sl=9320.95 alert=retest2 |

### Cycle 3 — SELL (started 2023-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 12:15:00 | 9289.05 | 9337.69 | 9341.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-01 09:15:00 | 9243.00 | 9315.05 | 9329.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-01 12:15:00 | 9362.50 | 9309.45 | 9321.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 12:15:00 | 9362.50 | 9309.45 | 9321.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 12:15:00 | 9362.50 | 9309.45 | 9321.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 12:30:00 | 9358.85 | 9309.45 | 9321.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 13:15:00 | 9353.00 | 9318.16 | 9324.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-01 14:30:00 | 9331.90 | 9320.53 | 9325.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-02 09:45:00 | 9324.60 | 9324.94 | 9326.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-02 10:15:00 | 9365.95 | 9333.14 | 9329.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2023-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 10:15:00 | 9365.95 | 9333.14 | 9329.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-02 12:15:00 | 9422.25 | 9355.51 | 9340.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-07 13:15:00 | 9679.10 | 9716.22 | 9652.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-07 13:30:00 | 9700.55 | 9716.22 | 9652.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 14:15:00 | 9720.00 | 9716.98 | 9658.70 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 14:15:00 | 9625.00 | 9656.56 | 9659.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-12 09:15:00 | 9584.30 | 9638.66 | 9650.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-13 09:15:00 | 9594.00 | 9578.95 | 9606.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 09:15:00 | 9594.00 | 9578.95 | 9606.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 09:15:00 | 9594.00 | 9578.95 | 9606.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 09:45:00 | 9598.00 | 9578.95 | 9606.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 10:15:00 | 9550.50 | 9573.26 | 9601.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 10:45:00 | 9554.00 | 9573.26 | 9601.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 10:15:00 | 9572.75 | 9546.47 | 9571.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-14 10:45:00 | 9592.50 | 9546.47 | 9571.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 11:15:00 | 9556.15 | 9548.41 | 9569.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-14 12:15:00 | 9549.80 | 9548.41 | 9569.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-15 09:15:00 | 9635.80 | 9560.18 | 9566.31 | SL hit (close>static) qty=1.00 sl=9576.00 alert=retest2 |

### Cycle 6 — BUY (started 2023-06-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-15 10:15:00 | 9679.00 | 9583.94 | 9576.55 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 11:15:00 | 9573.35 | 9589.76 | 9591.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-19 14:15:00 | 9528.50 | 9571.93 | 9582.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 15:15:00 | 9515.05 | 9504.29 | 9532.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-21 09:15:00 | 9517.35 | 9504.29 | 9532.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 9490.95 | 9501.62 | 9529.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-21 10:15:00 | 9474.90 | 9501.62 | 9529.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-26 14:45:00 | 9474.95 | 9403.06 | 9406.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-26 15:15:00 | 9470.00 | 9416.45 | 9411.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2023-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 15:15:00 | 9470.00 | 9416.45 | 9411.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 09:15:00 | 9488.90 | 9446.76 | 9430.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 09:15:00 | 9686.15 | 9709.14 | 9620.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-03 10:00:00 | 9686.15 | 9709.14 | 9620.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 9651.55 | 9682.65 | 9649.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 09:30:00 | 9662.70 | 9682.65 | 9649.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 10:15:00 | 9645.20 | 9675.16 | 9649.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-05 09:15:00 | 9723.05 | 9654.08 | 9646.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-10 13:15:00 | 9772.00 | 9828.02 | 9828.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2023-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 13:15:00 | 9772.00 | 9828.02 | 9828.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 14:15:00 | 9735.60 | 9809.53 | 9819.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 9852.00 | 9810.85 | 9818.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 09:15:00 | 9852.00 | 9810.85 | 9818.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 9852.00 | 9810.85 | 9818.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 10:00:00 | 9852.00 | 9810.85 | 9818.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2023-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 10:15:00 | 9915.00 | 9831.68 | 9827.06 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-07-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 15:15:00 | 9833.00 | 9852.24 | 9853.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 09:15:00 | 9778.00 | 9837.39 | 9846.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-17 14:15:00 | 9693.25 | 9624.67 | 9666.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 14:15:00 | 9693.25 | 9624.67 | 9666.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 14:15:00 | 9693.25 | 9624.67 | 9666.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 15:00:00 | 9693.25 | 9624.67 | 9666.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 15:15:00 | 9690.00 | 9637.74 | 9668.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-18 09:15:00 | 9685.35 | 9637.74 | 9668.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 10:15:00 | 9675.25 | 9653.91 | 9670.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-18 11:30:00 | 9617.70 | 9649.28 | 9667.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-19 09:15:00 | 9616.60 | 9656.38 | 9665.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-19 10:15:00 | 9641.35 | 9653.96 | 9663.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-19 11:15:00 | 9642.30 | 9651.93 | 9661.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 11:15:00 | 9653.90 | 9652.33 | 9660.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-19 11:30:00 | 9661.05 | 9652.33 | 9660.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 13:15:00 | 9610.00 | 9643.74 | 9655.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-19 13:30:00 | 9650.95 | 9643.74 | 9655.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 11:15:00 | 9663.60 | 9632.47 | 9643.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 11:30:00 | 9643.75 | 9632.47 | 9643.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 12:15:00 | 9691.75 | 9644.33 | 9647.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 13:00:00 | 9691.75 | 9644.33 | 9647.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-07-20 13:15:00 | 9738.80 | 9663.22 | 9655.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2023-07-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 13:15:00 | 9738.80 | 9663.22 | 9655.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-20 14:15:00 | 9762.05 | 9682.99 | 9665.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-21 13:15:00 | 9736.15 | 9749.84 | 9713.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-21 14:00:00 | 9736.15 | 9749.84 | 9713.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 9770.00 | 9760.27 | 9728.04 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 09:15:00 | 9685.00 | 9720.44 | 9722.31 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 11:15:00 | 9757.85 | 9726.25 | 9724.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-26 09:15:00 | 9787.00 | 9750.11 | 9737.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-26 10:15:00 | 9744.55 | 9749.00 | 9738.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-26 10:45:00 | 9742.00 | 9749.00 | 9738.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 11:15:00 | 9783.40 | 9755.88 | 9742.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-27 09:30:00 | 9810.10 | 9764.08 | 9751.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-27 10:30:00 | 9791.40 | 9767.98 | 9754.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-27 12:15:00 | 9725.00 | 9755.72 | 9751.12 | SL hit (close<static) qty=1.00 sl=9730.00 alert=retest2 |

### Cycle 15 — SELL (started 2023-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 09:15:00 | 9699.90 | 9742.07 | 9745.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-28 10:15:00 | 9657.00 | 9725.06 | 9737.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-28 14:15:00 | 9664.65 | 9663.75 | 9699.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-28 15:00:00 | 9664.65 | 9663.75 | 9699.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 9729.95 | 9676.71 | 9699.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 10:00:00 | 9729.95 | 9676.71 | 9699.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 10:15:00 | 9768.05 | 9694.98 | 9705.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 11:00:00 | 9768.05 | 9694.98 | 9705.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2023-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 11:15:00 | 9806.20 | 9717.22 | 9714.61 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 15:15:00 | 9704.00 | 9738.11 | 9740.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 10:15:00 | 9650.60 | 9717.15 | 9730.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-02 14:15:00 | 9648.05 | 9644.53 | 9685.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-02 15:00:00 | 9648.05 | 9644.53 | 9685.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 09:15:00 | 9568.40 | 9528.75 | 9548.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-08 10:00:00 | 9568.40 | 9528.75 | 9548.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 10:15:00 | 9573.25 | 9537.65 | 9550.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-08 10:30:00 | 9582.85 | 9537.65 | 9550.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 09:15:00 | 9440.80 | 9422.57 | 9456.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-11 10:00:00 | 9440.80 | 9422.57 | 9456.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 13:15:00 | 9372.40 | 9328.07 | 9353.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 14:00:00 | 9372.40 | 9328.07 | 9353.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 14:15:00 | 9408.45 | 9344.14 | 9358.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 15:00:00 | 9408.45 | 9344.14 | 9358.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2023-08-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 09:15:00 | 9463.35 | 9377.30 | 9371.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 09:15:00 | 9482.80 | 9445.59 | 9426.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 09:15:00 | 9474.75 | 9476.22 | 9454.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 09:15:00 | 9474.75 | 9476.22 | 9454.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 09:15:00 | 9474.75 | 9476.22 | 9454.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 10:00:00 | 9474.75 | 9476.22 | 9454.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 9599.90 | 9584.71 | 9553.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-28 11:00:00 | 9627.50 | 9575.50 | 9561.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-28 13:15:00 | 9624.95 | 9589.95 | 9571.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-29 09:30:00 | 9650.20 | 9606.45 | 9586.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-29 10:00:00 | 9629.00 | 9606.45 | 9586.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 09:15:00 | 9693.00 | 9634.54 | 9610.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-30 10:30:00 | 9706.00 | 9644.84 | 9617.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-30 12:00:00 | 9708.95 | 9657.66 | 9625.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-06 13:15:00 | 10136.00 | 10210.04 | 10219.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2023-09-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 13:15:00 | 10136.00 | 10210.04 | 10219.14 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-09-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 09:15:00 | 10272.00 | 10222.63 | 10222.46 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-09-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 12:15:00 | 10214.00 | 10222.65 | 10222.73 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-09-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 14:15:00 | 10270.70 | 10231.24 | 10226.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-07 15:15:00 | 10293.00 | 10243.60 | 10232.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 09:15:00 | 10444.50 | 10471.91 | 10402.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-12 09:45:00 | 10442.70 | 10471.91 | 10402.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 12:15:00 | 10491.50 | 10472.31 | 10420.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-12 13:15:00 | 10517.00 | 10472.31 | 10420.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-14 09:15:00 | 10511.50 | 10474.01 | 10456.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-15 09:30:00 | 10537.50 | 10500.54 | 10483.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-20 09:15:00 | 10440.20 | 10515.46 | 10515.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2023-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 09:15:00 | 10440.20 | 10515.46 | 10515.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 10:15:00 | 10384.80 | 10489.33 | 10503.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 09:15:00 | 10455.00 | 10329.84 | 10371.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 09:15:00 | 10455.00 | 10329.84 | 10371.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 10455.00 | 10329.84 | 10371.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 10:00:00 | 10455.00 | 10329.84 | 10371.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 10436.00 | 10351.08 | 10377.07 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 12:15:00 | 10507.20 | 10403.01 | 10397.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-22 13:15:00 | 10557.00 | 10433.81 | 10411.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 09:15:00 | 10539.80 | 10574.51 | 10525.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-26 10:00:00 | 10539.80 | 10574.51 | 10525.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 11:15:00 | 10535.00 | 10564.43 | 10529.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 12:00:00 | 10535.00 | 10564.43 | 10529.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 12:15:00 | 10560.00 | 10563.54 | 10532.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 12:30:00 | 10536.00 | 10563.54 | 10532.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 10551.70 | 10559.94 | 10540.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 12:15:00 | 10614.00 | 10572.41 | 10549.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 13:45:00 | 10620.80 | 10584.74 | 10559.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-28 13:30:00 | 10613.50 | 10624.56 | 10600.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 11:00:00 | 10611.00 | 10594.72 | 10590.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 14:15:00 | 10591.20 | 10625.75 | 10609.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-29 15:00:00 | 10591.20 | 10625.75 | 10609.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 15:15:00 | 10586.70 | 10617.94 | 10607.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 09:15:00 | 10413.80 | 10617.94 | 10607.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-10-03 09:15:00 | 10329.50 | 10560.25 | 10582.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2023-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 09:15:00 | 10329.50 | 10560.25 | 10582.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 09:15:00 | 10155.90 | 10355.85 | 10452.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 13:15:00 | 10214.50 | 10183.98 | 10263.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-05 13:30:00 | 10205.70 | 10183.98 | 10263.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 15:15:00 | 10235.10 | 10193.87 | 10254.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 09:15:00 | 10319.00 | 10193.87 | 10254.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 10316.40 | 10218.38 | 10259.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 09:30:00 | 10305.70 | 10218.38 | 10259.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2023-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 15:15:00 | 10300.00 | 10277.55 | 10276.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-09 09:15:00 | 10304.00 | 10282.84 | 10278.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 11:15:00 | 10253.50 | 10279.19 | 10278.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 11:15:00 | 10253.50 | 10279.19 | 10278.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 11:15:00 | 10253.50 | 10279.19 | 10278.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-09 12:00:00 | 10253.50 | 10279.19 | 10278.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2023-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 12:15:00 | 10219.50 | 10267.25 | 10272.78 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 09:15:00 | 10399.50 | 10280.13 | 10275.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 09:15:00 | 10413.50 | 10360.04 | 10326.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-16 10:15:00 | 10668.20 | 10674.21 | 10591.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-16 11:00:00 | 10668.20 | 10674.21 | 10591.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 10664.80 | 10688.81 | 10636.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-17 10:00:00 | 10664.80 | 10688.81 | 10636.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 10:15:00 | 10673.80 | 10685.81 | 10640.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-17 10:30:00 | 10659.00 | 10685.81 | 10640.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 09:15:00 | 10761.90 | 10763.64 | 10727.11 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 15:15:00 | 10716.00 | 10739.52 | 10739.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 09:15:00 | 10650.20 | 10721.65 | 10731.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 10:15:00 | 10630.00 | 10625.06 | 10664.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-25 10:45:00 | 10620.30 | 10625.06 | 10664.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 14:15:00 | 10595.00 | 10590.35 | 10632.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-25 14:45:00 | 10595.80 | 10590.35 | 10632.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 15:15:00 | 10620.00 | 10596.28 | 10631.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-26 09:15:00 | 10513.00 | 10596.28 | 10631.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-27 13:15:00 | 10778.80 | 10556.37 | 10557.04 | SL hit (close>static) qty=1.00 sl=10635.00 alert=retest2 |

### Cycle 30 — BUY (started 2023-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 12:15:00 | 10306.40 | 10292.75 | 10291.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-08 14:15:00 | 10331.00 | 10305.08 | 10297.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-10 09:15:00 | 10340.70 | 10377.47 | 10352.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 09:15:00 | 10340.70 | 10377.47 | 10352.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 10340.70 | 10377.47 | 10352.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 09:30:00 | 10349.70 | 10377.47 | 10352.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 10:15:00 | 10378.10 | 10377.60 | 10354.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-10 11:15:00 | 10392.00 | 10377.60 | 10354.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-10 15:15:00 | 10395.00 | 10383.39 | 10365.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-13 10:45:00 | 10392.60 | 10392.18 | 10376.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-13 13:00:00 | 10392.00 | 10389.96 | 10377.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 10410.00 | 10398.56 | 10386.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-16 13:15:00 | 10510.80 | 10452.18 | 10426.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-17 09:15:00 | 10541.50 | 10469.49 | 10441.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-22 11:15:00 | 10447.50 | 10511.63 | 10518.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2023-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 11:15:00 | 10447.50 | 10511.63 | 10518.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-23 13:15:00 | 10415.00 | 10457.77 | 10482.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 14:15:00 | 10498.60 | 10465.94 | 10483.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 14:15:00 | 10498.60 | 10465.94 | 10483.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 14:15:00 | 10498.60 | 10465.94 | 10483.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 15:00:00 | 10498.60 | 10465.94 | 10483.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 15:15:00 | 10490.00 | 10470.75 | 10484.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 09:15:00 | 10507.30 | 10470.75 | 10484.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 10:15:00 | 10494.90 | 10487.17 | 10490.15 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 12:15:00 | 10500.20 | 10491.97 | 10491.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 09:15:00 | 10537.00 | 10507.21 | 10500.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-28 13:15:00 | 10519.20 | 10519.24 | 10509.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-28 14:00:00 | 10519.20 | 10519.24 | 10509.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 14:15:00 | 10535.30 | 10522.45 | 10511.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 14:45:00 | 10515.70 | 10522.45 | 10511.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 12:15:00 | 10552.00 | 10566.21 | 10549.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 13:00:00 | 10552.00 | 10566.21 | 10549.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 13:15:00 | 10573.40 | 10567.65 | 10552.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-30 14:45:00 | 10601.00 | 10574.26 | 10556.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-04 09:15:00 | 10485.00 | 10591.50 | 10589.84 | SL hit (close<static) qty=1.00 sl=10547.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-04 10:15:00 | 10495.00 | 10572.20 | 10581.21 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-05 09:15:00 | 10737.00 | 10604.53 | 10590.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 09:15:00 | 10768.80 | 10667.53 | 10650.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-07 13:15:00 | 10690.00 | 10698.30 | 10673.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-07 13:15:00 | 10690.00 | 10698.30 | 10673.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 13:15:00 | 10690.00 | 10698.30 | 10673.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-07 13:30:00 | 10680.00 | 10698.30 | 10673.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 15:15:00 | 10690.00 | 10697.71 | 10677.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 09:15:00 | 10680.00 | 10697.71 | 10677.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 09:15:00 | 10662.00 | 10690.57 | 10676.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 09:30:00 | 10666.50 | 10690.57 | 10676.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 10:15:00 | 10619.20 | 10676.30 | 10671.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 11:00:00 | 10619.20 | 10676.30 | 10671.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2023-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 11:15:00 | 10601.20 | 10661.28 | 10664.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-11 10:15:00 | 10585.80 | 10624.94 | 10642.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 14:15:00 | 10381.80 | 10370.60 | 10440.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-13 15:00:00 | 10381.80 | 10370.60 | 10440.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 10378.00 | 10378.23 | 10432.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-14 10:30:00 | 10327.50 | 10375.46 | 10426.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-14 12:30:00 | 10337.20 | 10369.85 | 10414.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-15 10:30:00 | 10327.50 | 10360.26 | 10393.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-22 13:15:00 | 10245.50 | 10172.26 | 10163.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2023-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 13:15:00 | 10245.50 | 10172.26 | 10163.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 14:15:00 | 10271.40 | 10230.34 | 10201.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 12:15:00 | 10245.90 | 10251.00 | 10224.80 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 15:00:00 | 10287.30 | 10262.18 | 10234.67 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 10:15:00 | 10218.70 | 10256.02 | 10239.03 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-12-28 10:15:00 | 10218.70 | 10256.02 | 10239.03 | SL hit (close<ema400) qty=1.00 sl=10239.03 alert=retest1 |

### Cycle 37 — SELL (started 2024-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 10:15:00 | 10191.70 | 10254.77 | 10263.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-02 13:15:00 | 10160.50 | 10217.86 | 10242.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-05 12:15:00 | 10040.00 | 10016.19 | 10061.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-05 13:00:00 | 10040.00 | 10016.19 | 10061.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 09:15:00 | 10035.00 | 10018.37 | 10048.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-08 11:45:00 | 9993.15 | 10012.90 | 10040.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-09 12:30:00 | 10000.30 | 9982.53 | 10002.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-09 13:45:00 | 9995.45 | 9986.07 | 10001.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-10 09:15:00 | 9950.00 | 10000.48 | 10006.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 10:15:00 | 10008.30 | 9998.43 | 10003.93 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-01-11 09:15:00 | 10088.00 | 10019.37 | 10011.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2024-01-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 09:15:00 | 10088.00 | 10019.37 | 10011.43 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2024-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 09:15:00 | 9967.95 | 10014.45 | 10016.53 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2024-01-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 13:15:00 | 10074.90 | 10007.44 | 10002.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 09:15:00 | 10220.30 | 10073.15 | 10036.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 09:15:00 | 10039.20 | 10136.16 | 10098.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 09:15:00 | 10039.20 | 10136.16 | 10098.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 10039.20 | 10136.16 | 10098.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 10:00:00 | 10039.20 | 10136.16 | 10098.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 10:15:00 | 10057.20 | 10120.37 | 10095.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 11:15:00 | 10043.20 | 10120.37 | 10095.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 13:15:00 | 10083.20 | 10100.92 | 10091.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 14:00:00 | 10083.20 | 10100.92 | 10091.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 14:15:00 | 10064.00 | 10093.53 | 10088.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 14:30:00 | 10048.80 | 10093.53 | 10088.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2024-01-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 15:15:00 | 10000.00 | 10074.83 | 10080.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 9860.05 | 10031.87 | 10060.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 9990.55 | 9960.11 | 10000.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 09:15:00 | 9990.55 | 9960.11 | 10000.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 9990.55 | 9960.11 | 10000.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 09:30:00 | 10008.30 | 9960.11 | 10000.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 10:15:00 | 10007.00 | 9969.48 | 10001.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 11:00:00 | 10007.00 | 9969.48 | 10001.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 11:15:00 | 10007.80 | 9977.15 | 10001.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 11:30:00 | 10012.80 | 9977.15 | 10001.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 12:15:00 | 10020.50 | 9985.82 | 10003.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 13:00:00 | 10020.50 | 9985.82 | 10003.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 14:15:00 | 10031.00 | 9999.84 | 10007.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 15:00:00 | 10031.00 | 9999.84 | 10007.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 15:15:00 | 10050.00 | 10009.87 | 10011.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-20 09:15:00 | 10005.00 | 10009.87 | 10011.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2024-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 10:15:00 | 10020.00 | 10012.29 | 10012.05 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 11:15:00 | 10003.20 | 10010.47 | 10011.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-20 13:15:00 | 9981.00 | 10003.08 | 10007.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 12:15:00 | 9884.15 | 9880.31 | 9919.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-24 13:00:00 | 9884.15 | 9880.31 | 9919.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 13:15:00 | 9917.55 | 9887.75 | 9919.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 14:00:00 | 9917.55 | 9887.75 | 9919.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 9978.20 | 9905.84 | 9924.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 15:00:00 | 9978.20 | 9905.84 | 9924.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 15:15:00 | 9950.00 | 9914.67 | 9926.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:15:00 | 9950.00 | 9914.67 | 9926.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 10:15:00 | 9900.70 | 9916.00 | 9925.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 12:00:00 | 9851.00 | 9903.00 | 9918.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 14:15:00 | 9875.00 | 9898.62 | 9913.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 15:15:00 | 9879.75 | 9898.89 | 9912.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-29 10:15:00 | 9986.55 | 9918.46 | 9918.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2024-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 10:15:00 | 9986.55 | 9918.46 | 9918.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 11:15:00 | 10016.80 | 9938.13 | 9927.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 14:15:00 | 9950.65 | 9982.34 | 9966.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 14:15:00 | 9950.65 | 9982.34 | 9966.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 14:15:00 | 9950.65 | 9982.34 | 9966.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 15:00:00 | 9950.65 | 9982.34 | 9966.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 15:15:00 | 9950.00 | 9975.87 | 9965.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 09:15:00 | 10033.00 | 9975.87 | 9965.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 09:15:00 | 10053.80 | 9991.46 | 9973.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 10:45:00 | 10087.70 | 10006.69 | 9981.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 11:15:00 | 10120.00 | 10006.69 | 9981.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-09 14:15:00 | 10732.60 | 10737.37 | 10737.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2024-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 14:15:00 | 10732.60 | 10737.37 | 10737.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 15:15:00 | 10710.00 | 10731.89 | 10735.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-12 13:15:00 | 10719.90 | 10714.50 | 10724.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-12 14:00:00 | 10719.90 | 10714.50 | 10724.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 09:15:00 | 10778.00 | 10725.26 | 10726.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 10:00:00 | 10778.00 | 10725.26 | 10726.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2024-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 10:15:00 | 10802.20 | 10740.65 | 10733.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-13 13:15:00 | 10810.70 | 10765.46 | 10747.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-14 10:15:00 | 10782.40 | 10785.11 | 10763.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 10:15:00 | 10782.40 | 10785.11 | 10763.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 10:15:00 | 10782.40 | 10785.11 | 10763.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-14 10:30:00 | 10792.80 | 10785.11 | 10763.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 11355.35 | 11447.05 | 11332.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 09:30:00 | 11344.05 | 11447.05 | 11332.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 10:15:00 | 11446.00 | 11446.84 | 11342.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 12:45:00 | 11488.80 | 11429.15 | 11390.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 15:00:00 | 11473.90 | 11441.76 | 11403.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-22 09:15:00 | 11505.30 | 11443.45 | 11407.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-22 10:15:00 | 11471.00 | 11444.51 | 11411.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 10:15:00 | 11460.00 | 11447.61 | 11415.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 11:00:00 | 11460.00 | 11447.61 | 11415.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 13:15:00 | 11509.25 | 11585.21 | 11533.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-23 14:00:00 | 11509.25 | 11585.21 | 11533.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 14:15:00 | 11544.90 | 11577.15 | 11534.96 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-02-26 12:15:00 | 11496.05 | 11517.47 | 11517.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2024-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 12:15:00 | 11496.05 | 11517.47 | 11517.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 11:15:00 | 11435.00 | 11482.83 | 11498.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 14:15:00 | 11500.00 | 11472.85 | 11488.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 14:15:00 | 11500.00 | 11472.85 | 11488.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 14:15:00 | 11500.00 | 11472.85 | 11488.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 15:00:00 | 11500.00 | 11472.85 | 11488.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 15:15:00 | 11456.15 | 11469.51 | 11485.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 09:30:00 | 11425.05 | 11453.03 | 11476.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 11:15:00 | 11450.85 | 11336.68 | 11342.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-01 11:15:00 | 11524.65 | 11374.28 | 11358.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2024-03-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 11:15:00 | 11524.65 | 11374.28 | 11358.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 13:15:00 | 11567.95 | 11436.73 | 11391.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 09:15:00 | 11577.85 | 11638.77 | 11595.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 09:15:00 | 11577.85 | 11638.77 | 11595.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 11577.85 | 11638.77 | 11595.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:00:00 | 11577.85 | 11638.77 | 11595.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 10:15:00 | 11598.00 | 11630.62 | 11595.54 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2024-03-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-07 09:15:00 | 11524.00 | 11579.57 | 11582.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-07 11:15:00 | 11465.20 | 11544.97 | 11565.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-12 09:15:00 | 11458.00 | 11438.34 | 11480.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-12 09:30:00 | 11457.70 | 11438.34 | 11480.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 10:15:00 | 11485.00 | 11447.67 | 11480.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-12 10:45:00 | 11474.95 | 11447.67 | 11480.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 11:15:00 | 11557.95 | 11469.73 | 11487.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-12 12:00:00 | 11557.95 | 11469.73 | 11487.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 12:15:00 | 11548.80 | 11485.54 | 11493.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-12 12:30:00 | 11555.80 | 11485.54 | 11493.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 15:15:00 | 11502.00 | 11495.86 | 11496.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-13 09:15:00 | 11465.45 | 11495.86 | 11496.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 11410.05 | 11478.70 | 11488.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-13 11:15:00 | 11361.20 | 11462.57 | 11480.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-13 13:30:00 | 11360.80 | 11410.37 | 11449.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-14 09:45:00 | 11358.35 | 11394.43 | 11432.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-14 10:45:00 | 11363.05 | 11386.78 | 11425.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 12:15:00 | 11449.45 | 11403.02 | 11426.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 13:00:00 | 11449.45 | 11403.02 | 11426.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 13:15:00 | 11440.00 | 11410.42 | 11427.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 13:30:00 | 11438.35 | 11410.42 | 11427.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 14:15:00 | 11398.95 | 11408.13 | 11424.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 14:30:00 | 11443.25 | 11408.13 | 11424.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 11405.00 | 11402.04 | 11418.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 11:00:00 | 11325.00 | 11386.63 | 11410.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-15 13:15:00 | 11467.35 | 11398.96 | 11409.58 | SL hit (close>static) qty=1.00 sl=11429.00 alert=retest2 |

### Cycle 50 — BUY (started 2024-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 09:15:00 | 11448.75 | 11417.88 | 11416.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-18 13:15:00 | 11506.60 | 11450.21 | 11433.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 09:15:00 | 11503.00 | 11514.03 | 11471.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-19 09:15:00 | 11503.00 | 11514.03 | 11471.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 09:15:00 | 11503.00 | 11514.03 | 11471.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 10:00:00 | 11503.00 | 11514.03 | 11471.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 14:15:00 | 12587.05 | 12592.55 | 12525.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-01 14:30:00 | 12542.10 | 12592.55 | 12525.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 09:15:00 | 12567.15 | 12587.92 | 12534.83 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2024-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-03 11:15:00 | 12452.00 | 12510.91 | 12518.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 09:15:00 | 12380.25 | 12451.34 | 12483.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-04 12:15:00 | 12462.25 | 12429.06 | 12462.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-04 12:15:00 | 12462.25 | 12429.06 | 12462.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 12:15:00 | 12462.25 | 12429.06 | 12462.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-04 13:00:00 | 12462.25 | 12429.06 | 12462.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 13:15:00 | 12599.00 | 12463.05 | 12475.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-04 14:00:00 | 12599.00 | 12463.05 | 12475.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 14:15:00 | 12557.65 | 12481.97 | 12482.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-04 14:45:00 | 12614.90 | 12481.97 | 12482.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2024-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-04 15:15:00 | 12570.00 | 12499.57 | 12490.65 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-04-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 11:15:00 | 12421.80 | 12482.84 | 12485.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-05 14:15:00 | 12417.10 | 12455.45 | 12470.68 | Break + close below crossover candle low |

### Cycle 54 — BUY (started 2024-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 09:15:00 | 12598.45 | 12479.04 | 12478.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 10:15:00 | 12747.10 | 12532.65 | 12502.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-10 09:15:00 | 12828.00 | 12847.99 | 12761.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-10 10:00:00 | 12828.00 | 12847.99 | 12761.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 11:15:00 | 12664.80 | 12804.85 | 12756.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-10 12:00:00 | 12664.80 | 12804.85 | 12756.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 12:15:00 | 12695.60 | 12783.00 | 12750.96 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2024-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 15:15:00 | 12675.00 | 12725.09 | 12729.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 09:15:00 | 12540.00 | 12688.07 | 12711.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-15 14:15:00 | 12415.50 | 12369.42 | 12463.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-15 14:15:00 | 12415.50 | 12369.42 | 12463.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 14:15:00 | 12415.50 | 12369.42 | 12463.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 14:30:00 | 12435.10 | 12369.42 | 12463.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 12419.85 | 12386.00 | 12454.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 09:30:00 | 12525.45 | 12386.00 | 12454.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 12448.35 | 12398.47 | 12454.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:00:00 | 12448.35 | 12398.47 | 12454.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 11:15:00 | 12445.25 | 12407.83 | 12453.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:30:00 | 12459.00 | 12407.83 | 12453.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 12:15:00 | 12444.65 | 12415.19 | 12452.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 12:30:00 | 12440.70 | 12415.19 | 12452.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 13:15:00 | 12511.70 | 12434.49 | 12458.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 14:00:00 | 12511.70 | 12434.49 | 12458.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 14:15:00 | 12509.40 | 12449.47 | 12462.74 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2024-04-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 09:15:00 | 12511.35 | 12471.53 | 12471.07 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-04-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 15:15:00 | 12420.00 | 12471.97 | 12476.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 09:15:00 | 12316.00 | 12440.78 | 12462.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 11:15:00 | 12490.00 | 12420.11 | 12447.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 11:15:00 | 12490.00 | 12420.11 | 12447.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 11:15:00 | 12490.00 | 12420.11 | 12447.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 11:45:00 | 12482.20 | 12420.11 | 12447.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 12:15:00 | 12524.35 | 12440.96 | 12454.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 13:15:00 | 12550.90 | 12440.96 | 12454.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2024-04-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-19 13:15:00 | 12622.20 | 12477.20 | 12469.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-19 14:15:00 | 12706.45 | 12523.05 | 12490.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 12:15:00 | 12945.00 | 12945.26 | 12844.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-24 13:00:00 | 12945.00 | 12945.26 | 12844.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 12804.45 | 12913.13 | 12861.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:00:00 | 12804.45 | 12913.13 | 12861.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 12762.75 | 12883.05 | 12852.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:45:00 | 12762.00 | 12883.05 | 12852.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 12953.80 | 12902.99 | 12873.51 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-04-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 12:15:00 | 12733.35 | 12860.02 | 12860.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-26 14:15:00 | 12696.30 | 12808.07 | 12835.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-30 09:15:00 | 12882.35 | 12745.29 | 12771.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-30 09:15:00 | 12882.35 | 12745.29 | 12771.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 09:15:00 | 12882.35 | 12745.29 | 12771.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 09:45:00 | 12925.00 | 12745.29 | 12771.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 10:15:00 | 12802.95 | 12756.82 | 12774.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 10:45:00 | 12844.25 | 12756.82 | 12774.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2024-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 12:15:00 | 12865.90 | 12796.10 | 12790.25 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 09:15:00 | 12669.75 | 12780.48 | 12786.66 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 11:15:00 | 12824.50 | 12790.62 | 12790.28 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-05-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 09:15:00 | 12728.45 | 12781.76 | 12787.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 10:15:00 | 12682.00 | 12761.81 | 12778.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-07 15:15:00 | 12392.00 | 12380.31 | 12466.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 09:15:00 | 12473.00 | 12380.31 | 12466.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 12415.00 | 12387.25 | 12461.43 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-09 09:15:00 | 12690.00 | 12517.38 | 12495.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 09:15:00 | 12808.80 | 12661.25 | 12615.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 09:15:00 | 12749.85 | 12764.55 | 12703.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 10:00:00 | 12749.85 | 12764.55 | 12703.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 12742.30 | 12760.10 | 12706.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:45:00 | 12705.15 | 12760.10 | 12706.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 12537.45 | 12722.93 | 12715.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 09:30:00 | 12540.05 | 12722.93 | 12715.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2024-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 10:15:00 | 12485.95 | 12675.53 | 12694.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 12:15:00 | 12391.95 | 12593.93 | 12652.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-17 10:15:00 | 12530.00 | 12513.17 | 12581.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-17 11:00:00 | 12530.00 | 12513.17 | 12581.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 12:15:00 | 12615.70 | 12532.68 | 12578.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 12:45:00 | 12624.30 | 12532.68 | 12578.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 13:15:00 | 12605.05 | 12547.16 | 12580.97 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 09:15:00 | 12614.50 | 12599.17 | 12597.54 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 12:15:00 | 12554.95 | 12589.61 | 12594.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 13:15:00 | 12484.80 | 12568.65 | 12584.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 11:15:00 | 12552.40 | 12530.77 | 12555.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 11:15:00 | 12552.40 | 12530.77 | 12555.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 12552.40 | 12530.77 | 12555.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:45:00 | 12543.10 | 12530.77 | 12555.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 12530.70 | 12530.76 | 12552.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-22 14:00:00 | 12501.60 | 12524.93 | 12548.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 09:15:00 | 12501.00 | 12522.97 | 12543.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 09:15:00 | 12590.00 | 12536.37 | 12547.40 | SL hit (close>static) qty=1.00 sl=12561.00 alert=retest2 |

### Cycle 68 — BUY (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 11:15:00 | 12665.00 | 12571.82 | 12562.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 12:15:00 | 12851.05 | 12627.67 | 12588.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 09:15:00 | 12805.60 | 12907.78 | 12814.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 09:15:00 | 12805.60 | 12907.78 | 12814.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 12805.60 | 12907.78 | 12814.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:00:00 | 12805.60 | 12907.78 | 12814.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 12875.40 | 12901.31 | 12819.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 11:45:00 | 12928.55 | 12903.72 | 12828.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 12:15:00 | 12920.10 | 12903.72 | 12828.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 11:00:00 | 12973.00 | 12919.78 | 12869.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 13:00:00 | 12927.45 | 12917.28 | 12877.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 12896.40 | 12913.10 | 12878.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:00:00 | 12896.40 | 12913.10 | 12878.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 12819.00 | 12894.28 | 12873.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 15:00:00 | 12819.00 | 12894.28 | 12873.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 12833.95 | 12882.22 | 12869.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:15:00 | 12791.65 | 12882.22 | 12869.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-29 10:15:00 | 12815.90 | 12853.09 | 12857.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 10:15:00 | 12815.90 | 12853.09 | 12857.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 12733.30 | 12793.63 | 12822.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 12:15:00 | 12515.00 | 12495.98 | 12573.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-03 13:00:00 | 12515.00 | 12495.98 | 12573.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 12364.75 | 12463.91 | 12533.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 11:00:00 | 12104.40 | 12392.01 | 12494.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 11:45:00 | 12192.65 | 12318.60 | 12451.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 11583.02 | 12297.87 | 12430.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 15:15:00 | 12204.65 | 12252.45 | 12384.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-05 09:15:00 | 12555.15 | 12305.35 | 12385.24 | SL hit (close>ema200) qty=0.50 sl=12305.35 alert=retest2 |

### Cycle 70 — BUY (started 2024-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 13:15:00 | 12517.25 | 12432.06 | 12428.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 12569.05 | 12481.12 | 12453.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 11:15:00 | 12721.40 | 12738.68 | 12667.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 12:00:00 | 12721.40 | 12738.68 | 12667.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 12:15:00 | 12648.25 | 12720.60 | 12666.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 13:00:00 | 12648.25 | 12720.60 | 12666.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 13:15:00 | 12726.05 | 12721.69 | 12671.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 13:30:00 | 12646.05 | 12721.69 | 12671.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 12691.05 | 12715.56 | 12673.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:30:00 | 12682.00 | 12715.56 | 12673.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 12719.00 | 12720.32 | 12683.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:45:00 | 12686.20 | 12720.32 | 12683.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 12830.00 | 12850.00 | 12812.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 10:00:00 | 12830.00 | 12850.00 | 12812.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 12767.35 | 12833.47 | 12808.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 10:30:00 | 12742.60 | 12833.47 | 12808.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 12800.05 | 12826.78 | 12807.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:15:00 | 12820.30 | 12826.78 | 12807.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:45:00 | 12811.65 | 12825.38 | 12814.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 11:15:00 | 12830.10 | 12820.70 | 12813.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-18 09:15:00 | 12660.00 | 12803.90 | 12812.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2024-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 09:15:00 | 12660.00 | 12803.90 | 12812.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 14:15:00 | 12551.90 | 12662.50 | 12731.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 11:15:00 | 12203.85 | 12197.24 | 12312.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-21 12:00:00 | 12203.85 | 12197.24 | 12312.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 12202.60 | 12199.22 | 12263.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 12:15:00 | 12181.25 | 12198.90 | 12257.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 09:15:00 | 12158.95 | 12197.13 | 12236.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 13:15:00 | 12279.20 | 12215.69 | 12208.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2024-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 13:15:00 | 12279.20 | 12215.69 | 12208.45 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 09:15:00 | 12096.30 | 12187.26 | 12196.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 14:15:00 | 12021.25 | 12114.06 | 12143.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 09:15:00 | 12114.20 | 12099.84 | 12131.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 12114.20 | 12099.84 | 12131.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 12114.20 | 12099.84 | 12131.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:30:00 | 12175.25 | 12099.84 | 12131.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 12073.00 | 12094.47 | 12125.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 12:15:00 | 12055.00 | 12087.78 | 12120.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 13:00:00 | 12050.00 | 12080.22 | 12113.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 13:15:00 | 12206.70 | 12105.52 | 12122.14 | SL hit (close>static) qty=1.00 sl=12143.05 alert=retest2 |

### Cycle 74 — BUY (started 2024-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 15:15:00 | 12134.00 | 12090.13 | 12086.70 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 13:15:00 | 12069.40 | 12089.20 | 12091.61 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-07-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 14:15:00 | 12125.35 | 12096.43 | 12094.68 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 09:15:00 | 12027.00 | 12083.11 | 12088.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 11:15:00 | 11988.25 | 12054.26 | 12074.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 15:15:00 | 12045.00 | 12034.86 | 12056.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-09 09:15:00 | 12492.25 | 12034.86 | 12056.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 78 — BUY (started 2024-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 09:15:00 | 12583.00 | 12144.49 | 12104.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 10:15:00 | 12712.80 | 12258.15 | 12159.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 14:15:00 | 12767.30 | 12829.67 | 12631.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-10 14:45:00 | 12766.75 | 12829.67 | 12631.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 14:15:00 | 12720.00 | 12785.51 | 12704.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 15:00:00 | 12720.00 | 12785.51 | 12704.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 12730.25 | 12774.46 | 12706.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:15:00 | 12580.00 | 12774.46 | 12706.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 12577.00 | 12734.97 | 12695.07 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2024-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 12:15:00 | 12563.05 | 12669.57 | 12672.09 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 13:15:00 | 12675.95 | 12658.49 | 12656.76 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 14:15:00 | 12640.20 | 12654.83 | 12655.25 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 10:15:00 | 12681.30 | 12658.72 | 12656.74 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 14:15:00 | 12628.00 | 12651.73 | 12654.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 09:15:00 | 12575.20 | 12634.55 | 12646.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 13:15:00 | 12665.75 | 12599.59 | 12621.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 13:15:00 | 12665.75 | 12599.59 | 12621.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 12665.75 | 12599.59 | 12621.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:00:00 | 12665.75 | 12599.59 | 12621.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 12646.30 | 12608.93 | 12623.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 09:15:00 | 12607.55 | 12617.43 | 12625.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 10:30:00 | 12605.00 | 12624.35 | 12627.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 11:30:00 | 12596.80 | 12615.88 | 12623.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 11:00:00 | 12572.20 | 12573.11 | 12593.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 12586.00 | 12575.68 | 12593.04 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-22 14:15:00 | 12651.85 | 12606.94 | 12604.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2024-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 14:15:00 | 12651.85 | 12606.94 | 12604.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 10:15:00 | 12680.00 | 12624.18 | 12613.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 12:15:00 | 12550.60 | 12623.03 | 12615.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 12:15:00 | 12550.60 | 12623.03 | 12615.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 12550.60 | 12623.03 | 12615.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 12536.50 | 12623.03 | 12615.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 12607.05 | 12619.84 | 12614.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:45:00 | 12571.25 | 12619.84 | 12614.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 12611.70 | 12618.21 | 12614.33 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 09:15:00 | 12483.75 | 12592.47 | 12603.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 10:15:00 | 12455.00 | 12510.22 | 12549.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-25 14:15:00 | 12498.45 | 12491.75 | 12527.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-25 15:00:00 | 12498.45 | 12491.75 | 12527.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 12490.00 | 12491.40 | 12524.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 12424.95 | 12491.40 | 12524.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 09:15:00 | 12577.05 | 12508.53 | 12528.86 | SL hit (close>static) qty=1.00 sl=12525.00 alert=retest2 |

### Cycle 86 — BUY (started 2024-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 11:15:00 | 12647.00 | 12542.16 | 12540.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 12:15:00 | 12736.60 | 12581.05 | 12558.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 09:15:00 | 12950.00 | 13182.81 | 13078.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 09:15:00 | 12950.00 | 13182.81 | 13078.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 12950.00 | 13182.81 | 13078.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 10:15:00 | 12829.50 | 13182.81 | 13078.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 12841.60 | 13114.57 | 13056.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 10:45:00 | 12850.70 | 13114.57 | 13056.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2024-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 12:15:00 | 12756.80 | 12988.51 | 13005.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 15:15:00 | 12697.10 | 12863.24 | 12938.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 12386.95 | 12359.30 | 12570.48 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 11:00:00 | 12322.00 | 12351.84 | 12547.89 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 12:30:00 | 12305.15 | 12339.81 | 12508.46 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 12365.00 | 12290.07 | 12410.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:30:00 | 12384.30 | 12290.07 | 12410.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 12261.00 | 12319.60 | 12376.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 11:30:00 | 12213.65 | 12298.37 | 12356.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:00:00 | 12200.65 | 12278.83 | 12342.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 15:00:00 | 12217.95 | 12255.23 | 12319.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 14:15:00 | 12215.60 | 12246.54 | 12286.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 12286.70 | 12245.61 | 12275.23 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-12 09:15:00 | 12286.70 | 12245.61 | 12275.23 | SL hit (close>ema400) qty=1.00 sl=12275.23 alert=retest1 |

### Cycle 88 — BUY (started 2024-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 10:15:00 | 12301.85 | 12286.34 | 12285.94 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 11:15:00 | 12265.00 | 12282.07 | 12284.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 12:15:00 | 12213.85 | 12268.43 | 12277.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 12:15:00 | 12214.35 | 12196.68 | 12227.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 13:00:00 | 12214.35 | 12196.68 | 12227.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 15:15:00 | 12211.65 | 12200.64 | 12221.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:15:00 | 12300.25 | 12200.64 | 12221.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 12184.00 | 12197.31 | 12218.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 10:15:00 | 12131.70 | 12197.31 | 12218.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 12:15:00 | 12180.00 | 12182.05 | 12206.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 10:30:00 | 12175.00 | 12196.53 | 12205.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 11:30:00 | 12158.60 | 12187.42 | 12200.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 12234.85 | 12180.52 | 12190.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 10:00:00 | 12234.85 | 12180.52 | 12190.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 12246.00 | 12193.62 | 12195.50 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-20 11:15:00 | 12238.65 | 12202.62 | 12199.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2024-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 11:15:00 | 12238.65 | 12202.62 | 12199.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 13:15:00 | 12285.00 | 12224.86 | 12210.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 15:15:00 | 12208.00 | 12229.95 | 12215.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 15:15:00 | 12208.00 | 12229.95 | 12215.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 12208.00 | 12229.95 | 12215.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 15:15:00 | 12305.05 | 12239.84 | 12228.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 09:45:00 | 12290.00 | 12261.83 | 12240.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 14:45:00 | 12300.55 | 12295.35 | 12268.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:30:00 | 12305.60 | 12290.01 | 12270.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 12270.70 | 12286.15 | 12270.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:00:00 | 12270.70 | 12286.15 | 12270.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 11:15:00 | 12285.00 | 12285.92 | 12272.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:30:00 | 12278.55 | 12285.92 | 12272.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 12234.05 | 12278.55 | 12272.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 15:00:00 | 12234.05 | 12278.55 | 12272.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 15:15:00 | 12250.05 | 12272.85 | 12270.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 09:15:00 | 12255.00 | 12272.85 | 12270.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 09:15:00 | 12292.15 | 12361.64 | 12361.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 09:15:00 | 12292.15 | 12361.64 | 12361.74 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 14:15:00 | 12458.40 | 12363.93 | 12359.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 09:15:00 | 12494.30 | 12403.77 | 12379.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 12:15:00 | 12405.00 | 12424.23 | 12396.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 12:15:00 | 12405.00 | 12424.23 | 12396.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 12:15:00 | 12405.00 | 12424.23 | 12396.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 13:00:00 | 12405.00 | 12424.23 | 12396.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 13:15:00 | 12421.85 | 12423.75 | 12398.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 14:45:00 | 12451.00 | 12424.92 | 12401.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 15:15:00 | 12461.80 | 12424.92 | 12401.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 10:45:00 | 12452.85 | 12431.21 | 12411.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 11:15:00 | 12445.50 | 12431.21 | 12411.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 12374.75 | 12419.92 | 12407.83 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-02 11:15:00 | 12374.75 | 12419.92 | 12407.83 | SL hit (close<static) qty=1.00 sl=12395.05 alert=retest2 |

### Cycle 93 — SELL (started 2024-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 15:15:00 | 12390.00 | 12406.24 | 12407.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 09:15:00 | 12324.35 | 12389.86 | 12400.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 15:15:00 | 12370.00 | 12339.82 | 12363.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 15:15:00 | 12370.00 | 12339.82 | 12363.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 15:15:00 | 12370.00 | 12339.82 | 12363.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 09:15:00 | 12319.20 | 12339.82 | 12363.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 15:15:00 | 12278.90 | 12202.30 | 12201.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2024-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 15:15:00 | 12278.90 | 12202.30 | 12201.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 09:15:00 | 12298.80 | 12221.60 | 12210.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 12205.10 | 12245.03 | 12228.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 13:15:00 | 12205.10 | 12245.03 | 12228.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 12205.10 | 12245.03 | 12228.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 12205.10 | 12245.03 | 12228.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 12227.95 | 12241.61 | 12228.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 15:15:00 | 12260.55 | 12241.61 | 12228.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 09:15:00 | 12193.70 | 12235.06 | 12227.78 | SL hit (close<static) qty=1.00 sl=12201.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 09:15:00 | 12200.20 | 12273.06 | 12280.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 15:15:00 | 12170.20 | 12219.57 | 12239.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 12324.00 | 12240.46 | 12247.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 12324.00 | 12240.46 | 12247.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 12324.00 | 12240.46 | 12247.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:45:00 | 12320.00 | 12240.46 | 12247.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 10:15:00 | 12347.20 | 12261.81 | 12256.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 09:15:00 | 12552.00 | 12363.90 | 12313.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 12670.30 | 12710.39 | 12652.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 10:00:00 | 12670.30 | 12710.39 | 12652.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 12643.10 | 12696.93 | 12651.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:30:00 | 12634.00 | 12696.93 | 12651.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 12665.00 | 12690.55 | 12652.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 14:00:00 | 12719.90 | 12691.05 | 12659.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 10:15:00 | 13006.00 | 13224.14 | 13228.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 10:15:00 | 13006.00 | 13224.14 | 13228.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 09:15:00 | 12926.80 | 13115.41 | 13166.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 09:15:00 | 12834.40 | 12832.08 | 12970.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 10:00:00 | 12834.40 | 12832.08 | 12970.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 12682.85 | 12568.85 | 12612.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:30:00 | 12739.80 | 12568.85 | 12612.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 12712.00 | 12597.48 | 12621.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:30:00 | 12705.00 | 12597.48 | 12621.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 11:15:00 | 12692.00 | 12616.38 | 12627.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 12:15:00 | 12734.90 | 12616.38 | 12627.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2024-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 12:15:00 | 12780.00 | 12649.11 | 12641.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 13:15:00 | 12848.05 | 12688.90 | 12660.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 12:15:00 | 12850.00 | 12869.79 | 12810.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 13:00:00 | 12850.00 | 12869.79 | 12810.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 13:15:00 | 12799.20 | 12855.67 | 12809.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 14:00:00 | 12799.20 | 12855.67 | 12809.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 12796.30 | 12843.80 | 12808.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 15:00:00 | 12796.30 | 12843.80 | 12808.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 12792.10 | 12833.46 | 12807.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:15:00 | 12735.00 | 12833.46 | 12807.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — SELL (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 10:15:00 | 12569.05 | 12751.22 | 12772.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 14:15:00 | 12548.05 | 12645.26 | 12709.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 14:15:00 | 12380.00 | 12370.31 | 12462.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-16 15:00:00 | 12380.00 | 12370.31 | 12462.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 12032.65 | 12097.44 | 12178.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 10:15:00 | 11902.40 | 12104.93 | 12147.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 14:15:00 | 11922.25 | 12042.62 | 12102.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 09:30:00 | 11912.05 | 11997.86 | 12063.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 14:15:00 | 11934.20 | 12017.50 | 12055.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 09:15:00 | 11307.28 | 11472.57 | 11585.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 09:15:00 | 11326.14 | 11472.57 | 11585.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 09:15:00 | 11316.45 | 11472.57 | 11585.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 09:15:00 | 11337.49 | 11472.57 | 11585.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 11155.00 | 11167.11 | 11349.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-30 10:15:00 | 11346.55 | 11203.00 | 11348.87 | SL hit (close>ema200) qty=0.50 sl=11203.00 alert=retest2 |

### Cycle 100 — BUY (started 2024-11-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 13:15:00 | 11187.20 | 11139.34 | 11137.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 11286.55 | 11179.66 | 11157.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 11236.90 | 11297.71 | 11245.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 11236.90 | 11297.71 | 11245.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 11236.90 | 11297.71 | 11245.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 11236.90 | 11297.71 | 11245.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 11241.05 | 11286.38 | 11244.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:00:00 | 11241.05 | 11286.38 | 11244.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 11281.10 | 11285.32 | 11248.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 14:30:00 | 11325.00 | 11280.76 | 11266.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 09:15:00 | 11453.55 | 11285.01 | 11269.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-12 09:45:00 | 11362.20 | 11392.39 | 11353.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-12 11:00:00 | 11323.45 | 11378.60 | 11350.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 11317.45 | 11366.37 | 11347.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 12:00:00 | 11317.45 | 11366.37 | 11347.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-12 12:15:00 | 11215.05 | 11336.11 | 11335.39 | SL hit (close<static) qty=1.00 sl=11235.05 alert=retest2 |

### Cycle 101 — SELL (started 2024-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 13:15:00 | 11182.30 | 11305.35 | 11321.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 14:15:00 | 11135.90 | 11271.46 | 11304.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 15:15:00 | 11092.90 | 11091.21 | 11169.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 09:15:00 | 10966.50 | 11091.21 | 11169.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 11:15:00 | 11117.70 | 11036.29 | 11074.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 11:45:00 | 11112.00 | 11036.29 | 11074.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 11137.00 | 11056.43 | 11080.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:45:00 | 11153.25 | 11056.43 | 11080.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 15:15:00 | 11116.60 | 11084.91 | 11088.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:15:00 | 11157.60 | 11084.91 | 11088.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 11158.55 | 11099.64 | 11095.17 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 10980.10 | 11090.13 | 11095.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 15:15:00 | 10940.00 | 11060.10 | 11080.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 11:15:00 | 10944.90 | 10918.66 | 10969.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 12:00:00 | 10944.90 | 10918.66 | 10969.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 10987.65 | 10932.46 | 10970.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 13:00:00 | 10987.65 | 10932.46 | 10970.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 11045.20 | 10955.01 | 10977.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:00:00 | 11045.20 | 10955.01 | 10977.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 11087.25 | 10981.45 | 10987.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:45:00 | 11075.00 | 10981.45 | 10987.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2024-11-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 15:15:00 | 11070.05 | 10999.17 | 10995.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 11117.95 | 11022.93 | 11006.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 14:15:00 | 11019.85 | 11077.44 | 11047.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 14:15:00 | 11019.85 | 11077.44 | 11047.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 11019.85 | 11077.44 | 11047.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 15:00:00 | 11019.85 | 11077.44 | 11047.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 11003.20 | 11062.59 | 11043.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 09:15:00 | 11050.00 | 11062.59 | 11043.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 10942.00 | 11038.47 | 11034.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 10:00:00 | 10942.00 | 11038.47 | 11034.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 11031.00 | 11036.98 | 11033.78 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2024-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 11:15:00 | 10974.85 | 11024.55 | 11028.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 12:15:00 | 10939.15 | 11007.47 | 11020.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 09:15:00 | 11036.85 | 10985.37 | 11002.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 09:15:00 | 11036.85 | 10985.37 | 11002.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 11036.85 | 10985.37 | 11002.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:00:00 | 11036.85 | 10985.37 | 11002.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 11061.05 | 11000.50 | 11007.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:45:00 | 11085.00 | 11000.50 | 11007.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2024-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 11:15:00 | 11069.90 | 11014.38 | 11013.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 12:15:00 | 11105.10 | 11032.53 | 11021.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 09:15:00 | 11022.70 | 11044.80 | 11032.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 09:15:00 | 11022.70 | 11044.80 | 11032.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 11022.70 | 11044.80 | 11032.52 | EMA400 retest candle locked (from upside) |

### Cycle 107 — SELL (started 2024-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 11:15:00 | 10969.90 | 11017.77 | 11021.62 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2024-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 11:15:00 | 11094.55 | 11016.25 | 11012.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 09:15:00 | 11268.70 | 11096.59 | 11055.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 11:15:00 | 11222.05 | 11223.30 | 11165.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-03 12:00:00 | 11222.05 | 11223.30 | 11165.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 11199.00 | 11238.96 | 11202.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:45:00 | 11199.95 | 11238.96 | 11202.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 11167.50 | 11224.67 | 11199.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:00:00 | 11167.50 | 11224.67 | 11199.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 11180.95 | 11215.93 | 11197.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:45:00 | 11165.70 | 11215.93 | 11197.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 13:15:00 | 11175.85 | 11207.91 | 11195.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 13:45:00 | 11176.05 | 11207.91 | 11195.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 11139.00 | 11194.13 | 11190.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 15:00:00 | 11139.00 | 11194.13 | 11190.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2024-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 15:15:00 | 11123.00 | 11179.90 | 11184.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 09:15:00 | 11070.50 | 11158.02 | 11173.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 13:15:00 | 11140.00 | 11136.35 | 11156.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 13:15:00 | 11140.00 | 11136.35 | 11156.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 13:15:00 | 11140.00 | 11136.35 | 11156.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:00:00 | 11140.00 | 11136.35 | 11156.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 11214.00 | 11151.88 | 11161.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:30:00 | 11235.15 | 11151.88 | 11161.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 15:15:00 | 11176.00 | 11156.70 | 11163.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:15:00 | 11204.05 | 11156.70 | 11163.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 11170.00 | 11156.00 | 11161.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 11:00:00 | 11170.00 | 11156.00 | 11161.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2024-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 11:15:00 | 11250.00 | 11174.80 | 11169.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 12:15:00 | 11331.15 | 11206.07 | 11184.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 14:15:00 | 11277.00 | 11279.03 | 11247.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-09 14:45:00 | 11275.25 | 11279.03 | 11247.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 11286.80 | 11277.54 | 11251.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 10:15:00 | 11301.20 | 11277.54 | 11251.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 14:15:00 | 11198.85 | 11239.95 | 11242.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2024-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 14:15:00 | 11198.85 | 11239.95 | 11242.17 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 09:15:00 | 11272.20 | 11243.17 | 11243.05 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 10:15:00 | 11120.00 | 11230.49 | 11243.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 11088.90 | 11161.67 | 11199.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 11:15:00 | 11238.65 | 11167.73 | 11194.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 11:15:00 | 11238.65 | 11167.73 | 11194.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 11238.65 | 11167.73 | 11194.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:00:00 | 11238.65 | 11167.73 | 11194.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 11217.05 | 11177.60 | 11196.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:00:00 | 11217.05 | 11177.60 | 11196.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 11273.35 | 11196.75 | 11203.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:00:00 | 11273.35 | 11196.75 | 11203.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2024-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 14:15:00 | 11270.35 | 11211.47 | 11209.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 09:15:00 | 11298.20 | 11238.38 | 11222.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 13:15:00 | 11248.35 | 11250.95 | 11234.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-16 13:45:00 | 11249.90 | 11250.95 | 11234.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 11196.75 | 11244.50 | 11236.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:00:00 | 11196.75 | 11244.50 | 11236.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 11178.85 | 11231.37 | 11230.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:30:00 | 11180.20 | 11231.37 | 11230.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — SELL (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 11:15:00 | 11140.00 | 11213.10 | 11222.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 11074.65 | 11174.51 | 11202.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 11:15:00 | 11008.25 | 11006.41 | 11066.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 11:30:00 | 11018.05 | 11006.41 | 11066.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 10981.50 | 10978.20 | 11027.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:30:00 | 11061.70 | 10978.20 | 11027.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 11045.45 | 10991.65 | 11029.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 11:00:00 | 11045.45 | 10991.65 | 11029.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 11048.00 | 11002.92 | 11030.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 12:00:00 | 11048.00 | 11002.92 | 11030.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 10940.00 | 10990.34 | 11022.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 12:30:00 | 11012.10 | 10990.34 | 11022.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 10877.45 | 10931.72 | 10974.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 12:45:00 | 10841.10 | 10908.01 | 10959.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 09:30:00 | 10825.00 | 10860.00 | 10916.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 11:30:00 | 10843.00 | 10853.49 | 10903.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 15:15:00 | 10911.80 | 10884.07 | 10882.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2024-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 15:15:00 | 10911.80 | 10884.07 | 10882.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 11103.40 | 10927.94 | 10902.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 14:15:00 | 10934.15 | 10964.87 | 10936.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 14:15:00 | 10934.15 | 10964.87 | 10936.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 10934.15 | 10964.87 | 10936.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 15:00:00 | 10934.15 | 10964.87 | 10936.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 10957.95 | 10963.49 | 10938.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:15:00 | 10870.60 | 10963.49 | 10938.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 10829.90 | 10936.77 | 10928.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:45:00 | 10820.10 | 10936.77 | 10928.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 10897.00 | 10928.82 | 10925.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 11:30:00 | 10924.00 | 10924.41 | 10923.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 12:15:00 | 10898.55 | 10919.24 | 10921.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2024-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 12:15:00 | 10898.55 | 10919.24 | 10921.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 10856.45 | 10906.68 | 10915.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 10887.30 | 10846.00 | 10870.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 13:15:00 | 10887.30 | 10846.00 | 10870.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 10887.30 | 10846.00 | 10870.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:00:00 | 10887.30 | 10846.00 | 10870.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 10843.65 | 10845.53 | 10867.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 09:30:00 | 10831.95 | 10845.02 | 10863.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 10:15:00 | 10910.25 | 10858.07 | 10868.11 | SL hit (close>static) qty=1.00 sl=10893.70 alert=retest2 |

### Cycle 118 — BUY (started 2025-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 13:15:00 | 11128.00 | 10914.59 | 10891.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 14:15:00 | 11193.40 | 10970.35 | 10919.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 10:15:00 | 11796.95 | 11866.27 | 11673.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 11:00:00 | 11796.95 | 11866.27 | 11673.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 11742.05 | 11804.98 | 11702.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 14:45:00 | 11763.15 | 11804.98 | 11702.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 11707.05 | 11765.04 | 11719.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 12:45:00 | 11721.95 | 11765.04 | 11719.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 11697.35 | 11751.51 | 11717.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:45:00 | 11678.70 | 11751.51 | 11717.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 11711.25 | 11743.45 | 11717.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 10:00:00 | 11767.95 | 11743.83 | 11721.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 10:30:00 | 11765.90 | 11753.54 | 11728.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 12:15:00 | 11761.30 | 11760.17 | 11750.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-10 09:15:00 | 11640.00 | 11733.87 | 11742.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 11640.00 | 11733.87 | 11742.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 11:15:00 | 11588.85 | 11693.77 | 11722.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 11655.75 | 11576.04 | 11620.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 11655.75 | 11576.04 | 11620.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 11655.75 | 11576.04 | 11620.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 09:30:00 | 11697.40 | 11576.04 | 11620.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 11734.15 | 11607.66 | 11631.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 11734.15 | 11607.66 | 11631.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2025-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 12:15:00 | 11763.75 | 11652.99 | 11648.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 09:15:00 | 12088.00 | 11776.96 | 11710.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 12023.70 | 12046.39 | 11961.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 11:00:00 | 12023.70 | 12046.39 | 11961.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 12033.40 | 12079.34 | 12019.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:30:00 | 12018.65 | 12079.34 | 12019.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 11993.00 | 12062.07 | 12017.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 11:00:00 | 11993.00 | 12062.07 | 12017.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 11:15:00 | 12108.55 | 12071.37 | 12025.47 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 11933.80 | 12010.66 | 12013.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 11881.60 | 11980.50 | 11998.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 10:15:00 | 11963.85 | 11961.54 | 11984.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 11:00:00 | 11963.85 | 11961.54 | 11984.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 11:15:00 | 11993.75 | 11967.98 | 11984.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:00:00 | 11993.75 | 11967.98 | 11984.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 12:15:00 | 12035.30 | 11981.44 | 11989.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:30:00 | 12044.35 | 11981.44 | 11989.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2025-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 14:15:00 | 12026.05 | 11994.97 | 11994.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 11:15:00 | 12041.60 | 12015.44 | 12005.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-23 15:15:00 | 12010.00 | 12027.91 | 12015.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 15:15:00 | 12010.00 | 12027.91 | 12015.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 12010.00 | 12027.91 | 12015.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:15:00 | 12006.15 | 12027.91 | 12015.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 11991.00 | 12020.52 | 12013.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 11:45:00 | 12072.85 | 12023.81 | 12016.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 13:15:00 | 11970.90 | 12009.32 | 12010.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2025-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 13:15:00 | 11970.90 | 12009.32 | 12010.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 11945.35 | 11996.52 | 12004.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 11:15:00 | 12001.70 | 11983.31 | 11994.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 11:15:00 | 12001.70 | 11983.31 | 11994.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 11:15:00 | 12001.70 | 11983.31 | 11994.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 12:00:00 | 12001.70 | 11983.31 | 11994.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 12:15:00 | 12005.25 | 11987.70 | 11995.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 12:30:00 | 11995.00 | 11987.70 | 11995.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 13:15:00 | 12002.65 | 11990.69 | 11995.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 14:00:00 | 12002.65 | 11990.69 | 11995.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 14:15:00 | 11997.50 | 11992.05 | 11996.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 14:45:00 | 11999.55 | 11992.05 | 11996.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 15:15:00 | 11990.00 | 11991.64 | 11995.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 09:15:00 | 11935.10 | 11991.64 | 11995.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 11922.15 | 11977.74 | 11988.88 | EMA400 retest candle locked (from downside) |

### Cycle 124 — BUY (started 2025-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 12:15:00 | 12092.20 | 11991.90 | 11991.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-28 14:15:00 | 12135.20 | 12035.89 | 12012.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-29 13:15:00 | 11995.65 | 12127.72 | 12079.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 13:15:00 | 11995.65 | 12127.72 | 12079.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 13:15:00 | 11995.65 | 12127.72 | 12079.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-29 14:00:00 | 11995.65 | 12127.72 | 12079.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 11983.10 | 12098.80 | 12070.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-29 14:30:00 | 11989.70 | 12098.80 | 12070.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 10:15:00 | 12001.85 | 12044.52 | 12049.64 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 10:15:00 | 12168.90 | 12064.15 | 12051.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 11:15:00 | 12274.35 | 12106.19 | 12072.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-04 10:15:00 | 13040.00 | 13051.56 | 12837.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-04 10:45:00 | 12995.90 | 13051.56 | 12837.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 13073.05 | 13070.07 | 13035.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:45:00 | 13065.40 | 13070.07 | 13035.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 13060.00 | 13068.05 | 13038.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 13083.85 | 13068.05 | 13038.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 12940.00 | 13028.25 | 13030.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 12940.00 | 13028.25 | 13030.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 13:15:00 | 12896.80 | 12975.00 | 13002.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 11:15:00 | 12725.50 | 12700.76 | 12764.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 12:00:00 | 12725.50 | 12700.76 | 12764.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 12701.00 | 12687.64 | 12732.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:30:00 | 12666.00 | 12680.31 | 12725.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 15:15:00 | 12660.00 | 12664.97 | 12701.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 13:15:00 | 12686.30 | 12664.33 | 12685.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-17 15:15:00 | 12759.95 | 12704.75 | 12700.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2025-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 15:15:00 | 12759.95 | 12704.75 | 12700.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 09:15:00 | 12795.25 | 12722.85 | 12708.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-19 09:15:00 | 12723.00 | 12756.34 | 12737.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 09:15:00 | 12723.00 | 12756.34 | 12737.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 12723.00 | 12756.34 | 12737.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:30:00 | 12725.50 | 12756.34 | 12737.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 12704.15 | 12745.90 | 12734.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-19 11:00:00 | 12704.15 | 12745.90 | 12734.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 11:15:00 | 12707.65 | 12738.25 | 12732.02 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 12:15:00 | 12675.25 | 12725.65 | 12726.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-20 09:15:00 | 12397.00 | 12644.44 | 12687.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 09:15:00 | 12403.30 | 12373.29 | 12452.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 12400.00 | 12355.19 | 12399.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 12400.00 | 12355.19 | 12399.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:30:00 | 12419.00 | 12355.19 | 12399.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 12468.95 | 12377.94 | 12405.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 11:00:00 | 12468.95 | 12377.94 | 12405.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 11:15:00 | 12444.85 | 12391.33 | 12409.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 12:45:00 | 12408.10 | 12399.36 | 12411.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 14:15:00 | 12479.15 | 12423.81 | 12420.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2025-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 14:15:00 | 12479.15 | 12423.81 | 12420.89 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 10:15:00 | 12396.15 | 12420.07 | 12420.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 12:15:00 | 12361.45 | 12402.14 | 12411.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 10:15:00 | 11710.35 | 11682.29 | 11801.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-05 11:00:00 | 11710.35 | 11682.29 | 11801.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 11:15:00 | 11804.90 | 11706.81 | 11801.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 12:00:00 | 11804.90 | 11706.81 | 11801.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 12:15:00 | 11750.10 | 11715.47 | 11797.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 13:30:00 | 11684.00 | 11697.37 | 11781.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 10:15:00 | 11647.10 | 11630.76 | 11628.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 10:15:00 | 11647.10 | 11630.76 | 11628.56 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 13:15:00 | 11554.75 | 11615.32 | 11621.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 14:15:00 | 11513.00 | 11594.86 | 11612.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 14:15:00 | 11549.90 | 11545.59 | 11573.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-17 15:00:00 | 11549.90 | 11545.59 | 11573.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 11624.80 | 11559.15 | 11574.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:15:00 | 11643.95 | 11559.15 | 11574.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 11627.60 | 11572.84 | 11579.54 | EMA400 retest candle locked (from downside) |

### Cycle 134 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 11647.20 | 11587.71 | 11585.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 12:15:00 | 11687.55 | 11607.68 | 11594.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 09:15:00 | 11651.40 | 11654.10 | 11624.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-19 10:00:00 | 11651.40 | 11654.10 | 11624.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 13:15:00 | 11590.00 | 11643.64 | 11629.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 13:45:00 | 11561.65 | 11643.64 | 11629.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 14:15:00 | 11613.15 | 11637.54 | 11628.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 09:15:00 | 11646.75 | 11633.03 | 11627.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 10:45:00 | 11635.00 | 11632.28 | 11627.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 11:15:00 | 11790.35 | 11837.38 | 11838.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 11:15:00 | 11790.35 | 11837.38 | 11838.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 13:15:00 | 11771.60 | 11815.04 | 11827.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 09:15:00 | 11805.85 | 11792.29 | 11812.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 09:15:00 | 11805.85 | 11792.29 | 11812.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 11805.85 | 11792.29 | 11812.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:00:00 | 11805.85 | 11792.29 | 11812.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 11797.25 | 11793.28 | 11811.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 12:45:00 | 11728.00 | 11777.50 | 11800.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 13:15:00 | 11670.00 | 11601.15 | 11594.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — BUY (started 2025-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 13:15:00 | 11670.00 | 11601.15 | 11594.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 11724.75 | 11625.87 | 11606.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 12:15:00 | 11686.25 | 11699.58 | 11655.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-03 13:00:00 | 11686.25 | 11699.58 | 11655.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 15:15:00 | 11646.00 | 11688.23 | 11660.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:15:00 | 11448.95 | 11688.23 | 11660.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 11348.90 | 11620.36 | 11632.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 11147.90 | 11428.56 | 11516.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 11353.90 | 11320.50 | 11418.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 15:00:00 | 11353.90 | 11320.50 | 11418.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 11444.25 | 11350.44 | 11415.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 11337.50 | 11350.35 | 11409.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 10:15:00 | 11516.80 | 11449.14 | 11440.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — BUY (started 2025-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 10:15:00 | 11516.80 | 11449.14 | 11440.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 11601.90 | 11491.60 | 11465.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 09:15:00 | 11678.00 | 11780.80 | 11694.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-16 09:15:00 | 11678.00 | 11780.80 | 11694.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 09:15:00 | 11678.00 | 11780.80 | 11694.28 | EMA400 retest candle locked (from upside) |

### Cycle 139 — SELL (started 2025-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-17 09:15:00 | 11548.00 | 11667.82 | 11671.06 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 13:15:00 | 11700.00 | 11671.59 | 11670.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 11:15:00 | 11723.00 | 11689.66 | 11680.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 09:15:00 | 11631.00 | 11702.92 | 11692.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-22 09:15:00 | 11631.00 | 11702.92 | 11692.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 11631.00 | 11702.92 | 11692.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 10:00:00 | 11631.00 | 11702.92 | 11692.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 10:15:00 | 11658.00 | 11693.93 | 11689.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 10:30:00 | 11649.00 | 11693.93 | 11689.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 12:15:00 | 11676.00 | 11690.36 | 11688.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 12:45:00 | 11681.00 | 11690.36 | 11688.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 13:15:00 | 11740.00 | 11700.29 | 11693.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 11847.00 | 11714.90 | 11701.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 12:45:00 | 11822.00 | 11850.59 | 11834.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 14:15:00 | 11732.00 | 11817.18 | 11821.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2025-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 14:15:00 | 11732.00 | 11817.18 | 11821.14 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 15:15:00 | 11809.00 | 11803.72 | 11803.01 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 11:15:00 | 11760.00 | 11796.85 | 11800.17 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 12:15:00 | 11849.00 | 11807.28 | 11804.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 09:15:00 | 11915.00 | 11838.62 | 11821.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-05 11:15:00 | 12405.00 | 12406.61 | 12271.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-05 11:45:00 | 12391.00 | 12406.61 | 12271.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 12531.00 | 12530.29 | 12445.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 10:30:00 | 12610.00 | 12542.03 | 12458.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 13:15:00 | 12338.00 | 12456.51 | 12472.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 13:15:00 | 12338.00 | 12456.51 | 12472.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 12311.00 | 12406.65 | 12443.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 12501.00 | 12337.08 | 12374.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 12501.00 | 12337.08 | 12374.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 12501.00 | 12337.08 | 12374.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 12501.00 | 12337.08 | 12374.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 12534.00 | 12407.49 | 12402.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 12545.00 | 12434.99 | 12415.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 12529.00 | 12533.54 | 12479.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 10:30:00 | 12543.00 | 12533.54 | 12479.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 12453.00 | 12518.46 | 12482.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:45:00 | 12441.00 | 12518.46 | 12482.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 12443.00 | 12503.37 | 12478.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 14:45:00 | 12506.00 | 12499.10 | 12479.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 12500.00 | 12492.28 | 12477.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 12700.00 | 12845.83 | 12861.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 12700.00 | 12845.83 | 12861.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 12640.00 | 12804.66 | 12841.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 14:15:00 | 12416.00 | 12387.47 | 12432.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-27 15:00:00 | 12416.00 | 12387.47 | 12432.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 12391.00 | 12357.08 | 12386.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:00:00 | 12391.00 | 12357.08 | 12386.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 12361.00 | 12357.86 | 12383.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 11:15:00 | 12338.00 | 12357.86 | 12383.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 13:15:00 | 12339.00 | 12349.31 | 12374.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 15:15:00 | 12427.00 | 12370.17 | 12378.14 | SL hit (close>static) qty=1.00 sl=12415.00 alert=retest2 |

### Cycle 148 — BUY (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 09:15:00 | 12438.00 | 12383.73 | 12383.58 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 12337.00 | 12374.39 | 12379.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 11:15:00 | 12280.00 | 12355.51 | 12370.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 14:15:00 | 12279.00 | 12272.41 | 12304.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 14:15:00 | 12279.00 | 12272.41 | 12304.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 12279.00 | 12272.41 | 12304.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 12279.00 | 12272.41 | 12304.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 12273.00 | 12272.53 | 12301.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 12217.00 | 12272.53 | 12301.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 12217.00 | 12261.42 | 12294.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 10:45:00 | 12154.00 | 12235.34 | 12279.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 14:45:00 | 12164.00 | 12176.74 | 12233.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 09:30:00 | 12157.00 | 12172.20 | 12221.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 11:45:00 | 12169.00 | 12177.57 | 12215.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 12212.00 | 12186.92 | 12213.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:00:00 | 12212.00 | 12186.92 | 12213.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 12167.00 | 12182.94 | 12209.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:30:00 | 12189.00 | 12182.94 | 12209.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 12062.00 | 12158.60 | 12193.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:45:00 | 12193.00 | 12158.60 | 12193.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 12230.00 | 12148.54 | 12165.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:00:00 | 12230.00 | 12148.54 | 12165.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-06 10:15:00 | 12471.00 | 12213.03 | 12193.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 12471.00 | 12213.03 | 12193.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 11:15:00 | 12515.00 | 12273.43 | 12222.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 12570.00 | 12572.10 | 12469.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 11:00:00 | 12570.00 | 12572.10 | 12469.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 12519.00 | 12570.35 | 12502.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 12519.00 | 12570.35 | 12502.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 12472.00 | 12550.68 | 12499.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:15:00 | 12541.00 | 12550.68 | 12499.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 12487.00 | 12537.94 | 12498.30 | EMA400 retest candle locked (from upside) |

### Cycle 151 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 12424.00 | 12474.90 | 12480.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 12:15:00 | 12383.00 | 12439.91 | 12461.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 12:15:00 | 12386.00 | 12371.18 | 12407.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 12:15:00 | 12386.00 | 12371.18 | 12407.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 12:15:00 | 12386.00 | 12371.18 | 12407.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 12:45:00 | 12398.00 | 12371.18 | 12407.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 12390.00 | 12374.94 | 12406.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 13:45:00 | 12386.00 | 12374.94 | 12406.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 12416.00 | 12383.15 | 12406.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 12416.00 | 12383.15 | 12406.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 12405.00 | 12387.52 | 12406.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 12391.00 | 12387.52 | 12406.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 12435.00 | 12397.02 | 12409.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:00:00 | 12435.00 | 12397.02 | 12409.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2025-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 10:15:00 | 12518.00 | 12421.21 | 12419.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 11:15:00 | 12545.00 | 12445.97 | 12430.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 15:15:00 | 12782.00 | 12783.12 | 12716.50 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 09:15:00 | 12849.00 | 12783.12 | 12716.50 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 12803.00 | 12779.48 | 12735.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:30:00 | 12749.00 | 12779.48 | 12735.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 12698.00 | 12766.30 | 12743.91 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 12698.00 | 12766.30 | 12743.91 | SL hit (close<ema400) qty=1.00 sl=12743.91 alert=retest1 |

### Cycle 153 — SELL (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 14:15:00 | 12706.00 | 12727.63 | 12729.88 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 12802.00 | 12736.96 | 12733.39 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 13:15:00 | 12636.00 | 12718.82 | 12727.21 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 13:15:00 | 12787.00 | 12726.28 | 12720.76 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 09:15:00 | 12635.00 | 12716.13 | 12724.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 09:15:00 | 12579.00 | 12661.42 | 12690.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 12:15:00 | 12435.00 | 12433.57 | 12518.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-01 12:30:00 | 12435.00 | 12433.57 | 12518.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 12484.00 | 12443.76 | 12496.34 | EMA400 retest candle locked (from downside) |

### Cycle 158 — BUY (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 14:15:00 | 12631.00 | 12535.43 | 12525.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 09:15:00 | 12743.00 | 12589.67 | 12552.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 09:15:00 | 12638.00 | 12699.17 | 12642.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 12638.00 | 12699.17 | 12642.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 12638.00 | 12699.17 | 12642.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:00:00 | 12638.00 | 12699.17 | 12642.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 12631.00 | 12685.54 | 12641.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:30:00 | 12638.00 | 12685.54 | 12641.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 12637.00 | 12675.83 | 12640.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 13:00:00 | 12656.00 | 12671.86 | 12642.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 14:00:00 | 12650.00 | 12667.49 | 12642.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 09:15:00 | 12508.00 | 12628.57 | 12630.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 12508.00 | 12628.57 | 12630.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 12:15:00 | 12497.00 | 12571.05 | 12601.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 12485.00 | 12450.64 | 12496.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 12485.00 | 12450.64 | 12496.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 12485.00 | 12450.64 | 12496.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 12485.00 | 12450.64 | 12496.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 12515.00 | 12468.21 | 12497.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:30:00 | 12519.00 | 12468.21 | 12497.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 12471.00 | 12468.77 | 12494.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 13:30:00 | 12459.00 | 12470.02 | 12492.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 15:15:00 | 12453.00 | 12470.01 | 12490.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 12589.00 | 12491.09 | 12496.67 | SL hit (close>static) qty=1.00 sl=12518.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 10:15:00 | 12570.00 | 12506.87 | 12503.33 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 11:15:00 | 12462.00 | 12541.73 | 12548.11 | EMA200 below EMA400 |

### Cycle 162 — BUY (started 2025-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 15:15:00 | 12569.00 | 12533.42 | 12530.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 12580.00 | 12542.74 | 12535.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 12:15:00 | 12540.00 | 12548.47 | 12540.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 12:15:00 | 12540.00 | 12548.47 | 12540.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 12540.00 | 12548.47 | 12540.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 12540.00 | 12548.47 | 12540.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 12501.00 | 12538.98 | 12536.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 14:00:00 | 12501.00 | 12538.98 | 12536.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — SELL (started 2025-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 14:15:00 | 12476.00 | 12526.38 | 12531.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 12466.00 | 12508.49 | 12521.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 11:15:00 | 12423.00 | 12402.31 | 12427.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 12:00:00 | 12423.00 | 12402.31 | 12427.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 12476.00 | 12417.05 | 12431.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:30:00 | 12510.00 | 12417.05 | 12431.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 12498.00 | 12433.24 | 12437.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:45:00 | 12496.00 | 12433.24 | 12437.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — BUY (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 14:15:00 | 12488.00 | 12444.19 | 12442.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 09:15:00 | 12661.00 | 12497.12 | 12467.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 10:15:00 | 12594.00 | 12603.12 | 12552.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 11:00:00 | 12594.00 | 12603.12 | 12552.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 12545.00 | 12591.50 | 12551.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:00:00 | 12545.00 | 12591.50 | 12551.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 12559.00 | 12585.00 | 12552.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:00:00 | 12559.00 | 12585.00 | 12552.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 12553.00 | 12578.60 | 12552.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:30:00 | 12546.00 | 12578.60 | 12552.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 12558.00 | 12574.48 | 12552.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:45:00 | 12545.00 | 12574.48 | 12552.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 12565.00 | 12572.58 | 12554.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 12485.00 | 12572.58 | 12554.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 12450.00 | 12548.07 | 12544.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:30:00 | 12452.00 | 12548.07 | 12544.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 12405.00 | 12519.45 | 12531.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 12:15:00 | 12333.00 | 12418.83 | 12460.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 12368.00 | 12344.81 | 12393.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:00:00 | 12368.00 | 12344.81 | 12393.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 12443.00 | 12364.45 | 12398.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 12443.00 | 12364.45 | 12398.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 12455.00 | 12382.56 | 12403.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:30:00 | 12446.00 | 12382.56 | 12403.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 12527.00 | 12428.32 | 12421.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 10:15:00 | 12593.00 | 12461.25 | 12437.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 12479.00 | 12539.17 | 12496.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 12479.00 | 12539.17 | 12496.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 12479.00 | 12539.17 | 12496.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:00:00 | 12479.00 | 12539.17 | 12496.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 12525.00 | 12536.34 | 12499.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:15:00 | 12574.00 | 12536.34 | 12499.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 12:45:00 | 12560.00 | 12544.94 | 12509.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 14:00:00 | 12573.00 | 12550.55 | 12515.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 12706.00 | 12543.27 | 12518.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 12504.00 | 12535.42 | 12516.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 12504.00 | 12535.42 | 12516.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 12442.00 | 12516.73 | 12510.10 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-01 10:15:00 | 12442.00 | 12516.73 | 12510.10 | SL hit (close<static) qty=1.00 sl=12464.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 12431.00 | 12499.59 | 12502.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 13:15:00 | 12370.00 | 12458.06 | 12482.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 13:15:00 | 12360.00 | 12348.81 | 12400.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 14:00:00 | 12360.00 | 12348.81 | 12400.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 12413.00 | 12364.39 | 12394.79 | EMA400 retest candle locked (from downside) |

### Cycle 168 — BUY (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 12:15:00 | 12556.00 | 12440.20 | 12424.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 09:15:00 | 12581.00 | 12507.11 | 12464.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 12:15:00 | 12506.00 | 12518.02 | 12481.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-06 13:00:00 | 12506.00 | 12518.02 | 12481.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 12488.00 | 12509.91 | 12488.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:00:00 | 12488.00 | 12509.91 | 12488.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 12501.00 | 12508.13 | 12489.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:30:00 | 12466.00 | 12508.13 | 12489.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 12479.00 | 12502.30 | 12488.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:45:00 | 12475.00 | 12502.30 | 12488.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 12473.00 | 12496.44 | 12487.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 12:45:00 | 12473.00 | 12496.44 | 12487.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 12488.00 | 12494.75 | 12487.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:30:00 | 12456.00 | 12494.75 | 12487.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 12632.00 | 12522.20 | 12500.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 11:45:00 | 12672.00 | 12582.21 | 12539.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 09:30:00 | 12695.00 | 12636.86 | 12591.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-18 09:15:00 | 13939.20 | 13075.12 | 12924.04 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 169 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 14772.00 | 14821.59 | 14826.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 14645.00 | 14786.28 | 14809.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 14840.00 | 14780.02 | 14801.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 14840.00 | 14780.02 | 14801.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 14840.00 | 14780.02 | 14801.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:45:00 | 14860.00 | 14780.02 | 14801.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 14863.00 | 14796.61 | 14807.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:00:00 | 14863.00 | 14796.61 | 14807.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 14777.00 | 14788.43 | 14801.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:45:00 | 14786.00 | 14788.43 | 14801.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 14808.00 | 14792.35 | 14802.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 14808.00 | 14792.35 | 14802.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — BUY (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 14:15:00 | 14903.00 | 14814.48 | 14811.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 15056.00 | 14876.95 | 14841.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 09:15:00 | 15282.00 | 15287.55 | 15171.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 11:15:00 | 15190.00 | 15264.91 | 15181.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 15190.00 | 15264.91 | 15181.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 15190.00 | 15264.91 | 15181.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 15183.00 | 15248.53 | 15181.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:45:00 | 15185.00 | 15248.53 | 15181.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 15179.00 | 15234.63 | 15181.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:15:00 | 15104.00 | 15234.63 | 15181.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 15099.00 | 15207.50 | 15173.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:45:00 | 15087.00 | 15207.50 | 15173.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 15134.00 | 15192.80 | 15169.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:00:00 | 15060.00 | 15166.24 | 15159.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — SELL (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 10:15:00 | 15064.00 | 15145.79 | 15151.25 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 15311.00 | 15162.38 | 15154.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 10:15:00 | 15325.00 | 15194.91 | 15169.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 11:15:00 | 15280.00 | 15289.57 | 15246.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 12:00:00 | 15280.00 | 15289.57 | 15246.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 15269.00 | 15286.38 | 15255.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 15:00:00 | 15269.00 | 15286.38 | 15255.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 15251.00 | 15279.31 | 15255.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 15310.00 | 15279.31 | 15255.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 11:15:00 | 16065.00 | 16198.27 | 16208.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — SELL (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 11:15:00 | 16065.00 | 16198.27 | 16208.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 14:15:00 | 15973.00 | 16138.29 | 16177.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 15:15:00 | 16035.00 | 16029.74 | 16083.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-01 09:15:00 | 16059.00 | 16029.74 | 16083.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 15930.00 | 16009.79 | 16069.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 11:45:00 | 15877.00 | 15969.62 | 16039.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 09:15:00 | 15849.00 | 15952.67 | 16007.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 09:45:00 | 15817.00 | 15923.34 | 15988.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 13:15:00 | 16042.00 | 15946.71 | 15942.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — BUY (started 2025-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 13:15:00 | 16042.00 | 15946.71 | 15942.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 16128.00 | 15998.93 | 15968.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 13:15:00 | 16079.00 | 16115.74 | 16072.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 13:15:00 | 16079.00 | 16115.74 | 16072.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 16079.00 | 16115.74 | 16072.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:00:00 | 16079.00 | 16115.74 | 16072.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 16005.00 | 16093.59 | 16066.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:45:00 | 16027.00 | 16093.59 | 16066.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 16014.00 | 16077.67 | 16061.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 15976.00 | 16077.67 | 16061.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — SELL (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 09:15:00 | 15906.00 | 16043.34 | 16047.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 11:15:00 | 15890.00 | 16002.22 | 16027.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 15974.00 | 15962.30 | 15999.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 15:00:00 | 15974.00 | 15962.30 | 15999.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 15948.00 | 15959.44 | 15995.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 16040.00 | 15972.35 | 15997.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 16116.00 | 16001.08 | 16008.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:45:00 | 16099.00 | 16001.08 | 16008.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 16179.00 | 16036.67 | 16023.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 16240.00 | 16077.33 | 16043.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 09:15:00 | 16170.00 | 16262.01 | 16202.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 16170.00 | 16262.01 | 16202.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 16170.00 | 16262.01 | 16202.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:00:00 | 16170.00 | 16262.01 | 16202.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 16163.00 | 16242.21 | 16199.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 14:15:00 | 16247.00 | 16215.44 | 16195.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 16330.00 | 16221.32 | 16202.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 16322.00 | 16248.33 | 16232.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 10:15:00 | 16310.00 | 16368.78 | 16373.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 16310.00 | 16368.78 | 16373.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 11:15:00 | 16282.00 | 16351.43 | 16364.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 16363.00 | 16308.21 | 16332.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 16363.00 | 16308.21 | 16332.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 16363.00 | 16308.21 | 16332.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 16363.00 | 16308.21 | 16332.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 16330.00 | 16312.57 | 16332.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 13:00:00 | 16309.00 | 16320.08 | 16332.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 14:00:00 | 16320.00 | 16320.07 | 16331.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 14:15:00 | 16400.00 | 16336.05 | 16337.80 | SL hit (close>static) qty=1.00 sl=16388.00 alert=retest2 |

### Cycle 178 — BUY (started 2025-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 15:15:00 | 16382.00 | 16345.24 | 16341.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 16423.00 | 16360.79 | 16349.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 11:15:00 | 16355.00 | 16365.27 | 16353.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 11:15:00 | 16355.00 | 16365.27 | 16353.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 16355.00 | 16365.27 | 16353.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 16355.00 | 16365.27 | 16353.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 16344.00 | 16361.01 | 16352.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:15:00 | 16332.00 | 16361.01 | 16352.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 16296.00 | 16348.01 | 16347.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:00:00 | 16296.00 | 16348.01 | 16347.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — SELL (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 14:15:00 | 16316.00 | 16341.61 | 16344.70 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 16416.00 | 16356.86 | 16350.59 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 13:15:00 | 16104.00 | 16305.56 | 16329.09 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 11:15:00 | 16365.00 | 16290.59 | 16283.14 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 15:15:00 | 16155.00 | 16256.60 | 16270.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 09:15:00 | 15664.00 | 16138.08 | 16215.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 13:15:00 | 15448.00 | 15440.18 | 15592.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 13:30:00 | 15459.00 | 15440.18 | 15592.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 15490.00 | 15439.99 | 15498.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 15441.00 | 15439.99 | 15498.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 15477.00 | 15447.39 | 15496.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:15:00 | 15511.00 | 15447.39 | 15496.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 15488.00 | 15455.52 | 15495.56 | EMA400 retest candle locked (from downside) |

### Cycle 184 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 15608.00 | 15522.98 | 15519.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 13:15:00 | 15644.00 | 15585.20 | 15556.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 09:15:00 | 15580.00 | 15601.96 | 15573.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 10:00:00 | 15580.00 | 15601.96 | 15573.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 15620.00 | 15605.57 | 15577.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:45:00 | 15581.00 | 15605.57 | 15577.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 15617.00 | 15706.51 | 15674.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:15:00 | 15593.00 | 15706.51 | 15674.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 15629.00 | 15691.01 | 15670.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:30:00 | 15606.00 | 15691.01 | 15670.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 15691.00 | 15669.64 | 15664.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 14:45:00 | 15616.00 | 15669.64 | 15664.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 15708.00 | 15677.31 | 15668.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 15733.00 | 15677.31 | 15668.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 15751.00 | 15692.05 | 15675.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 11:30:00 | 15801.00 | 15731.19 | 15696.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:30:00 | 15773.00 | 15833.59 | 15820.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 14:15:00 | 15776.00 | 15811.90 | 15812.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — SELL (started 2025-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 14:15:00 | 15776.00 | 15811.90 | 15812.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 15:15:00 | 15740.00 | 15797.52 | 15805.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 13:15:00 | 15762.00 | 15740.31 | 15769.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 13:15:00 | 15762.00 | 15740.31 | 15769.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 15762.00 | 15740.31 | 15769.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 14:00:00 | 15762.00 | 15740.31 | 15769.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 15809.00 | 15754.05 | 15772.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 15809.00 | 15754.05 | 15772.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 15822.00 | 15767.64 | 15777.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 15813.00 | 15767.64 | 15777.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — BUY (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 09:15:00 | 15889.00 | 15791.91 | 15787.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 11:15:00 | 16000.00 | 15851.30 | 15816.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 14:15:00 | 15945.00 | 15985.04 | 15936.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-24 15:00:00 | 15945.00 | 15985.04 | 15936.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 15950.00 | 15978.04 | 15937.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 15999.00 | 15978.04 | 15937.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 11:00:00 | 15978.00 | 15985.38 | 15948.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 14:15:00 | 15897.00 | 15970.31 | 15953.81 | SL hit (close<static) qty=1.00 sl=15911.00 alert=retest2 |

### Cycle 187 — SELL (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 14:15:00 | 15904.00 | 16008.77 | 16016.92 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 14:15:00 | 16103.00 | 15978.68 | 15974.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 10:15:00 | 16199.00 | 16070.80 | 16021.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 16099.00 | 16156.03 | 16094.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 16099.00 | 16156.03 | 16094.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 16099.00 | 16156.03 | 16094.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:45:00 | 16113.00 | 16156.03 | 16094.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 16038.00 | 16132.42 | 16089.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:00:00 | 16038.00 | 16132.42 | 16089.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 16041.00 | 16114.14 | 16085.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 14:00:00 | 16090.00 | 16102.25 | 16084.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 15:00:00 | 16087.00 | 16099.20 | 16084.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 09:15:00 | 16087.00 | 16094.36 | 16083.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 15987.00 | 16072.89 | 16074.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — SELL (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 09:15:00 | 15987.00 | 16072.89 | 16074.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 10:15:00 | 15980.00 | 16054.31 | 16066.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 09:15:00 | 16119.00 | 16022.32 | 16036.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 16119.00 | 16022.32 | 16036.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 16119.00 | 16022.32 | 16036.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:30:00 | 16075.00 | 16022.32 | 16036.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 16196.00 | 16057.05 | 16051.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 16279.00 | 16156.26 | 16105.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 16183.00 | 16184.60 | 16128.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 10:00:00 | 16183.00 | 16184.60 | 16128.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 16181.00 | 16178.24 | 16142.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:30:00 | 16117.00 | 16178.24 | 16142.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 16058.00 | 16155.27 | 16141.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 16005.00 | 16155.27 | 16141.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 16086.00 | 16141.42 | 16136.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:30:00 | 16069.00 | 16141.42 | 16136.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 16136.00 | 16142.87 | 16138.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 16124.00 | 16142.87 | 16138.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — SELL (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 14:15:00 | 16026.00 | 16119.50 | 16128.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 15:15:00 | 15995.00 | 16094.60 | 16116.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 12:15:00 | 16098.00 | 16081.46 | 16101.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 12:15:00 | 16098.00 | 16081.46 | 16101.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 16098.00 | 16081.46 | 16101.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:30:00 | 16033.00 | 16067.57 | 16093.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 16176.00 | 16072.39 | 16087.77 | SL hit (close>static) qty=1.00 sl=16110.00 alert=retest2 |

### Cycle 192 — BUY (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 10:15:00 | 16206.00 | 16099.11 | 16098.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 11:15:00 | 16260.00 | 16131.29 | 16113.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 10:15:00 | 16438.00 | 16442.20 | 16343.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 11:00:00 | 16438.00 | 16442.20 | 16343.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 16400.00 | 16424.62 | 16371.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 16398.00 | 16424.62 | 16371.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 16357.00 | 16411.10 | 16370.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 16357.00 | 16411.10 | 16370.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 16392.00 | 16407.28 | 16372.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 16454.00 | 16378.30 | 16371.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 09:15:00 | 16251.00 | 16365.88 | 16373.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 16251.00 | 16365.88 | 16373.91 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 16448.00 | 16378.75 | 16371.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 16518.00 | 16427.47 | 16400.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 16586.00 | 16586.28 | 16530.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 14:45:00 | 16610.00 | 16586.28 | 16530.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 16592.00 | 16647.66 | 16608.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 16592.00 | 16647.66 | 16608.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 16578.00 | 16633.73 | 16605.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 13:00:00 | 16616.00 | 16630.18 | 16606.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 15:00:00 | 16613.00 | 16620.32 | 16606.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 16664.00 | 16614.25 | 16604.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 10:15:00 | 16530.00 | 16594.00 | 16596.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 16530.00 | 16594.00 | 16596.81 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 10:15:00 | 16686.00 | 16604.48 | 16597.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 16976.00 | 16766.79 | 16714.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 09:15:00 | 17152.00 | 17208.62 | 17121.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 17152.00 | 17208.62 | 17121.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 17152.00 | 17208.62 | 17121.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:15:00 | 17069.00 | 17208.62 | 17121.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 17030.00 | 17172.90 | 17113.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 17030.00 | 17172.90 | 17113.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 17024.00 | 17143.12 | 17105.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:00:00 | 17024.00 | 17143.12 | 17105.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 16787.00 | 17071.90 | 17076.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 13:15:00 | 16668.00 | 16991.12 | 17039.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 16542.00 | 16464.52 | 16585.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 13:00:00 | 16542.00 | 16464.52 | 16585.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 16557.00 | 16492.37 | 16577.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 16614.00 | 16492.37 | 16577.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 16605.00 | 16514.90 | 16580.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 16611.00 | 16514.90 | 16580.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 16560.00 | 16523.92 | 16578.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:45:00 | 16499.00 | 16515.74 | 16569.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 10:15:00 | 15674.05 | 15736.37 | 15816.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-28 09:15:00 | 14849.10 | 15200.74 | 15396.97 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 198 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 14870.00 | 14508.38 | 14479.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 14978.00 | 14731.91 | 14612.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 14948.00 | 14971.08 | 14822.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:45:00 | 14920.00 | 14971.08 | 14822.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 14989.00 | 15024.46 | 14928.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:30:00 | 15089.00 | 15034.37 | 14950.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:30:00 | 15136.00 | 15025.55 | 14990.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 15218.00 | 15278.09 | 15282.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — SELL (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 09:15:00 | 15218.00 | 15278.09 | 15282.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 10:15:00 | 15070.00 | 15236.47 | 15262.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 10:15:00 | 15157.00 | 15113.47 | 15168.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 10:15:00 | 15157.00 | 15113.47 | 15168.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 15157.00 | 15113.47 | 15168.04 | EMA400 retest candle locked (from downside) |

### Cycle 200 — BUY (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 10:15:00 | 15180.00 | 15162.33 | 15161.90 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 15101.00 | 15150.06 | 15156.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 14905.00 | 15092.23 | 15127.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 15046.00 | 15020.19 | 15065.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 13:15:00 | 15046.00 | 15020.19 | 15065.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 15046.00 | 15020.19 | 15065.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:45:00 | 15060.00 | 15020.19 | 15065.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 15051.00 | 15020.44 | 15053.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 15053.00 | 15020.44 | 15053.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 15053.00 | 15026.95 | 15053.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:00:00 | 15020.00 | 15032.29 | 15051.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:30:00 | 15025.00 | 15028.23 | 15048.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 15013.00 | 15039.91 | 15050.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 10:00:00 | 14989.00 | 14962.55 | 14994.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 15044.00 | 14978.84 | 14998.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:00:00 | 15044.00 | 14978.84 | 14998.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 15127.00 | 15008.47 | 15010.32 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-25 11:15:00 | 15127.00 | 15008.47 | 15010.32 | SL hit (close>static) qty=1.00 sl=15080.00 alert=retest2 |

### Cycle 202 — BUY (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 12:15:00 | 15028.00 | 15012.38 | 15011.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 15158.00 | 15066.29 | 15038.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 15070.00 | 15076.27 | 15048.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 11:15:00 | 15070.00 | 15076.27 | 15048.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 15070.00 | 15076.27 | 15048.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:45:00 | 15061.00 | 15076.27 | 15048.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 14984.00 | 15112.58 | 15083.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:15:00 | 14912.00 | 15112.58 | 15083.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 14970.00 | 15084.06 | 15073.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:30:00 | 14918.00 | 15084.06 | 15073.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 14929.00 | 15053.05 | 15060.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 14865.00 | 14998.08 | 15031.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 14:15:00 | 14165.00 | 14153.62 | 14408.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 14:45:00 | 14168.00 | 14153.62 | 14408.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 14347.00 | 14222.99 | 14320.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 14469.00 | 14222.99 | 14320.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 14471.00 | 14272.59 | 14334.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 14318.00 | 14272.59 | 14334.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 12:15:00 | 14320.00 | 14316.49 | 14340.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:00:00 | 14326.00 | 14318.39 | 14339.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 13602.10 | 14104.18 | 14231.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 13604.00 | 14104.18 | 14231.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 13609.70 | 14104.18 | 14231.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 13694.00 | 13644.52 | 13857.43 | SL hit (close>ema200) qty=0.50 sl=13644.52 alert=retest2 |

### Cycle 204 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 13105.00 | 12934.07 | 12911.97 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 12643.00 | 12910.99 | 12934.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 11:15:00 | 12612.00 | 12851.19 | 12905.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 12420.00 | 12406.13 | 12510.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:45:00 | 12444.00 | 12406.13 | 12510.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 12570.00 | 12438.90 | 12515.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 12594.00 | 12438.90 | 12515.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 12543.00 | 12459.72 | 12518.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 14:15:00 | 12510.00 | 12459.72 | 12518.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 12720.00 | 12514.99 | 12528.73 | SL hit (close>static) qty=1.00 sl=12614.00 alert=retest2 |

### Cycle 206 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 12779.00 | 12567.80 | 12551.48 | EMA200 above EMA400 |

### Cycle 207 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 12433.00 | 12581.08 | 12581.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 12385.00 | 12541.87 | 12563.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 12655.00 | 12410.15 | 12442.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 12655.00 | 12410.15 | 12442.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 12655.00 | 12410.15 | 12442.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 12593.00 | 12410.15 | 12442.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 12526.00 | 12462.11 | 12460.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 12526.00 | 12462.11 | 12460.57 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 13:15:00 | 12434.00 | 12456.49 | 12458.16 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 12511.00 | 12467.39 | 12462.96 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 12397.00 | 12458.53 | 12460.03 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 12632.00 | 12454.37 | 12451.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 12659.00 | 12535.47 | 12499.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 12582.00 | 12609.94 | 12552.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 12582.00 | 12609.94 | 12552.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 12582.00 | 12609.94 | 12552.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 11:45:00 | 12681.00 | 12617.32 | 12566.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 12:45:00 | 12655.00 | 12632.26 | 12577.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 11:15:00 | 13174.00 | 13427.57 | 13436.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 213 — SELL (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 11:15:00 | 13174.00 | 13427.57 | 13436.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 12:15:00 | 13076.00 | 13357.25 | 13403.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 13342.00 | 13241.74 | 13322.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 13342.00 | 13241.74 | 13322.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 13342.00 | 13241.74 | 13322.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:15:00 | 13427.00 | 13241.74 | 13322.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 13378.00 | 13268.99 | 13327.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:45:00 | 13461.00 | 13268.99 | 13327.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 12:15:00 | 13311.00 | 13289.23 | 13327.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 13:15:00 | 13278.00 | 13289.23 | 13327.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 14:00:00 | 13279.00 | 13287.19 | 13322.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 15:00:00 | 13287.00 | 13287.15 | 13319.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 09:45:00 | 13248.00 | 13291.78 | 13315.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 13340.00 | 13301.42 | 13318.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 11:00:00 | 13340.00 | 13301.42 | 13318.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 13325.00 | 13306.14 | 13318.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 11:30:00 | 13349.00 | 13306.14 | 13318.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 13310.00 | 13306.91 | 13317.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 12:30:00 | 13369.00 | 13306.91 | 13317.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 13312.00 | 13307.93 | 13317.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 13:45:00 | 13332.00 | 13307.93 | 13317.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 13330.00 | 13312.34 | 13318.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 14:45:00 | 13337.00 | 13312.34 | 13318.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 15:15:00 | 13336.00 | 13317.07 | 13320.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 09:15:00 | 13478.00 | 13317.07 | 13320.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 13611.00 | 13375.86 | 13346.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 214 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 13611.00 | 13375.86 | 13346.58 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 13343.00 | 13434.87 | 13442.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 09:15:00 | 13276.00 | 13374.40 | 13407.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 13109.00 | 13085.67 | 13170.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 13109.00 | 13085.67 | 13170.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 13109.00 | 13085.67 | 13170.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 10:30:00 | 13055.00 | 13090.94 | 13164.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 12:15:00 | 13235.00 | 13131.60 | 13171.15 | SL hit (close>static) qty=1.00 sl=13189.00 alert=retest2 |

### Cycle 216 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 13256.00 | 13195.81 | 13192.91 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 11:15:00 | 13136.00 | 13189.64 | 13190.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 12:15:00 | 13122.00 | 13176.11 | 13184.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 13489.00 | 13157.81 | 13166.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 13489.00 | 13157.81 | 13166.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 13489.00 | 13157.81 | 13166.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:45:00 | 13497.00 | 13157.81 | 13166.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 218 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 13478.00 | 13221.85 | 13194.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 13510.00 | 13279.48 | 13223.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 14:15:00 | 13280.00 | 13305.94 | 13252.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 14:45:00 | 13261.00 | 13305.94 | 13252.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 13265.00 | 13297.75 | 13253.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 13044.00 | 13297.75 | 13253.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 13015.00 | 13241.20 | 13231.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:30:00 | 13014.00 | 13241.20 | 13231.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 219 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 13040.00 | 13200.96 | 13214.25 | EMA200 below EMA400 |

### Cycle 220 — BUY (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 15:15:00 | 13320.00 | 13230.32 | 13218.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 13748.00 | 13333.86 | 13267.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 13448.00 | 13494.88 | 13403.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 13448.00 | 13494.88 | 13403.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 13448.00 | 13494.88 | 13403.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:30:00 | 13584.00 | 13487.95 | 13440.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 14:15:00 | 13600.00 | 13480.94 | 13451.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-18 11:00:00 | 9088.75 | 2023-05-22 10:15:00 | 9158.50 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2023-05-30 09:30:00 | 9350.90 | 2023-05-31 11:15:00 | 9299.45 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2023-05-30 11:15:00 | 9354.70 | 2023-05-31 11:15:00 | 9299.45 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2023-06-01 14:30:00 | 9331.90 | 2023-06-02 10:15:00 | 9365.95 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2023-06-02 09:45:00 | 9324.60 | 2023-06-02 10:15:00 | 9365.95 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2023-06-14 12:15:00 | 9549.80 | 2023-06-15 09:15:00 | 9635.80 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2023-06-21 10:15:00 | 9474.90 | 2023-06-26 15:15:00 | 9470.00 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2023-06-26 14:45:00 | 9474.95 | 2023-06-26 15:15:00 | 9470.00 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2023-07-05 09:15:00 | 9723.05 | 2023-07-10 13:15:00 | 9772.00 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest2 | 2023-07-18 11:30:00 | 9617.70 | 2023-07-20 13:15:00 | 9738.80 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2023-07-19 09:15:00 | 9616.60 | 2023-07-20 13:15:00 | 9738.80 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2023-07-19 10:15:00 | 9641.35 | 2023-07-20 13:15:00 | 9738.80 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2023-07-19 11:15:00 | 9642.30 | 2023-07-20 13:15:00 | 9738.80 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2023-07-27 09:30:00 | 9810.10 | 2023-07-27 12:15:00 | 9725.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2023-07-27 10:30:00 | 9791.40 | 2023-07-27 12:15:00 | 9725.00 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2023-08-28 11:00:00 | 9627.50 | 2023-09-06 13:15:00 | 10136.00 | STOP_HIT | 1.00 | 5.28% |
| BUY | retest2 | 2023-08-28 13:15:00 | 9624.95 | 2023-09-06 13:15:00 | 10136.00 | STOP_HIT | 1.00 | 5.31% |
| BUY | retest2 | 2023-08-29 09:30:00 | 9650.20 | 2023-09-06 13:15:00 | 10136.00 | STOP_HIT | 1.00 | 5.03% |
| BUY | retest2 | 2023-08-29 10:00:00 | 9629.00 | 2023-09-06 13:15:00 | 10136.00 | STOP_HIT | 1.00 | 5.27% |
| BUY | retest2 | 2023-08-30 10:30:00 | 9706.00 | 2023-09-06 13:15:00 | 10136.00 | STOP_HIT | 1.00 | 4.43% |
| BUY | retest2 | 2023-08-30 12:00:00 | 9708.95 | 2023-09-06 13:15:00 | 10136.00 | STOP_HIT | 1.00 | 4.40% |
| BUY | retest2 | 2023-09-12 13:15:00 | 10517.00 | 2023-09-20 09:15:00 | 10440.20 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2023-09-14 09:15:00 | 10511.50 | 2023-09-20 09:15:00 | 10440.20 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2023-09-15 09:30:00 | 10537.50 | 2023-09-20 09:15:00 | 10440.20 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2023-09-27 12:15:00 | 10614.00 | 2023-10-03 09:15:00 | 10329.50 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2023-09-27 13:45:00 | 10620.80 | 2023-10-03 09:15:00 | 10329.50 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2023-09-28 13:30:00 | 10613.50 | 2023-10-03 09:15:00 | 10329.50 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2023-09-29 11:00:00 | 10611.00 | 2023-10-03 09:15:00 | 10329.50 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2023-10-26 09:15:00 | 10513.00 | 2023-10-27 13:15:00 | 10778.80 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2023-10-27 15:00:00 | 10545.60 | 2023-11-08 12:15:00 | 10306.40 | STOP_HIT | 1.00 | 2.27% |
| BUY | retest2 | 2023-11-10 11:15:00 | 10392.00 | 2023-11-22 11:15:00 | 10447.50 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2023-11-10 15:15:00 | 10395.00 | 2023-11-22 11:15:00 | 10447.50 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest2 | 2023-11-13 10:45:00 | 10392.60 | 2023-11-22 11:15:00 | 10447.50 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2023-11-13 13:00:00 | 10392.00 | 2023-11-22 11:15:00 | 10447.50 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2023-11-16 13:15:00 | 10510.80 | 2023-11-22 11:15:00 | 10447.50 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2023-11-17 09:15:00 | 10541.50 | 2023-11-22 11:15:00 | 10447.50 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2023-11-30 14:45:00 | 10601.00 | 2023-12-04 09:15:00 | 10485.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2023-12-14 10:30:00 | 10327.50 | 2023-12-22 13:15:00 | 10245.50 | STOP_HIT | 1.00 | 0.79% |
| SELL | retest2 | 2023-12-14 12:30:00 | 10337.20 | 2023-12-22 13:15:00 | 10245.50 | STOP_HIT | 1.00 | 0.89% |
| SELL | retest2 | 2023-12-15 10:30:00 | 10327.50 | 2023-12-22 13:15:00 | 10245.50 | STOP_HIT | 1.00 | 0.79% |
| BUY | retest1 | 2023-12-27 15:00:00 | 10287.30 | 2023-12-28 10:15:00 | 10218.70 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2023-12-28 14:15:00 | 10265.00 | 2024-01-02 10:15:00 | 10191.70 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2023-12-28 15:00:00 | 10267.50 | 2024-01-02 10:15:00 | 10191.70 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-01-01 12:00:00 | 10260.80 | 2024-01-02 10:15:00 | 10191.70 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-01-01 12:30:00 | 10274.70 | 2024-01-02 10:15:00 | 10191.70 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-01-01 14:15:00 | 10264.50 | 2024-01-02 10:15:00 | 10191.70 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-01-08 11:45:00 | 9993.15 | 2024-01-11 09:15:00 | 10088.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-01-09 12:30:00 | 10000.30 | 2024-01-11 09:15:00 | 10088.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-01-09 13:45:00 | 9995.45 | 2024-01-11 09:15:00 | 10088.00 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-01-10 09:15:00 | 9950.00 | 2024-01-11 09:15:00 | 10088.00 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-01-25 12:00:00 | 9851.00 | 2024-01-29 10:15:00 | 9986.55 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-01-25 14:15:00 | 9875.00 | 2024-01-29 10:15:00 | 9986.55 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-01-25 15:15:00 | 9879.75 | 2024-01-29 10:15:00 | 9986.55 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-01-31 10:45:00 | 10087.70 | 2024-02-09 14:15:00 | 10732.60 | STOP_HIT | 1.00 | 6.39% |
| BUY | retest2 | 2024-01-31 11:15:00 | 10120.00 | 2024-02-09 14:15:00 | 10732.60 | STOP_HIT | 1.00 | 6.05% |
| BUY | retest2 | 2024-02-21 12:45:00 | 11488.80 | 2024-02-26 12:15:00 | 11496.05 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2024-02-21 15:00:00 | 11473.90 | 2024-02-26 12:15:00 | 11496.05 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2024-02-22 09:15:00 | 11505.30 | 2024-02-26 12:15:00 | 11496.05 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2024-02-22 10:15:00 | 11471.00 | 2024-02-26 12:15:00 | 11496.05 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2024-02-28 09:30:00 | 11425.05 | 2024-03-01 11:15:00 | 11524.65 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-03-01 11:15:00 | 11450.85 | 2024-03-01 11:15:00 | 11524.65 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-03-13 11:15:00 | 11361.20 | 2024-03-15 13:15:00 | 11467.35 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-03-13 13:30:00 | 11360.80 | 2024-03-18 09:15:00 | 11448.75 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-03-14 09:45:00 | 11358.35 | 2024-03-18 09:15:00 | 11448.75 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-03-14 10:45:00 | 11363.05 | 2024-03-18 09:15:00 | 11448.75 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-03-15 11:00:00 | 11325.00 | 2024-03-18 09:15:00 | 11448.75 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-05-22 14:00:00 | 12501.60 | 2024-05-23 09:15:00 | 12590.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-05-23 09:15:00 | 12501.00 | 2024-05-23 09:15:00 | 12590.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-05-27 11:45:00 | 12928.55 | 2024-05-29 10:15:00 | 12815.90 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-05-27 12:15:00 | 12920.10 | 2024-05-29 10:15:00 | 12815.90 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-05-28 11:00:00 | 12973.00 | 2024-05-29 10:15:00 | 12815.90 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-05-28 13:00:00 | 12927.45 | 2024-05-29 10:15:00 | 12815.90 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-06-04 11:00:00 | 12104.40 | 2024-06-04 12:15:00 | 11583.02 | PARTIAL | 0.50 | 4.31% |
| SELL | retest2 | 2024-06-04 11:00:00 | 12104.40 | 2024-06-05 09:15:00 | 12555.15 | STOP_HIT | 0.50 | -3.72% |
| SELL | retest2 | 2024-06-04 11:45:00 | 12192.65 | 2024-06-05 13:15:00 | 12517.25 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2024-06-04 15:15:00 | 12204.65 | 2024-06-05 13:15:00 | 12517.25 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2024-06-13 12:15:00 | 12820.30 | 2024-06-18 09:15:00 | 12660.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-06-14 09:45:00 | 12811.65 | 2024-06-18 09:15:00 | 12660.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-06-14 11:15:00 | 12830.10 | 2024-06-18 09:15:00 | 12660.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-06-24 12:15:00 | 12181.25 | 2024-06-26 13:15:00 | 12279.20 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-06-25 09:15:00 | 12158.95 | 2024-06-26 13:15:00 | 12279.20 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-07-01 12:15:00 | 12055.00 | 2024-07-01 13:15:00 | 12206.70 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-07-01 13:00:00 | 12050.00 | 2024-07-01 13:15:00 | 12206.70 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-07-02 09:15:00 | 12047.05 | 2024-07-03 15:15:00 | 12134.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-07-02 12:00:00 | 12020.00 | 2024-07-03 15:15:00 | 12134.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-07-19 09:15:00 | 12607.55 | 2024-07-22 14:15:00 | 12651.85 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2024-07-19 10:30:00 | 12605.00 | 2024-07-22 14:15:00 | 12651.85 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2024-07-19 11:30:00 | 12596.80 | 2024-07-22 14:15:00 | 12651.85 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2024-07-22 11:00:00 | 12572.20 | 2024-07-22 14:15:00 | 12651.85 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-07-26 09:15:00 | 12424.95 | 2024-07-26 09:15:00 | 12577.05 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest1 | 2024-08-06 11:00:00 | 12322.00 | 2024-08-12 09:15:00 | 12286.70 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest1 | 2024-08-06 12:30:00 | 12305.15 | 2024-08-12 09:15:00 | 12286.70 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2024-08-08 11:30:00 | 12213.65 | 2024-08-13 10:15:00 | 12301.85 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-08-08 13:00:00 | 12200.65 | 2024-08-13 10:15:00 | 12301.85 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-08-08 15:00:00 | 12217.95 | 2024-08-13 10:15:00 | 12301.85 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-08-09 14:15:00 | 12215.60 | 2024-08-13 10:15:00 | 12301.85 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-08-16 10:15:00 | 12131.70 | 2024-08-20 11:15:00 | 12238.65 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-08-16 12:15:00 | 12180.00 | 2024-08-20 11:15:00 | 12238.65 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2024-08-19 10:30:00 | 12175.00 | 2024-08-20 11:15:00 | 12238.65 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-08-19 11:30:00 | 12158.60 | 2024-08-20 11:15:00 | 12238.65 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-08-22 15:15:00 | 12305.05 | 2024-08-29 09:15:00 | 12292.15 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2024-08-23 09:45:00 | 12290.00 | 2024-08-29 09:15:00 | 12292.15 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2024-08-23 14:45:00 | 12300.55 | 2024-08-29 09:15:00 | 12292.15 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2024-08-26 09:30:00 | 12305.60 | 2024-08-29 09:15:00 | 12292.15 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2024-08-27 09:15:00 | 12255.00 | 2024-08-29 09:15:00 | 12292.15 | STOP_HIT | 1.00 | 0.30% |
| BUY | retest2 | 2024-08-30 14:45:00 | 12451.00 | 2024-09-02 11:15:00 | 12374.75 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-08-30 15:15:00 | 12461.80 | 2024-09-02 11:15:00 | 12374.75 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-09-02 10:45:00 | 12452.85 | 2024-09-02 11:15:00 | 12374.75 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-09-02 11:15:00 | 12445.50 | 2024-09-02 11:15:00 | 12374.75 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-09-02 15:00:00 | 12429.15 | 2024-09-03 15:15:00 | 12390.00 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2024-09-03 11:00:00 | 12430.00 | 2024-09-03 15:15:00 | 12390.00 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2024-09-05 09:15:00 | 12319.20 | 2024-09-10 15:15:00 | 12278.90 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2024-09-11 15:15:00 | 12260.55 | 2024-09-12 09:15:00 | 12193.70 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-09-12 14:00:00 | 12334.70 | 2024-09-17 09:15:00 | 12200.20 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-09-25 14:00:00 | 12719.90 | 2024-10-01 10:15:00 | 13006.00 | STOP_HIT | 1.00 | 2.25% |
| SELL | retest2 | 2024-10-22 10:15:00 | 11902.40 | 2024-10-29 09:15:00 | 11307.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 14:15:00 | 11922.25 | 2024-10-29 09:15:00 | 11326.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 09:30:00 | 11912.05 | 2024-10-29 09:15:00 | 11316.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 14:15:00 | 11934.20 | 2024-10-29 09:15:00 | 11337.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 10:15:00 | 11902.40 | 2024-10-30 10:15:00 | 11346.55 | STOP_HIT | 0.50 | 4.67% |
| SELL | retest2 | 2024-10-22 14:15:00 | 11922.25 | 2024-10-30 10:15:00 | 11346.55 | STOP_HIT | 0.50 | 4.83% |
| SELL | retest2 | 2024-10-23 09:30:00 | 11912.05 | 2024-10-30 10:15:00 | 11346.55 | STOP_HIT | 0.50 | 4.75% |
| SELL | retest2 | 2024-10-23 14:15:00 | 11934.20 | 2024-10-30 10:15:00 | 11346.55 | STOP_HIT | 0.50 | 4.92% |
| SELL | retest2 | 2024-11-04 09:15:00 | 10942.50 | 2024-11-05 13:15:00 | 11187.20 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2024-11-04 10:00:00 | 10932.15 | 2024-11-05 13:15:00 | 11187.20 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2024-11-04 12:00:00 | 10924.75 | 2024-11-05 13:15:00 | 11187.20 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2024-11-04 13:15:00 | 10930.90 | 2024-11-05 13:15:00 | 11187.20 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2024-11-05 11:15:00 | 11178.25 | 2024-11-05 13:15:00 | 11187.20 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2024-11-08 14:30:00 | 11325.00 | 2024-11-12 12:15:00 | 11215.05 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-11-11 09:15:00 | 11453.55 | 2024-11-12 12:15:00 | 11215.05 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2024-11-12 09:45:00 | 11362.20 | 2024-11-12 12:15:00 | 11215.05 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-11-12 11:00:00 | 11323.45 | 2024-11-12 12:15:00 | 11215.05 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-12-10 10:15:00 | 11301.20 | 2024-12-10 14:15:00 | 11198.85 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-12-23 12:45:00 | 10841.10 | 2024-12-26 15:15:00 | 10911.80 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-12-24 09:30:00 | 10825.00 | 2024-12-26 15:15:00 | 10911.80 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-12-24 11:30:00 | 10843.00 | 2024-12-26 15:15:00 | 10911.80 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-12-30 11:30:00 | 10924.00 | 2024-12-30 12:15:00 | 10898.55 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-01-01 09:30:00 | 10831.95 | 2025-01-01 10:15:00 | 10910.25 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-01-08 10:00:00 | 11767.95 | 2025-01-10 09:15:00 | 11640.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-01-08 10:30:00 | 11765.90 | 2025-01-10 09:15:00 | 11640.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-01-09 12:15:00 | 11761.30 | 2025-01-10 09:15:00 | 11640.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-01-24 11:45:00 | 12072.85 | 2025-01-24 13:15:00 | 11970.90 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-02-07 11:15:00 | 13083.85 | 2025-02-10 09:15:00 | 12940.00 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-02-14 10:30:00 | 12666.00 | 2025-02-17 15:15:00 | 12759.95 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-02-14 15:15:00 | 12660.00 | 2025-02-17 15:15:00 | 12759.95 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-02-17 13:15:00 | 12686.30 | 2025-02-17 15:15:00 | 12759.95 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-02-25 12:45:00 | 12408.10 | 2025-02-25 14:15:00 | 12479.15 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-03-05 13:30:00 | 11684.00 | 2025-03-13 10:15:00 | 11647.10 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2025-03-20 09:15:00 | 11646.75 | 2025-03-26 11:15:00 | 11790.35 | STOP_HIT | 1.00 | 1.23% |
| BUY | retest2 | 2025-03-20 10:45:00 | 11635.00 | 2025-03-26 11:15:00 | 11790.35 | STOP_HIT | 1.00 | 1.34% |
| SELL | retest2 | 2025-03-27 12:45:00 | 11728.00 | 2025-04-02 13:15:00 | 11670.00 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2025-04-08 10:30:00 | 11337.50 | 2025-04-09 10:15:00 | 11516.80 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-04-23 09:15:00 | 11847.00 | 2025-04-25 14:15:00 | 11732.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-04-25 12:45:00 | 11822.00 | 2025-04-25 14:15:00 | 11732.00 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-05-07 10:30:00 | 12610.00 | 2025-05-08 13:15:00 | 12338.00 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-05-13 14:45:00 | 12506.00 | 2025-05-20 13:15:00 | 12700.00 | STOP_HIT | 1.00 | 1.55% |
| BUY | retest2 | 2025-05-14 09:15:00 | 12500.00 | 2025-05-20 13:15:00 | 12700.00 | STOP_HIT | 1.00 | 1.60% |
| SELL | retest2 | 2025-05-29 11:15:00 | 12338.00 | 2025-05-29 15:15:00 | 12427.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-05-29 13:15:00 | 12339.00 | 2025-05-29 15:15:00 | 12427.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-06-03 10:45:00 | 12154.00 | 2025-06-06 10:15:00 | 12471.00 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2025-06-03 14:45:00 | 12164.00 | 2025-06-06 10:15:00 | 12471.00 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-06-04 09:30:00 | 12157.00 | 2025-06-06 10:15:00 | 12471.00 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2025-06-04 11:45:00 | 12169.00 | 2025-06-06 10:15:00 | 12471.00 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest1 | 2025-06-20 09:15:00 | 12849.00 | 2025-06-23 09:15:00 | 12698.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-07-04 13:00:00 | 12656.00 | 2025-07-07 09:15:00 | 12508.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-07-04 14:00:00 | 12650.00 | 2025-07-07 09:15:00 | 12508.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-07-09 13:30:00 | 12459.00 | 2025-07-10 09:15:00 | 12589.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-07-09 15:15:00 | 12453.00 | 2025-07-10 09:15:00 | 12589.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-07-31 11:15:00 | 12574.00 | 2025-08-01 10:15:00 | 12442.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-07-31 12:45:00 | 12560.00 | 2025-08-01 10:15:00 | 12442.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-07-31 14:00:00 | 12573.00 | 2025-08-01 10:15:00 | 12442.00 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-08-01 09:15:00 | 12706.00 | 2025-08-01 10:15:00 | 12442.00 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-08-08 11:45:00 | 12672.00 | 2025-08-18 09:15:00 | 13939.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-12 09:30:00 | 12695.00 | 2025-08-18 09:15:00 | 13964.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-16 09:15:00 | 15310.00 | 2025-09-29 11:15:00 | 16065.00 | STOP_HIT | 1.00 | 4.93% |
| SELL | retest2 | 2025-10-01 11:45:00 | 15877.00 | 2025-10-06 13:15:00 | 16042.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-10-03 09:15:00 | 15849.00 | 2025-10-06 13:15:00 | 16042.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-10-03 09:45:00 | 15817.00 | 2025-10-06 13:15:00 | 16042.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-10-14 14:15:00 | 16247.00 | 2025-10-24 10:15:00 | 16310.00 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest2 | 2025-10-15 09:15:00 | 16330.00 | 2025-10-24 10:15:00 | 16310.00 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-10-16 09:15:00 | 16322.00 | 2025-10-24 10:15:00 | 16310.00 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-10-27 13:00:00 | 16309.00 | 2025-10-27 14:15:00 | 16400.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-10-27 14:00:00 | 16320.00 | 2025-10-27 14:15:00 | 16400.00 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-11-17 11:30:00 | 15801.00 | 2025-11-19 14:15:00 | 15776.00 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-11-19 12:30:00 | 15773.00 | 2025-11-19 14:15:00 | 15776.00 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2025-11-25 09:15:00 | 15999.00 | 2025-11-25 14:15:00 | 15897.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-11-25 11:00:00 | 15978.00 | 2025-11-25 14:15:00 | 15897.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-11-26 09:15:00 | 16020.00 | 2025-11-27 12:15:00 | 15901.00 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-11-27 13:45:00 | 15970.00 | 2025-11-27 14:15:00 | 15904.00 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-12-03 14:00:00 | 16090.00 | 2025-12-04 09:15:00 | 15987.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-12-03 15:00:00 | 16087.00 | 2025-12-04 09:15:00 | 15987.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-12-04 09:15:00 | 16087.00 | 2025-12-04 09:15:00 | 15987.00 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-12-10 13:30:00 | 16033.00 | 2025-12-11 09:15:00 | 16176.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-12-17 09:15:00 | 16454.00 | 2025-12-18 09:15:00 | 16251.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-12-26 13:00:00 | 16616.00 | 2025-12-29 10:15:00 | 16530.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-12-26 15:00:00 | 16613.00 | 2025-12-29 10:15:00 | 16530.00 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-12-29 09:15:00 | 16664.00 | 2025-12-29 10:15:00 | 16530.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-01-13 10:45:00 | 16499.00 | 2026-01-23 10:15:00 | 15674.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 10:45:00 | 16499.00 | 2026-01-28 09:15:00 | 14849.10 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-06 11:30:00 | 15089.00 | 2026-02-16 09:15:00 | 15218.00 | STOP_HIT | 1.00 | 0.85% |
| BUY | retest2 | 2026-02-10 09:30:00 | 15136.00 | 2026-02-16 09:15:00 | 15218.00 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2026-02-23 13:00:00 | 15020.00 | 2026-02-25 11:15:00 | 15127.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-02-23 13:30:00 | 15025.00 | 2026-02-25 11:15:00 | 15127.00 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2026-02-24 09:15:00 | 15013.00 | 2026-02-25 11:15:00 | 15127.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-02-25 10:00:00 | 14989.00 | 2026-02-25 11:15:00 | 15127.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-03-06 09:15:00 | 14318.00 | 2026-03-09 09:15:00 | 13602.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 12:15:00 | 14320.00 | 2026-03-09 09:15:00 | 13604.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 13:00:00 | 14326.00 | 2026-03-09 09:15:00 | 13609.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 09:15:00 | 14318.00 | 2026-03-10 10:15:00 | 13694.00 | STOP_HIT | 0.50 | 4.36% |
| SELL | retest2 | 2026-03-06 12:15:00 | 14320.00 | 2026-03-10 10:15:00 | 13694.00 | STOP_HIT | 0.50 | 4.37% |
| SELL | retest2 | 2026-03-06 13:00:00 | 14326.00 | 2026-03-10 10:15:00 | 13694.00 | STOP_HIT | 0.50 | 4.41% |
| SELL | retest2 | 2026-03-24 14:15:00 | 12510.00 | 2026-03-25 09:15:00 | 12720.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-04-01 10:15:00 | 12593.00 | 2026-04-01 12:15:00 | 12526.00 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2026-04-07 11:45:00 | 12681.00 | 2026-04-13 11:15:00 | 13174.00 | STOP_HIT | 1.00 | 3.89% |
| BUY | retest2 | 2026-04-07 12:45:00 | 12655.00 | 2026-04-13 11:15:00 | 13174.00 | STOP_HIT | 1.00 | 4.10% |
| SELL | retest2 | 2026-04-15 13:15:00 | 13278.00 | 2026-04-17 09:15:00 | 13611.00 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2026-04-15 14:00:00 | 13279.00 | 2026-04-17 09:15:00 | 13611.00 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2026-04-15 15:00:00 | 13287.00 | 2026-04-17 09:15:00 | 13611.00 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2026-04-16 09:45:00 | 13248.00 | 2026-04-17 09:15:00 | 13611.00 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2026-04-27 10:30:00 | 13055.00 | 2026-04-27 12:15:00 | 13235.00 | STOP_HIT | 1.00 | -1.38% |
