# Bajaj Holdings & Investment Ltd. (BAJAJHLDNG)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 10678.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 166 |
| ALERT1 | 105 |
| ALERT2 | 105 |
| ALERT2_SKIP | 58 |
| ALERT3 | 264 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 9 |
| ENTRY2 | 134 |
| PARTIAL | 18 |
| TARGET_HIT | 13 |
| STOP_HIT | 131 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 161 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 72 / 89
- **Target hits / Stop hits / Partials:** 13 / 130 / 18
- **Avg / median % per leg:** 1.07% / -0.55%
- **Sum % (uncompounded):** 172.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 68 | 24 | 35.3% | 7 | 61 | 0 | 0.73% | 49.5% |
| BUY @ 2nd Alert (retest1) | 7 | 1 | 14.3% | 1 | 6 | 0 | 0.46% | 3.2% |
| BUY @ 3rd Alert (retest2) | 61 | 23 | 37.7% | 6 | 55 | 0 | 0.76% | 46.3% |
| SELL (all) | 93 | 48 | 51.6% | 6 | 69 | 18 | 1.33% | 123.3% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.96% | -1.9% |
| SELL @ 3rd Alert (retest2) | 91 | 48 | 52.7% | 6 | 67 | 18 | 1.38% | 125.2% |
| retest1 (combined) | 9 | 1 | 11.1% | 1 | 8 | 0 | 0.14% | 1.3% |
| retest2 (combined) | 152 | 71 | 46.7% | 12 | 122 | 18 | 1.13% | 171.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 10:15:00 | 8349.95 | 8404.18 | 8411.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 14:15:00 | 8335.05 | 8367.31 | 8389.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-18 11:15:00 | 8279.95 | 8254.41 | 8299.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-18 11:30:00 | 8230.55 | 8254.41 | 8299.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 8138.55 | 8236.60 | 8283.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 15:15:00 | 8019.05 | 8084.54 | 8118.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 14:15:00 | 8002.90 | 7929.90 | 7921.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 14:15:00 | 8002.90 | 7929.90 | 7921.02 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 12:15:00 | 7836.60 | 7905.65 | 7913.46 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 15:15:00 | 7965.05 | 7926.32 | 7921.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 8023.45 | 7945.75 | 7930.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 7946.15 | 7988.73 | 7966.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 7946.15 | 7988.73 | 7966.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 7946.15 | 7988.73 | 7966.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 7879.20 | 7988.73 | 7966.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 7875.60 | 7966.10 | 7958.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 7875.60 | 7966.10 | 7958.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 7773.30 | 7927.54 | 7941.65 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 09:15:00 | 8078.10 | 7944.45 | 7941.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 8126.45 | 8051.41 | 8007.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 11:15:00 | 8430.00 | 8433.63 | 8343.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 12:00:00 | 8430.00 | 8433.63 | 8343.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 8359.65 | 8411.66 | 8355.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 15:00:00 | 8359.65 | 8411.66 | 8355.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 8400.00 | 8409.33 | 8359.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 09:15:00 | 8346.00 | 8409.33 | 8359.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 8375.00 | 8402.46 | 8361.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 09:45:00 | 8350.00 | 8402.46 | 8361.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 10:15:00 | 8360.45 | 8394.06 | 8360.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 10:30:00 | 8350.40 | 8394.06 | 8360.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 11:15:00 | 8350.00 | 8385.25 | 8359.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 11:30:00 | 8364.70 | 8385.25 | 8359.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 12:15:00 | 8355.00 | 8379.20 | 8359.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 09:15:00 | 8412.55 | 8366.24 | 8357.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-13 12:15:00 | 8339.45 | 8358.62 | 8357.26 | SL hit (close<static) qty=1.00 sl=8339.80 alert=retest2 |

### Cycle 7 — SELL (started 2024-06-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 13:15:00 | 8314.80 | 8349.86 | 8353.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-13 14:15:00 | 8240.00 | 8327.89 | 8343.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 09:15:00 | 8348.65 | 8313.77 | 8322.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 09:15:00 | 8348.65 | 8313.77 | 8322.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 8348.65 | 8313.77 | 8322.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:00:00 | 8348.65 | 8313.77 | 8322.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 8341.05 | 8319.23 | 8324.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:30:00 | 8350.75 | 8319.23 | 8324.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 13:15:00 | 8298.80 | 8319.28 | 8323.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-18 15:15:00 | 8273.60 | 8311.97 | 8319.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 09:30:00 | 8271.85 | 8291.88 | 8308.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 09:15:00 | 8606.80 | 8306.57 | 8267.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 09:15:00 | 8606.80 | 8306.57 | 8267.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 10:15:00 | 8860.00 | 8417.25 | 8321.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 11:15:00 | 8634.05 | 8639.04 | 8519.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-25 12:00:00 | 8634.05 | 8639.04 | 8519.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 12:15:00 | 8589.10 | 8629.06 | 8525.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 12:30:00 | 8540.90 | 8629.06 | 8525.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 8551.80 | 8600.07 | 8543.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:45:00 | 8550.00 | 8600.07 | 8543.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 8600.35 | 8600.13 | 8548.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 09:15:00 | 8673.55 | 8528.97 | 8528.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 11:15:00 | 8499.15 | 8542.96 | 8537.31 | SL hit (close<static) qty=1.00 sl=8546.50 alert=retest2 |

### Cycle 9 — SELL (started 2024-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 12:15:00 | 8495.55 | 8533.48 | 8533.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 8450.00 | 8516.78 | 8525.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 09:15:00 | 8609.75 | 8516.14 | 8521.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 09:15:00 | 8609.75 | 8516.14 | 8521.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 8609.75 | 8516.14 | 8521.56 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 10:15:00 | 8599.40 | 8532.79 | 8528.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 8700.00 | 8578.55 | 8555.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 10:15:00 | 9697.80 | 9702.40 | 9431.97 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 13:30:00 | 9794.00 | 9729.35 | 9600.17 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 09:30:00 | 9839.15 | 9804.77 | 9671.47 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 12:30:00 | 9754.40 | 9782.00 | 9693.19 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 13:30:00 | 9776.00 | 9781.18 | 9700.89 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 9707.45 | 9776.93 | 9720.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-10 09:15:00 | 9707.45 | 9776.93 | 9720.21 | SL hit (close<ema400) qty=1.00 sl=9720.21 alert=retest1 |

### Cycle 11 — SELL (started 2024-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 13:15:00 | 9697.00 | 9714.51 | 9715.45 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2024-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 14:15:00 | 9775.60 | 9726.72 | 9720.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 10:15:00 | 9870.05 | 9762.09 | 9739.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 10:15:00 | 9845.00 | 9846.85 | 9802.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 11:00:00 | 9845.00 | 9846.85 | 9802.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 12:15:00 | 9860.95 | 9852.29 | 9812.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 13:00:00 | 9860.95 | 9852.29 | 9812.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 9804.50 | 9860.84 | 9831.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:30:00 | 9809.25 | 9860.84 | 9831.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 9838.00 | 9856.27 | 9831.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 11:00:00 | 9838.00 | 9856.27 | 9831.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 11:15:00 | 9814.95 | 9848.01 | 9830.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:00:00 | 9814.95 | 9848.01 | 9830.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 9810.00 | 9840.41 | 9828.33 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2024-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 15:15:00 | 9792.95 | 9820.56 | 9821.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 09:15:00 | 9661.00 | 9788.65 | 9806.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 13:15:00 | 9715.10 | 9677.90 | 9737.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-18 14:00:00 | 9715.10 | 9677.90 | 9737.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 9789.20 | 9700.16 | 9742.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 15:00:00 | 9789.20 | 9700.16 | 9742.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 9713.00 | 9702.73 | 9739.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:15:00 | 9842.00 | 9702.73 | 9739.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 9870.00 | 9736.18 | 9751.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 9870.00 | 9736.18 | 9751.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 9825.00 | 9753.94 | 9758.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 11:30:00 | 9787.00 | 9753.63 | 9757.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-22 09:15:00 | 9903.00 | 9752.65 | 9750.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2024-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 09:15:00 | 9903.00 | 9752.65 | 9750.51 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 15:15:00 | 9740.00 | 9815.92 | 9816.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-24 13:15:00 | 9643.40 | 9767.77 | 9792.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 09:15:00 | 9513.75 | 9483.86 | 9581.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-26 09:30:00 | 9539.95 | 9483.86 | 9581.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 15:15:00 | 9470.10 | 9471.42 | 9531.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 09:15:00 | 9697.55 | 9471.42 | 9531.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 9584.45 | 9494.02 | 9536.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 09:30:00 | 9669.95 | 9494.02 | 9536.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 10:15:00 | 9580.85 | 9511.39 | 9540.74 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2024-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 13:15:00 | 9640.70 | 9563.66 | 9559.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 14:15:00 | 9665.80 | 9584.09 | 9569.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 09:15:00 | 9585.00 | 9645.19 | 9620.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 09:15:00 | 9585.00 | 9645.19 | 9620.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 9585.00 | 9645.19 | 9620.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:30:00 | 9538.60 | 9645.19 | 9620.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 9661.00 | 9648.35 | 9624.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 9801.90 | 9638.47 | 9628.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 10:15:00 | 9556.70 | 9627.23 | 9625.42 | SL hit (close<static) qty=1.00 sl=9583.45 alert=retest2 |

### Cycle 17 — SELL (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 11:15:00 | 9486.10 | 9599.01 | 9612.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 13:15:00 | 9414.40 | 9548.32 | 9586.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 09:15:00 | 9522.95 | 9502.26 | 9552.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 09:15:00 | 9522.95 | 9502.26 | 9552.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 9522.95 | 9502.26 | 9552.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:30:00 | 9450.55 | 9502.26 | 9552.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 9524.85 | 9506.78 | 9549.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 15:00:00 | 9461.10 | 9496.91 | 9531.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 13:15:00 | 9401.55 | 9338.51 | 9334.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 13:15:00 | 9401.55 | 9338.51 | 9334.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 11:15:00 | 9516.75 | 9391.11 | 9361.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 9371.45 | 9417.76 | 9383.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 14:15:00 | 9371.45 | 9417.76 | 9383.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 9371.45 | 9417.76 | 9383.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 15:00:00 | 9371.45 | 9417.76 | 9383.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 9402.00 | 9414.61 | 9385.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:15:00 | 9443.05 | 9414.61 | 9385.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 9451.10 | 9421.91 | 9391.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 14:00:00 | 9491.40 | 9447.43 | 9414.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 09:15:00 | 9243.95 | 9388.25 | 9405.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 09:15:00 | 9243.95 | 9388.25 | 9405.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 14:15:00 | 9142.45 | 9261.09 | 9329.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 10:15:00 | 9249.95 | 9240.27 | 9300.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 10:45:00 | 9249.50 | 9240.27 | 9300.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 9299.95 | 9252.21 | 9300.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:45:00 | 9310.00 | 9252.21 | 9300.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 9304.25 | 9262.62 | 9300.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 12:45:00 | 9313.90 | 9262.62 | 9300.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 13:15:00 | 9312.40 | 9272.57 | 9301.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 13:45:00 | 9318.25 | 9272.57 | 9301.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 14:15:00 | 9376.05 | 9293.27 | 9308.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 15:00:00 | 9376.05 | 9293.27 | 9308.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — BUY (started 2024-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 15:15:00 | 9436.00 | 9321.82 | 9320.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 09:15:00 | 9494.60 | 9356.37 | 9336.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 11:15:00 | 9700.10 | 9714.46 | 9648.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 11:45:00 | 9743.90 | 9714.46 | 9648.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 9879.55 | 9859.76 | 9813.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 10:15:00 | 9898.90 | 9859.76 | 9813.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 11:45:00 | 9900.00 | 9868.96 | 9826.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 14:30:00 | 9909.55 | 9876.10 | 9839.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-03 10:15:00 | 10888.79 | 10573.88 | 10355.30 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2024-09-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 12:15:00 | 10746.30 | 10828.81 | 10832.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 10692.50 | 10790.05 | 10813.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 15:15:00 | 10600.00 | 10513.39 | 10624.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 15:15:00 | 10600.00 | 10513.39 | 10624.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 10600.00 | 10513.39 | 10624.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 09:15:00 | 10284.80 | 10513.39 | 10624.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-16 15:15:00 | 10440.00 | 10412.25 | 10410.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2024-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 15:15:00 | 10440.00 | 10412.25 | 10410.80 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 09:15:00 | 10370.70 | 10403.94 | 10407.16 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2024-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 10:15:00 | 10495.05 | 10399.57 | 10395.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 12:15:00 | 10756.00 | 10477.64 | 10431.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 13:15:00 | 10840.15 | 10901.89 | 10777.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 13:15:00 | 10840.15 | 10901.89 | 10777.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 13:15:00 | 10840.15 | 10901.89 | 10777.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 15:00:00 | 11154.65 | 10952.44 | 10811.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 15:00:00 | 11108.95 | 11057.80 | 10949.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 10:45:00 | 11034.25 | 11046.81 | 10971.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-24 15:15:00 | 10870.00 | 10925.61 | 10932.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2024-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 15:15:00 | 10870.00 | 10925.61 | 10932.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 09:15:00 | 10670.00 | 10874.49 | 10908.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 09:15:00 | 10747.20 | 10741.72 | 10809.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 09:15:00 | 10747.20 | 10741.72 | 10809.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 10747.20 | 10741.72 | 10809.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:45:00 | 10801.65 | 10741.72 | 10809.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 10780.35 | 10739.82 | 10781.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 10780.35 | 10739.82 | 10781.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 10825.00 | 10756.85 | 10785.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 10761.65 | 10756.85 | 10785.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 10796.15 | 10764.71 | 10786.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 15:00:00 | 10526.00 | 10722.82 | 10760.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 10:15:00 | 10563.85 | 10399.30 | 10387.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 10563.85 | 10399.30 | 10387.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 10:15:00 | 10648.00 | 10515.22 | 10459.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 09:15:00 | 10711.10 | 10769.04 | 10683.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 09:15:00 | 10711.10 | 10769.04 | 10683.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 10711.10 | 10769.04 | 10683.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:45:00 | 10686.30 | 10769.04 | 10683.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 10677.40 | 10750.71 | 10682.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 12:15:00 | 10735.10 | 10739.76 | 10683.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 13:30:00 | 10711.45 | 10730.84 | 10689.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 14:15:00 | 10647.15 | 10714.11 | 10685.38 | SL hit (close<static) qty=1.00 sl=10660.00 alert=retest2 |

### Cycle 27 — SELL (started 2024-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 09:15:00 | 10640.00 | 10679.41 | 10679.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 14:15:00 | 10530.40 | 10623.45 | 10650.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 14:15:00 | 10401.50 | 10390.03 | 10492.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-17 15:00:00 | 10401.50 | 10390.03 | 10492.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 10262.10 | 10251.96 | 10341.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 14:30:00 | 10153.45 | 10246.18 | 10286.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 09:15:00 | 10136.95 | 10252.62 | 10285.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:15:00 | 10154.25 | 10230.08 | 10268.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 12:45:00 | 10158.60 | 10205.20 | 10250.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 10084.45 | 10149.40 | 10206.96 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-28 11:15:00 | 10286.55 | 10198.69 | 10197.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 11:15:00 | 10286.55 | 10198.69 | 10197.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 14:15:00 | 10377.55 | 10296.06 | 10257.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 12:15:00 | 10311.25 | 10319.24 | 10286.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 12:45:00 | 10309.70 | 10319.24 | 10286.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 13:15:00 | 10308.00 | 10316.99 | 10288.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 13:45:00 | 10315.35 | 10316.99 | 10288.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 10250.05 | 10303.60 | 10284.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 15:00:00 | 10250.05 | 10303.60 | 10284.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 10228.00 | 10288.48 | 10279.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:15:00 | 10150.45 | 10288.48 | 10279.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2024-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 09:15:00 | 10159.65 | 10262.72 | 10268.68 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-11-01 17:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 17:15:00 | 10330.00 | 10258.80 | 10256.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 18:15:00 | 10350.00 | 10277.04 | 10265.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 09:15:00 | 10419.15 | 10493.42 | 10411.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 09:15:00 | 10419.15 | 10493.42 | 10411.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 10419.15 | 10493.42 | 10411.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:00:00 | 10419.15 | 10493.42 | 10411.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 10595.00 | 10513.73 | 10428.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 11:15:00 | 10603.95 | 10513.73 | 10428.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 12:30:00 | 10633.95 | 10532.24 | 10452.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 13:30:00 | 10600.95 | 10543.58 | 10464.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 14:15:00 | 10609.25 | 10543.58 | 10464.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 10510.45 | 10575.29 | 10537.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:30:00 | 10448.70 | 10575.29 | 10537.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 10542.85 | 10568.80 | 10537.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 11:30:00 | 10557.25 | 10568.36 | 10540.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 12:00:00 | 10566.60 | 10568.36 | 10540.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-07 12:15:00 | 10464.30 | 10547.55 | 10533.58 | SL hit (close<static) qty=1.00 sl=10500.15 alert=retest2 |

### Cycle 31 — SELL (started 2024-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 15:15:00 | 10486.00 | 10522.41 | 10524.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 10408.25 | 10499.58 | 10513.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 12:15:00 | 10482.50 | 10472.24 | 10495.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 12:15:00 | 10482.50 | 10472.24 | 10495.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 10482.50 | 10472.24 | 10495.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:45:00 | 10481.00 | 10472.24 | 10495.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 13:15:00 | 10588.45 | 10495.48 | 10504.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 13:45:00 | 10569.30 | 10495.48 | 10504.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 10493.70 | 10495.12 | 10503.15 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 09:15:00 | 10614.00 | 10516.88 | 10511.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-11 10:15:00 | 10781.10 | 10569.72 | 10536.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-12 12:15:00 | 10679.30 | 10741.01 | 10671.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-12 13:00:00 | 10679.30 | 10741.01 | 10671.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 13:15:00 | 10726.90 | 10738.18 | 10676.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 13:30:00 | 10632.30 | 10738.18 | 10676.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 10648.75 | 10725.10 | 10686.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 09:45:00 | 10591.75 | 10725.10 | 10686.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 10679.75 | 10716.03 | 10685.89 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2024-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 14:15:00 | 10517.10 | 10647.05 | 10660.18 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2024-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 09:15:00 | 10844.15 | 10671.70 | 10668.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-14 10:15:00 | 11043.25 | 10746.01 | 10702.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-14 14:15:00 | 10802.30 | 10835.53 | 10767.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 14:15:00 | 10802.30 | 10835.53 | 10767.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 14:15:00 | 10802.30 | 10835.53 | 10767.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-14 15:00:00 | 10802.30 | 10835.53 | 10767.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 15:15:00 | 10850.00 | 10838.42 | 10774.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-18 09:45:00 | 10726.45 | 10824.35 | 10774.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 10:15:00 | 10756.55 | 10810.79 | 10772.43 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2024-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 15:15:00 | 10652.00 | 10739.93 | 10750.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 10:15:00 | 10604.90 | 10691.39 | 10724.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 10531.85 | 10500.55 | 10570.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 10531.85 | 10500.55 | 10570.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 10531.85 | 10500.55 | 10570.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 15:15:00 | 10400.40 | 10508.21 | 10550.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 10:15:00 | 10439.65 | 10516.82 | 10520.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 12:15:00 | 10422.30 | 10512.40 | 10517.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 09:15:00 | 10312.20 | 10490.00 | 10503.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 10338.50 | 10459.70 | 10488.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 10:30:00 | 10300.00 | 10427.22 | 10471.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 13:15:00 | 10309.80 | 10372.47 | 10436.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 13:45:00 | 10299.90 | 10357.98 | 10424.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 09:30:00 | 10247.55 | 10293.12 | 10334.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 10300.45 | 10279.21 | 10312.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 14:15:00 | 10320.85 | 10279.21 | 10312.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 10507.05 | 10324.78 | 10330.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 15:00:00 | 10507.05 | 10324.78 | 10330.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-29 15:15:00 | 10500.00 | 10359.82 | 10345.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2024-11-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 15:15:00 | 10500.00 | 10359.82 | 10345.59 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 10:15:00 | 10301.00 | 10330.25 | 10333.51 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2024-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 13:15:00 | 10367.70 | 10338.77 | 10336.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 14:15:00 | 10405.80 | 10352.18 | 10342.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 11:15:00 | 10600.00 | 10608.45 | 10522.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 12:00:00 | 10600.00 | 10608.45 | 10522.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 13:15:00 | 10632.90 | 10605.73 | 10535.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 13:30:00 | 10555.00 | 10605.73 | 10535.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 10580.15 | 10596.53 | 10548.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 12:45:00 | 10698.00 | 10602.23 | 10561.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 15:00:00 | 10766.95 | 10644.98 | 10588.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 10:45:00 | 10694.60 | 10671.05 | 10616.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 09:15:00 | 10791.00 | 10683.55 | 10644.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 11208.20 | 11239.33 | 11168.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:30:00 | 11186.95 | 11239.33 | 11168.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 11106.85 | 11212.84 | 11162.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 11:00:00 | 11106.85 | 11212.84 | 11162.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 11122.20 | 11194.71 | 11158.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 11:30:00 | 11110.40 | 11194.71 | 11158.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 11120.80 | 11182.40 | 11159.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:45:00 | 11111.45 | 11182.40 | 11159.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 11122.20 | 11170.36 | 11156.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:30:00 | 11122.05 | 11170.36 | 11156.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 11080.00 | 11152.29 | 11149.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 09:15:00 | 11206.05 | 11152.29 | 11149.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 10:15:00 | 11105.40 | 11141.47 | 11144.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2024-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 10:15:00 | 11105.40 | 11141.47 | 11144.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 14:15:00 | 11056.00 | 11102.23 | 11123.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-17 10:15:00 | 11127.95 | 11095.42 | 11114.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 10:15:00 | 11127.95 | 11095.42 | 11114.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 11127.95 | 11095.42 | 11114.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 11:00:00 | 11127.95 | 11095.42 | 11114.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 11:15:00 | 11302.05 | 11136.74 | 11131.23 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2024-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 09:15:00 | 10861.00 | 11129.81 | 11154.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 12:15:00 | 10853.30 | 10997.81 | 11081.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 13:15:00 | 11052.45 | 11008.74 | 11078.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 14:00:00 | 11052.45 | 11008.74 | 11078.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 11344.65 | 11075.92 | 11102.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:30:00 | 11415.10 | 11075.92 | 11102.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 11280.00 | 11116.74 | 11119.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 09:45:00 | 11198.45 | 11110.55 | 11116.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 11174.60 | 11123.36 | 11121.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2024-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 10:15:00 | 11174.60 | 11123.36 | 11121.32 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 12:15:00 | 11071.55 | 11115.56 | 11118.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 13:15:00 | 10957.40 | 11083.93 | 11103.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 09:15:00 | 11040.95 | 10960.91 | 11032.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-23 09:15:00 | 11040.95 | 10960.91 | 11032.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 11040.95 | 10960.91 | 11032.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 10:00:00 | 11040.95 | 10960.91 | 11032.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 11069.95 | 10982.72 | 11035.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:00:00 | 11069.95 | 10982.72 | 11035.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 11092.50 | 11004.68 | 11040.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 09:15:00 | 10971.40 | 11055.53 | 11056.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 09:15:00 | 11087.10 | 11061.85 | 11059.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2024-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 09:15:00 | 11087.10 | 11061.85 | 11059.04 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 15:15:00 | 11029.15 | 11066.24 | 11066.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 09:15:00 | 10988.40 | 11050.68 | 11059.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 12:15:00 | 11137.10 | 11047.58 | 11053.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 12:15:00 | 11137.10 | 11047.58 | 11053.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 12:15:00 | 11137.10 | 11047.58 | 11053.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 13:00:00 | 11137.10 | 11047.58 | 11053.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2024-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 13:15:00 | 11152.80 | 11068.62 | 11062.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 11289.55 | 11134.21 | 11096.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 15:15:00 | 11261.60 | 11269.67 | 11194.55 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 09:15:00 | 11449.80 | 11269.67 | 11194.55 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 11617.00 | 11339.13 | 11232.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 10:30:00 | 11739.00 | 11394.29 | 11267.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-30 14:15:00 | 12594.78 | 11765.38 | 11484.10 | Target hit (10%) qty=1.00 alert=retest1 |

### Cycle 47 — SELL (started 2025-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 14:15:00 | 11591.20 | 11812.60 | 11827.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 15:15:00 | 11562.00 | 11762.48 | 11803.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 11262.60 | 11250.85 | 11438.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 11:00:00 | 11262.60 | 11250.85 | 11438.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 11461.00 | 11251.45 | 11352.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:45:00 | 11507.00 | 11251.45 | 11352.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 11471.10 | 11295.38 | 11363.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 10:45:00 | 11491.10 | 11295.38 | 11363.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 11:15:00 | 11620.80 | 11360.46 | 11387.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 12:00:00 | 11620.80 | 11360.46 | 11387.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2025-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 13:15:00 | 11496.60 | 11409.73 | 11406.19 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2025-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 14:15:00 | 11264.50 | 11380.68 | 11393.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 15:15:00 | 11190.00 | 11342.55 | 11374.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 10520.00 | 10518.47 | 10740.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 10520.00 | 10518.47 | 10740.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 10640.00 | 10548.26 | 10684.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 15:00:00 | 10640.00 | 10548.26 | 10684.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 10688.00 | 10576.21 | 10684.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 10842.35 | 10576.21 | 10684.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 10828.10 | 10626.59 | 10697.96 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2025-01-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 13:15:00 | 10820.00 | 10739.46 | 10735.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 15:15:00 | 10840.00 | 10773.28 | 10752.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 10823.00 | 10877.52 | 10829.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 10823.00 | 10877.52 | 10829.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 10823.00 | 10877.52 | 10829.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:00:00 | 10823.00 | 10877.52 | 10829.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 10824.40 | 10866.90 | 10829.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:45:00 | 10795.00 | 10866.90 | 10829.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 10849.90 | 10863.50 | 10830.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 12:15:00 | 10831.45 | 10863.50 | 10830.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 12:15:00 | 10805.30 | 10851.86 | 10828.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 13:00:00 | 10805.30 | 10851.86 | 10828.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 10821.20 | 10845.73 | 10827.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 14:45:00 | 10859.70 | 10836.98 | 10825.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-17 15:15:00 | 10800.00 | 10829.58 | 10823.29 | SL hit (close<static) qty=1.00 sl=10805.30 alert=retest2 |

### Cycle 51 — SELL (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 12:15:00 | 10786.15 | 10842.14 | 10847.19 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 09:15:00 | 11277.70 | 10913.97 | 10876.43 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 13:15:00 | 11050.00 | 11176.69 | 11190.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 14:15:00 | 11021.95 | 11145.74 | 11175.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 11299.90 | 11169.25 | 11180.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 11299.90 | 11169.25 | 11180.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 11299.90 | 11169.25 | 11180.25 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 11278.85 | 11191.17 | 11189.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 11456.95 | 11278.98 | 11234.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 10:15:00 | 11323.00 | 11354.51 | 11285.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 10:15:00 | 11323.00 | 11354.51 | 11285.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 11323.00 | 11354.51 | 11285.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 11:00:00 | 11323.00 | 11354.51 | 11285.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 11:15:00 | 11377.30 | 11359.07 | 11293.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 11:30:00 | 11352.50 | 11359.07 | 11293.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 12:15:00 | 11286.00 | 11344.46 | 11293.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 13:00:00 | 11286.00 | 11344.46 | 11293.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 11290.10 | 11333.59 | 11292.91 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 11:15:00 | 11207.15 | 11261.62 | 11267.66 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-01-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 13:15:00 | 11310.05 | 11271.76 | 11271.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 14:15:00 | 11431.00 | 11303.60 | 11285.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-04 09:15:00 | 11881.50 | 11991.95 | 11806.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-04 10:00:00 | 11881.50 | 11991.95 | 11806.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 11740.00 | 11941.56 | 11800.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:45:00 | 11778.15 | 11941.56 | 11800.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 11597.15 | 11872.68 | 11782.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 12:00:00 | 11597.15 | 11872.68 | 11782.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 11740.00 | 11789.07 | 11765.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 11597.55 | 11789.07 | 11765.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 09:15:00 | 11475.20 | 11726.29 | 11738.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-05 14:15:00 | 11365.35 | 11548.29 | 11636.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 09:15:00 | 11539.25 | 11515.56 | 11604.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 11539.25 | 11515.56 | 11604.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 11539.25 | 11515.56 | 11604.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 12:45:00 | 11435.90 | 11493.78 | 11571.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 13:15:00 | 11403.80 | 11346.95 | 11392.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 13:45:00 | 11450.00 | 11381.56 | 11404.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 15:15:00 | 11523.10 | 11428.98 | 11423.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2025-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-10 15:15:00 | 11523.10 | 11428.98 | 11423.18 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2025-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 10:15:00 | 11345.75 | 11422.69 | 11432.43 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2025-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 13:15:00 | 11595.05 | 11464.38 | 11449.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-12 14:15:00 | 11945.45 | 11560.59 | 11494.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 11:15:00 | 12058.40 | 12075.88 | 11905.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-14 11:45:00 | 12058.65 | 12075.88 | 11905.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 14:15:00 | 12052.95 | 12052.38 | 11935.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 14:45:00 | 12055.00 | 12052.38 | 11935.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 15:15:00 | 12032.60 | 12048.42 | 11944.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-17 09:15:00 | 11847.35 | 12048.42 | 11944.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 09:15:00 | 11900.20 | 12018.78 | 11940.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-17 09:30:00 | 11858.85 | 12018.78 | 11940.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 11830.00 | 11981.02 | 11930.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-17 11:00:00 | 11830.00 | 11981.02 | 11930.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 11:15:00 | 11740.15 | 11932.85 | 11912.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-17 12:00:00 | 11740.15 | 11932.85 | 11912.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2025-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 13:15:00 | 11830.25 | 11894.81 | 11897.89 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 14:15:00 | 12050.00 | 11925.85 | 11911.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 11:15:00 | 12180.10 | 12014.24 | 11960.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-18 13:15:00 | 11882.80 | 11996.28 | 11962.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-18 13:15:00 | 11882.80 | 11996.28 | 11962.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 13:15:00 | 11882.80 | 11996.28 | 11962.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 14:00:00 | 11882.80 | 11996.28 | 11962.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 11994.30 | 11995.89 | 11965.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-19 09:15:00 | 12131.15 | 12000.71 | 11970.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 13:15:00 | 11819.50 | 11965.19 | 11970.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 13:15:00 | 11819.50 | 11965.19 | 11970.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-19 14:15:00 | 11749.85 | 11922.12 | 11950.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 09:15:00 | 12100.40 | 11929.43 | 11947.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 09:15:00 | 12100.40 | 11929.43 | 11947.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 12100.40 | 11929.43 | 11947.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 09:30:00 | 12099.35 | 11929.43 | 11947.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 10:15:00 | 12084.15 | 11960.37 | 11959.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 11:15:00 | 12238.00 | 12015.90 | 11984.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 12145.00 | 12216.68 | 12112.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 12145.00 | 12216.68 | 12112.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 12230.00 | 12245.48 | 12178.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 12060.80 | 12245.48 | 12178.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 12080.00 | 12212.39 | 12169.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:30:00 | 11943.40 | 12212.39 | 12169.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 12332.85 | 12236.48 | 12184.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 12:30:00 | 12526.75 | 12241.79 | 12194.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 13:15:00 | 11990.10 | 12191.45 | 12175.97 | SL hit (close<static) qty=1.00 sl=12035.50 alert=retest2 |

### Cycle 65 — SELL (started 2025-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 14:15:00 | 11985.00 | 12150.16 | 12158.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 09:15:00 | 11900.00 | 12073.54 | 12120.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 09:15:00 | 12198.30 | 11866.84 | 11955.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 09:15:00 | 12198.30 | 11866.84 | 11955.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 12198.30 | 11866.84 | 11955.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 10:00:00 | 12198.30 | 11866.84 | 11955.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 10:15:00 | 12340.00 | 11961.47 | 11990.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 10:30:00 | 12341.95 | 11961.47 | 11990.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2025-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 11:15:00 | 12223.15 | 12013.81 | 12011.22 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 11456.45 | 11944.30 | 11990.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 13:15:00 | 11368.00 | 11503.07 | 11658.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 09:15:00 | 11333.85 | 11275.27 | 11407.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-05 09:15:00 | 11333.85 | 11275.27 | 11407.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 11333.85 | 11275.27 | 11407.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:45:00 | 11370.25 | 11275.27 | 11407.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 11357.15 | 11291.64 | 11402.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:30:00 | 11343.90 | 11291.64 | 11402.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 11:15:00 | 11460.00 | 11325.31 | 11407.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:30:00 | 11454.60 | 11325.31 | 11407.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 12:15:00 | 11484.00 | 11357.05 | 11414.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 12:45:00 | 11507.65 | 11357.05 | 11414.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2025-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 09:15:00 | 11648.05 | 11476.98 | 11458.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 10:15:00 | 11906.65 | 11562.92 | 11499.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 13:15:00 | 11488.00 | 11651.09 | 11564.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-06 13:15:00 | 11488.00 | 11651.09 | 11564.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 13:15:00 | 11488.00 | 11651.09 | 11564.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 14:00:00 | 11488.00 | 11651.09 | 11564.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 14:15:00 | 11375.95 | 11596.06 | 11547.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 14:30:00 | 11371.45 | 11596.06 | 11547.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 11425.00 | 11561.85 | 11536.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 09:15:00 | 11563.95 | 11561.85 | 11536.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 09:15:00 | 11373.80 | 11636.77 | 11653.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 11373.80 | 11636.77 | 11653.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 14:15:00 | 11318.35 | 11388.54 | 11440.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 11390.30 | 11380.96 | 11427.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 11390.30 | 11380.96 | 11427.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 11390.30 | 11380.96 | 11427.39 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2025-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 12:15:00 | 11630.55 | 11480.54 | 11465.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 11795.25 | 11579.51 | 11519.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-18 14:15:00 | 11619.90 | 11633.89 | 11575.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-18 14:30:00 | 11601.50 | 11633.89 | 11575.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 11703.20 | 11640.73 | 11588.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-19 11:15:00 | 11825.25 | 11673.89 | 11607.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 12:15:00 | 12230.00 | 12417.36 | 12442.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 12:15:00 | 12230.00 | 12417.36 | 12442.01 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 12602.00 | 12464.76 | 12449.88 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 09:15:00 | 12162.40 | 12418.10 | 12437.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 10:15:00 | 12077.00 | 12349.88 | 12404.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-04 12:15:00 | 11498.10 | 11474.70 | 11613.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-04 12:45:00 | 11494.80 | 11474.70 | 11613.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 11499.95 | 11482.53 | 11593.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-04 15:00:00 | 11499.95 | 11482.53 | 11593.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 10660.45 | 10883.22 | 11149.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 10587.60 | 10864.16 | 11024.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:45:00 | 10601.75 | 10837.97 | 10997.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 10:15:00 | 11204.70 | 11069.08 | 11054.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 11204.70 | 11069.08 | 11054.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 11316.55 | 11118.58 | 11077.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 10:15:00 | 11594.00 | 11615.59 | 11451.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 11592.00 | 11600.11 | 11513.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 11592.00 | 11600.11 | 11513.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 10:30:00 | 11743.00 | 11644.69 | 11541.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 15:15:00 | 12151.00 | 12286.29 | 12300.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2025-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 15:15:00 | 12151.00 | 12286.29 | 12300.19 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2025-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 12:15:00 | 12354.00 | 12313.29 | 12309.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 12401.00 | 12340.68 | 12324.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 11:15:00 | 12329.00 | 12340.80 | 12327.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 11:15:00 | 12329.00 | 12340.80 | 12327.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 12329.00 | 12340.80 | 12327.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 12:00:00 | 12329.00 | 12340.80 | 12327.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 12:15:00 | 12335.00 | 12339.64 | 12327.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-29 13:45:00 | 12347.00 | 12339.11 | 12328.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-29 14:15:00 | 12367.00 | 12339.11 | 12328.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 09:15:00 | 12123.00 | 12290.00 | 12308.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 09:15:00 | 12123.00 | 12290.00 | 12308.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 10:15:00 | 12101.00 | 12252.20 | 12289.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 10:15:00 | 12086.00 | 12057.86 | 12152.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-02 10:45:00 | 12025.00 | 12057.86 | 12152.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 11:15:00 | 12065.00 | 12059.29 | 12144.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 11:30:00 | 12116.00 | 12059.29 | 12144.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 12011.00 | 12025.57 | 12094.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 10:30:00 | 11960.00 | 12010.66 | 12081.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 13:30:00 | 11907.00 | 11957.10 | 12035.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:45:00 | 11948.00 | 11940.93 | 12007.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 12:00:00 | 11941.00 | 11956.16 | 12003.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 11981.00 | 11906.68 | 11947.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:45:00 | 11983.00 | 11906.68 | 11947.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 11951.00 | 11915.55 | 11947.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 14:00:00 | 11948.00 | 11922.04 | 11947.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 11362.00 | 11616.04 | 11759.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 11311.65 | 11616.04 | 11759.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 11350.60 | 11616.04 | 11759.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 11343.95 | 11616.04 | 11759.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 11350.60 | 11616.04 | 11759.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 11758.00 | 11455.51 | 11582.04 | SL hit (close>ema200) qty=0.50 sl=11455.51 alert=retest2 |

### Cycle 78 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 12019.00 | 11713.67 | 11676.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 12114.00 | 11793.74 | 11716.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 14:15:00 | 11915.00 | 11953.38 | 11856.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 15:00:00 | 11915.00 | 11953.38 | 11856.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 11979.00 | 11949.97 | 11870.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 12:00:00 | 12055.00 | 11971.14 | 11894.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 15:15:00 | 12069.00 | 11990.61 | 11923.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-16 13:15:00 | 13260.50 | 12906.30 | 12561.41 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 09:15:00 | 13561.00 | 13709.83 | 13729.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 13:15:00 | 13475.00 | 13605.50 | 13668.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 10:15:00 | 13740.00 | 13521.42 | 13596.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 10:15:00 | 13740.00 | 13521.42 | 13596.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 13740.00 | 13521.42 | 13596.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:00:00 | 13740.00 | 13521.42 | 13596.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 13970.00 | 13611.14 | 13630.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:00:00 | 13970.00 | 13611.14 | 13630.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 12:15:00 | 13928.00 | 13674.51 | 13657.19 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 15:15:00 | 13500.00 | 13634.82 | 13645.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 13394.00 | 13586.66 | 13622.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 11:15:00 | 13324.00 | 13257.78 | 13346.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 12:00:00 | 13324.00 | 13257.78 | 13346.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 13240.00 | 13254.23 | 13336.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:30:00 | 13302.00 | 13254.23 | 13336.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 13272.00 | 13257.78 | 13331.01 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 13450.00 | 13367.65 | 13364.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 14:15:00 | 13485.00 | 13410.68 | 13386.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 12:15:00 | 14364.00 | 14373.31 | 14251.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 13:00:00 | 14364.00 | 14373.31 | 14251.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 14229.00 | 14358.51 | 14299.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 14229.00 | 14358.51 | 14299.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 14253.00 | 14337.40 | 14295.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:00:00 | 14253.00 | 14337.40 | 14295.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 14084.00 | 14286.72 | 14276.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 14084.00 | 14286.72 | 14276.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 14081.00 | 14245.58 | 14258.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 14026.00 | 14193.25 | 14231.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 14124.00 | 14068.77 | 14142.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 14:15:00 | 14124.00 | 14068.77 | 14142.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 14124.00 | 14068.77 | 14142.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 14124.00 | 14068.77 | 14142.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 14020.00 | 14059.01 | 14131.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 13808.00 | 14059.01 | 14131.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 13:30:00 | 13992.00 | 14013.59 | 14073.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 14:45:00 | 13980.00 | 14004.87 | 14064.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 13862.00 | 14003.90 | 14058.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 13879.00 | 13978.92 | 14042.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 13811.00 | 13927.39 | 14006.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 14:30:00 | 13840.00 | 13876.28 | 13960.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 14:00:00 | 13709.00 | 13750.63 | 13849.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 13:15:00 | 13680.00 | 13582.27 | 13579.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2025-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 13:15:00 | 13680.00 | 13582.27 | 13579.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 14:15:00 | 13781.00 | 13622.01 | 13597.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 11:15:00 | 13605.00 | 13673.63 | 13635.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 11:15:00 | 13605.00 | 13673.63 | 13635.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 13605.00 | 13673.63 | 13635.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:00:00 | 13605.00 | 13673.63 | 13635.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 13648.00 | 13668.50 | 13637.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 15:00:00 | 13713.00 | 13679.40 | 13647.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 15:15:00 | 14191.00 | 14298.75 | 14301.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2025-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 15:15:00 | 14191.00 | 14298.75 | 14301.32 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 09:15:00 | 14325.00 | 14304.00 | 14303.47 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 10:15:00 | 14236.00 | 14290.40 | 14297.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 14214.00 | 14275.12 | 14289.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 15:15:00 | 14262.00 | 14251.73 | 14272.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-07 09:15:00 | 14107.00 | 14251.73 | 14272.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 14194.00 | 14240.18 | 14264.93 | EMA400 retest candle locked (from downside) |

### Cycle 88 — BUY (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 15:15:00 | 14338.00 | 14265.86 | 14265.56 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 14138.00 | 14240.29 | 14253.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 14017.00 | 14195.63 | 14232.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 12:15:00 | 13830.00 | 13821.14 | 13918.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 13:00:00 | 13830.00 | 13821.14 | 13918.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 13814.00 | 13836.85 | 13895.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:15:00 | 13777.00 | 13836.85 | 13895.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 11:15:00 | 13788.00 | 13832.08 | 13888.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 12:15:00 | 13781.00 | 13826.86 | 13880.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 13:00:00 | 13795.00 | 13820.49 | 13873.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 13898.00 | 13810.51 | 13848.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 13876.00 | 13810.51 | 13848.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 13847.00 | 13817.81 | 13848.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:15:00 | 13958.00 | 13817.81 | 13848.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 14070.00 | 13868.24 | 13868.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:00:00 | 14070.00 | 13868.24 | 13868.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-14 12:15:00 | 14060.00 | 13906.60 | 13885.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 12:15:00 | 14060.00 | 13906.60 | 13885.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 14:15:00 | 14130.00 | 13980.62 | 13925.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 15:15:00 | 14140.00 | 14169.15 | 14076.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 09:15:00 | 14098.00 | 14169.15 | 14076.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 14096.00 | 14143.78 | 14080.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:45:00 | 14088.00 | 14143.78 | 14080.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 14116.00 | 14138.22 | 14083.90 | EMA400 retest candle locked (from upside) |

### Cycle 91 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 14016.00 | 14077.97 | 14080.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 13:15:00 | 13975.00 | 14029.00 | 14053.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 13:15:00 | 13980.00 | 13896.19 | 13954.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 13:15:00 | 13980.00 | 13896.19 | 13954.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 13980.00 | 13896.19 | 13954.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:15:00 | 14031.00 | 13896.19 | 13954.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 14052.00 | 13927.35 | 13963.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:45:00 | 14050.00 | 13927.35 | 13963.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 14001.00 | 13942.08 | 13966.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 13984.00 | 13942.08 | 13966.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 13993.00 | 13954.01 | 13967.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 13993.00 | 13954.01 | 13967.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 11:15:00 | 14071.00 | 13977.41 | 13977.28 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 13:15:00 | 13950.00 | 13994.48 | 13996.18 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 11:15:00 | 14035.00 | 13999.48 | 13996.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 14:15:00 | 14185.00 | 14052.29 | 14022.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 14015.00 | 14062.22 | 14033.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 14015.00 | 14062.22 | 14033.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 14015.00 | 14062.22 | 14033.50 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 13820.00 | 14013.78 | 14014.09 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2025-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 11:15:00 | 14091.00 | 13995.40 | 13986.12 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-07-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 15:15:00 | 13779.00 | 13972.23 | 13980.49 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 14001.00 | 13941.57 | 13938.86 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 11:15:00 | 13881.00 | 13942.64 | 13947.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 13:15:00 | 13845.00 | 13916.45 | 13934.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 14:15:00 | 13953.00 | 13923.76 | 13936.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 14:15:00 | 13953.00 | 13923.76 | 13936.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 13953.00 | 13923.76 | 13936.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 13953.00 | 13923.76 | 13936.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 15:15:00 | 14059.00 | 13950.81 | 13947.39 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 10:15:00 | 13898.00 | 13936.60 | 13941.30 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 11:15:00 | 13980.00 | 13945.28 | 13944.82 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 13828.00 | 13924.78 | 13936.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 15:15:00 | 13800.00 | 13899.83 | 13923.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 10:15:00 | 13960.00 | 13879.89 | 13908.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 10:15:00 | 13960.00 | 13879.89 | 13908.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 13960.00 | 13879.89 | 13908.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:00:00 | 13960.00 | 13879.89 | 13908.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 14003.00 | 13904.51 | 13917.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:30:00 | 14188.00 | 13904.51 | 13917.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 13903.00 | 13910.11 | 13917.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 13903.00 | 13910.11 | 13917.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 13883.00 | 13904.67 | 13913.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 12:15:00 | 13843.00 | 13904.95 | 13912.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 13:30:00 | 13848.00 | 13876.69 | 13897.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:00:00 | 13797.00 | 13855.34 | 13882.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 09:15:00 | 14171.00 | 13802.57 | 13821.71 | SL hit (close>static) qty=1.00 sl=13976.00 alert=retest2 |

### Cycle 104 — BUY (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 10:15:00 | 14080.00 | 13858.06 | 13845.19 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 12:15:00 | 13792.00 | 13827.88 | 13832.66 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 09:15:00 | 13997.00 | 13847.25 | 13837.30 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 14:15:00 | 13718.00 | 13843.59 | 13844.50 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 13975.00 | 13846.24 | 13839.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 14:15:00 | 14059.00 | 13907.96 | 13870.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 12:15:00 | 13888.00 | 13935.15 | 13902.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 12:15:00 | 13888.00 | 13935.15 | 13902.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 13888.00 | 13935.15 | 13902.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:30:00 | 13889.00 | 13935.15 | 13902.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 13855.00 | 13919.12 | 13898.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:00:00 | 13855.00 | 13919.12 | 13898.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 13824.00 | 13884.24 | 13885.13 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 13900.00 | 13887.39 | 13886.48 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 11:15:00 | 13882.00 | 13885.29 | 13885.62 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 13:15:00 | 13890.00 | 13886.19 | 13885.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 14:15:00 | 13934.00 | 13895.75 | 13890.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 14292.00 | 14471.12 | 14303.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 14292.00 | 14471.12 | 14303.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 14292.00 | 14471.12 | 14303.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 14292.00 | 14471.12 | 14303.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 14176.00 | 14412.10 | 14292.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:45:00 | 14200.00 | 14412.10 | 14292.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 14003.00 | 14330.28 | 14265.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:45:00 | 13998.00 | 14330.28 | 14265.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 13:15:00 | 13795.00 | 14150.54 | 14190.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 09:15:00 | 13527.00 | 13925.61 | 14069.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 13744.00 | 13738.06 | 13881.18 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 12:30:00 | 13657.00 | 13704.17 | 13828.71 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 13790.00 | 13709.60 | 13789.29 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-22 09:15:00 | 13790.00 | 13709.60 | 13789.29 | SL hit (close>ema400) qty=1.00 sl=13789.29 alert=retest1 |

### Cycle 114 — BUY (started 2025-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 14:15:00 | 13056.00 | 12805.37 | 12779.09 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 12875.00 | 12889.70 | 12890.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 14:15:00 | 12736.00 | 12858.96 | 12876.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 12903.00 | 12847.77 | 12867.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 12903.00 | 12847.77 | 12867.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 12903.00 | 12847.77 | 12867.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:45:00 | 12945.00 | 12847.77 | 12867.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 12921.00 | 12862.42 | 12872.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:45:00 | 12921.00 | 12862.42 | 12872.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 12925.00 | 12874.93 | 12876.92 | EMA400 retest candle locked (from downside) |

### Cycle 116 — BUY (started 2025-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 12:15:00 | 12938.00 | 12887.55 | 12882.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 13:15:00 | 13035.00 | 12917.04 | 12896.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 15:15:00 | 12950.00 | 12951.94 | 12917.55 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:15:00 | 13170.00 | 12951.94 | 12917.55 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 12972.00 | 13029.61 | 12980.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-10 13:15:00 | 12972.00 | 13029.61 | 12980.70 | SL hit (close<ema400) qty=1.00 sl=12980.70 alert=retest1 |

### Cycle 117 — SELL (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 10:15:00 | 13273.00 | 13433.92 | 13450.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 12:15:00 | 13235.00 | 13374.15 | 13419.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 14:15:00 | 13266.00 | 13265.34 | 13318.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 14:30:00 | 13265.00 | 13265.34 | 13318.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 13314.00 | 13279.81 | 13316.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:00:00 | 13314.00 | 13279.81 | 13316.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 13253.00 | 13274.45 | 13310.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 11:15:00 | 13230.00 | 13274.45 | 13310.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 14:15:00 | 13242.00 | 13273.20 | 13301.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 14:45:00 | 13236.00 | 13269.36 | 13297.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 15:15:00 | 13242.00 | 13269.36 | 13297.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 13233.00 | 13257.71 | 13286.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 12:00:00 | 13183.00 | 13241.69 | 13274.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:30:00 | 13149.00 | 13194.46 | 13238.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 15:15:00 | 12568.50 | 12773.44 | 12916.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 15:15:00 | 12579.90 | 12773.44 | 12916.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 15:15:00 | 12574.20 | 12773.44 | 12916.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 15:15:00 | 12579.90 | 12773.44 | 12916.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 12857.00 | 12790.15 | 12910.64 | SL hit (close>ema200) qty=0.50 sl=12790.15 alert=retest2 |

### Cycle 118 — BUY (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 15:15:00 | 12145.00 | 11977.23 | 11968.30 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 11900.00 | 12005.90 | 12011.43 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 12112.00 | 12009.89 | 12001.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 13:15:00 | 12179.00 | 12088.72 | 12053.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 11:15:00 | 12132.00 | 12203.95 | 12136.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 11:15:00 | 12132.00 | 12203.95 | 12136.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 12132.00 | 12203.95 | 12136.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:00:00 | 12132.00 | 12203.95 | 12136.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 12139.00 | 12190.96 | 12137.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 14:15:00 | 12269.00 | 12181.57 | 12137.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 12193.00 | 12170.08 | 12140.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:45:00 | 12174.00 | 12168.47 | 12142.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:15:00 | 12179.00 | 12168.47 | 12142.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 12178.00 | 12170.37 | 12145.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 12:00:00 | 12197.00 | 12175.70 | 12150.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 14:15:00 | 12258.00 | 12176.45 | 12155.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 12330.00 | 12180.37 | 12160.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 13:15:00 | 12810.00 | 12946.10 | 12957.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 13:15:00 | 12810.00 | 12946.10 | 12957.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 15:15:00 | 12750.00 | 12883.02 | 12925.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 12:15:00 | 12164.00 | 12132.80 | 12239.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-04 12:45:00 | 12151.00 | 12132.80 | 12239.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 12343.00 | 12167.23 | 12221.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 12343.00 | 12167.23 | 12221.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 10:15:00 | 12675.00 | 12268.78 | 12262.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 11:15:00 | 12882.00 | 12391.43 | 12318.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 12623.00 | 12782.44 | 12578.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-07 10:00:00 | 12623.00 | 12782.44 | 12578.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 12708.00 | 12767.55 | 12590.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:30:00 | 12699.00 | 12767.55 | 12590.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 12560.00 | 12689.78 | 12606.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 12560.00 | 12689.78 | 12606.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 12650.00 | 12681.83 | 12610.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 13022.00 | 12681.83 | 12610.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 12:45:00 | 12736.00 | 12690.21 | 12638.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 13:45:00 | 12742.00 | 12709.17 | 12652.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 12250.00 | 12583.05 | 12605.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 12250.00 | 12583.05 | 12605.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 14:15:00 | 12127.00 | 12307.47 | 12443.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 09:15:00 | 12081.00 | 12071.49 | 12210.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-13 09:45:00 | 12076.00 | 12071.49 | 12210.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 12227.00 | 12060.65 | 12129.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 12227.00 | 12060.65 | 12129.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 12410.00 | 12130.52 | 12155.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:00:00 | 12410.00 | 12130.52 | 12155.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 11:15:00 | 12434.00 | 12191.21 | 12180.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 12:15:00 | 12539.00 | 12260.77 | 12213.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 09:15:00 | 12312.00 | 12383.99 | 12298.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-17 10:00:00 | 12312.00 | 12383.99 | 12298.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 12322.00 | 12371.59 | 12301.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:45:00 | 12319.00 | 12371.59 | 12301.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 12250.00 | 12347.27 | 12296.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:45:00 | 12254.00 | 12347.27 | 12296.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 12190.00 | 12315.82 | 12286.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:00:00 | 12190.00 | 12315.82 | 12286.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2025-11-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 14:15:00 | 12105.00 | 12254.16 | 12262.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 12045.00 | 12193.42 | 12232.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 15:15:00 | 11737.00 | 11727.02 | 11813.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 09:15:00 | 11755.00 | 11727.02 | 11813.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 11696.00 | 11720.82 | 11802.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 10:15:00 | 11650.00 | 11720.82 | 11802.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 12:00:00 | 11649.00 | 11675.17 | 11721.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:30:00 | 11655.00 | 11634.13 | 11691.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 15:00:00 | 11463.00 | 11634.13 | 11691.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 11576.00 | 11539.68 | 11611.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:45:00 | 11651.00 | 11539.68 | 11611.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 11643.00 | 11569.84 | 11613.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 11643.00 | 11569.84 | 11613.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 11635.00 | 11582.87 | 11615.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 11639.00 | 11582.87 | 11615.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 11651.00 | 11596.50 | 11618.67 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-26 15:15:00 | 11725.00 | 11639.69 | 11628.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2025-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 15:15:00 | 11725.00 | 11639.69 | 11628.11 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 09:15:00 | 11520.00 | 11615.76 | 11618.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 10:15:00 | 11436.00 | 11579.80 | 11601.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 12:15:00 | 11515.00 | 11343.75 | 11426.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 12:15:00 | 11515.00 | 11343.75 | 11426.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 11515.00 | 11343.75 | 11426.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:45:00 | 11536.00 | 11343.75 | 11426.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 11572.00 | 11389.40 | 11440.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 13:45:00 | 11570.00 | 11389.40 | 11440.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 11502.00 | 11433.45 | 11452.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:15:00 | 11743.00 | 11433.45 | 11452.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 11640.00 | 11474.76 | 11469.63 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 11351.00 | 11451.74 | 11462.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 14:15:00 | 11325.00 | 11418.76 | 11444.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 12:15:00 | 11132.00 | 11098.61 | 11194.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 13:00:00 | 11132.00 | 11098.61 | 11194.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 11167.00 | 11116.19 | 11185.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 11167.00 | 11116.19 | 11185.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 11300.00 | 11152.95 | 11196.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:15:00 | 11223.00 | 11152.95 | 11196.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 11353.00 | 11192.96 | 11210.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:45:00 | 11365.00 | 11192.96 | 11210.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 11323.00 | 11218.97 | 11220.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 11347.00 | 11218.97 | 11220.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 11:15:00 | 11270.00 | 11229.18 | 11225.18 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2025-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 12:15:00 | 11180.00 | 11219.34 | 11221.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 13:15:00 | 11144.00 | 11204.27 | 11214.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 11:15:00 | 11178.00 | 11135.42 | 11170.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 11:15:00 | 11178.00 | 11135.42 | 11170.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 11178.00 | 11135.42 | 11170.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:00:00 | 11178.00 | 11135.42 | 11170.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 11195.00 | 11147.33 | 11172.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:45:00 | 11243.00 | 11147.33 | 11172.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 11188.00 | 11155.47 | 11173.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 14:30:00 | 11140.00 | 11161.77 | 11175.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 11115.00 | 11164.42 | 11175.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 14:15:00 | 11062.00 | 11008.73 | 11003.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 14:15:00 | 11062.00 | 11008.73 | 11003.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 09:15:00 | 11114.00 | 11031.27 | 11014.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 14:15:00 | 11065.00 | 11071.65 | 11045.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 14:15:00 | 11065.00 | 11071.65 | 11045.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 11065.00 | 11071.65 | 11045.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 15:15:00 | 11101.00 | 11045.12 | 11041.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 11:15:00 | 11015.00 | 11047.09 | 11044.72 | SL hit (close<static) qty=1.00 sl=11031.00 alert=retest2 |

### Cycle 133 — SELL (started 2025-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 12:15:00 | 11027.00 | 11043.07 | 11043.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 13:15:00 | 10980.00 | 11030.46 | 11037.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 11022.00 | 10998.49 | 11018.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 11022.00 | 10998.49 | 11018.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 11022.00 | 10998.49 | 11018.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 11027.00 | 10998.49 | 11018.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 11169.00 | 11032.59 | 11032.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 12:15:00 | 11300.00 | 11194.56 | 11132.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 11223.00 | 11237.57 | 11181.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 13:15:00 | 11227.00 | 11239.05 | 11218.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 11227.00 | 11239.05 | 11218.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:30:00 | 11215.00 | 11239.05 | 11218.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 11228.00 | 11236.84 | 11219.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:45:00 | 11220.00 | 11236.84 | 11219.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 11220.00 | 11233.47 | 11219.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 11234.00 | 11233.47 | 11219.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 11255.00 | 11237.78 | 11222.65 | EMA400 retest candle locked (from upside) |

### Cycle 135 — SELL (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 13:15:00 | 11160.00 | 11205.27 | 11210.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 11152.00 | 11194.61 | 11205.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 11277.00 | 11207.15 | 11209.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 11277.00 | 11207.15 | 11209.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 11277.00 | 11207.15 | 11209.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:30:00 | 11284.00 | 11207.15 | 11209.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 11205.00 | 11206.72 | 11208.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:15:00 | 11190.00 | 11206.72 | 11208.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 11:15:00 | 11233.00 | 11211.98 | 11210.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — BUY (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 11:15:00 | 11233.00 | 11211.98 | 11210.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 14:15:00 | 11254.00 | 11224.13 | 11216.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 09:15:00 | 11190.00 | 11219.20 | 11216.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 09:15:00 | 11190.00 | 11219.20 | 11216.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 11190.00 | 11219.20 | 11216.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:45:00 | 11211.00 | 11219.20 | 11216.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 10:15:00 | 11159.00 | 11207.16 | 11210.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 13:15:00 | 11104.00 | 11170.58 | 11191.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 11295.00 | 11141.13 | 11163.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 11295.00 | 11141.13 | 11163.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 11295.00 | 11141.13 | 11163.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 11295.00 | 11141.13 | 11163.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 11369.00 | 11186.71 | 11182.26 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 11:15:00 | 11090.00 | 11224.94 | 11237.88 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 15:15:00 | 11266.00 | 11221.82 | 11218.11 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 11161.00 | 11215.40 | 11216.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 11128.00 | 11189.22 | 11203.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 14:15:00 | 11214.00 | 11194.17 | 11204.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 14:15:00 | 11214.00 | 11194.17 | 11204.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 11214.00 | 11194.17 | 11204.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 15:00:00 | 11214.00 | 11194.17 | 11204.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 11180.00 | 11191.34 | 11202.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 11075.00 | 11191.34 | 11202.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 11145.00 | 11182.07 | 11197.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 12:30:00 | 11037.00 | 11128.06 | 11166.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 11516.00 | 11228.83 | 11202.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 09:15:00 | 11516.00 | 11228.83 | 11202.32 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 10:15:00 | 11161.00 | 11199.52 | 11204.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 14:15:00 | 11043.00 | 11133.60 | 11168.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 15:15:00 | 10797.00 | 10749.62 | 10846.35 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 09:45:00 | 10649.00 | 10723.90 | 10825.87 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 10755.00 | 10730.12 | 10819.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:30:00 | 10709.00 | 10730.12 | 10819.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 10707.00 | 10746.79 | 10791.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:00:00 | 10676.00 | 10724.25 | 10769.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 15:00:00 | 10665.00 | 10708.68 | 10753.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 10523.00 | 10704.74 | 10748.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 10750.00 | 10703.12 | 10739.13 | SL hit (close>ema400) qty=1.00 sl=10739.13 alert=retest1 |

### Cycle 144 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 10681.00 | 10637.16 | 10636.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 12:15:00 | 10727.00 | 10666.14 | 10650.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 10628.00 | 10686.30 | 10667.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 10628.00 | 10686.30 | 10667.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 10628.00 | 10686.30 | 10667.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:00:00 | 10628.00 | 10686.30 | 10667.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 10688.00 | 10686.64 | 10669.39 | EMA400 retest candle locked (from upside) |

### Cycle 145 — SELL (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 09:15:00 | 10525.00 | 10644.55 | 10654.81 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 10685.00 | 10631.65 | 10631.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 10840.00 | 10732.33 | 10693.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 10749.00 | 10767.36 | 10734.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 10749.00 | 10767.36 | 10734.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 10749.00 | 10767.36 | 10734.56 | EMA400 retest candle locked (from upside) |

### Cycle 147 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 10608.00 | 10708.78 | 10715.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 10555.00 | 10678.02 | 10700.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 10562.00 | 10561.23 | 10619.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 14:00:00 | 10562.00 | 10561.23 | 10619.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 10660.00 | 10580.98 | 10623.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 10660.00 | 10580.98 | 10623.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 10640.00 | 10592.79 | 10625.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 10811.00 | 10592.79 | 10625.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 10927.00 | 10695.50 | 10668.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 10974.00 | 10817.07 | 10736.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 14:15:00 | 10889.00 | 10952.49 | 10867.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 14:15:00 | 10889.00 | 10952.49 | 10867.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 10889.00 | 10952.49 | 10867.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 09:15:00 | 11125.00 | 10936.99 | 10868.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 15:15:00 | 10825.00 | 10953.77 | 10924.77 | SL hit (close<static) qty=1.00 sl=10827.00 alert=retest2 |

### Cycle 149 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 10696.00 | 10902.22 | 10903.98 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2026-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 15:15:00 | 10999.00 | 10901.13 | 10892.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 11044.00 | 10929.70 | 10906.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 12:15:00 | 11040.00 | 11050.60 | 11004.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 13:00:00 | 11040.00 | 11050.60 | 11004.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 11060.00 | 11061.28 | 11025.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 10:30:00 | 11158.00 | 11082.03 | 11037.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:00:00 | 11165.00 | 11082.03 | 11037.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 10852.00 | 11023.58 | 11046.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 10852.00 | 11023.58 | 11046.96 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2026-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 12:15:00 | 11109.00 | 11014.88 | 11009.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 13:15:00 | 11208.00 | 11053.51 | 11027.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 11300.00 | 11320.25 | 11238.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 09:30:00 | 11280.00 | 11320.25 | 11238.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 11241.00 | 11296.41 | 11252.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:00:00 | 11241.00 | 11296.41 | 11252.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 11167.00 | 11270.53 | 11244.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 11167.00 | 11270.53 | 11244.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 11121.00 | 11240.62 | 11233.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 11000.00 | 11240.62 | 11233.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 11065.00 | 11205.50 | 11218.06 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 11299.00 | 11203.92 | 11203.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 11:15:00 | 11321.00 | 11227.34 | 11214.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 12:15:00 | 11358.00 | 11426.08 | 11356.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 12:15:00 | 11358.00 | 11426.08 | 11356.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 11358.00 | 11426.08 | 11356.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:00:00 | 11358.00 | 11426.08 | 11356.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 11326.00 | 11406.07 | 11353.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:30:00 | 11290.00 | 11406.07 | 11353.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 11321.00 | 11389.05 | 11350.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 14:45:00 | 11348.00 | 11389.05 | 11350.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 11347.00 | 11380.64 | 11350.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:30:00 | 11319.00 | 11360.51 | 11343.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 11229.00 | 11334.21 | 11333.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:30:00 | 11250.00 | 11334.21 | 11333.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — SELL (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 11:15:00 | 11205.00 | 11308.37 | 11321.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 12:15:00 | 11174.00 | 11281.50 | 11308.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 10685.00 | 10675.18 | 10817.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 14:45:00 | 10665.00 | 10675.18 | 10817.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 10630.00 | 10632.49 | 10707.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 10:30:00 | 10626.00 | 10635.79 | 10702.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 14:00:00 | 10598.00 | 10636.00 | 10686.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:30:00 | 10620.00 | 10644.25 | 10675.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 12:00:00 | 10615.00 | 10638.40 | 10669.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 10347.00 | 10241.52 | 10328.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 10360.00 | 10241.52 | 10328.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 10282.00 | 10249.62 | 10324.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:15:00 | 10262.00 | 10249.62 | 10324.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:45:00 | 10215.00 | 10237.29 | 10311.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 10094.70 | 10160.33 | 10241.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 10068.10 | 10160.33 | 10241.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 10089.00 | 10160.33 | 10241.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 10084.25 | 10160.33 | 10241.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:15:00 | 9748.90 | 9885.97 | 10045.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:15:00 | 9704.25 | 9885.97 | 10045.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-13 13:15:00 | 9563.40 | 9751.36 | 9926.43 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 156 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 9870.00 | 9711.15 | 9700.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 9895.00 | 9772.86 | 9732.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 9707.00 | 9796.10 | 9759.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 9707.00 | 9796.10 | 9759.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 9707.00 | 9796.10 | 9759.97 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 9658.00 | 9732.44 | 9739.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 13:15:00 | 9606.00 | 9686.87 | 9712.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 9276.00 | 9270.25 | 9410.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 10:45:00 | 9294.00 | 9270.25 | 9410.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 9392.00 | 9302.24 | 9401.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 9413.00 | 9302.24 | 9401.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 9363.00 | 9314.39 | 9398.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 9403.00 | 9314.39 | 9398.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 9650.00 | 9392.44 | 9413.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 9650.00 | 9392.44 | 9413.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 9625.00 | 9438.95 | 9432.90 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 9292.00 | 9434.83 | 9447.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 9220.00 | 9349.67 | 9401.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 8942.00 | 8910.37 | 9080.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 8942.00 | 8910.37 | 9080.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 8942.00 | 8910.37 | 9080.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:30:00 | 8860.00 | 8898.80 | 9059.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 15:15:00 | 8850.00 | 8924.54 | 9022.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 8854.50 | 8907.95 | 8942.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 9041.50 | 8973.60 | 8967.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 9041.50 | 8973.60 | 8967.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 9059.50 | 8990.78 | 8976.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 9901.00 | 9910.15 | 9722.64 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 10039.50 | 9910.15 | 9722.64 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 9960.00 | 9979.97 | 9869.16 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 14:15:00 | 9849.00 | 9925.75 | 9883.33 | SL hit (close<ema400) qty=1.00 sl=9883.33 alert=retest1 |

### Cycle 161 — SELL (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 12:15:00 | 10275.00 | 10358.47 | 10369.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 13:15:00 | 10266.00 | 10339.97 | 10359.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 10366.00 | 10326.29 | 10346.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 10366.00 | 10326.29 | 10346.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 10366.00 | 10326.29 | 10346.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 10366.00 | 10326.29 | 10346.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 10408.50 | 10342.73 | 10352.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 10408.50 | 10342.73 | 10352.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 10399.00 | 10359.23 | 10358.42 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 14:15:00 | 10315.00 | 10353.63 | 10356.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 12:15:00 | 10279.50 | 10333.34 | 10345.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 10340.00 | 10311.61 | 10328.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 10340.00 | 10311.61 | 10328.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 10340.00 | 10311.61 | 10328.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:15:00 | 10427.00 | 10311.61 | 10328.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 10382.50 | 10325.79 | 10333.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:30:00 | 10401.00 | 10325.79 | 10333.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 10399.00 | 10340.43 | 10339.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 12:15:00 | 10434.50 | 10359.24 | 10348.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 10305.50 | 10348.49 | 10344.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 10305.50 | 10348.49 | 10344.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 10305.50 | 10348.49 | 10344.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 10305.50 | 10348.49 | 10344.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — SELL (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 14:15:00 | 10301.00 | 10339.00 | 10340.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 10170.00 | 10305.36 | 10324.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 13:15:00 | 10260.50 | 10250.12 | 10287.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 13:45:00 | 10275.00 | 10250.12 | 10287.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 10236.00 | 10247.30 | 10282.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:45:00 | 10287.50 | 10247.30 | 10282.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 10241.00 | 10246.04 | 10278.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 10532.00 | 10246.04 | 10278.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 10646.00 | 10326.03 | 10312.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 10759.00 | 10486.46 | 10417.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 12:15:00 | 10539.00 | 10539.20 | 10463.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 13:00:00 | 10539.00 | 10539.20 | 10463.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 10525.00 | 10583.50 | 10548.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 10517.00 | 10583.50 | 10548.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 10625.00 | 10591.80 | 10555.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:30:00 | 10535.00 | 10591.80 | 10555.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 10772.00 | 10639.22 | 10587.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:30:00 | 10703.00 | 10639.22 | 10587.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 10607.00 | 10632.78 | 10589.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 10607.00 | 10632.78 | 10589.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 10678.00 | 10641.82 | 10597.34 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-24 15:15:00 | 8019.05 | 2024-05-30 14:15:00 | 8002.90 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2024-06-13 09:15:00 | 8412.55 | 2024-06-13 12:15:00 | 8339.45 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-06-18 15:15:00 | 8273.60 | 2024-06-24 09:15:00 | 8606.80 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2024-06-19 09:30:00 | 8271.85 | 2024-06-24 09:15:00 | 8606.80 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2024-06-27 09:15:00 | 8673.55 | 2024-06-27 11:15:00 | 8499.15 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest1 | 2024-07-08 13:30:00 | 9794.00 | 2024-07-10 09:15:00 | 9707.45 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest1 | 2024-07-09 09:30:00 | 9839.15 | 2024-07-10 09:15:00 | 9707.45 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest1 | 2024-07-09 12:30:00 | 9754.40 | 2024-07-10 09:15:00 | 9707.45 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-07-09 13:30:00 | 9776.00 | 2024-07-10 09:15:00 | 9707.45 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-07-10 14:15:00 | 9750.00 | 2024-07-11 09:15:00 | 9674.95 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-07-19 11:30:00 | 9787.00 | 2024-07-22 09:15:00 | 9903.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-08-01 09:15:00 | 9801.90 | 2024-08-01 10:15:00 | 9556.70 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2024-08-02 15:00:00 | 9461.10 | 2024-08-07 13:15:00 | 9401.55 | STOP_HIT | 1.00 | 0.63% |
| BUY | retest2 | 2024-08-09 14:00:00 | 9491.40 | 2024-08-13 09:15:00 | 9243.95 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2024-08-26 10:15:00 | 9898.90 | 2024-09-03 10:15:00 | 10888.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-26 11:45:00 | 9900.00 | 2024-09-03 10:15:00 | 10890.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-26 14:30:00 | 9909.55 | 2024-09-03 10:15:00 | 10900.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-10 09:15:00 | 10284.80 | 2024-09-16 15:15:00 | 10440.00 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-09-20 15:00:00 | 11154.65 | 2024-09-24 15:15:00 | 10870.00 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2024-09-23 15:00:00 | 11108.95 | 2024-09-24 15:15:00 | 10870.00 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2024-09-24 10:45:00 | 11034.25 | 2024-09-24 15:15:00 | 10870.00 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-09-27 15:00:00 | 10526.00 | 2024-10-09 10:15:00 | 10563.85 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2024-10-14 12:15:00 | 10735.10 | 2024-10-14 14:15:00 | 10647.15 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-10-14 13:30:00 | 10711.45 | 2024-10-14 14:15:00 | 10647.15 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-10-15 09:15:00 | 10731.35 | 2024-10-15 09:15:00 | 10650.05 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-10-15 13:30:00 | 10743.90 | 2024-10-16 09:15:00 | 10640.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-10-23 14:30:00 | 10153.45 | 2024-10-28 11:15:00 | 10286.55 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-10-24 09:15:00 | 10136.95 | 2024-10-28 11:15:00 | 10286.55 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-10-24 11:15:00 | 10154.25 | 2024-10-28 11:15:00 | 10286.55 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-10-24 12:45:00 | 10158.60 | 2024-10-28 11:15:00 | 10286.55 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-11-05 11:15:00 | 10603.95 | 2024-11-07 12:15:00 | 10464.30 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-11-05 12:30:00 | 10633.95 | 2024-11-07 12:15:00 | 10464.30 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-11-05 13:30:00 | 10600.95 | 2024-11-07 15:15:00 | 10486.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-11-05 14:15:00 | 10609.25 | 2024-11-07 15:15:00 | 10486.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-11-07 11:30:00 | 10557.25 | 2024-11-07 15:15:00 | 10486.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-11-07 12:00:00 | 10566.60 | 2024-11-07 15:15:00 | 10486.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-11-22 15:15:00 | 10400.40 | 2024-11-29 15:15:00 | 10500.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-11-26 10:15:00 | 10439.65 | 2024-11-29 15:15:00 | 10500.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-11-26 12:15:00 | 10422.30 | 2024-11-29 15:15:00 | 10500.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-11-27 09:15:00 | 10312.20 | 2024-11-29 15:15:00 | 10500.00 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-11-27 10:30:00 | 10300.00 | 2024-11-29 15:15:00 | 10500.00 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2024-11-27 13:15:00 | 10309.80 | 2024-11-29 15:15:00 | 10500.00 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-11-27 13:45:00 | 10299.90 | 2024-11-29 15:15:00 | 10500.00 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2024-11-29 09:30:00 | 10247.55 | 2024-11-29 15:15:00 | 10500.00 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2024-12-05 12:45:00 | 10698.00 | 2024-12-16 10:15:00 | 11105.40 | STOP_HIT | 1.00 | 3.81% |
| BUY | retest2 | 2024-12-05 15:00:00 | 10766.95 | 2024-12-16 10:15:00 | 11105.40 | STOP_HIT | 1.00 | 3.14% |
| BUY | retest2 | 2024-12-06 10:45:00 | 10694.60 | 2024-12-16 10:15:00 | 11105.40 | STOP_HIT | 1.00 | 3.84% |
| BUY | retest2 | 2024-12-09 09:15:00 | 10791.00 | 2024-12-16 10:15:00 | 11105.40 | STOP_HIT | 1.00 | 2.91% |
| BUY | retest2 | 2024-12-16 09:15:00 | 11206.05 | 2024-12-16 10:15:00 | 11105.40 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-12-20 09:45:00 | 11198.45 | 2024-12-20 10:15:00 | 11174.60 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2024-12-24 09:15:00 | 10971.40 | 2024-12-24 09:15:00 | 11087.10 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest1 | 2024-12-30 09:15:00 | 11449.80 | 2024-12-30 14:15:00 | 12594.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-30 10:30:00 | 11739.00 | 2024-12-30 14:15:00 | 12912.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-30 15:00:00 | 12967.55 | 2025-01-03 14:15:00 | 11591.20 | STOP_HIT | 1.00 | -10.61% |
| BUY | retest2 | 2024-12-31 12:00:00 | 11756.70 | 2025-01-03 14:15:00 | 11591.20 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-12-31 12:45:00 | 11740.90 | 2025-01-03 14:15:00 | 11591.20 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-01-17 14:45:00 | 10859.70 | 2025-01-17 15:15:00 | 10800.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-01-20 09:15:00 | 10927.90 | 2025-01-21 12:15:00 | 10786.15 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-01-20 15:15:00 | 10845.00 | 2025-01-21 12:15:00 | 10786.15 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-01-21 11:45:00 | 10843.00 | 2025-01-21 12:15:00 | 10786.15 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-02-06 12:45:00 | 11435.90 | 2025-02-10 15:15:00 | 11523.10 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-02-10 13:15:00 | 11403.80 | 2025-02-10 15:15:00 | 11523.10 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-02-10 13:45:00 | 11450.00 | 2025-02-10 15:15:00 | 11523.10 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-02-19 09:15:00 | 12131.15 | 2025-02-19 13:15:00 | 11819.50 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-02-24 12:30:00 | 12526.75 | 2025-02-24 13:15:00 | 11990.10 | STOP_HIT | 1.00 | -4.28% |
| BUY | retest2 | 2025-03-07 09:15:00 | 11563.95 | 2025-03-11 09:15:00 | 11373.80 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-03-19 11:15:00 | 11825.25 | 2025-03-27 12:15:00 | 12230.00 | STOP_HIT | 1.00 | 3.42% |
| SELL | retest2 | 2025-04-09 09:15:00 | 10587.60 | 2025-04-11 10:15:00 | 11204.70 | STOP_HIT | 1.00 | -5.83% |
| SELL | retest2 | 2025-04-09 09:45:00 | 10601.75 | 2025-04-11 10:15:00 | 11204.70 | STOP_HIT | 1.00 | -5.69% |
| BUY | retest2 | 2025-04-17 10:30:00 | 11743.00 | 2025-04-25 15:15:00 | 12151.00 | STOP_HIT | 1.00 | 3.47% |
| BUY | retest2 | 2025-04-29 13:45:00 | 12347.00 | 2025-04-30 09:15:00 | 12123.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-04-29 14:15:00 | 12367.00 | 2025-04-30 09:15:00 | 12123.00 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-05-05 10:30:00 | 11960.00 | 2025-05-09 09:15:00 | 11362.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-05 13:30:00 | 11907.00 | 2025-05-09 09:15:00 | 11311.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 09:45:00 | 11948.00 | 2025-05-09 09:15:00 | 11350.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 12:00:00 | 11941.00 | 2025-05-09 09:15:00 | 11343.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-07 14:00:00 | 11948.00 | 2025-05-09 09:15:00 | 11350.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-05 10:30:00 | 11960.00 | 2025-05-12 09:15:00 | 11758.00 | STOP_HIT | 0.50 | 1.69% |
| SELL | retest2 | 2025-05-05 13:30:00 | 11907.00 | 2025-05-12 09:15:00 | 11758.00 | STOP_HIT | 0.50 | 1.25% |
| SELL | retest2 | 2025-05-06 09:45:00 | 11948.00 | 2025-05-12 09:15:00 | 11758.00 | STOP_HIT | 0.50 | 1.59% |
| SELL | retest2 | 2025-05-06 12:00:00 | 11941.00 | 2025-05-12 09:15:00 | 11758.00 | STOP_HIT | 0.50 | 1.53% |
| SELL | retest2 | 2025-05-07 14:00:00 | 11948.00 | 2025-05-12 09:15:00 | 11758.00 | STOP_HIT | 0.50 | 1.59% |
| BUY | retest2 | 2025-05-14 12:00:00 | 12055.00 | 2025-05-16 13:15:00 | 13260.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-14 15:15:00 | 12069.00 | 2025-05-16 13:15:00 | 13275.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-16 09:15:00 | 13808.00 | 2025-06-24 13:15:00 | 13680.00 | STOP_HIT | 1.00 | 0.93% |
| SELL | retest2 | 2025-06-16 13:30:00 | 13992.00 | 2025-06-24 13:15:00 | 13680.00 | STOP_HIT | 1.00 | 2.23% |
| SELL | retest2 | 2025-06-16 14:45:00 | 13980.00 | 2025-06-24 13:15:00 | 13680.00 | STOP_HIT | 1.00 | 2.15% |
| SELL | retest2 | 2025-06-17 09:15:00 | 13862.00 | 2025-06-24 13:15:00 | 13680.00 | STOP_HIT | 1.00 | 1.31% |
| SELL | retest2 | 2025-06-17 11:45:00 | 13811.00 | 2025-06-24 13:15:00 | 13680.00 | STOP_HIT | 1.00 | 0.95% |
| SELL | retest2 | 2025-06-17 14:30:00 | 13840.00 | 2025-06-24 13:15:00 | 13680.00 | STOP_HIT | 1.00 | 1.16% |
| SELL | retest2 | 2025-06-18 14:00:00 | 13709.00 | 2025-06-24 13:15:00 | 13680.00 | STOP_HIT | 1.00 | 0.21% |
| BUY | retest2 | 2025-06-25 15:00:00 | 13713.00 | 2025-07-03 15:15:00 | 14191.00 | STOP_HIT | 1.00 | 3.49% |
| SELL | retest2 | 2025-07-11 10:15:00 | 13777.00 | 2025-07-14 12:15:00 | 14060.00 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-07-11 11:15:00 | 13788.00 | 2025-07-14 12:15:00 | 14060.00 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-07-11 12:15:00 | 13781.00 | 2025-07-14 12:15:00 | 14060.00 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-07-11 13:00:00 | 13795.00 | 2025-07-14 12:15:00 | 14060.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-08-05 12:15:00 | 13843.00 | 2025-08-07 09:15:00 | 14171.00 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-08-05 13:30:00 | 13848.00 | 2025-08-07 09:15:00 | 14171.00 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-08-06 10:00:00 | 13797.00 | 2025-08-07 09:15:00 | 14171.00 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest1 | 2025-08-21 12:30:00 | 13657.00 | 2025-08-22 09:15:00 | 13790.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-08-22 11:15:00 | 13682.00 | 2025-08-26 14:15:00 | 12997.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-22 11:15:00 | 13682.00 | 2025-09-02 13:15:00 | 12680.00 | STOP_HIT | 0.50 | 7.32% |
| BUY | retest1 | 2025-09-10 09:15:00 | 13170.00 | 2025-09-10 13:15:00 | 12972.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-09-10 15:15:00 | 13099.00 | 2025-09-19 10:15:00 | 13273.00 | STOP_HIT | 1.00 | 1.33% |
| SELL | retest2 | 2025-09-23 11:15:00 | 13230.00 | 2025-09-26 15:15:00 | 12568.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 14:15:00 | 13242.00 | 2025-09-26 15:15:00 | 12579.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 14:45:00 | 13236.00 | 2025-09-26 15:15:00 | 12574.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 15:15:00 | 13242.00 | 2025-09-26 15:15:00 | 12579.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 11:15:00 | 13230.00 | 2025-09-29 09:15:00 | 12857.00 | STOP_HIT | 0.50 | 2.82% |
| SELL | retest2 | 2025-09-23 14:15:00 | 13242.00 | 2025-09-29 09:15:00 | 12857.00 | STOP_HIT | 0.50 | 2.91% |
| SELL | retest2 | 2025-09-23 14:45:00 | 13236.00 | 2025-09-29 09:15:00 | 12857.00 | STOP_HIT | 0.50 | 2.86% |
| SELL | retest2 | 2025-09-23 15:15:00 | 13242.00 | 2025-09-29 09:15:00 | 12857.00 | STOP_HIT | 0.50 | 2.91% |
| SELL | retest2 | 2025-09-24 12:00:00 | 13183.00 | 2025-09-29 11:15:00 | 12523.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 09:30:00 | 13149.00 | 2025-09-29 11:15:00 | 12491.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 12:00:00 | 13183.00 | 2025-10-03 09:15:00 | 11864.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-25 09:30:00 | 13149.00 | 2025-10-03 09:15:00 | 11834.10 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-10-14 14:15:00 | 12269.00 | 2025-10-28 13:15:00 | 12810.00 | STOP_HIT | 1.00 | 4.41% |
| BUY | retest2 | 2025-10-15 09:15:00 | 12193.00 | 2025-10-28 13:15:00 | 12810.00 | STOP_HIT | 1.00 | 5.06% |
| BUY | retest2 | 2025-10-15 09:45:00 | 12174.00 | 2025-10-28 13:15:00 | 12810.00 | STOP_HIT | 1.00 | 5.22% |
| BUY | retest2 | 2025-10-15 10:15:00 | 12179.00 | 2025-10-28 13:15:00 | 12810.00 | STOP_HIT | 1.00 | 5.18% |
| BUY | retest2 | 2025-10-15 12:00:00 | 12197.00 | 2025-10-28 13:15:00 | 12810.00 | STOP_HIT | 1.00 | 5.03% |
| BUY | retest2 | 2025-10-15 14:15:00 | 12258.00 | 2025-10-28 13:15:00 | 12810.00 | STOP_HIT | 1.00 | 4.50% |
| BUY | retest2 | 2025-10-16 09:15:00 | 12330.00 | 2025-10-28 13:15:00 | 12810.00 | STOP_HIT | 1.00 | 3.89% |
| BUY | retest2 | 2025-11-10 09:15:00 | 13022.00 | 2025-11-11 09:15:00 | 12250.00 | STOP_HIT | 1.00 | -5.93% |
| BUY | retest2 | 2025-11-10 12:45:00 | 12736.00 | 2025-11-11 09:15:00 | 12250.00 | STOP_HIT | 1.00 | -3.82% |
| BUY | retest2 | 2025-11-10 13:45:00 | 12742.00 | 2025-11-11 09:15:00 | 12250.00 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2025-11-21 10:15:00 | 11650.00 | 2025-11-26 15:15:00 | 11725.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-11-24 12:00:00 | 11649.00 | 2025-11-26 15:15:00 | 11725.00 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-11-24 14:30:00 | 11655.00 | 2025-11-26 15:15:00 | 11725.00 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-11-24 15:00:00 | 11463.00 | 2025-11-26 15:15:00 | 11725.00 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2025-12-05 14:30:00 | 11140.00 | 2025-12-15 14:15:00 | 11062.00 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2025-12-08 09:15:00 | 11115.00 | 2025-12-15 14:15:00 | 11062.00 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2025-12-17 15:15:00 | 11101.00 | 2025-12-18 11:15:00 | 11015.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-12-29 11:15:00 | 11190.00 | 2025-12-29 11:15:00 | 11233.00 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2026-01-07 12:30:00 | 11037.00 | 2026-01-08 09:15:00 | 11516.00 | STOP_HIT | 1.00 | -4.34% |
| SELL | retest1 | 2026-01-14 09:45:00 | 10649.00 | 2026-01-19 10:15:00 | 10750.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-01-16 13:00:00 | 10676.00 | 2026-01-22 10:15:00 | 10681.00 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2026-01-16 15:00:00 | 10665.00 | 2026-01-22 10:15:00 | 10681.00 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2026-01-19 09:15:00 | 10523.00 | 2026-01-22 10:15:00 | 10681.00 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-01-19 13:15:00 | 10678.00 | 2026-01-22 10:15:00 | 10681.00 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2026-02-05 09:15:00 | 11125.00 | 2026-02-05 15:15:00 | 10825.00 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2026-02-11 10:30:00 | 11158.00 | 2026-02-13 09:15:00 | 10852.00 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2026-02-11 11:00:00 | 11165.00 | 2026-02-13 09:15:00 | 10852.00 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2026-03-05 10:30:00 | 10626.00 | 2026-03-12 09:15:00 | 10094.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 14:00:00 | 10598.00 | 2026-03-12 09:15:00 | 10068.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 10:30:00 | 10620.00 | 2026-03-12 09:15:00 | 10089.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 12:00:00 | 10615.00 | 2026-03-12 09:15:00 | 10084.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 10:15:00 | 10262.00 | 2026-03-13 09:15:00 | 9748.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 10:45:00 | 10215.00 | 2026-03-13 09:15:00 | 9704.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 10:30:00 | 10626.00 | 2026-03-13 13:15:00 | 9563.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-05 14:00:00 | 10598.00 | 2026-03-13 13:15:00 | 9558.00 | TARGET_HIT | 0.50 | 9.81% |
| SELL | retest2 | 2026-03-06 10:30:00 | 10620.00 | 2026-03-13 13:15:00 | 9553.50 | TARGET_HIT | 0.50 | 10.04% |
| SELL | retest2 | 2026-03-06 12:00:00 | 10615.00 | 2026-03-13 14:15:00 | 9538.20 | TARGET_HIT | 0.50 | 10.14% |
| SELL | retest2 | 2026-03-11 10:15:00 | 10262.00 | 2026-03-16 13:15:00 | 9538.00 | STOP_HIT | 0.50 | 7.06% |
| SELL | retest2 | 2026-03-11 10:45:00 | 10215.00 | 2026-03-16 13:15:00 | 9538.00 | STOP_HIT | 0.50 | 6.63% |
| SELL | retest2 | 2026-04-01 10:30:00 | 8860.00 | 2026-04-06 11:15:00 | 9041.50 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2026-04-01 15:15:00 | 8850.00 | 2026-04-06 11:15:00 | 9041.50 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2026-04-06 09:15:00 | 8854.50 | 2026-04-06 11:15:00 | 9041.50 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest1 | 2026-04-10 09:15:00 | 10039.50 | 2026-04-13 14:15:00 | 9849.00 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-04-15 09:15:00 | 10062.50 | 2026-04-24 12:15:00 | 10275.00 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest2 | 2026-04-15 14:30:00 | 10058.50 | 2026-04-24 12:15:00 | 10275.00 | STOP_HIT | 1.00 | 2.15% |
