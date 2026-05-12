# Oracle Financial Services Software Ltd. (OFSS)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 9321.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 42 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 25 |
| PARTIAL | 7 |
| TARGET_HIT | 6 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 19
- **Target hits / Stop hits / Partials:** 6 / 19 / 7
- **Avg / median % per leg:** 1.72% / -0.84%
- **Sum % (uncompounded):** 55.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.62% | -2.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.62% | -2.6% |
| SELL (all) | 31 | 13 | 41.9% | 6 | 18 | 7 | 1.86% | 57.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 31 | 13 | 41.9% | 6 | 18 | 7 | 1.86% | 57.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 32 | 13 | 40.6% | 6 | 19 | 7 | 1.72% | 55.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 9374.00 | 8523.82 | 8523.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 11:15:00 | 9436.00 | 8532.89 | 8527.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 13:15:00 | 9010.00 | 9024.14 | 8839.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 14:00:00 | 9010.00 | 9024.14 | 8839.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 8894.50 | 9017.27 | 8851.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:00:00 | 8894.50 | 9017.27 | 8851.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 8894.50 | 9014.13 | 8852.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 8879.00 | 9014.13 | 8852.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 8852.50 | 9010.92 | 8879.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:00:00 | 8852.50 | 9010.92 | 8879.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 8809.00 | 9008.91 | 8878.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:45:00 | 8798.00 | 9008.91 | 8878.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 8815.00 | 8949.62 | 8861.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:30:00 | 8810.00 | 8949.62 | 8861.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 8838.00 | 8931.71 | 8857.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:45:00 | 8825.50 | 8931.71 | 8857.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 8835.00 | 8909.35 | 8851.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:45:00 | 8841.00 | 8909.35 | 8851.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 8815.50 | 8908.41 | 8851.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 8815.50 | 8908.41 | 8851.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 8734.50 | 8904.55 | 8850.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:30:00 | 8744.00 | 8904.55 | 8850.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 8860.50 | 8884.20 | 8843.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 09:15:00 | 8984.50 | 8885.89 | 8845.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 12:15:00 | 8749.50 | 8883.70 | 8845.05 | SL hit (close<static) qty=1.00 sl=8811.50 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 8471.50 | 8811.13 | 8811.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 15:15:00 | 8453.50 | 8807.57 | 8809.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 10:15:00 | 8670.00 | 8663.16 | 8724.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-14 11:00:00 | 8670.00 | 8663.16 | 8724.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 8774.50 | 8651.33 | 8711.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:00:00 | 8774.50 | 8651.33 | 8711.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 8797.00 | 8652.78 | 8712.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:00:00 | 8797.00 | 8652.78 | 8712.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 8803.50 | 8658.37 | 8713.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 8803.50 | 8658.37 | 8713.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 8715.00 | 8663.14 | 8714.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 8699.50 | 8663.14 | 8714.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 8712.50 | 8663.63 | 8714.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 11:00:00 | 8639.50 | 8663.39 | 8714.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 11:30:00 | 8652.00 | 8663.44 | 8714.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 12:30:00 | 8650.00 | 8663.36 | 8713.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 14:30:00 | 8622.00 | 8663.01 | 8713.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 8845.00 | 8664.51 | 8713.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 8845.00 | 8664.51 | 8713.49 | SL hit (close>static) qty=1.00 sl=8744.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 9141.50 | 8699.90 | 8699.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 14:15:00 | 9185.00 | 8744.12 | 8722.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 8742.50 | 8833.36 | 8775.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 8742.50 | 8833.36 | 8775.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 8742.50 | 8833.36 | 8775.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:00:00 | 8742.50 | 8833.36 | 8775.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 8687.50 | 8831.91 | 8774.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:00:00 | 8687.50 | 8831.91 | 8774.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 8872.50 | 8758.99 | 8743.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:30:00 | 8745.00 | 8758.99 | 8743.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 8809.00 | 8922.83 | 8840.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:00:00 | 8809.00 | 8922.83 | 8840.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 8798.00 | 8921.59 | 8840.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:45:00 | 8831.00 | 8921.59 | 8840.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 8754.50 | 8919.93 | 8840.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:45:00 | 8752.00 | 8919.93 | 8840.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 8823.50 | 8914.41 | 8838.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:15:00 | 8788.50 | 8914.41 | 8838.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 8801.50 | 8913.28 | 8838.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:30:00 | 8792.00 | 8913.28 | 8838.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 09:15:00 | 8589.00 | 8784.23 | 8785.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 10:15:00 | 8558.50 | 8781.99 | 8783.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 8489.00 | 8485.57 | 8602.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 10:00:00 | 8489.00 | 8485.57 | 8602.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 7959.50 | 7799.95 | 8009.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:45:00 | 7996.00 | 7799.95 | 8009.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 7983.50 | 7803.19 | 8009.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 11:45:00 | 7900.50 | 7805.38 | 8008.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 11:15:00 | 7900.50 | 7813.99 | 8006.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 13:15:00 | 7866.50 | 7816.02 | 8005.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:30:00 | 7899.00 | 7808.01 | 7990.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 7997.00 | 7815.52 | 7988.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:00:00 | 7997.00 | 7815.52 | 7988.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 8016.50 | 7817.52 | 7988.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 12:00:00 | 8016.50 | 7817.52 | 7988.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 7950.00 | 7822.39 | 7988.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:45:00 | 8022.00 | 7822.39 | 7988.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 8041.00 | 7825.73 | 7988.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-27 09:15:00 | 8041.00 | 7825.73 | 7988.13 | SL hit (close>static) qty=1.00 sl=8022.50 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 15:15:00 | 7980.00 | 7201.84 | 7200.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 09:15:00 | 8091.50 | 7210.69 | 7204.58 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-14 12:15:00 | 8501.00 | 2025-05-15 15:15:00 | 8563.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-05-14 12:45:00 | 8460.00 | 2025-05-15 15:15:00 | 8563.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-05-15 09:45:00 | 8497.00 | 2025-05-15 15:15:00 | 8563.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-05-15 11:00:00 | 8491.50 | 2025-05-15 15:15:00 | 8563.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-05-22 09:15:00 | 8232.50 | 2025-05-23 09:15:00 | 8513.50 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2025-06-03 09:45:00 | 8323.50 | 2025-06-04 09:15:00 | 8547.00 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-06-03 10:30:00 | 8313.00 | 2025-06-04 09:15:00 | 8547.00 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2025-07-25 09:15:00 | 8984.50 | 2025-07-25 12:15:00 | 8749.50 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-08-22 11:00:00 | 8639.50 | 2025-08-25 09:15:00 | 8845.00 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2025-08-22 11:30:00 | 8652.00 | 2025-08-25 09:15:00 | 8845.00 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2025-08-22 12:30:00 | 8650.00 | 2025-08-25 09:15:00 | 8845.00 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-08-22 14:30:00 | 8622.00 | 2025-08-25 09:15:00 | 8845.00 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-08-26 12:30:00 | 8669.00 | 2025-09-05 09:15:00 | 8235.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 12:30:00 | 8669.00 | 2025-09-10 09:15:00 | 9035.00 | STOP_HIT | 0.50 | -4.22% |
| SELL | retest2 | 2026-01-19 11:45:00 | 7900.50 | 2026-01-27 09:15:00 | 8041.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-01-20 11:15:00 | 7900.50 | 2026-01-27 09:15:00 | 8041.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-01-20 13:15:00 | 7866.50 | 2026-01-27 09:15:00 | 8041.00 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2026-01-22 10:30:00 | 7899.00 | 2026-01-27 09:15:00 | 8041.00 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2026-01-27 13:45:00 | 7953.50 | 2026-01-27 15:15:00 | 8054.50 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-01-28 10:00:00 | 7954.50 | 2026-02-04 09:15:00 | 7556.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-28 13:30:00 | 7947.00 | 2026-02-04 09:15:00 | 7549.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-29 09:15:00 | 7907.00 | 2026-02-04 09:15:00 | 7511.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-03 13:30:00 | 7855.00 | 2026-02-05 09:15:00 | 7462.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-03 14:30:00 | 7851.00 | 2026-02-05 09:15:00 | 7458.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-03 15:00:00 | 7832.00 | 2026-02-05 11:15:00 | 7440.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-28 10:00:00 | 7954.50 | 2026-02-06 12:15:00 | 7159.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-28 13:30:00 | 7947.00 | 2026-02-06 12:15:00 | 7152.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-29 09:15:00 | 7907.00 | 2026-02-12 09:15:00 | 7116.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-03 13:30:00 | 7855.00 | 2026-02-12 09:15:00 | 7069.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-03 14:30:00 | 7851.00 | 2026-02-12 09:15:00 | 7065.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-03 15:00:00 | 7832.00 | 2026-02-12 09:15:00 | 7048.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-17 09:30:00 | 7847.00 | 2026-04-17 15:15:00 | 8020.00 | STOP_HIT | 1.00 | -2.20% |
