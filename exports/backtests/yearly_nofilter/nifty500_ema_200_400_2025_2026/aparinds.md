# Apar Industries Ltd. (APARINDS)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 12760.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 0 |
| ALERT3 | 37 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 16 |
| PARTIAL | 0 |
| TARGET_HIT | 8 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 8
- **Target hits / Stop hits / Partials:** 8 / 8 / 0
- **Avg / median % per leg:** 4.20% / 8.82%
- **Sum % (uncompounded):** 67.18%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 8 | 50.0% | 8 | 8 | 0 | 4.20% | 67.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 8 | 50.0% | 8 | 8 | 0 | 4.20% | 67.2% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 16 | 8 | 50.0% | 8 | 8 | 0 | 4.20% | 67.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 10:15:00 | 7567.50 | 6392.56 | 6387.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 11:15:00 | 7647.00 | 6405.04 | 6393.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 14:15:00 | 8654.00 | 8663.33 | 8166.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-28 15:00:00 | 8654.00 | 8663.33 | 8166.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 8420.00 | 8745.49 | 8426.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 8420.00 | 8745.49 | 8426.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 8454.00 | 8742.59 | 8427.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:30:00 | 8448.00 | 8742.59 | 8427.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 8450.50 | 8736.86 | 8427.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:15:00 | 8430.50 | 8736.86 | 8427.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 8440.00 | 8733.91 | 8427.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:30:00 | 8450.00 | 8733.91 | 8427.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 8448.00 | 8731.06 | 8427.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:15:00 | 8407.00 | 8731.06 | 8427.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 8496.50 | 8728.73 | 8427.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 8417.00 | 8728.73 | 8427.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 8440.00 | 8722.99 | 8427.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:45:00 | 8431.00 | 8722.99 | 8427.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 8413.50 | 8719.91 | 8427.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:30:00 | 8411.00 | 8719.91 | 8427.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 8404.50 | 8716.77 | 8427.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:00:00 | 8404.50 | 8716.77 | 8427.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 8361.00 | 8713.23 | 8427.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 8361.00 | 8713.23 | 8427.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 8405.00 | 8702.39 | 8427.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:45:00 | 8380.00 | 8702.39 | 8427.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 8334.00 | 8698.73 | 8427.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:15:00 | 8267.00 | 8698.73 | 8427.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 15:15:00 | 7803.00 | 8250.77 | 8251.84 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 13:15:00 | 8551.00 | 8253.80 | 8252.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 09:15:00 | 8559.50 | 8261.16 | 8256.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 11:15:00 | 8486.50 | 8508.93 | 8403.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 12:00:00 | 8486.50 | 8508.93 | 8403.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 8380.50 | 8507.04 | 8403.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:00:00 | 8380.50 | 8507.04 | 8403.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 8391.50 | 8505.89 | 8403.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 15:15:00 | 8319.00 | 8505.89 | 8403.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 8319.00 | 8504.03 | 8403.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:15:00 | 8356.50 | 8504.03 | 8403.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 8437.00 | 8502.49 | 8403.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 09:30:00 | 8450.50 | 8450.62 | 8388.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 10:00:00 | 8450.00 | 8450.62 | 8388.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 13:15:00 | 8340.50 | 8448.06 | 8388.04 | SL hit (close<static) qty=1.00 sl=8367.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 11:15:00 | 8350.00 | 8725.95 | 8727.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 13:15:00 | 8253.00 | 8717.35 | 8722.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 10:15:00 | 7925.50 | 7851.31 | 8181.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 11:00:00 | 7925.50 | 7851.31 | 8181.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 8083.00 | 7866.73 | 8168.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:45:00 | 8113.50 | 7866.73 | 8168.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 8162.00 | 7871.69 | 8163.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 8162.00 | 7871.69 | 8163.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 8190.00 | 7874.86 | 8163.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 9322.50 | 7874.86 | 8163.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 9452.00 | 7890.55 | 8170.23 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 11:15:00 | 9500.00 | 8407.66 | 8405.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 10:15:00 | 9575.00 | 8472.66 | 8438.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 9782.00 | 9811.86 | 9322.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 10:45:00 | 9822.00 | 9811.86 | 9322.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 9268.00 | 9798.15 | 9325.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 9268.00 | 9798.15 | 9325.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 9245.00 | 9792.65 | 9324.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 9041.00 | 9792.65 | 9324.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 9112.00 | 9753.48 | 9323.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 9112.00 | 9753.48 | 9323.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 9022.00 | 9746.20 | 9321.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:00:00 | 9022.00 | 9746.20 | 9321.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 9486.00 | 9649.06 | 9310.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 12:30:00 | 9630.00 | 9647.65 | 9314.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 09:45:00 | 9634.00 | 9647.20 | 9321.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 14:45:00 | 9614.00 | 9644.94 | 9328.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 15:15:00 | 9631.00 | 9644.94 | 9328.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 9402.00 | 9642.48 | 9342.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:45:00 | 9403.00 | 9642.48 | 9342.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 12:15:00 | 9380.00 | 9637.72 | 9342.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 13:00:00 | 9380.00 | 9637.72 | 9342.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Target hit | 2026-03-27 14:15:00 | 10593.00 | 9710.05 | 9412.38 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-10-06 09:30:00 | 8450.50 | 2025-10-06 13:15:00 | 8340.50 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-10-06 10:00:00 | 8450.00 | 2025-10-06 13:15:00 | 8340.50 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-10-06 14:45:00 | 8477.00 | 2025-10-09 11:15:00 | 8349.00 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-10-07 09:30:00 | 8470.50 | 2025-10-09 11:15:00 | 8349.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-10-14 09:15:00 | 8585.50 | 2025-10-14 12:15:00 | 8363.00 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2025-10-16 09:45:00 | 8700.00 | 2025-10-29 13:15:00 | 9467.15 | TARGET_HIT | 1.00 | 8.82% |
| BUY | retest2 | 2025-10-20 11:30:00 | 8606.50 | 2025-10-29 13:15:00 | 9462.20 | TARGET_HIT | 1.00 | 9.94% |
| BUY | retest2 | 2025-10-20 14:30:00 | 8602.00 | 2025-10-29 14:15:00 | 9570.00 | TARGET_HIT | 1.00 | 11.25% |
| BUY | retest2 | 2025-11-11 15:00:00 | 8567.50 | 2025-11-19 09:15:00 | 9424.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-09 10:00:00 | 8574.50 | 2025-12-30 10:15:00 | 8430.50 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-12-29 12:30:00 | 8553.50 | 2025-12-30 10:15:00 | 8430.50 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-12-29 15:00:00 | 8566.50 | 2025-12-30 10:15:00 | 8430.50 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2026-03-18 12:30:00 | 9630.00 | 2026-03-27 14:15:00 | 10593.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-19 09:45:00 | 9634.00 | 2026-03-27 14:15:00 | 10597.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-19 14:45:00 | 9614.00 | 2026-03-27 14:15:00 | 10575.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-19 15:15:00 | 9631.00 | 2026-03-27 14:15:00 | 10594.10 | TARGET_HIT | 1.00 | 10.00% |
