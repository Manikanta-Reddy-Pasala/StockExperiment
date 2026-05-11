# Bajaj Auto Ltd. (BAJAJ-AUTO)

## Backtest Summary

- **Window:** 2022-04-07 13:15:00 → 2026-05-08 15:15:00 (7050 bars)
- **Last close:** 10696.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 2 |
| ALERT3 | 37 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 38 |
| PARTIAL | 5 |
| TARGET_HIT | 7 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 42 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 13 / 29
- **Target hits / Stop hits / Partials:** 6 / 31 / 5
- **Avg / median % per leg:** 0.73% / -0.84%
- **Sum % (uncompounded):** 30.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 3 | 12.5% | 1 | 23 | 0 | -0.45% | -10.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 24 | 3 | 12.5% | 1 | 23 | 0 | -0.45% | -10.9% |
| SELL (all) | 18 | 10 | 55.6% | 5 | 8 | 5 | 2.32% | 41.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 10 | 55.6% | 5 | 8 | 5 | 2.32% | 41.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 42 | 13 | 31.0% | 6 | 31 | 5 | 0.73% | 30.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 9416.60 | 10783.00 | 10783.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 09:15:00 | 9200.00 | 10066.47 | 10346.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 09:15:00 | 8775.00 | 8707.47 | 9053.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-30 09:45:00 | 8770.00 | 8707.47 | 9053.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 8998.60 | 8726.74 | 9036.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:00:00 | 8998.60 | 8726.74 | 9036.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 9062.85 | 8730.09 | 9036.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 9095.00 | 8730.09 | 9036.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 9130.00 | 8734.07 | 9037.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 14:00:00 | 9130.00 | 8734.07 | 9037.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 9174.95 | 8738.45 | 9038.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 15:00:00 | 9174.95 | 8738.45 | 9038.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 8968.90 | 8747.95 | 9038.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 11:45:00 | 8925.05 | 8750.26 | 9038.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 15:15:00 | 8900.00 | 8756.21 | 9036.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 10:45:00 | 8920.00 | 8761.42 | 9035.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 14:45:00 | 8914.00 | 8770.31 | 9034.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 8976.00 | 8792.79 | 9023.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:45:00 | 9009.20 | 8792.79 | 9023.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 8951.15 | 8800.82 | 9021.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 10:45:00 | 8923.65 | 8801.94 | 9021.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 13:15:00 | 8478.80 | 8779.72 | 8978.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 13:15:00 | 8455.00 | 8779.72 | 8978.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 13:15:00 | 8474.00 | 8779.72 | 8978.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 13:15:00 | 8468.30 | 8779.72 | 8978.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 13:15:00 | 8477.47 | 8779.72 | 8978.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-28 11:15:00 | 8032.54 | 8614.03 | 8832.84 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 14:15:00 | 8706.00 | 8165.04 | 8163.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 15:15:00 | 8744.00 | 8170.81 | 8165.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 8444.00 | 8497.01 | 8377.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 8444.00 | 8497.01 | 8377.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 8444.00 | 8497.01 | 8377.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 15:15:00 | 8482.50 | 8494.30 | 8379.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 09:45:00 | 8470.00 | 8494.24 | 8380.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 10:30:00 | 8481.00 | 8497.13 | 8386.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 09:15:00 | 8472.00 | 8500.00 | 8394.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 8327.00 | 8499.15 | 8398.07 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 8327.00 | 8499.15 | 8398.07 | SL hit (close<static) qty=1.00 sl=8350.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 8321.00 | 8363.82 | 8363.97 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 14:15:00 | 8444.50 | 8364.81 | 8364.41 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 14:15:00 | 8301.00 | 8363.71 | 8363.88 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 13:15:00 | 8397.00 | 8364.26 | 8364.15 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 14:15:00 | 8285.50 | 8363.43 | 8363.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 8060.00 | 8359.67 | 8361.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 15:15:00 | 8245.00 | 8242.30 | 8291.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-08 09:15:00 | 8211.00 | 8242.30 | 8291.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 8274.00 | 8238.95 | 8286.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:45:00 | 8288.50 | 8238.95 | 8286.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 8235.00 | 8238.91 | 8286.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:30:00 | 8284.50 | 8239.50 | 8286.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 8258.00 | 8239.69 | 8286.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 12:00:00 | 8246.00 | 8239.75 | 8286.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 13:30:00 | 8249.00 | 8239.98 | 8285.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 14:00:00 | 8237.50 | 8239.98 | 8285.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 10:00:00 | 8242.00 | 8239.04 | 8284.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 8285.00 | 8239.50 | 8284.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:00:00 | 8285.00 | 8239.50 | 8284.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 8255.50 | 8239.66 | 8284.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 10:00:00 | 8217.00 | 8240.25 | 8283.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 10:30:00 | 8214.00 | 8239.97 | 8283.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 12:30:00 | 8217.50 | 8239.68 | 8282.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 13:15:00 | 8220.00 | 8239.68 | 8282.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 8573.00 | 8242.35 | 8283.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 8573.00 | 8242.35 | 8283.13 | SL hit (close>static) qty=1.00 sl=8310.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 13:15:00 | 8826.50 | 8322.29 | 8321.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 8913.50 | 8469.25 | 8403.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 10:15:00 | 8872.50 | 8904.41 | 8716.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-24 11:00:00 | 8872.50 | 8904.41 | 8716.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 8697.50 | 8891.87 | 8725.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 15:00:00 | 8697.50 | 8891.87 | 8725.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 8715.00 | 8890.11 | 8725.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 09:15:00 | 8774.00 | 8890.11 | 8725.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 11:45:00 | 8719.00 | 8885.90 | 8726.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 12:15:00 | 8732.00 | 8885.90 | 8726.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 12:45:00 | 8740.00 | 8884.57 | 8726.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 8709.50 | 8881.82 | 8726.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 8709.50 | 8881.82 | 8726.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 8729.00 | 8880.30 | 8726.52 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-30 11:15:00 | 8674.00 | 8874.77 | 8726.02 | SL hit (close<static) qty=1.00 sl=8690.50 alert=retest2 |

### Cycle 9 — SELL (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 14:15:00 | 9048.50 | 9437.80 | 9439.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 8836.50 | 9428.24 | 9434.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 9446.50 | 9187.50 | 9292.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 9446.50 | 9187.50 | 9292.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 9446.50 | 9187.50 | 9292.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:45:00 | 9413.50 | 9187.50 | 9292.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 9429.00 | 9189.90 | 9293.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:45:00 | 9440.00 | 9189.90 | 9293.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 11:15:00 | 9770.00 | 9375.89 | 9374.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 10:15:00 | 9799.00 | 9398.74 | 9386.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 14:15:00 | 9490.00 | 9492.89 | 9442.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 15:00:00 | 9490.00 | 9492.89 | 9442.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 9374.00 | 9498.26 | 9447.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 9374.00 | 9498.26 | 9447.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 9678.00 | 9500.05 | 9448.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 12:45:00 | 9852.50 | 9505.89 | 9452.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-02-03 11:45:00 | 8925.05 | 2025-02-14 13:15:00 | 8478.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-03 15:15:00 | 8900.00 | 2025-02-14 13:15:00 | 8455.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-04 10:45:00 | 8920.00 | 2025-02-14 13:15:00 | 8474.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-04 14:45:00 | 8914.00 | 2025-02-14 13:15:00 | 8468.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 10:45:00 | 8923.65 | 2025-02-14 13:15:00 | 8477.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-03 11:45:00 | 8925.05 | 2025-02-28 11:15:00 | 8032.54 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-03 15:15:00 | 8900.00 | 2025-02-28 11:15:00 | 8028.00 | TARGET_HIT | 0.50 | 9.80% |
| SELL | retest2 | 2025-02-04 10:45:00 | 8920.00 | 2025-02-28 11:15:00 | 8022.60 | TARGET_HIT | 0.50 | 10.06% |
| SELL | retest2 | 2025-02-04 14:45:00 | 8914.00 | 2025-02-28 11:15:00 | 8031.28 | TARGET_HIT | 0.50 | 9.90% |
| SELL | retest2 | 2025-02-10 10:45:00 | 8923.65 | 2025-02-28 12:15:00 | 8010.00 | TARGET_HIT | 0.50 | 10.24% |
| BUY | retest2 | 2025-06-13 15:15:00 | 8482.50 | 2025-06-20 09:15:00 | 8327.00 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-06-16 09:45:00 | 8470.00 | 2025-06-20 09:15:00 | 8327.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-06-17 10:30:00 | 8481.00 | 2025-06-20 09:15:00 | 8327.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-06-19 09:15:00 | 8472.00 | 2025-06-20 09:15:00 | 8327.00 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-06-25 09:30:00 | 8432.00 | 2025-07-01 10:15:00 | 8338.50 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-06-25 12:15:00 | 8386.50 | 2025-07-01 10:15:00 | 8338.50 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-06-25 14:00:00 | 8385.50 | 2025-07-01 10:15:00 | 8338.50 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-06-26 13:45:00 | 8385.50 | 2025-07-01 10:15:00 | 8338.50 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-08-12 12:00:00 | 8246.00 | 2025-08-18 09:15:00 | 8573.00 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2025-08-12 13:30:00 | 8249.00 | 2025-08-18 09:15:00 | 8573.00 | STOP_HIT | 1.00 | -3.93% |
| SELL | retest2 | 2025-08-12 14:00:00 | 8237.50 | 2025-08-18 09:15:00 | 8573.00 | STOP_HIT | 1.00 | -4.07% |
| SELL | retest2 | 2025-08-13 10:00:00 | 8242.00 | 2025-08-18 09:15:00 | 8573.00 | STOP_HIT | 1.00 | -4.02% |
| SELL | retest2 | 2025-08-14 10:00:00 | 8217.00 | 2025-08-18 09:15:00 | 8573.00 | STOP_HIT | 1.00 | -4.33% |
| SELL | retest2 | 2025-08-14 10:30:00 | 8214.00 | 2025-08-18 09:15:00 | 8573.00 | STOP_HIT | 1.00 | -4.37% |
| SELL | retest2 | 2025-08-14 12:30:00 | 8217.50 | 2025-08-18 09:15:00 | 8573.00 | STOP_HIT | 1.00 | -4.33% |
| SELL | retest2 | 2025-08-14 13:15:00 | 8220.00 | 2025-08-18 09:15:00 | 8573.00 | STOP_HIT | 1.00 | -4.29% |
| BUY | retest2 | 2025-09-29 09:15:00 | 8774.00 | 2025-09-30 11:15:00 | 8674.00 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-09-29 11:45:00 | 8719.00 | 2025-09-30 11:15:00 | 8674.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-09-29 12:15:00 | 8732.00 | 2025-09-30 11:15:00 | 8674.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-09-29 12:45:00 | 8740.00 | 2025-09-30 11:15:00 | 8674.00 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-10-01 09:15:00 | 8840.00 | 2025-10-01 11:15:00 | 8670.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-10-06 12:45:00 | 8731.00 | 2025-11-07 09:15:00 | 8624.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-10-06 13:45:00 | 8741.00 | 2025-11-07 09:15:00 | 8624.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-10-09 10:30:00 | 8736.50 | 2025-11-07 09:15:00 | 8624.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-11-13 09:15:00 | 8876.50 | 2025-11-14 10:15:00 | 8802.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-11-13 09:45:00 | 8886.50 | 2025-11-14 10:15:00 | 8802.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-11-13 10:45:00 | 8882.00 | 2025-11-14 10:15:00 | 8802.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-11-13 11:30:00 | 8873.50 | 2025-11-14 10:15:00 | 8802.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-11-17 09:15:00 | 8975.00 | 2025-12-18 09:15:00 | 8745.00 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2025-12-19 09:15:00 | 8881.00 | 2026-01-06 09:15:00 | 9769.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-13 14:00:00 | 8869.00 | 2026-03-20 14:15:00 | 9048.50 | STOP_HIT | 1.00 | 2.02% |
| BUY | retest2 | 2026-03-13 14:30:00 | 8885.50 | 2026-03-20 14:15:00 | 9048.50 | STOP_HIT | 1.00 | 1.83% |
