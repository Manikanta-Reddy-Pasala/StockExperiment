# Oracle Financial Services Software Ltd. (OFSS)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 9321.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 74 |
| ALERT1 | 54 |
| ALERT2 | 51 |
| ALERT2_SKIP | 28 |
| ALERT3 | 135 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 57 |
| PARTIAL | 6 |
| TARGET_HIT | 2 |
| STOP_HIT | 60 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 65 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 43
- **Target hits / Stop hits / Partials:** 2 / 57 / 6
- **Avg / median % per leg:** 0.28% / -0.72%
- **Sum % (uncompounded):** 18.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 4 | 18.2% | 1 | 21 | 0 | -0.53% | -11.6% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.92% | -0.9% |
| BUY @ 3rd Alert (retest2) | 21 | 4 | 19.0% | 1 | 20 | 0 | -0.51% | -10.7% |
| SELL (all) | 43 | 18 | 41.9% | 1 | 36 | 6 | 0.69% | 29.7% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.98% | 2.0% |
| SELL @ 3rd Alert (retest2) | 42 | 17 | 40.5% | 1 | 35 | 6 | 0.66% | 27.8% |
| retest1 (combined) | 2 | 1 | 50.0% | 0 | 2 | 0 | 0.53% | 1.1% |
| retest2 (combined) | 63 | 21 | 33.3% | 2 | 55 | 6 | 0.27% | 17.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 8545.00 | 8277.13 | 8272.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 8633.00 | 8348.30 | 8305.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 8532.50 | 8538.25 | 8450.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 12:45:00 | 8503.50 | 8538.25 | 8450.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 8512.00 | 8509.24 | 8461.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 8608.00 | 8520.82 | 8499.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 11:30:00 | 8594.00 | 8545.84 | 8517.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 8502.00 | 8600.34 | 8603.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 8502.00 | 8600.34 | 8603.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 8436.00 | 8567.47 | 8587.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 8513.50 | 8348.99 | 8393.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 8513.50 | 8348.99 | 8393.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 8513.50 | 8348.99 | 8393.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 8513.50 | 8348.99 | 8393.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 8462.00 | 8371.59 | 8400.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:45:00 | 8446.00 | 8371.59 | 8400.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 8389.00 | 8384.58 | 8399.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 14:15:00 | 8421.00 | 8384.58 | 8399.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 8369.50 | 8381.57 | 8397.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 15:15:00 | 8356.00 | 8381.57 | 8397.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 09:15:00 | 8455.00 | 8392.16 | 8399.01 | SL hit (close>static) qty=1.00 sl=8428.50 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 11:15:00 | 8440.50 | 8406.84 | 8404.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 13:15:00 | 8462.50 | 8421.68 | 8412.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 8406.50 | 8430.90 | 8419.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 8406.50 | 8430.90 | 8419.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 8406.50 | 8430.90 | 8419.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:30:00 | 8441.50 | 8430.90 | 8419.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 8425.00 | 8429.72 | 8420.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 8469.00 | 8429.72 | 8420.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 15:15:00 | 8405.00 | 8416.82 | 8417.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 15:15:00 | 8405.00 | 8416.82 | 8417.27 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 8479.00 | 8429.25 | 8422.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 12:15:00 | 8570.50 | 8473.40 | 8446.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 14:15:00 | 8494.50 | 8505.90 | 8485.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 14:45:00 | 8499.00 | 8505.90 | 8485.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 8519.00 | 8508.52 | 8488.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 8513.50 | 8508.52 | 8488.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 8490.50 | 8504.92 | 8488.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 11:00:00 | 8559.50 | 8515.83 | 8495.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 12:15:00 | 8537.50 | 8515.97 | 8497.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 8430.50 | 8496.84 | 8493.05 | SL hit (close<static) qty=1.00 sl=8458.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 15:15:00 | 8442.00 | 8485.87 | 8488.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 8413.00 | 8471.29 | 8481.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 14:15:00 | 8371.50 | 8359.35 | 8395.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-03 15:00:00 | 8371.50 | 8359.35 | 8395.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 8547.00 | 8396.99 | 8406.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:00:00 | 8547.00 | 8396.99 | 8406.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 8714.50 | 8460.49 | 8434.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 10:15:00 | 8845.00 | 8661.43 | 8563.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 8728.00 | 8736.30 | 8661.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 10:45:00 | 8750.50 | 8736.30 | 8661.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 9588.00 | 9443.33 | 9368.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 9640.50 | 9443.33 | 9368.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:15:00 | 9625.00 | 9507.98 | 9445.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 11:00:00 | 9610.50 | 9528.49 | 9460.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 12:00:00 | 9627.00 | 9548.19 | 9475.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 9645.00 | 9689.18 | 9645.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:45:00 | 9618.00 | 9689.18 | 9645.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 9615.00 | 9674.34 | 9642.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 14:00:00 | 9615.00 | 9674.34 | 9642.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 9662.50 | 9671.97 | 9644.46 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-19 09:15:00 | 9459.00 | 9624.98 | 9627.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 09:15:00 | 9459.00 | 9624.98 | 9627.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 9364.00 | 9572.79 | 9603.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 14:15:00 | 9403.50 | 9377.41 | 9441.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 15:00:00 | 9403.50 | 9377.41 | 9441.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 9271.50 | 9214.03 | 9266.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 12:00:00 | 9190.00 | 9209.06 | 9255.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 14:15:00 | 9189.50 | 9209.96 | 9247.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 12:15:00 | 9082.50 | 9002.87 | 8993.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 12:15:00 | 9082.50 | 9002.87 | 8993.95 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 15:15:00 | 9011.00 | 9028.99 | 9030.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 09:15:00 | 8951.00 | 9013.40 | 9023.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 12:15:00 | 9053.00 | 9002.45 | 9014.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 12:15:00 | 9053.00 | 9002.45 | 9014.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 9053.00 | 9002.45 | 9014.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:00:00 | 9053.00 | 9002.45 | 9014.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 8981.50 | 8998.26 | 9011.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 14:15:00 | 8973.00 | 8998.26 | 9011.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 8975.00 | 9000.17 | 9010.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 10:00:00 | 8972.50 | 8994.63 | 9006.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:15:00 | 8974.50 | 8988.95 | 9001.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:15:00 | 8524.35 | 8666.06 | 8751.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:15:00 | 8526.25 | 8666.06 | 8751.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:15:00 | 8523.88 | 8666.06 | 8751.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:15:00 | 8525.77 | 8666.06 | 8751.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 10:15:00 | 8679.50 | 8668.75 | 8745.03 | SL hit (close>ema200) qty=0.50 sl=8668.75 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 8815.00 | 8750.07 | 8750.04 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 15:15:00 | 8735.50 | 8776.17 | 8780.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 10:15:00 | 8717.00 | 8759.03 | 8771.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 11:15:00 | 8779.50 | 8763.12 | 8772.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 11:15:00 | 8779.50 | 8763.12 | 8772.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 8779.50 | 8763.12 | 8772.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:45:00 | 8775.50 | 8763.12 | 8772.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 8822.50 | 8775.00 | 8776.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:30:00 | 8823.00 | 8775.00 | 8776.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 13:15:00 | 8823.50 | 8784.70 | 8781.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 11:15:00 | 8845.50 | 8800.08 | 8789.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 8764.50 | 8807.29 | 8799.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 8764.50 | 8807.29 | 8799.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 8764.50 | 8807.29 | 8799.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:00:00 | 8764.50 | 8807.29 | 8799.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 10:15:00 | 8734.50 | 8792.73 | 8793.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 11:15:00 | 8700.00 | 8774.19 | 8784.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 10:15:00 | 8727.00 | 8724.25 | 8750.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 10:15:00 | 8727.00 | 8724.25 | 8750.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 8727.00 | 8724.25 | 8750.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:45:00 | 8745.00 | 8724.25 | 8750.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 8749.00 | 8729.20 | 8750.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 12:00:00 | 8749.00 | 8729.20 | 8750.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 8767.00 | 8736.76 | 8751.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 12:45:00 | 8800.00 | 8736.76 | 8751.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 8749.50 | 8739.31 | 8751.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:45:00 | 8768.50 | 8739.31 | 8751.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 8785.00 | 8748.45 | 8754.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:00:00 | 8785.00 | 8748.45 | 8754.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 8767.50 | 8752.26 | 8755.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 8817.00 | 8752.26 | 8755.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 09:15:00 | 8851.50 | 8772.11 | 8764.38 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 13:15:00 | 8670.00 | 8808.34 | 8813.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 14:15:00 | 8625.00 | 8771.68 | 8796.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 14:15:00 | 8696.00 | 8677.68 | 8726.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 15:00:00 | 8696.00 | 8677.68 | 8726.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 8602.00 | 8655.08 | 8707.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:30:00 | 8687.00 | 8655.08 | 8707.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 8662.50 | 8651.13 | 8684.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:30:00 | 8692.00 | 8651.13 | 8684.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 8607.00 | 8641.08 | 8673.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 8592.00 | 8641.08 | 8673.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 14:00:00 | 8605.00 | 8628.86 | 8657.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 14:30:00 | 8601.50 | 8619.88 | 8650.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 15:15:00 | 8570.00 | 8501.62 | 8497.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 8570.00 | 8501.62 | 8497.45 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 8456.00 | 8507.97 | 8511.78 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 8570.00 | 8513.26 | 8510.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 15:15:00 | 8590.00 | 8528.61 | 8517.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 09:15:00 | 8505.00 | 8523.89 | 8516.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 09:15:00 | 8505.00 | 8523.89 | 8516.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 8505.00 | 8523.89 | 8516.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:00:00 | 8505.00 | 8523.89 | 8516.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 8527.50 | 8524.61 | 8517.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:45:00 | 8509.00 | 8524.61 | 8517.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 8509.00 | 8521.49 | 8516.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:00:00 | 8509.00 | 8521.49 | 8516.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 8507.00 | 8518.59 | 8515.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:15:00 | 8497.00 | 8518.59 | 8515.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 13:15:00 | 8479.00 | 8510.67 | 8512.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 8415.00 | 8491.54 | 8503.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 8617.50 | 8462.29 | 8469.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 8617.50 | 8462.29 | 8469.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 8617.50 | 8462.29 | 8469.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:00:00 | 8617.50 | 8462.29 | 8469.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 10:15:00 | 8560.00 | 8481.83 | 8477.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 09:15:00 | 8624.50 | 8565.83 | 8536.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 14:15:00 | 8585.00 | 8595.10 | 8565.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 15:00:00 | 8585.00 | 8595.10 | 8565.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 8541.00 | 8588.79 | 8572.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:00:00 | 8541.00 | 8588.79 | 8572.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 8589.00 | 8588.83 | 8573.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:30:00 | 8535.50 | 8588.83 | 8573.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 13:15:00 | 8574.50 | 8585.96 | 8573.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 14:00:00 | 8574.50 | 8585.96 | 8573.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 8553.50 | 8579.47 | 8571.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 15:00:00 | 8553.50 | 8579.47 | 8571.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 8584.50 | 8580.48 | 8573.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 8510.00 | 8572.78 | 8570.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 8552.00 | 8568.63 | 8568.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 11:45:00 | 8576.00 | 8577.10 | 8572.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 13:15:00 | 8666.50 | 8690.34 | 8691.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 13:15:00 | 8666.50 | 8690.34 | 8691.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 8625.00 | 8677.27 | 8685.71 | Break + close below crossover candle low |

### Cycle 23 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 8845.00 | 8703.57 | 8695.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 10:15:00 | 8894.00 | 8741.66 | 8713.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 14:15:00 | 8753.50 | 8777.72 | 8743.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 14:15:00 | 8753.50 | 8777.72 | 8743.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 8753.50 | 8777.72 | 8743.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:00:00 | 8753.50 | 8777.72 | 8743.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 8751.00 | 8772.38 | 8744.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 8689.00 | 8772.38 | 8744.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 8723.00 | 8762.50 | 8742.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 8685.50 | 8762.50 | 8742.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 8696.00 | 8749.20 | 8738.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:45:00 | 8686.00 | 8749.20 | 8738.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 8685.00 | 8736.36 | 8733.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:00:00 | 8685.00 | 8736.36 | 8733.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 12:15:00 | 8655.00 | 8720.09 | 8726.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 8575.00 | 8691.07 | 8712.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 8445.50 | 8442.71 | 8530.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 8445.50 | 8442.71 | 8530.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 8417.00 | 8384.08 | 8457.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 8451.50 | 8384.08 | 8457.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 8471.00 | 8401.46 | 8459.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 8471.00 | 8401.46 | 8459.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 8425.00 | 8406.17 | 8455.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 12:30:00 | 8420.00 | 8415.54 | 8455.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 15:15:00 | 8500.00 | 8444.03 | 8459.51 | SL hit (close>static) qty=1.00 sl=8478.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 8540.00 | 8475.99 | 8471.39 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 10:15:00 | 8410.50 | 8462.84 | 8468.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 10:15:00 | 8359.50 | 8415.62 | 8438.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 8318.00 | 8286.25 | 8325.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 8318.00 | 8286.25 | 8325.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 8318.00 | 8286.25 | 8325.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:45:00 | 8336.50 | 8286.25 | 8325.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 8370.00 | 8303.00 | 8329.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:00:00 | 8370.00 | 8303.00 | 8329.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 8389.00 | 8320.20 | 8335.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:00:00 | 8389.00 | 8320.20 | 8335.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 8294.00 | 8322.14 | 8333.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:45:00 | 8315.50 | 8322.14 | 8333.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 8360.00 | 8326.17 | 8332.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:45:00 | 8396.00 | 8326.17 | 8332.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 8381.00 | 8337.13 | 8337.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:30:00 | 8388.50 | 8337.13 | 8337.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 11:15:00 | 8357.50 | 8341.21 | 8339.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 13:15:00 | 8394.50 | 8355.19 | 8346.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 9075.00 | 9103.23 | 8890.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 13:00:00 | 9075.00 | 9103.23 | 8890.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 8987.50 | 9092.56 | 8953.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:45:00 | 8940.50 | 9092.56 | 8953.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 8946.00 | 9043.40 | 8954.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:00:00 | 8946.00 | 9043.40 | 8954.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 8945.00 | 9023.72 | 8953.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 13:30:00 | 8973.00 | 9010.97 | 8954.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 14:15:00 | 8980.00 | 9010.97 | 8954.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 8922.50 | 8981.54 | 8953.90 | SL hit (close<static) qty=1.00 sl=8927.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 13:15:00 | 9055.00 | 9092.42 | 9093.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 14:15:00 | 9009.00 | 9075.74 | 9086.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 11:15:00 | 9089.00 | 9047.70 | 9066.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 11:15:00 | 9089.00 | 9047.70 | 9066.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 9089.00 | 9047.70 | 9066.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:00:00 | 9089.00 | 9047.70 | 9066.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 9063.50 | 9050.86 | 9066.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 14:30:00 | 9032.50 | 9054.83 | 9065.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 09:15:00 | 9102.00 | 9065.09 | 9068.46 | SL hit (close>static) qty=1.00 sl=9099.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 12:15:00 | 9119.00 | 9078.62 | 9073.84 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 10:15:00 | 8994.50 | 9059.97 | 9068.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 12:15:00 | 8967.00 | 9029.59 | 9047.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 11:15:00 | 8533.00 | 8437.21 | 8510.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 11:15:00 | 8533.00 | 8437.21 | 8510.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 8533.00 | 8437.21 | 8510.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 8533.00 | 8437.21 | 8510.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 8515.00 | 8452.77 | 8510.83 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 8735.00 | 8581.47 | 8561.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 8872.50 | 8639.68 | 8589.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 10:15:00 | 9167.50 | 9196.20 | 9083.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 11:00:00 | 9167.50 | 9196.20 | 9083.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 9183.00 | 9186.11 | 9114.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:45:00 | 9121.50 | 9186.11 | 9114.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 9236.50 | 9269.86 | 9223.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:30:00 | 9243.50 | 9269.86 | 9223.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 9199.00 | 9255.69 | 9221.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:00:00 | 9199.00 | 9255.69 | 9221.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 9204.50 | 9245.45 | 9219.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 9255.50 | 9236.56 | 9218.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 10:15:00 | 9151.50 | 9219.38 | 9213.48 | SL hit (close<static) qty=1.00 sl=9190.50 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 9118.00 | 9199.10 | 9204.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 9055.00 | 9148.32 | 9175.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 9034.50 | 9030.38 | 9094.98 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-15 11:45:00 | 8854.50 | 8979.72 | 9060.35 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 8644.50 | 8629.20 | 8693.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 13:30:00 | 8575.00 | 8603.52 | 8660.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 12:45:00 | 8583.00 | 8571.73 | 8615.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 8679.50 | 8592.32 | 8610.58 | SL hit (close>ema400) qty=1.00 sl=8610.58 alert=retest1 |

### Cycle 33 — BUY (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 12:15:00 | 8671.00 | 8629.47 | 8624.73 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 12:15:00 | 8586.00 | 8624.39 | 8628.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 13:15:00 | 8552.00 | 8609.92 | 8621.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 15:15:00 | 8624.50 | 8611.09 | 8620.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 15:15:00 | 8624.50 | 8611.09 | 8620.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 8624.50 | 8611.09 | 8620.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 8664.00 | 8611.09 | 8620.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 8620.00 | 8612.87 | 8620.08 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 8680.50 | 8635.86 | 8629.86 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 8609.00 | 8629.53 | 8631.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 8589.00 | 8614.08 | 8622.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 10:15:00 | 8235.00 | 8140.67 | 8201.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 10:15:00 | 8235.00 | 8140.67 | 8201.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 8235.00 | 8140.67 | 8201.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 8235.00 | 8140.67 | 8201.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 8211.50 | 8154.83 | 8202.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 12:30:00 | 8205.50 | 8161.57 | 8201.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:30:00 | 8200.00 | 8170.05 | 8201.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 14:30:00 | 8205.00 | 8169.04 | 8198.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:30:00 | 8198.00 | 8185.59 | 8200.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 8191.00 | 8186.67 | 8199.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 11:15:00 | 8176.00 | 8186.67 | 8199.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 14:00:00 | 8184.00 | 8191.54 | 8199.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 15:15:00 | 8248.00 | 8210.11 | 8206.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 8248.00 | 8210.11 | 8206.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 8310.00 | 8230.09 | 8216.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 8357.00 | 8360.04 | 8316.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 12:45:00 | 8355.00 | 8360.04 | 8316.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 8298.50 | 8344.29 | 8317.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 8298.50 | 8344.29 | 8317.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 8300.50 | 8335.53 | 8315.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 8265.00 | 8335.53 | 8315.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 8264.00 | 8297.20 | 8300.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 8173.00 | 8272.36 | 8289.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 8297.50 | 8256.69 | 8273.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 8297.50 | 8256.69 | 8273.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 8297.50 | 8256.69 | 8273.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:30:00 | 8288.00 | 8256.69 | 8273.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 8306.50 | 8266.65 | 8276.93 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 13:15:00 | 8310.00 | 8288.21 | 8285.31 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 8210.50 | 8274.14 | 8279.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 13:15:00 | 8200.00 | 8238.96 | 8259.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 8296.00 | 8230.48 | 8248.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 8296.00 | 8230.48 | 8248.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 8296.00 | 8230.48 | 8248.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:45:00 | 8278.50 | 8230.48 | 8248.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 8282.50 | 8240.88 | 8251.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:45:00 | 8312.00 | 8240.88 | 8251.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 11:15:00 | 8336.00 | 8259.91 | 8259.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 13:15:00 | 8348.00 | 8290.98 | 8274.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 14:15:00 | 8370.50 | 8386.96 | 8344.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-20 15:00:00 | 8370.50 | 8386.96 | 8344.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 8272.50 | 8361.04 | 8340.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 8272.50 | 8361.04 | 8340.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 8259.00 | 8340.63 | 8332.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:30:00 | 8242.50 | 8340.63 | 8332.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 8242.00 | 8320.90 | 8324.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 12:15:00 | 8215.50 | 8299.82 | 8314.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 8172.00 | 8122.05 | 8164.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 8172.00 | 8122.05 | 8164.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 8172.00 | 8122.05 | 8164.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:45:00 | 8157.00 | 8122.05 | 8164.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 8133.50 | 8124.34 | 8161.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:15:00 | 8165.00 | 8124.34 | 8161.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 8185.00 | 8136.47 | 8164.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:00:00 | 8185.00 | 8136.47 | 8164.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 8175.50 | 8144.28 | 8165.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:00:00 | 8175.50 | 8144.28 | 8165.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 8181.00 | 8151.62 | 8166.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 14:00:00 | 8181.00 | 8151.62 | 8166.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 8200.00 | 8165.52 | 8170.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:15:00 | 8162.00 | 8165.52 | 8170.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 8225.00 | 8177.41 | 8175.45 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 11:15:00 | 8143.00 | 8168.31 | 8171.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 10:15:00 | 8105.00 | 8141.85 | 8155.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 8143.50 | 8124.92 | 8138.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 8143.50 | 8124.92 | 8138.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 8143.50 | 8124.92 | 8138.86 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 13:15:00 | 8174.50 | 8148.77 | 8146.87 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 8120.50 | 8142.67 | 8144.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 8105.00 | 8135.13 | 8141.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 15:15:00 | 8133.00 | 8117.16 | 8128.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 15:15:00 | 8133.00 | 8117.16 | 8128.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 8133.00 | 8117.16 | 8128.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 8127.50 | 8117.16 | 8128.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 8081.50 | 8110.03 | 8124.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 11:00:00 | 8078.00 | 8103.62 | 8120.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 13:45:00 | 8075.50 | 8094.28 | 8111.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 15:15:00 | 8075.00 | 8092.03 | 8109.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 8213.00 | 8113.50 | 8115.77 | SL hit (close>static) qty=1.00 sl=8190.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 10:15:00 | 8193.00 | 8129.40 | 8122.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 10:15:00 | 8221.00 | 8180.31 | 8156.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 8135.50 | 8189.18 | 8173.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 8135.50 | 8189.18 | 8173.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 8135.50 | 8189.18 | 8173.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 8135.50 | 8189.18 | 8173.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 8084.50 | 8168.24 | 8165.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 8084.50 | 8168.24 | 8165.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 8062.00 | 8146.99 | 8156.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 8035.00 | 8124.60 | 8145.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 8053.50 | 8044.45 | 8087.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 8053.50 | 8044.45 | 8087.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 8082.50 | 8053.99 | 8084.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:00:00 | 8082.50 | 8053.99 | 8084.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 8059.50 | 8055.09 | 8081.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:15:00 | 8062.00 | 8055.09 | 8081.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 8062.00 | 8056.47 | 8080.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 8068.00 | 8056.47 | 8080.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 8065.50 | 8058.28 | 8078.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 7988.00 | 8045.82 | 8071.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 09:15:00 | 7588.60 | 7941.80 | 8001.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 10:15:00 | 7961.50 | 7945.74 | 7997.73 | SL hit (close>ema200) qty=0.50 sl=7945.74 alert=retest2 |

### Cycle 49 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 7896.00 | 7769.99 | 7759.79 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 7779.50 | 7802.34 | 7802.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 7732.00 | 7783.10 | 7793.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 12:15:00 | 7782.00 | 7775.99 | 7786.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 12:45:00 | 7785.50 | 7775.99 | 7786.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 7760.00 | 7772.79 | 7784.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 13:30:00 | 7786.00 | 7772.79 | 7784.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 7778.50 | 7773.93 | 7783.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 15:00:00 | 7778.50 | 7773.93 | 7783.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 7760.00 | 7771.15 | 7781.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 7745.50 | 7771.15 | 7781.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 7718.00 | 7760.52 | 7775.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 10:15:00 | 7706.00 | 7760.52 | 7775.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 7725.00 | 7685.89 | 7684.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 7725.00 | 7685.89 | 7684.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 15:15:00 | 7740.00 | 7711.89 | 7698.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 7685.50 | 7706.61 | 7697.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 7685.50 | 7706.61 | 7697.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 7685.50 | 7706.61 | 7697.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:45:00 | 7638.00 | 7706.61 | 7697.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 7677.00 | 7700.69 | 7695.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:30:00 | 7673.00 | 7700.69 | 7695.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 7697.50 | 7700.05 | 7695.89 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 7670.00 | 7691.63 | 7692.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 11:15:00 | 7641.50 | 7668.02 | 7679.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 7655.00 | 7640.25 | 7660.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 7655.00 | 7640.25 | 7660.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 7655.00 | 7640.25 | 7660.06 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 11:15:00 | 7754.50 | 7676.50 | 7673.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 12:15:00 | 7776.50 | 7696.50 | 7683.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 7671.00 | 7746.88 | 7720.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 10:15:00 | 7671.00 | 7746.88 | 7720.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 7671.00 | 7746.88 | 7720.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 7671.00 | 7746.88 | 7720.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 7662.00 | 7729.90 | 7714.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:30:00 | 7675.00 | 7729.90 | 7714.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 7603.50 | 7690.24 | 7698.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 7575.00 | 7651.95 | 7678.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 12:15:00 | 7608.50 | 7607.47 | 7646.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 13:00:00 | 7608.50 | 7607.47 | 7646.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 7578.50 | 7565.73 | 7595.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 7600.00 | 7565.73 | 7595.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 7600.00 | 7572.58 | 7596.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 7633.50 | 7572.58 | 7596.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 7634.50 | 7584.97 | 7599.55 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2026-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 14:15:00 | 7699.00 | 7623.87 | 7614.09 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 12:15:00 | 7584.00 | 7612.68 | 7613.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 13:15:00 | 7533.50 | 7596.84 | 7605.84 | Break + close below crossover candle low |

### Cycle 57 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 7759.00 | 7610.31 | 7607.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 10:15:00 | 7910.00 | 7670.25 | 7635.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 7937.50 | 7937.75 | 7858.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 09:45:00 | 7905.50 | 7937.75 | 7858.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 13:15:00 | 7811.50 | 7902.85 | 7866.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:00:00 | 7811.50 | 7902.85 | 7866.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 7818.00 | 7885.88 | 7861.83 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 09:15:00 | 7652.00 | 7824.88 | 7837.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 10:15:00 | 7600.00 | 7779.91 | 7815.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 7985.00 | 7761.90 | 7781.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 7985.00 | 7761.90 | 7781.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 7985.00 | 7761.90 | 7781.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 7996.00 | 7761.90 | 7781.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 7891.50 | 7810.72 | 7801.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 10:15:00 | 7997.00 | 7898.85 | 7855.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 15:15:00 | 7940.00 | 7947.68 | 7901.62 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 09:15:00 | 8029.50 | 7947.68 | 7901.62 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 8004.00 | 7979.18 | 7941.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 15:15:00 | 8054.50 | 7979.18 | 7941.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 09:45:00 | 8006.00 | 7986.29 | 7951.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 13:15:00 | 7955.50 | 7982.30 | 7961.34 | SL hit (close<ema400) qty=1.00 sl=7961.34 alert=retest1 |

### Cycle 60 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 7797.00 | 7925.68 | 7941.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 11:15:00 | 7772.50 | 7895.04 | 7926.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 12:15:00 | 7750.50 | 7745.95 | 7790.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-01 13:00:00 | 7750.50 | 7745.95 | 7790.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 7759.50 | 7748.66 | 7787.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 7686.50 | 7752.20 | 7782.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 10:00:00 | 7701.50 | 7742.06 | 7774.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 14:30:00 | 7702.00 | 7710.39 | 7742.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 7869.00 | 7746.13 | 7753.51 | SL hit (close>static) qty=1.00 sl=7792.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 7919.50 | 7780.80 | 7768.60 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 09:15:00 | 7556.50 | 7772.70 | 7779.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 10:15:00 | 7463.00 | 7540.90 | 7629.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 7303.00 | 7283.51 | 7376.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 11:00:00 | 7303.00 | 7283.51 | 7376.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 7315.50 | 7284.90 | 7336.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:30:00 | 7363.50 | 7284.90 | 7336.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 7330.00 | 7293.92 | 7335.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:15:00 | 7348.00 | 7293.92 | 7335.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 7346.00 | 7304.34 | 7336.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:30:00 | 7354.50 | 7304.34 | 7336.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 7342.50 | 7311.97 | 7337.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 12:45:00 | 7354.50 | 7311.97 | 7337.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 7341.00 | 7324.02 | 7338.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:00:00 | 7341.00 | 7324.02 | 7338.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 7273.00 | 7314.77 | 7332.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 12:15:00 | 7257.50 | 7307.37 | 7325.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 10:15:00 | 6894.62 | 7138.25 | 7230.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-13 09:15:00 | 6531.75 | 6808.30 | 7007.62 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 63 — BUY (started 2026-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 15:15:00 | 6845.00 | 6808.29 | 6806.02 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 09:15:00 | 6739.50 | 6794.53 | 6799.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 14:15:00 | 6709.00 | 6744.04 | 6769.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-19 09:15:00 | 6772.00 | 6742.59 | 6764.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 6772.00 | 6742.59 | 6764.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 6772.00 | 6742.59 | 6764.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:00:00 | 6695.00 | 6731.06 | 6754.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:30:00 | 6695.00 | 6723.84 | 6749.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 6805.50 | 6617.17 | 6597.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 6805.50 | 6617.17 | 6597.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 12:15:00 | 6832.00 | 6719.06 | 6654.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 15:15:00 | 6897.00 | 6901.34 | 6814.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-02 09:15:00 | 6863.50 | 6901.34 | 6814.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 6949.00 | 6910.87 | 6827.15 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 10:15:00 | 6755.50 | 6820.49 | 6827.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 11:15:00 | 6734.00 | 6785.34 | 6804.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 14:15:00 | 6725.00 | 6692.96 | 6731.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-09 15:00:00 | 6725.00 | 6692.96 | 6731.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 6722.50 | 6698.87 | 6730.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:15:00 | 6738.00 | 6698.87 | 6730.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 6768.00 | 6712.70 | 6733.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:00:00 | 6768.00 | 6712.70 | 6733.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 6779.00 | 6725.96 | 6737.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:45:00 | 6781.00 | 6725.96 | 6737.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 6803.00 | 6756.15 | 6749.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 6814.00 | 6767.72 | 6755.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 6769.00 | 6781.43 | 6766.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 10:15:00 | 6769.00 | 6781.43 | 6766.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 6769.00 | 6781.43 | 6766.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:45:00 | 6787.50 | 6781.43 | 6766.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 6779.50 | 6781.04 | 6767.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:30:00 | 6761.50 | 6781.04 | 6767.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 6767.50 | 6778.34 | 6767.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 6767.50 | 6778.34 | 6767.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 6822.50 | 6787.17 | 6772.44 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 6675.50 | 6765.52 | 6766.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 6598.00 | 6700.60 | 6730.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 6611.00 | 6399.02 | 6444.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 6611.00 | 6399.02 | 6444.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 6611.00 | 6399.02 | 6444.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:00:00 | 6611.00 | 6399.02 | 6444.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 6637.50 | 6446.72 | 6462.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:45:00 | 6640.50 | 6446.72 | 6462.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 6610.00 | 6479.37 | 6475.77 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 6419.50 | 6485.04 | 6490.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 6395.50 | 6454.97 | 6475.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 6584.50 | 6458.32 | 6469.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 6584.50 | 6458.32 | 6469.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 6584.50 | 6458.32 | 6469.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 6584.50 | 6458.32 | 6469.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 6642.00 | 6495.06 | 6485.28 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-03-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 12:15:00 | 6414.50 | 6506.36 | 6508.55 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 11:15:00 | 6591.00 | 6507.22 | 6504.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 12:15:00 | 6626.00 | 6530.98 | 6515.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 6593.50 | 6622.16 | 6585.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 15:00:00 | 6593.50 | 6622.16 | 6585.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 6605.00 | 6618.73 | 6587.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 09:15:00 | 7014.50 | 6618.73 | 6587.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 09:15:00 | 7715.95 | 7487.61 | 7371.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 09:15:00 | 9494.50 | 9664.68 | 9681.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 10:15:00 | 9341.00 | 9491.44 | 9567.31 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 10:30:00 | 8488.50 | 2025-05-12 11:15:00 | 8545.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-05-16 09:15:00 | 8608.00 | 2025-05-20 13:15:00 | 8502.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-05-16 11:30:00 | 8594.00 | 2025-05-20 13:15:00 | 8502.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-05-23 15:15:00 | 8356.00 | 2025-05-26 09:15:00 | 8455.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-05-27 11:15:00 | 8469.00 | 2025-05-27 15:15:00 | 8405.00 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-05-30 11:00:00 | 8559.50 | 2025-05-30 14:15:00 | 8430.50 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-05-30 12:15:00 | 8537.50 | 2025-05-30 14:15:00 | 8430.50 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-06-13 10:15:00 | 9640.50 | 2025-06-19 09:15:00 | 9459.00 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-06-16 10:15:00 | 9625.00 | 2025-06-19 09:15:00 | 9459.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-06-16 11:00:00 | 9610.50 | 2025-06-19 09:15:00 | 9459.00 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-06-16 12:00:00 | 9627.00 | 2025-06-19 09:15:00 | 9459.00 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-06-25 12:00:00 | 9190.00 | 2025-07-03 12:15:00 | 9082.50 | STOP_HIT | 1.00 | 1.17% |
| SELL | retest2 | 2025-06-25 14:15:00 | 9189.50 | 2025-07-03 12:15:00 | 9082.50 | STOP_HIT | 1.00 | 1.16% |
| SELL | retest2 | 2025-07-08 14:15:00 | 8973.00 | 2025-07-14 09:15:00 | 8524.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-09 09:15:00 | 8975.00 | 2025-07-14 09:15:00 | 8526.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-09 10:00:00 | 8972.50 | 2025-07-14 09:15:00 | 8523.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-09 12:15:00 | 8974.50 | 2025-07-14 09:15:00 | 8525.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-08 14:15:00 | 8973.00 | 2025-07-14 10:15:00 | 8679.50 | STOP_HIT | 0.50 | 3.27% |
| SELL | retest2 | 2025-07-09 09:15:00 | 8975.00 | 2025-07-14 10:15:00 | 8679.50 | STOP_HIT | 0.50 | 3.29% |
| SELL | retest2 | 2025-07-09 10:00:00 | 8972.50 | 2025-07-14 10:15:00 | 8679.50 | STOP_HIT | 0.50 | 3.27% |
| SELL | retest2 | 2025-07-09 12:15:00 | 8974.50 | 2025-07-14 10:15:00 | 8679.50 | STOP_HIT | 0.50 | 3.29% |
| SELL | retest2 | 2025-07-15 09:45:00 | 8700.50 | 2025-07-15 10:15:00 | 8836.50 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-07-30 10:15:00 | 8592.00 | 2025-08-04 15:15:00 | 8570.00 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2025-07-30 14:00:00 | 8605.00 | 2025-08-04 15:15:00 | 8570.00 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2025-07-30 14:30:00 | 8601.50 | 2025-08-04 15:15:00 | 8570.00 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2025-08-19 11:45:00 | 8576.00 | 2025-08-22 13:15:00 | 8666.50 | STOP_HIT | 1.00 | 1.06% |
| SELL | retest2 | 2025-09-01 12:30:00 | 8420.00 | 2025-09-01 15:15:00 | 8500.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-09-02 09:15:00 | 8407.50 | 2025-09-02 09:15:00 | 8490.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-09-12 13:30:00 | 8973.00 | 2025-09-15 09:15:00 | 8922.50 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-09-12 14:15:00 | 8980.00 | 2025-09-15 09:15:00 | 8922.50 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-09-15 09:45:00 | 8970.00 | 2025-09-19 12:15:00 | 9056.00 | STOP_HIT | 1.00 | 0.96% |
| BUY | retest2 | 2025-09-15 14:45:00 | 8966.00 | 2025-09-19 12:15:00 | 9056.00 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest2 | 2025-09-18 14:15:00 | 9090.00 | 2025-09-19 13:15:00 | 9055.00 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-09-19 11:15:00 | 9090.00 | 2025-09-19 13:15:00 | 9055.00 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-09-22 14:30:00 | 9032.50 | 2025-09-23 09:15:00 | 9102.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-09-23 11:00:00 | 9053.50 | 2025-09-23 12:15:00 | 9119.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-09-23 11:30:00 | 9045.00 | 2025-09-23 12:15:00 | 9119.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-10-13 09:15:00 | 9255.50 | 2025-10-13 10:15:00 | 9151.50 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest1 | 2025-10-15 11:45:00 | 8854.50 | 2025-10-27 09:15:00 | 8679.50 | STOP_HIT | 1.00 | 1.98% |
| SELL | retest2 | 2025-10-23 13:30:00 | 8575.00 | 2025-10-27 12:15:00 | 8671.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-10-24 12:45:00 | 8583.00 | 2025-10-27 12:15:00 | 8671.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-11-10 12:30:00 | 8205.50 | 2025-11-11 15:15:00 | 8248.00 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-11-10 13:30:00 | 8200.00 | 2025-11-11 15:15:00 | 8248.00 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-11-10 14:30:00 | 8205.00 | 2025-11-11 15:15:00 | 8248.00 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-11-11 09:30:00 | 8198.00 | 2025-11-11 15:15:00 | 8248.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-11-11 11:15:00 | 8176.00 | 2025-11-11 15:15:00 | 8248.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-11-11 14:00:00 | 8184.00 | 2025-11-11 15:15:00 | 8248.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-12-03 11:00:00 | 8078.00 | 2025-12-04 09:15:00 | 8213.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-12-03 13:45:00 | 8075.50 | 2025-12-04 09:15:00 | 8213.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-12-03 15:15:00 | 8075.00 | 2025-12-04 09:15:00 | 8213.00 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-12-10 10:45:00 | 7988.00 | 2025-12-11 09:15:00 | 7588.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-10 10:45:00 | 7988.00 | 2025-12-11 10:15:00 | 7961.50 | STOP_HIT | 0.50 | 0.33% |
| SELL | retest2 | 2025-12-29 10:15:00 | 7706.00 | 2026-01-02 10:15:00 | 7725.00 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-01-27 09:15:00 | 8029.50 | 2026-01-28 13:15:00 | 7955.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-01-27 15:15:00 | 8054.50 | 2026-01-29 09:15:00 | 7816.00 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2026-01-28 09:45:00 | 8006.00 | 2026-01-29 09:15:00 | 7816.00 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2026-01-28 15:00:00 | 8019.00 | 2026-01-29 09:15:00 | 7816.00 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2026-02-02 09:15:00 | 7686.50 | 2026-02-03 09:15:00 | 7869.00 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2026-02-02 10:00:00 | 7701.50 | 2026-02-03 09:15:00 | 7869.00 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2026-02-02 14:30:00 | 7702.00 | 2026-02-03 09:15:00 | 7869.00 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2026-02-11 12:15:00 | 7257.50 | 2026-02-12 10:15:00 | 6894.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 12:15:00 | 7257.50 | 2026-02-13 09:15:00 | 6531.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-19 12:00:00 | 6695.00 | 2026-02-26 09:15:00 | 6805.50 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2026-02-19 12:30:00 | 6695.00 | 2026-02-26 09:15:00 | 6805.50 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2026-03-27 09:15:00 | 7014.50 | 2026-04-16 09:15:00 | 7715.95 | TARGET_HIT | 1.00 | 10.00% |
