# Go Digit General Insurance Ltd. (GODIGIT)

## Backtest Summary

- **Window:** 2024-05-23 09:15:00 → 2026-05-08 15:15:00 (3392 bars)
- **Last close:** 313.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 169 |
| ALERT1 | 112 |
| ALERT2 | 105 |
| ALERT2_SKIP | 67 |
| ALERT3 | 269 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 144 |
| PARTIAL | 12 |
| TARGET_HIT | 7 |
| STOP_HIT | 138 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 157 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 117
- **Target hits / Stop hits / Partials:** 7 / 138 / 12
- **Avg / median % per leg:** -0.05% / -0.96%
- **Sum % (uncompounded):** -7.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 70 | 10 | 14.3% | 2 | 68 | 0 | -0.76% | -53.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 70 | 10 | 14.3% | 2 | 68 | 0 | -0.76% | -53.4% |
| SELL (all) | 87 | 30 | 34.5% | 5 | 70 | 12 | 0.52% | 45.5% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.00% | 8.0% |
| SELL @ 3rd Alert (retest2) | 85 | 28 | 32.9% | 5 | 69 | 11 | 0.44% | 37.5% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.00% | 8.0% |
| retest2 (combined) | 155 | 38 | 24.5% | 7 | 137 | 11 | -0.10% | -16.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 14:15:00 | 301.05 | 300.54 | 300.52 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 297.05 | 300.08 | 300.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 12:15:00 | 294.10 | 297.89 | 299.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 15:15:00 | 300.00 | 297.82 | 298.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 15:15:00 | 300.00 | 297.82 | 298.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 15:15:00 | 300.00 | 297.82 | 298.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 09:45:00 | 296.60 | 297.59 | 298.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 10:30:00 | 296.70 | 297.45 | 298.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 11:45:00 | 296.85 | 297.41 | 298.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 10:15:00 | 300.00 | 298.74 | 298.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 300.00 | 298.74 | 298.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 12:15:00 | 300.55 | 299.32 | 298.95 | Break + close above crossover candle high |

### Cycle 4 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 292.40 | 298.69 | 298.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 286.75 | 296.30 | 297.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 12:15:00 | 296.35 | 296.31 | 297.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 12:15:00 | 296.35 | 296.31 | 297.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 296.35 | 296.31 | 297.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 13:00:00 | 296.35 | 296.31 | 297.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 13:15:00 | 298.05 | 296.66 | 297.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 13:45:00 | 298.45 | 296.66 | 297.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 14:15:00 | 297.55 | 296.84 | 297.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-05 09:30:00 | 296.45 | 297.62 | 297.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-05 10:15:00 | 299.80 | 298.05 | 298.08 | SL hit (close>static) qty=1.00 sl=299.25 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 11:15:00 | 300.10 | 298.46 | 298.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 14:15:00 | 301.55 | 299.54 | 298.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 344.05 | 344.13 | 334.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 15:00:00 | 344.05 | 344.13 | 334.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 338.00 | 342.25 | 338.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 15:00:00 | 338.00 | 342.25 | 338.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 336.70 | 341.14 | 338.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 350.65 | 341.14 | 338.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 11:45:00 | 338.50 | 340.74 | 340.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-14 12:15:00 | 339.70 | 340.53 | 340.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 12:15:00 | 339.70 | 340.53 | 340.60 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 13:15:00 | 341.95 | 340.81 | 340.73 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 14:15:00 | 339.45 | 340.54 | 340.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 15:15:00 | 338.40 | 340.11 | 340.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 324.70 | 322.21 | 326.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 324.70 | 322.21 | 326.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 324.70 | 322.21 | 326.29 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 14:15:00 | 334.20 | 327.74 | 327.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 10:15:00 | 338.10 | 332.49 | 331.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 14:15:00 | 335.40 | 335.41 | 333.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-26 15:00:00 | 335.40 | 335.41 | 333.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 15:15:00 | 335.50 | 335.43 | 333.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:15:00 | 333.55 | 335.43 | 333.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 337.75 | 335.89 | 333.83 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2024-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 09:15:00 | 331.50 | 333.35 | 333.49 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 10:15:00 | 337.50 | 334.18 | 333.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 350.30 | 339.08 | 336.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 343.05 | 347.80 | 343.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 09:15:00 | 343.05 | 347.80 | 343.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 343.05 | 347.80 | 343.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:00:00 | 343.05 | 347.80 | 343.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 343.40 | 346.92 | 343.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 345.05 | 343.07 | 342.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 12:45:00 | 344.75 | 343.45 | 342.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 11:30:00 | 344.10 | 343.42 | 343.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 12:15:00 | 342.00 | 343.14 | 343.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 12:15:00 | 342.00 | 343.14 | 343.14 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 09:15:00 | 350.60 | 344.60 | 343.79 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 13:15:00 | 343.95 | 345.80 | 345.83 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 15:15:00 | 348.00 | 346.20 | 346.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 09:15:00 | 355.45 | 348.05 | 346.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 14:15:00 | 363.00 | 364.74 | 359.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-11 15:00:00 | 363.00 | 364.74 | 359.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 348.85 | 361.51 | 358.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:45:00 | 351.00 | 361.51 | 358.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 347.00 | 358.61 | 357.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 11:00:00 | 347.00 | 358.61 | 357.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 11:15:00 | 347.00 | 356.29 | 356.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 09:15:00 | 339.95 | 346.62 | 351.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 09:15:00 | 340.70 | 339.63 | 342.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 09:15:00 | 340.70 | 339.63 | 342.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 340.70 | 339.63 | 342.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 10:00:00 | 340.70 | 339.63 | 342.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 339.75 | 339.65 | 342.15 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-19 10:15:00 | 347.80 | 343.03 | 342.64 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 09:15:00 | 340.15 | 344.03 | 344.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 12:15:00 | 335.50 | 341.23 | 342.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 348.20 | 341.57 | 342.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 348.20 | 341.57 | 342.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 348.20 | 341.57 | 342.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 348.20 | 341.57 | 342.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 347.70 | 342.79 | 342.74 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 09:15:00 | 338.75 | 343.23 | 343.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 11:15:00 | 337.20 | 341.28 | 342.30 | Break + close below crossover candle low |

### Cycle 21 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 358.00 | 342.15 | 341.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 09:15:00 | 365.45 | 359.67 | 355.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 12:15:00 | 357.30 | 359.61 | 356.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 13:00:00 | 357.30 | 359.61 | 356.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 356.30 | 358.95 | 356.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 13:45:00 | 357.45 | 358.95 | 356.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 352.05 | 357.57 | 355.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 352.05 | 357.57 | 355.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 350.10 | 356.07 | 355.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 358.45 | 356.07 | 355.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 10:15:00 | 351.75 | 354.61 | 354.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 10:15:00 | 351.75 | 354.61 | 354.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 14:15:00 | 345.15 | 350.91 | 352.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 326.65 | 324.40 | 329.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 10:15:00 | 326.65 | 324.40 | 329.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 326.65 | 324.40 | 329.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:45:00 | 327.35 | 324.40 | 329.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 328.05 | 325.43 | 328.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 15:15:00 | 325.00 | 326.15 | 328.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:30:00 | 324.75 | 325.55 | 327.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 12:15:00 | 329.70 | 326.77 | 327.89 | SL hit (close>static) qty=1.00 sl=328.95 alert=retest2 |

### Cycle 23 — BUY (started 2024-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 10:15:00 | 330.00 | 328.46 | 328.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 09:15:00 | 332.05 | 329.97 | 329.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 12:15:00 | 349.50 | 350.26 | 345.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-14 12:30:00 | 348.90 | 350.26 | 345.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 349.45 | 350.62 | 348.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:00:00 | 349.45 | 350.62 | 348.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 349.75 | 350.44 | 348.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 15:00:00 | 349.75 | 350.44 | 348.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 348.90 | 350.02 | 348.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:30:00 | 348.85 | 350.02 | 348.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 344.95 | 349.00 | 348.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 11:00:00 | 344.95 | 349.00 | 348.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 11:15:00 | 342.30 | 347.66 | 347.85 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 10:15:00 | 353.35 | 348.16 | 347.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 09:15:00 | 369.65 | 357.21 | 352.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 12:15:00 | 359.05 | 359.60 | 355.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 13:00:00 | 359.05 | 359.60 | 355.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 13:15:00 | 357.15 | 359.11 | 355.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 14:00:00 | 357.15 | 359.11 | 355.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 359.75 | 360.37 | 358.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 09:15:00 | 370.40 | 360.06 | 358.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 11:30:00 | 362.05 | 363.83 | 362.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-02 13:15:00 | 398.26 | 383.88 | 378.42 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 09:15:00 | 382.40 | 385.45 | 385.51 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 12:15:00 | 393.25 | 386.86 | 386.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-06 10:15:00 | 395.85 | 391.34 | 388.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 14:15:00 | 389.70 | 392.40 | 390.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 14:15:00 | 389.70 | 392.40 | 390.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 389.70 | 392.40 | 390.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 15:00:00 | 389.70 | 392.40 | 390.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 387.15 | 391.35 | 390.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:15:00 | 383.25 | 391.35 | 390.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 11:15:00 | 385.90 | 389.04 | 389.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-10 09:15:00 | 384.25 | 386.92 | 387.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 10:15:00 | 388.40 | 387.22 | 388.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 10:15:00 | 388.40 | 387.22 | 388.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 388.40 | 387.22 | 388.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 11:00:00 | 388.40 | 387.22 | 388.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 11:15:00 | 382.95 | 386.36 | 387.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 13:00:00 | 379.25 | 384.94 | 386.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 13:15:00 | 360.29 | 366.90 | 369.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-09-19 11:15:00 | 341.32 | 354.18 | 361.67 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 29 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 369.45 | 359.20 | 358.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 11:15:00 | 373.25 | 362.01 | 359.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 09:15:00 | 368.30 | 370.71 | 365.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 10:00:00 | 368.30 | 370.71 | 365.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 365.05 | 368.89 | 365.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 12:00:00 | 365.05 | 368.89 | 365.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 370.55 | 369.22 | 366.09 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2024-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 14:15:00 | 362.50 | 365.58 | 365.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 15:15:00 | 360.05 | 364.47 | 365.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 11:15:00 | 356.80 | 356.14 | 358.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-27 11:45:00 | 356.90 | 356.14 | 358.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 31 — BUY (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 14:15:00 | 378.60 | 361.16 | 360.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-03 11:15:00 | 394.00 | 382.44 | 376.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 10:15:00 | 385.85 | 389.49 | 383.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-04 11:00:00 | 385.85 | 389.49 | 383.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 383.35 | 387.69 | 384.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 13:45:00 | 384.30 | 387.69 | 384.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 377.60 | 385.67 | 383.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:30:00 | 378.40 | 385.67 | 383.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 15:15:00 | 378.20 | 384.17 | 383.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:15:00 | 380.45 | 384.17 | 383.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 378.70 | 381.97 | 382.14 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 382.00 | 380.31 | 380.17 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 10:15:00 | 378.65 | 380.27 | 380.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 11:15:00 | 377.50 | 379.71 | 380.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 09:15:00 | 374.75 | 372.44 | 374.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 09:15:00 | 374.75 | 372.44 | 374.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 374.75 | 372.44 | 374.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:45:00 | 374.85 | 372.44 | 374.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 373.65 | 372.68 | 374.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:30:00 | 374.90 | 372.68 | 374.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 371.35 | 372.58 | 373.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 14:30:00 | 368.95 | 370.58 | 372.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 09:30:00 | 368.10 | 369.71 | 371.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 14:00:00 | 368.75 | 368.79 | 370.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 10:00:00 | 369.25 | 368.46 | 369.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 363.15 | 367.39 | 369.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 13:15:00 | 362.65 | 366.06 | 368.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 14:15:00 | 362.20 | 365.53 | 367.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 350.50 | 361.41 | 365.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 349.69 | 361.41 | 365.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 350.31 | 361.41 | 365.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 350.79 | 361.41 | 365.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 09:30:00 | 356.40 | 361.41 | 365.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 10:15:00 | 361.50 | 361.43 | 364.94 | SL hit (close>ema200) qty=0.50 sl=361.43 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 338.90 | 332.78 | 332.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 11:15:00 | 339.85 | 334.80 | 333.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 09:15:00 | 337.70 | 339.08 | 336.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 09:15:00 | 337.70 | 339.08 | 336.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 337.70 | 339.08 | 336.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:30:00 | 335.60 | 339.08 | 336.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 335.85 | 338.44 | 336.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:00:00 | 335.85 | 338.44 | 336.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 333.75 | 337.50 | 336.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:00:00 | 333.75 | 337.50 | 336.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 333.05 | 336.61 | 335.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:45:00 | 331.25 | 336.61 | 335.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2024-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 13:15:00 | 330.70 | 335.43 | 335.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 14:15:00 | 330.50 | 334.44 | 335.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 338.50 | 334.54 | 334.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 338.50 | 334.54 | 334.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 338.50 | 334.54 | 334.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 338.50 | 334.54 | 334.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2024-11-01 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 18:15:00 | 339.65 | 335.56 | 335.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-04 10:15:00 | 340.15 | 336.78 | 335.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 09:15:00 | 337.30 | 338.90 | 337.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 09:15:00 | 337.30 | 338.90 | 337.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 337.30 | 338.90 | 337.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:00:00 | 337.30 | 338.90 | 337.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 338.15 | 338.75 | 337.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:30:00 | 336.75 | 338.75 | 337.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 335.90 | 338.18 | 337.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 11:45:00 | 334.40 | 338.18 | 337.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 12:15:00 | 337.00 | 337.94 | 337.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 13:15:00 | 337.90 | 337.94 | 337.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 09:45:00 | 337.75 | 338.22 | 337.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 13:15:00 | 338.15 | 338.08 | 337.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 15:15:00 | 334.95 | 339.15 | 339.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 15:15:00 | 334.95 | 339.15 | 339.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 15:15:00 | 330.10 | 334.99 | 336.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 328.05 | 325.93 | 329.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 328.05 | 325.93 | 329.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 14:15:00 | 329.40 | 326.04 | 328.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:15:00 | 325.00 | 327.43 | 328.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-21 14:15:00 | 325.60 | 320.64 | 320.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 14:15:00 | 325.60 | 320.64 | 320.62 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 10:15:00 | 318.40 | 320.29 | 320.50 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 333.35 | 322.74 | 321.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 11:15:00 | 340.95 | 326.38 | 323.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 12:15:00 | 325.45 | 326.19 | 323.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 13:00:00 | 325.45 | 326.19 | 323.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 335.25 | 339.33 | 335.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:45:00 | 335.00 | 339.33 | 335.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 12:15:00 | 335.00 | 338.46 | 335.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 12:30:00 | 335.10 | 338.46 | 335.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 332.65 | 337.30 | 335.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 13:30:00 | 332.90 | 337.30 | 335.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 334.65 | 336.77 | 335.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 15:15:00 | 333.05 | 336.77 | 335.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 15:15:00 | 333.05 | 336.03 | 335.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 09:15:00 | 340.35 | 336.03 | 335.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 14:15:00 | 335.70 | 336.79 | 336.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:15:00 | 339.45 | 335.97 | 335.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 15:15:00 | 339.95 | 343.39 | 343.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 15:15:00 | 339.95 | 343.39 | 343.86 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-12-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 13:15:00 | 344.95 | 343.98 | 343.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 14:15:00 | 347.25 | 344.64 | 344.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 12:15:00 | 348.00 | 348.13 | 346.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 12:15:00 | 348.00 | 348.13 | 346.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 12:15:00 | 348.00 | 348.13 | 346.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 12:45:00 | 348.20 | 348.13 | 346.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 12:15:00 | 342.20 | 347.95 | 347.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 13:00:00 | 342.20 | 347.95 | 347.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 13:15:00 | 341.35 | 346.63 | 346.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 15:15:00 | 339.95 | 344.32 | 345.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 13:15:00 | 343.15 | 341.97 | 343.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-11 14:00:00 | 343.15 | 341.97 | 343.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 346.75 | 342.92 | 344.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 15:00:00 | 346.75 | 342.92 | 344.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 15:15:00 | 344.75 | 343.29 | 344.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 09:15:00 | 341.90 | 343.29 | 344.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 12:15:00 | 324.80 | 330.13 | 335.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-16 10:15:00 | 327.25 | 326.99 | 331.54 | SL hit (close>ema200) qty=0.50 sl=326.99 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 09:15:00 | 350.30 | 335.46 | 333.97 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 10:15:00 | 334.30 | 338.66 | 339.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 13:15:00 | 333.25 | 336.23 | 337.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 09:15:00 | 340.60 | 335.69 | 337.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 09:15:00 | 340.60 | 335.69 | 337.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 340.60 | 335.69 | 337.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 10:00:00 | 340.60 | 335.69 | 337.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 344.90 | 337.53 | 337.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 10:30:00 | 349.45 | 337.53 | 337.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2024-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 11:15:00 | 342.80 | 338.59 | 338.22 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 09:15:00 | 329.45 | 337.86 | 338.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 15:15:00 | 327.25 | 331.08 | 334.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 09:15:00 | 325.00 | 321.78 | 326.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 09:15:00 | 325.00 | 321.78 | 326.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 325.00 | 321.78 | 326.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 13:30:00 | 318.25 | 320.31 | 324.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 15:15:00 | 329.00 | 323.93 | 323.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2024-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 15:15:00 | 329.00 | 323.93 | 323.89 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 09:15:00 | 322.65 | 323.67 | 323.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 320.95 | 322.99 | 323.42 | Break + close below crossover candle low |

### Cycle 51 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 327.95 | 323.98 | 323.83 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 15:15:00 | 322.50 | 323.69 | 323.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 09:15:00 | 317.60 | 322.47 | 323.15 | Break + close below crossover candle low |

### Cycle 53 — BUY (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 09:15:00 | 344.15 | 324.82 | 323.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 10:15:00 | 348.65 | 329.59 | 325.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 14:15:00 | 325.05 | 330.09 | 327.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-01 14:15:00 | 325.05 | 330.09 | 327.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 14:15:00 | 325.05 | 330.09 | 327.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 15:00:00 | 325.05 | 330.09 | 327.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 15:15:00 | 328.55 | 329.78 | 327.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 09:15:00 | 336.15 | 329.78 | 327.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 09:15:00 | 323.55 | 328.26 | 328.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 323.55 | 328.26 | 328.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 318.80 | 326.37 | 327.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 15:15:00 | 297.85 | 297.61 | 303.95 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:15:00 | 291.60 | 297.61 | 303.95 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 15:15:00 | 277.02 | 282.68 | 288.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-01-14 09:15:00 | 282.85 | 282.71 | 288.21 | SL hit (close>ema200) qty=0.50 sl=282.71 alert=retest1 |

### Cycle 55 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 298.55 | 289.15 | 288.49 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 15:15:00 | 290.00 | 290.86 | 290.92 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 09:15:00 | 296.85 | 292.06 | 291.46 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 282.50 | 289.91 | 290.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 10:15:00 | 278.40 | 287.61 | 289.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 286.30 | 285.20 | 287.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 286.30 | 285.20 | 287.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 286.30 | 285.20 | 287.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 286.30 | 285.20 | 287.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 301.50 | 288.63 | 288.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 308.50 | 288.63 | 288.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 304.65 | 291.84 | 290.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 11:15:00 | 313.00 | 296.07 | 292.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 312.80 | 314.61 | 304.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 10:15:00 | 311.45 | 314.61 | 304.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 305.45 | 313.85 | 309.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 10:00:00 | 305.45 | 313.85 | 309.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 10:15:00 | 302.15 | 311.51 | 308.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 10:30:00 | 297.80 | 311.51 | 308.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 15:15:00 | 301.35 | 306.49 | 307.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 284.95 | 302.18 | 305.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 297.50 | 296.89 | 301.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:45:00 | 295.15 | 296.89 | 301.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 296.60 | 296.83 | 301.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 12:45:00 | 291.70 | 297.47 | 298.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 15:00:00 | 293.50 | 295.53 | 297.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 10:15:00 | 300.90 | 298.01 | 297.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 10:15:00 | 300.90 | 298.01 | 297.72 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 295.25 | 297.51 | 297.55 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 13:15:00 | 298.60 | 297.73 | 297.64 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 14:15:00 | 296.25 | 297.43 | 297.52 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 10:15:00 | 298.85 | 297.71 | 297.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 14:15:00 | 300.10 | 298.60 | 298.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 303.85 | 303.91 | 301.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 303.85 | 303.91 | 301.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 303.85 | 303.91 | 301.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:30:00 | 304.40 | 303.91 | 301.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 301.65 | 303.46 | 301.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:00:00 | 301.65 | 303.46 | 301.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 302.25 | 303.22 | 301.94 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 299.85 | 301.51 | 301.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 297.55 | 300.64 | 301.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 10:15:00 | 299.25 | 298.21 | 299.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-11 10:15:00 | 299.25 | 298.21 | 299.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 10:15:00 | 299.25 | 298.21 | 299.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 10:30:00 | 299.60 | 298.21 | 299.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 11:15:00 | 296.50 | 297.87 | 298.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 14:30:00 | 294.60 | 296.83 | 298.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 15:00:00 | 295.35 | 296.83 | 298.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 09:15:00 | 290.55 | 296.67 | 297.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 12:15:00 | 294.25 | 295.51 | 297.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 297.50 | 295.91 | 297.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:45:00 | 297.30 | 295.91 | 297.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 295.15 | 295.75 | 296.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:30:00 | 295.75 | 295.75 | 296.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 299.55 | 296.51 | 297.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-12 14:15:00 | 299.55 | 296.51 | 297.13 | SL hit (close>static) qty=1.00 sl=299.25 alert=retest2 |

### Cycle 67 — BUY (started 2025-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 09:15:00 | 301.35 | 298.04 | 297.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-14 09:15:00 | 311.55 | 301.76 | 299.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 11:15:00 | 302.10 | 302.67 | 300.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-14 11:15:00 | 302.10 | 302.67 | 300.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 11:15:00 | 302.10 | 302.67 | 300.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 11:30:00 | 300.10 | 302.67 | 300.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 12:15:00 | 300.00 | 302.13 | 300.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 13:00:00 | 300.00 | 302.13 | 300.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 13:15:00 | 301.15 | 301.94 | 300.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 13:30:00 | 301.85 | 301.94 | 300.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 14:15:00 | 301.85 | 301.92 | 300.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 15:00:00 | 301.85 | 301.92 | 300.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 15:15:00 | 289.90 | 299.52 | 299.81 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 14:15:00 | 304.85 | 299.62 | 299.38 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 10:15:00 | 295.50 | 298.82 | 299.11 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 12:15:00 | 299.85 | 299.32 | 299.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 13:15:00 | 301.85 | 299.83 | 299.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-19 15:15:00 | 306.55 | 306.60 | 303.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 09:15:00 | 308.40 | 306.60 | 303.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 303.45 | 305.97 | 303.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 10:00:00 | 303.45 | 305.97 | 303.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 10:15:00 | 304.20 | 305.62 | 303.94 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2025-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 14:15:00 | 300.00 | 302.92 | 303.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 09:15:00 | 299.10 | 301.69 | 302.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-21 15:15:00 | 302.20 | 300.86 | 301.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 15:15:00 | 302.20 | 300.86 | 301.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 302.20 | 300.86 | 301.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 299.00 | 300.86 | 301.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 299.45 | 300.58 | 301.38 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2025-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 12:15:00 | 303.30 | 300.72 | 300.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-27 09:15:00 | 306.35 | 302.78 | 301.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-27 12:15:00 | 300.95 | 302.85 | 302.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 12:15:00 | 300.95 | 302.85 | 302.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 12:15:00 | 300.95 | 302.85 | 302.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 13:00:00 | 300.95 | 302.85 | 302.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 13:15:00 | 299.70 | 302.22 | 301.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 14:00:00 | 299.70 | 302.22 | 301.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2025-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 15:15:00 | 297.00 | 301.03 | 301.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 295.55 | 299.93 | 300.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 14:15:00 | 309.40 | 299.59 | 300.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 14:15:00 | 309.40 | 299.59 | 300.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 309.40 | 299.59 | 300.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 309.40 | 299.59 | 300.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 295.40 | 298.75 | 299.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 11:15:00 | 290.30 | 296.02 | 298.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-04 10:15:00 | 304.15 | 298.07 | 297.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 10:15:00 | 304.15 | 298.07 | 297.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 14:15:00 | 312.90 | 303.24 | 300.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-04 15:15:00 | 300.60 | 302.71 | 300.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 15:15:00 | 300.60 | 302.71 | 300.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 15:15:00 | 300.60 | 302.71 | 300.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:15:00 | 302.30 | 302.71 | 300.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 302.60 | 302.69 | 300.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 11:00:00 | 306.90 | 302.36 | 301.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 15:00:00 | 306.15 | 304.13 | 302.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 09:15:00 | 306.95 | 304.22 | 302.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 10:15:00 | 305.65 | 304.33 | 303.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 305.10 | 309.00 | 306.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 305.10 | 309.00 | 306.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 305.30 | 308.26 | 306.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 11:15:00 | 306.40 | 308.26 | 306.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 14:45:00 | 307.90 | 307.72 | 306.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 09:30:00 | 306.30 | 307.94 | 306.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 11:15:00 | 302.00 | 306.25 | 306.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 11:15:00 | 302.00 | 306.25 | 306.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 15:15:00 | 300.10 | 303.65 | 304.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 299.75 | 297.64 | 300.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-17 10:00:00 | 299.75 | 297.64 | 300.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 305.80 | 299.28 | 300.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:00:00 | 305.80 | 299.28 | 300.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 306.45 | 300.71 | 301.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:30:00 | 306.00 | 300.71 | 301.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 300.85 | 300.31 | 300.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 15:00:00 | 300.85 | 300.31 | 300.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 299.80 | 300.21 | 300.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:15:00 | 298.65 | 300.21 | 300.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 299.50 | 300.07 | 300.70 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2025-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 12:15:00 | 303.10 | 300.98 | 300.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 305.05 | 302.29 | 301.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 299.70 | 303.61 | 303.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 09:15:00 | 299.70 | 303.61 | 303.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 299.70 | 303.61 | 303.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 09:45:00 | 300.10 | 303.61 | 303.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 299.85 | 302.86 | 302.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 11:15:00 | 303.15 | 302.86 | 302.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-20 11:15:00 | 300.85 | 302.46 | 302.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-20 11:15:00 | 300.85 | 302.46 | 302.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-21 14:15:00 | 297.45 | 300.37 | 301.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 09:15:00 | 300.60 | 300.05 | 300.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-24 09:15:00 | 300.60 | 300.05 | 300.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 300.60 | 300.05 | 300.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-24 15:00:00 | 297.65 | 299.36 | 300.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-01 09:15:00 | 282.77 | 289.53 | 290.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-01 15:15:00 | 288.25 | 287.82 | 289.20 | SL hit (close>ema200) qty=0.50 sl=287.82 alert=retest2 |

### Cycle 79 — BUY (started 2025-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 11:15:00 | 283.90 | 280.15 | 279.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 290.55 | 282.77 | 281.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-21 09:15:00 | 295.25 | 296.69 | 293.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-21 09:15:00 | 295.25 | 296.69 | 293.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 295.25 | 296.69 | 293.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 10:15:00 | 298.30 | 296.69 | 293.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 10:15:00 | 298.25 | 299.29 | 298.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 13:00:00 | 298.45 | 298.55 | 298.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 296.65 | 298.80 | 298.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 296.65 | 298.80 | 298.90 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 10:15:00 | 300.35 | 299.12 | 298.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 11:15:00 | 302.15 | 299.73 | 299.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 10:15:00 | 300.50 | 303.63 | 301.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 10:15:00 | 300.50 | 303.63 | 301.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 300.50 | 303.63 | 301.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 11:00:00 | 300.50 | 303.63 | 301.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 299.40 | 302.78 | 301.62 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2025-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 13:15:00 | 295.90 | 300.49 | 300.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 14:15:00 | 292.55 | 298.90 | 299.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 10:15:00 | 288.90 | 287.11 | 290.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-02 10:45:00 | 289.00 | 287.11 | 290.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 15:15:00 | 286.30 | 286.20 | 288.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:15:00 | 283.30 | 286.20 | 288.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 288.30 | 286.62 | 288.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 09:30:00 | 282.00 | 285.89 | 287.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 15:15:00 | 293.00 | 287.83 | 287.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 15:15:00 | 293.00 | 287.83 | 287.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 294.60 | 289.18 | 288.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 14:15:00 | 290.45 | 292.36 | 290.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 14:15:00 | 290.45 | 292.36 | 290.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 290.45 | 292.36 | 290.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 290.45 | 292.36 | 290.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 289.10 | 291.71 | 290.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 284.25 | 291.71 | 290.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 286.40 | 290.64 | 290.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 10:45:00 | 290.10 | 290.65 | 290.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 13:30:00 | 289.60 | 290.15 | 290.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 14:30:00 | 289.40 | 290.22 | 290.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-15 15:15:00 | 295.40 | 296.19 | 296.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-05-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 15:15:00 | 295.40 | 296.19 | 296.25 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 09:15:00 | 297.80 | 296.51 | 296.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 14:15:00 | 299.60 | 298.16 | 297.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 09:15:00 | 297.45 | 298.14 | 297.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 09:15:00 | 297.45 | 298.14 | 297.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 297.45 | 298.14 | 297.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 295.65 | 298.14 | 297.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 298.65 | 298.25 | 297.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 11:15:00 | 301.00 | 298.25 | 297.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-21 14:15:00 | 331.10 | 317.10 | 310.13 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 12:15:00 | 334.30 | 335.55 | 335.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 14:15:00 | 331.45 | 334.71 | 335.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 10:15:00 | 334.80 | 334.01 | 334.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 10:15:00 | 334.80 | 334.01 | 334.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 334.80 | 334.01 | 334.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:45:00 | 335.20 | 334.01 | 334.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 337.30 | 334.67 | 334.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:45:00 | 337.50 | 334.67 | 334.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 335.75 | 334.88 | 335.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 13:45:00 | 334.50 | 334.85 | 335.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 09:15:00 | 337.00 | 335.32 | 335.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 337.00 | 335.32 | 335.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 12:15:00 | 337.90 | 336.32 | 335.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 15:15:00 | 336.10 | 337.15 | 336.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 15:15:00 | 336.10 | 337.15 | 336.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 336.10 | 337.15 | 336.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:15:00 | 334.20 | 337.15 | 336.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 334.05 | 336.53 | 336.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:00:00 | 334.05 | 336.53 | 336.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 10:15:00 | 332.60 | 335.75 | 335.80 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 13:15:00 | 343.45 | 336.62 | 336.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 14:15:00 | 352.40 | 339.78 | 337.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 347.85 | 348.15 | 344.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 11:00:00 | 347.85 | 348.15 | 344.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 344.10 | 346.91 | 345.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 14:00:00 | 344.10 | 346.91 | 345.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 347.05 | 346.93 | 345.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:15:00 | 345.90 | 346.93 | 345.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 345.90 | 346.73 | 345.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:15:00 | 344.60 | 346.73 | 345.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 343.10 | 346.00 | 345.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:15:00 | 343.05 | 346.00 | 345.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 345.35 | 345.87 | 345.15 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 342.30 | 344.95 | 345.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 340.80 | 343.62 | 344.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 10:15:00 | 343.15 | 341.68 | 342.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 10:15:00 | 343.15 | 341.68 | 342.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 343.15 | 341.68 | 342.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:30:00 | 343.60 | 341.68 | 342.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 343.00 | 341.94 | 342.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 12:15:00 | 343.00 | 341.94 | 342.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 12:15:00 | 342.90 | 342.13 | 342.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 12:45:00 | 342.05 | 342.13 | 342.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 345.90 | 342.89 | 343.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 13:45:00 | 345.60 | 342.89 | 343.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 341.85 | 342.68 | 343.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 14:45:00 | 346.50 | 342.68 | 343.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 343.80 | 342.90 | 343.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 341.25 | 342.90 | 343.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 339.65 | 342.25 | 342.76 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 343.70 | 342.71 | 342.62 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 11:15:00 | 342.00 | 342.47 | 342.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 12:15:00 | 340.25 | 342.02 | 342.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 342.70 | 341.47 | 341.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 342.70 | 341.47 | 341.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 342.70 | 341.47 | 341.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:15:00 | 348.30 | 341.47 | 341.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 343.85 | 341.94 | 342.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:30:00 | 343.35 | 341.94 | 342.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 11:15:00 | 348.00 | 343.16 | 342.61 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 338.55 | 342.08 | 342.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 09:15:00 | 335.70 | 338.79 | 340.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 11:15:00 | 340.50 | 338.62 | 340.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 11:15:00 | 340.50 | 338.62 | 340.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 340.50 | 338.62 | 340.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:30:00 | 340.70 | 338.62 | 340.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 339.85 | 338.86 | 340.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:15:00 | 342.80 | 338.86 | 340.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 348.35 | 340.76 | 340.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:00:00 | 348.35 | 340.76 | 340.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 356.45 | 343.90 | 342.23 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 349.10 | 354.00 | 354.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 12:15:00 | 344.65 | 352.13 | 353.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 14:15:00 | 341.70 | 339.61 | 342.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 14:15:00 | 341.70 | 339.61 | 342.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 341.70 | 339.61 | 342.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 15:00:00 | 341.70 | 339.61 | 342.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 335.40 | 338.88 | 341.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 10:30:00 | 334.20 | 336.46 | 338.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 15:15:00 | 333.15 | 334.78 | 337.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 15:15:00 | 334.00 | 332.63 | 333.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 340.40 | 334.40 | 333.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 09:15:00 | 340.40 | 334.40 | 333.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 10:15:00 | 343.20 | 336.16 | 334.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 11:15:00 | 347.45 | 348.86 | 345.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 12:00:00 | 347.45 | 348.86 | 345.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 353.15 | 349.72 | 346.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 13:45:00 | 354.20 | 350.71 | 347.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 14:00:00 | 354.35 | 351.65 | 349.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 15:00:00 | 353.85 | 353.83 | 352.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 09:30:00 | 354.15 | 353.21 | 352.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 352.50 | 353.07 | 352.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 09:45:00 | 354.60 | 352.78 | 352.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:15:00 | 355.25 | 352.78 | 352.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 14:45:00 | 354.95 | 353.39 | 352.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 09:15:00 | 355.00 | 353.51 | 352.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 352.85 | 353.38 | 352.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:45:00 | 352.85 | 353.38 | 352.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 351.85 | 353.07 | 352.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:45:00 | 351.95 | 353.07 | 352.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 352.70 | 353.00 | 352.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 12:30:00 | 353.75 | 352.97 | 352.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 13:45:00 | 354.55 | 353.28 | 352.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 09:45:00 | 353.95 | 353.76 | 353.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 348.65 | 355.50 | 356.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 348.65 | 355.50 | 356.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 15:15:00 | 346.25 | 350.77 | 353.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 350.80 | 350.78 | 353.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 09:30:00 | 350.85 | 350.78 | 353.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 348.00 | 345.61 | 349.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:15:00 | 361.30 | 345.61 | 349.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 362.35 | 348.96 | 350.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:30:00 | 370.60 | 348.96 | 350.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 10:15:00 | 360.40 | 351.25 | 351.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 14:15:00 | 371.55 | 362.41 | 358.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 10:15:00 | 363.70 | 364.36 | 360.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 10:45:00 | 365.30 | 364.36 | 360.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 365.45 | 367.16 | 363.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:45:00 | 363.25 | 367.16 | 363.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 362.80 | 367.73 | 365.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 362.80 | 367.73 | 365.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 363.00 | 366.78 | 365.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:15:00 | 361.20 | 366.78 | 365.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 10:15:00 | 355.80 | 362.95 | 363.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 12:15:00 | 351.80 | 356.21 | 358.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 15:15:00 | 357.00 | 355.69 | 357.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 09:15:00 | 358.75 | 355.69 | 357.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 357.70 | 356.09 | 357.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:45:00 | 357.30 | 356.09 | 357.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 358.50 | 356.57 | 357.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:00:00 | 358.50 | 356.57 | 357.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 360.60 | 357.38 | 357.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:30:00 | 359.85 | 357.38 | 357.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 13:15:00 | 363.15 | 359.02 | 358.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 14:15:00 | 366.30 | 360.48 | 359.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 09:15:00 | 360.60 | 361.34 | 359.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-08 10:00:00 | 360.60 | 361.34 | 359.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 359.50 | 360.97 | 359.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:00:00 | 359.50 | 360.97 | 359.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 359.50 | 360.68 | 359.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:45:00 | 357.60 | 360.68 | 359.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 356.15 | 359.77 | 359.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:00:00 | 356.15 | 359.77 | 359.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2025-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 13:15:00 | 356.75 | 359.17 | 359.30 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 10:15:00 | 362.85 | 359.15 | 358.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 12:15:00 | 363.80 | 360.69 | 359.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 357.65 | 360.71 | 360.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 357.65 | 360.71 | 360.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 357.65 | 360.71 | 360.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:00:00 | 357.65 | 360.71 | 360.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 356.25 | 359.82 | 359.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:45:00 | 357.10 | 359.82 | 359.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 11:15:00 | 356.45 | 359.14 | 359.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 14:15:00 | 355.10 | 357.35 | 358.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 15:15:00 | 360.00 | 357.88 | 358.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 15:15:00 | 360.00 | 357.88 | 358.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 360.00 | 357.88 | 358.58 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 367.85 | 359.91 | 358.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 10:15:00 | 376.45 | 366.75 | 363.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 13:15:00 | 366.30 | 367.31 | 364.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 13:45:00 | 367.15 | 367.31 | 364.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 363.90 | 366.63 | 364.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 363.90 | 366.63 | 364.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 360.10 | 365.32 | 364.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:15:00 | 368.80 | 365.32 | 364.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 361.25 | 368.14 | 368.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 361.25 | 368.14 | 368.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 10:15:00 | 359.90 | 366.49 | 367.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 14:15:00 | 370.10 | 363.55 | 365.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 14:15:00 | 370.10 | 363.55 | 365.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 370.10 | 363.55 | 365.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 370.10 | 363.55 | 365.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 377.40 | 366.32 | 366.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 365.70 | 366.32 | 366.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 09:15:00 | 369.80 | 367.02 | 366.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 09:15:00 | 369.80 | 367.02 | 366.96 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 12:15:00 | 362.75 | 366.55 | 366.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 361.15 | 365.13 | 366.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 354.30 | 352.82 | 355.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-03 09:45:00 | 353.90 | 352.82 | 355.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 354.65 | 353.28 | 354.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 15:00:00 | 354.65 | 353.28 | 354.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 354.10 | 353.44 | 354.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 363.20 | 353.44 | 354.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 357.30 | 354.21 | 354.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 10:30:00 | 355.25 | 354.48 | 354.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 11:15:00 | 354.25 | 352.94 | 352.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 11:15:00 | 354.25 | 352.94 | 352.93 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 12:15:00 | 352.00 | 352.75 | 352.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 13:15:00 | 349.85 | 352.17 | 352.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 357.60 | 352.50 | 352.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 357.60 | 352.50 | 352.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 357.60 | 352.50 | 352.55 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 355.05 | 353.01 | 352.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 13:15:00 | 358.55 | 355.89 | 354.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 15:15:00 | 355.20 | 356.12 | 354.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 09:15:00 | 355.00 | 356.12 | 354.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 355.65 | 356.03 | 355.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 10:30:00 | 359.75 | 356.16 | 355.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 13:30:00 | 360.15 | 357.06 | 355.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 14:45:00 | 359.95 | 357.56 | 356.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 13:15:00 | 354.80 | 355.95 | 355.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2025-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 13:15:00 | 354.80 | 355.95 | 355.96 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 10:15:00 | 357.95 | 356.17 | 356.02 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 12:15:00 | 352.90 | 355.43 | 355.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 13:15:00 | 352.20 | 354.78 | 355.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 14:15:00 | 354.10 | 353.04 | 353.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 14:15:00 | 354.10 | 353.04 | 353.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 354.10 | 353.04 | 353.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:00:00 | 354.10 | 353.04 | 353.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 354.75 | 353.38 | 353.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 353.15 | 353.38 | 353.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 353.80 | 353.46 | 353.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:00:00 | 350.05 | 352.29 | 352.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 15:00:00 | 349.00 | 351.63 | 352.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 12:15:00 | 354.60 | 353.01 | 352.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 12:15:00 | 354.60 | 353.01 | 352.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 14:15:00 | 357.00 | 354.09 | 353.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 352.80 | 354.18 | 353.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 352.80 | 354.18 | 353.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 352.80 | 354.18 | 353.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 353.20 | 354.18 | 353.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 353.30 | 354.01 | 353.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:15:00 | 352.80 | 354.01 | 353.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 354.25 | 354.06 | 353.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 353.05 | 354.06 | 353.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 353.85 | 354.01 | 353.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:00:00 | 353.85 | 354.01 | 353.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 355.00 | 354.21 | 353.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:00:00 | 355.00 | 354.21 | 353.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 363.20 | 356.31 | 354.86 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 10:15:00 | 351.50 | 355.26 | 355.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 12:15:00 | 350.55 | 353.64 | 354.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 14:15:00 | 346.25 | 341.85 | 344.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 14:15:00 | 346.25 | 341.85 | 344.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 346.25 | 341.85 | 344.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 346.25 | 341.85 | 344.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 346.30 | 342.74 | 344.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 342.75 | 342.74 | 344.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 10:15:00 | 348.45 | 343.26 | 343.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 348.45 | 343.26 | 343.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 09:15:00 | 353.50 | 348.97 | 347.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 09:15:00 | 356.80 | 356.92 | 353.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 358.60 | 358.35 | 355.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 358.60 | 358.35 | 355.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 14:30:00 | 360.40 | 359.01 | 357.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 14:45:00 | 360.85 | 358.86 | 357.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 361.00 | 358.97 | 357.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:00:00 | 360.85 | 359.34 | 358.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 358.80 | 359.65 | 358.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:45:00 | 358.70 | 359.65 | 358.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 359.25 | 359.57 | 358.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:30:00 | 359.15 | 359.57 | 358.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 359.55 | 359.56 | 358.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 15:15:00 | 363.25 | 359.56 | 358.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 357.85 | 359.81 | 359.10 | SL hit (close<static) qty=1.00 sl=358.30 alert=retest2 |

### Cycle 118 — SELL (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 10:15:00 | 356.45 | 358.54 | 358.79 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 11:15:00 | 366.40 | 360.12 | 359.48 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 351.60 | 358.41 | 358.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 14:15:00 | 349.25 | 355.46 | 357.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 13:15:00 | 354.60 | 351.89 | 354.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 13:15:00 | 354.60 | 351.89 | 354.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 354.60 | 351.89 | 354.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:00:00 | 354.60 | 351.89 | 354.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 353.20 | 352.16 | 354.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:30:00 | 355.00 | 352.16 | 354.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 352.90 | 352.30 | 354.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 353.65 | 352.78 | 354.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 355.00 | 353.23 | 354.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 355.00 | 353.23 | 354.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 355.00 | 353.58 | 354.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 355.00 | 353.58 | 354.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 355.00 | 353.87 | 354.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 15:15:00 | 353.10 | 354.42 | 354.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 10:00:00 | 354.05 | 354.14 | 354.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 11:00:00 | 353.90 | 354.09 | 354.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 15:15:00 | 351.15 | 352.45 | 352.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 351.15 | 352.19 | 352.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 357.65 | 352.19 | 352.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 359.70 | 353.69 | 353.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 09:15:00 | 359.70 | 353.69 | 353.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 15:15:00 | 363.75 | 359.32 | 356.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 362.70 | 363.39 | 360.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 362.70 | 363.39 | 360.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 362.70 | 363.39 | 360.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 14:30:00 | 365.90 | 362.81 | 361.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:30:00 | 364.85 | 363.32 | 361.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 10:00:00 | 365.40 | 363.32 | 361.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 11:30:00 | 364.45 | 363.82 | 362.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 358.20 | 363.08 | 362.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 358.20 | 363.08 | 362.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-31 15:15:00 | 356.10 | 361.68 | 361.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 15:15:00 | 356.10 | 361.68 | 361.84 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 13:15:00 | 363.60 | 362.01 | 361.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 09:15:00 | 364.70 | 362.44 | 362.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 13:15:00 | 360.60 | 362.52 | 362.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 13:15:00 | 360.60 | 362.52 | 362.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 360.60 | 362.52 | 362.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:00:00 | 360.60 | 362.52 | 362.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 14:15:00 | 360.85 | 362.19 | 362.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 15:15:00 | 358.50 | 361.45 | 361.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 13:15:00 | 359.35 | 359.13 | 360.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 13:45:00 | 358.60 | 359.13 | 360.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 359.95 | 358.10 | 359.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:30:00 | 360.00 | 358.10 | 359.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 359.00 | 358.28 | 359.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:30:00 | 358.45 | 359.49 | 359.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 11:15:00 | 360.50 | 359.77 | 359.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 360.50 | 359.77 | 359.69 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 12:15:00 | 358.60 | 359.54 | 359.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 15:15:00 | 357.00 | 359.08 | 359.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 14:15:00 | 355.25 | 355.01 | 356.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 14:15:00 | 355.25 | 355.01 | 356.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 355.25 | 355.01 | 356.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:45:00 | 355.75 | 355.01 | 356.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 358.00 | 355.61 | 356.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:15:00 | 354.50 | 355.61 | 356.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 355.00 | 355.49 | 356.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 15:00:00 | 353.15 | 354.94 | 355.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 351.20 | 354.78 | 355.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 349.15 | 352.23 | 352.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 13:15:00 | 353.45 | 352.32 | 352.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 13:15:00 | 353.45 | 352.32 | 352.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 14:15:00 | 354.70 | 352.80 | 352.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 12:15:00 | 352.95 | 353.27 | 352.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 12:15:00 | 352.95 | 353.27 | 352.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 12:15:00 | 352.95 | 353.27 | 352.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 13:00:00 | 352.95 | 353.27 | 352.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 354.00 | 353.42 | 352.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 14:15:00 | 354.85 | 353.42 | 352.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 15:15:00 | 355.00 | 353.52 | 353.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 351.95 | 353.45 | 353.11 | SL hit (close<static) qty=1.00 sl=352.50 alert=retest2 |

### Cycle 128 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 352.35 | 352.90 | 352.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 350.50 | 352.33 | 352.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 352.45 | 352.22 | 352.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 352.45 | 352.22 | 352.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 352.45 | 352.22 | 352.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:30:00 | 352.35 | 352.22 | 352.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 353.50 | 352.47 | 352.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:00:00 | 353.50 | 352.47 | 352.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 352.00 | 352.38 | 352.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 12:30:00 | 351.00 | 352.27 | 352.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:30:00 | 350.60 | 352.13 | 352.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 15:15:00 | 353.95 | 352.49 | 352.52 | SL hit (close>static) qty=1.00 sl=353.80 alert=retest2 |

### Cycle 129 — BUY (started 2025-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 11:15:00 | 354.65 | 352.78 | 352.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 14:15:00 | 355.10 | 353.75 | 353.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 355.70 | 356.01 | 354.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 09:45:00 | 355.30 | 356.01 | 354.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 355.00 | 355.81 | 354.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 355.00 | 355.81 | 354.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 355.30 | 355.71 | 355.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:30:00 | 355.25 | 355.71 | 355.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 355.15 | 355.60 | 355.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 355.15 | 355.60 | 355.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 355.35 | 355.55 | 355.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:00:00 | 355.35 | 355.55 | 355.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 357.95 | 356.03 | 355.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:30:00 | 355.40 | 356.03 | 355.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 356.70 | 356.16 | 355.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:15:00 | 355.05 | 356.16 | 355.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 355.45 | 356.02 | 355.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 355.45 | 356.02 | 355.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 358.40 | 356.50 | 355.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 14:45:00 | 359.10 | 356.58 | 355.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 15:15:00 | 359.75 | 356.58 | 355.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 354.30 | 356.63 | 356.14 | SL hit (close<static) qty=1.00 sl=355.15 alert=retest2 |

### Cycle 130 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 11:15:00 | 353.20 | 355.58 | 355.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 12:15:00 | 352.60 | 354.98 | 355.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 13:15:00 | 355.00 | 354.99 | 355.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 13:15:00 | 355.00 | 354.99 | 355.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 355.00 | 354.99 | 355.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:00:00 | 355.00 | 354.99 | 355.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 355.70 | 355.13 | 355.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:30:00 | 354.85 | 355.13 | 355.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 355.10 | 355.12 | 355.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 353.75 | 355.12 | 355.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 349.55 | 354.01 | 354.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 10:15:00 | 349.05 | 354.01 | 354.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 11:15:00 | 348.90 | 353.17 | 354.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 12:15:00 | 348.25 | 352.38 | 353.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 13:00:00 | 348.85 | 351.68 | 353.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 347.45 | 346.12 | 347.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:00:00 | 347.45 | 346.12 | 347.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 346.75 | 346.25 | 347.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 341.70 | 346.30 | 346.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 15:15:00 | 347.90 | 345.79 | 346.05 | SL hit (close>static) qty=1.00 sl=347.60 alert=retest2 |

### Cycle 131 — BUY (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 09:15:00 | 342.95 | 341.67 | 341.50 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 11:15:00 | 342.70 | 345.19 | 345.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 15:15:00 | 341.05 | 343.50 | 344.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 11:15:00 | 343.50 | 342.54 | 343.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 11:15:00 | 343.50 | 342.54 | 343.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 343.50 | 342.54 | 343.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 11:45:00 | 341.00 | 342.51 | 343.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 14:00:00 | 340.65 | 342.06 | 342.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 15:15:00 | 340.75 | 341.89 | 342.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 10:15:00 | 340.10 | 341.54 | 342.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 342.80 | 341.47 | 342.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 342.80 | 341.47 | 342.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 343.65 | 341.91 | 342.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:30:00 | 344.10 | 341.91 | 342.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-23 15:15:00 | 343.75 | 342.59 | 342.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 15:15:00 | 343.75 | 342.59 | 342.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 14:15:00 | 348.20 | 344.42 | 343.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 14:15:00 | 345.45 | 346.23 | 345.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 15:00:00 | 345.45 | 346.23 | 345.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 346.00 | 346.19 | 345.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 344.90 | 346.19 | 345.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 343.80 | 345.71 | 345.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:30:00 | 344.05 | 345.71 | 345.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 343.05 | 345.18 | 344.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:00:00 | 343.05 | 345.18 | 344.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 343.55 | 344.85 | 344.75 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 342.95 | 344.47 | 344.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 13:15:00 | 341.50 | 343.88 | 344.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 12:15:00 | 345.00 | 343.21 | 343.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 12:15:00 | 345.00 | 343.21 | 343.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 345.00 | 343.21 | 343.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:30:00 | 345.00 | 343.21 | 343.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 345.10 | 343.59 | 343.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 345.10 | 343.59 | 343.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 15:15:00 | 346.90 | 344.28 | 344.08 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 12:15:00 | 343.15 | 343.88 | 343.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-31 15:15:00 | 341.50 | 343.46 | 343.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 09:15:00 | 343.75 | 343.52 | 343.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 343.75 | 343.52 | 343.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 343.75 | 343.52 | 343.75 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 11:15:00 | 346.45 | 344.16 | 344.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 15:15:00 | 347.75 | 345.78 | 344.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 09:15:00 | 345.50 | 345.73 | 344.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 09:15:00 | 345.50 | 345.73 | 344.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 345.50 | 345.73 | 344.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:45:00 | 344.00 | 345.73 | 344.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 345.05 | 345.71 | 345.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:30:00 | 345.20 | 345.71 | 345.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 345.25 | 345.62 | 345.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:00:00 | 345.25 | 345.62 | 345.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 346.05 | 345.70 | 345.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:45:00 | 345.00 | 345.70 | 345.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 345.50 | 345.66 | 345.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:15:00 | 345.30 | 345.66 | 345.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 347.50 | 346.03 | 345.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 11:00:00 | 348.30 | 346.48 | 345.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 342.80 | 347.28 | 346.72 | SL hit (close<static) qty=1.00 sl=343.05 alert=retest2 |

### Cycle 138 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 342.25 | 346.28 | 346.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 11:15:00 | 341.00 | 345.22 | 345.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 15:15:00 | 335.35 | 335.01 | 336.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 09:15:00 | 331.10 | 335.01 | 336.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 328.40 | 333.69 | 336.11 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2026-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 12:15:00 | 337.15 | 334.89 | 334.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 14:15:00 | 338.65 | 335.96 | 335.38 | Break + close above crossover candle high |

### Cycle 140 — SELL (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 09:15:00 | 328.35 | 334.94 | 335.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 10:15:00 | 326.70 | 333.29 | 334.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 12:15:00 | 325.50 | 325.33 | 328.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-16 12:45:00 | 326.00 | 325.33 | 328.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 326.75 | 324.85 | 326.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 14:30:00 | 326.80 | 324.85 | 326.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 326.00 | 325.08 | 326.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:30:00 | 324.55 | 324.93 | 326.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 11:15:00 | 324.00 | 324.90 | 326.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 10:00:00 | 323.35 | 323.19 | 324.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 324.55 | 322.56 | 323.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 325.40 | 323.13 | 323.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 325.40 | 323.13 | 323.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 325.10 | 323.52 | 323.67 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-22 12:15:00 | 325.00 | 323.82 | 323.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 12:15:00 | 325.00 | 323.82 | 323.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 10:15:00 | 334.80 | 326.29 | 324.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-27 09:15:00 | 327.95 | 329.53 | 327.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 327.95 | 329.53 | 327.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 327.95 | 329.53 | 327.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:15:00 | 325.30 | 329.53 | 327.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 332.80 | 330.19 | 328.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 10:00:00 | 336.00 | 332.58 | 330.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 09:15:00 | 325.40 | 330.12 | 330.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 325.40 | 330.12 | 330.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 12:15:00 | 323.70 | 327.57 | 328.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 12:15:00 | 325.35 | 324.52 | 326.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 12:15:00 | 325.35 | 324.52 | 326.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 325.35 | 324.52 | 326.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 11:45:00 | 321.40 | 324.03 | 325.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 13:15:00 | 324.15 | 322.68 | 322.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2026-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 13:15:00 | 324.15 | 322.68 | 322.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 14:15:00 | 324.45 | 323.03 | 322.79 | Break + close above crossover candle high |

### Cycle 144 — SELL (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 09:15:00 | 318.60 | 322.30 | 322.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 10:15:00 | 316.70 | 321.18 | 321.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 314.50 | 312.15 | 314.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 14:15:00 | 314.50 | 312.15 | 314.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 314.50 | 312.15 | 314.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 314.50 | 312.15 | 314.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 309.80 | 311.68 | 313.87 | EMA400 retest candle locked (from downside) |

### Cycle 145 — BUY (started 2026-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 14:15:00 | 318.70 | 314.34 | 314.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 15:15:00 | 321.05 | 315.68 | 314.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 318.30 | 319.86 | 318.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 318.30 | 319.86 | 318.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 318.30 | 319.86 | 318.10 | EMA400 retest candle locked (from upside) |

### Cycle 146 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 313.60 | 318.62 | 318.87 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 11:15:00 | 323.30 | 319.10 | 319.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 15:15:00 | 325.00 | 321.95 | 320.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 13:15:00 | 330.55 | 331.04 | 328.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-17 14:00:00 | 330.55 | 331.04 | 328.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 331.55 | 331.14 | 328.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:30:00 | 329.65 | 331.14 | 328.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 327.25 | 330.04 | 328.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:00:00 | 327.25 | 330.04 | 328.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 329.65 | 329.96 | 328.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 13:00:00 | 332.00 | 329.83 | 328.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 12:15:00 | 333.60 | 338.86 | 339.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2026-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 12:15:00 | 333.60 | 338.86 | 339.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 331.35 | 336.45 | 337.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 12:15:00 | 321.85 | 321.59 | 325.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 13:00:00 | 321.85 | 321.59 | 325.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 322.00 | 321.88 | 324.94 | EMA400 retest candle locked (from downside) |

### Cycle 149 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 333.40 | 327.05 | 326.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 13:15:00 | 334.70 | 328.58 | 327.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 324.00 | 329.38 | 328.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 324.00 | 329.38 | 328.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 324.00 | 329.38 | 328.20 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 325.25 | 327.24 | 327.35 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2026-03-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 15:15:00 | 337.60 | 327.95 | 327.45 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2026-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 14:15:00 | 333.70 | 335.60 | 335.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 332.05 | 334.47 | 335.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 13:15:00 | 332.95 | 332.95 | 334.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:00:00 | 332.95 | 332.95 | 334.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 339.45 | 334.25 | 334.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 15:00:00 | 339.45 | 334.25 | 334.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2026-03-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 15:15:00 | 341.70 | 335.74 | 335.26 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 10:15:00 | 332.55 | 334.94 | 334.97 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 335.20 | 334.99 | 334.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 12:15:00 | 336.35 | 335.27 | 335.11 | Break + close above crossover candle high |

### Cycle 156 — SELL (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 14:15:00 | 331.30 | 334.60 | 334.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-18 10:15:00 | 329.70 | 332.77 | 333.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-19 09:15:00 | 332.30 | 330.61 | 332.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 332.30 | 330.61 | 332.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 332.30 | 330.61 | 332.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 331.00 | 330.61 | 332.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 328.85 | 330.26 | 331.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 15:15:00 | 328.00 | 329.99 | 331.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 13:15:00 | 324.65 | 323.34 | 323.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 324.65 | 323.34 | 323.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 15:15:00 | 327.00 | 324.35 | 323.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 321.35 | 323.75 | 323.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 321.35 | 323.75 | 323.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 321.35 | 323.75 | 323.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 11:00:00 | 329.30 | 324.86 | 324.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 10:15:00 | 326.50 | 325.71 | 325.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 11:45:00 | 325.00 | 326.15 | 325.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 13:15:00 | 325.00 | 325.86 | 325.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 325.30 | 325.69 | 325.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 325.40 | 325.69 | 325.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 323.55 | 325.26 | 325.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:30:00 | 323.50 | 325.26 | 325.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-01 11:15:00 | 323.65 | 324.94 | 325.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — SELL (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 11:15:00 | 323.65 | 324.94 | 325.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 12:15:00 | 322.40 | 324.43 | 324.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 14:15:00 | 321.30 | 319.04 | 321.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 14:15:00 | 321.30 | 319.04 | 321.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 321.30 | 319.04 | 321.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 15:00:00 | 321.30 | 319.04 | 321.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 319.90 | 319.21 | 320.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 314.75 | 319.21 | 320.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 13:30:00 | 319.20 | 317.93 | 319.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 15:00:00 | 318.35 | 318.02 | 319.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:30:00 | 319.65 | 318.41 | 319.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 318.50 | 318.43 | 319.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 10:30:00 | 319.25 | 318.43 | 319.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 319.30 | 318.60 | 319.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 11:30:00 | 318.85 | 318.60 | 319.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 319.00 | 318.68 | 319.23 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-07 14:15:00 | 325.70 | 320.32 | 319.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2026-04-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 14:15:00 | 325.70 | 320.32 | 319.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 15:15:00 | 327.50 | 321.76 | 320.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 15:15:00 | 323.50 | 324.23 | 322.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 15:15:00 | 323.50 | 324.23 | 322.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 15:15:00 | 323.50 | 324.23 | 322.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 09:15:00 | 324.65 | 324.23 | 322.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 323.50 | 324.08 | 322.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 11:15:00 | 325.85 | 324.22 | 323.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 12:15:00 | 321.45 | 323.83 | 323.06 | SL hit (close<static) qty=1.00 sl=321.60 alert=retest2 |

### Cycle 160 — SELL (started 2026-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 14:15:00 | 319.00 | 322.45 | 322.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 09:15:00 | 317.45 | 319.94 | 321.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 14:15:00 | 319.95 | 319.55 | 320.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 14:15:00 | 319.95 | 319.55 | 320.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 319.95 | 319.55 | 320.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 15:00:00 | 319.95 | 319.55 | 320.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 321.35 | 319.82 | 320.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 13:45:00 | 319.60 | 320.21 | 320.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 15:00:00 | 318.75 | 319.92 | 320.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 10:15:00 | 319.90 | 320.09 | 320.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 10:15:00 | 322.20 | 320.51 | 320.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 10:15:00 | 322.20 | 320.51 | 320.44 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2026-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 12:15:00 | 318.50 | 320.19 | 320.31 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2026-04-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 12:15:00 | 320.25 | 319.75 | 319.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 15:15:00 | 321.85 | 320.48 | 320.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 09:15:00 | 319.80 | 320.35 | 320.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 319.80 | 320.35 | 320.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 319.80 | 320.35 | 320.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 13:15:00 | 321.45 | 320.42 | 320.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 09:30:00 | 321.55 | 321.39 | 320.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 11:00:00 | 326.90 | 322.49 | 321.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 14:45:00 | 321.30 | 322.04 | 321.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 15:15:00 | 322.90 | 322.22 | 321.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:15:00 | 320.55 | 322.22 | 321.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 322.50 | 322.27 | 321.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-23 14:15:00 | 319.95 | 321.34 | 321.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 319.95 | 321.34 | 321.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 319.05 | 320.53 | 321.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 13:15:00 | 319.95 | 319.88 | 320.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 14:00:00 | 319.95 | 319.88 | 320.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 322.45 | 320.41 | 320.62 | EMA400 retest candle locked (from downside) |

### Cycle 165 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 322.25 | 320.77 | 320.77 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 14:15:00 | 320.00 | 320.64 | 320.73 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 322.55 | 320.95 | 320.84 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 11:15:00 | 317.80 | 320.32 | 320.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 13:15:00 | 316.40 | 319.22 | 320.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 12:15:00 | 318.75 | 318.38 | 319.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-29 13:00:00 | 318.75 | 318.38 | 319.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 314.50 | 317.63 | 318.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 314.50 | 317.63 | 318.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 304.35 | 303.68 | 305.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 11:00:00 | 303.00 | 303.55 | 305.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 11:15:00 | 310.85 | 305.61 | 305.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 11:15:00 | 310.85 | 305.61 | 305.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 12:15:00 | 314.25 | 307.34 | 306.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 15:15:00 | 313.05 | 313.75 | 311.43 | EMA200 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-31 09:45:00 | 296.60 | 2024-06-03 10:15:00 | 300.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-05-31 10:30:00 | 296.70 | 2024-06-03 10:15:00 | 300.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-05-31 11:45:00 | 296.85 | 2024-06-03 10:15:00 | 300.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-06-05 09:30:00 | 296.45 | 2024-06-05 10:15:00 | 299.80 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-06-12 09:15:00 | 350.65 | 2024-06-14 12:15:00 | 339.70 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2024-06-14 11:45:00 | 338.50 | 2024-06-14 12:15:00 | 339.70 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2024-07-03 09:15:00 | 345.05 | 2024-07-04 12:15:00 | 342.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-07-03 12:45:00 | 344.75 | 2024-07-04 12:15:00 | 342.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-07-04 11:30:00 | 344.10 | 2024-07-04 12:15:00 | 342.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-08-01 09:15:00 | 358.45 | 2024-08-01 10:15:00 | 351.75 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2024-08-07 15:15:00 | 325.00 | 2024-08-08 12:15:00 | 329.70 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-08-08 09:30:00 | 324.75 | 2024-08-08 12:15:00 | 329.70 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-08-23 09:15:00 | 370.40 | 2024-09-02 13:15:00 | 398.26 | TARGET_HIT | 1.00 | 7.52% |
| BUY | retest2 | 2024-08-26 11:30:00 | 362.05 | 2024-09-05 09:15:00 | 382.40 | STOP_HIT | 1.00 | 5.62% |
| SELL | retest2 | 2024-09-10 13:00:00 | 379.25 | 2024-09-18 13:15:00 | 360.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-10 13:00:00 | 379.25 | 2024-09-19 11:15:00 | 341.32 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-15 14:30:00 | 368.95 | 2024-10-18 09:15:00 | 350.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-16 09:30:00 | 368.10 | 2024-10-18 09:15:00 | 349.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-16 14:00:00 | 368.75 | 2024-10-18 09:15:00 | 350.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 10:00:00 | 369.25 | 2024-10-18 09:15:00 | 350.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 14:30:00 | 368.95 | 2024-10-18 10:15:00 | 361.50 | STOP_HIT | 0.50 | 2.02% |
| SELL | retest2 | 2024-10-16 09:30:00 | 368.10 | 2024-10-18 10:15:00 | 361.50 | STOP_HIT | 0.50 | 1.79% |
| SELL | retest2 | 2024-10-16 14:00:00 | 368.75 | 2024-10-18 10:15:00 | 361.50 | STOP_HIT | 0.50 | 1.97% |
| SELL | retest2 | 2024-10-17 10:00:00 | 369.25 | 2024-10-18 10:15:00 | 361.50 | STOP_HIT | 0.50 | 2.10% |
| SELL | retest2 | 2024-10-17 13:15:00 | 362.65 | 2024-10-22 12:15:00 | 344.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 14:15:00 | 362.20 | 2024-10-22 12:15:00 | 344.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 09:30:00 | 356.40 | 2024-10-22 12:15:00 | 344.04 | PARTIAL | 0.50 | 3.47% |
| SELL | retest2 | 2024-10-21 09:45:00 | 362.15 | 2024-10-24 14:15:00 | 338.58 | PARTIAL | 0.50 | 6.51% |
| SELL | retest2 | 2024-10-17 13:15:00 | 362.65 | 2024-10-25 09:15:00 | 326.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-17 14:15:00 | 362.20 | 2024-10-25 09:15:00 | 325.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-18 09:30:00 | 356.40 | 2024-10-25 09:15:00 | 320.76 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-21 09:45:00 | 362.15 | 2024-10-25 09:15:00 | 325.94 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-11-05 13:15:00 | 337.90 | 2024-11-08 15:15:00 | 334.95 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-11-06 09:45:00 | 337.75 | 2024-11-08 15:15:00 | 334.95 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-11-06 13:15:00 | 338.15 | 2024-11-08 15:15:00 | 334.95 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-11-18 09:15:00 | 325.00 | 2024-11-21 14:15:00 | 325.60 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2024-11-28 09:15:00 | 340.35 | 2024-12-05 15:15:00 | 339.95 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2024-11-28 14:15:00 | 335.70 | 2024-12-05 15:15:00 | 339.95 | STOP_HIT | 1.00 | 1.27% |
| BUY | retest2 | 2024-11-29 09:15:00 | 339.45 | 2024-12-05 15:15:00 | 339.95 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2024-12-12 09:15:00 | 341.90 | 2024-12-13 12:15:00 | 324.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 09:15:00 | 341.90 | 2024-12-16 10:15:00 | 327.25 | STOP_HIT | 0.50 | 4.28% |
| SELL | retest2 | 2024-12-26 13:30:00 | 318.25 | 2024-12-27 15:15:00 | 329.00 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2025-01-02 09:15:00 | 336.15 | 2025-01-06 09:15:00 | 323.55 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest1 | 2025-01-10 09:15:00 | 291.60 | 2025-01-13 15:15:00 | 277.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-01-10 09:15:00 | 291.60 | 2025-01-14 09:15:00 | 282.85 | STOP_HIT | 0.50 | 3.00% |
| SELL | retest2 | 2025-01-14 12:15:00 | 285.40 | 2025-01-15 09:15:00 | 290.75 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-01-15 15:00:00 | 285.35 | 2025-01-16 09:15:00 | 298.55 | STOP_HIT | 1.00 | -4.63% |
| SELL | retest2 | 2025-01-30 12:45:00 | 291.70 | 2025-02-01 10:15:00 | 300.90 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2025-01-30 15:00:00 | 293.50 | 2025-02-01 10:15:00 | 300.90 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-02-11 14:30:00 | 294.60 | 2025-02-12 14:15:00 | 299.55 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-02-11 15:00:00 | 295.35 | 2025-02-12 14:15:00 | 299.55 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-02-12 09:15:00 | 290.55 | 2025-02-12 14:15:00 | 299.55 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-02-12 12:15:00 | 294.25 | 2025-02-12 14:15:00 | 299.55 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-03-03 11:15:00 | 290.30 | 2025-03-04 10:15:00 | 304.15 | STOP_HIT | 1.00 | -4.77% |
| BUY | retest2 | 2025-03-06 11:00:00 | 306.90 | 2025-03-11 11:15:00 | 302.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-03-06 15:00:00 | 306.15 | 2025-03-11 11:15:00 | 302.00 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-03-07 09:15:00 | 306.95 | 2025-03-11 11:15:00 | 302.00 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-03-07 10:15:00 | 305.65 | 2025-03-11 11:15:00 | 302.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-03-10 11:15:00 | 306.40 | 2025-03-11 11:15:00 | 302.00 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-03-10 14:45:00 | 307.90 | 2025-03-11 11:15:00 | 302.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-03-11 09:30:00 | 306.30 | 2025-03-11 11:15:00 | 302.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-03-20 11:15:00 | 303.15 | 2025-03-20 11:15:00 | 300.85 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-03-24 15:00:00 | 297.65 | 2025-04-01 09:15:00 | 282.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-24 15:00:00 | 297.65 | 2025-04-01 15:15:00 | 288.25 | STOP_HIT | 0.50 | 3.16% |
| BUY | retest2 | 2025-04-21 10:15:00 | 298.30 | 2025-04-25 10:15:00 | 296.65 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-04-23 10:15:00 | 298.25 | 2025-04-25 10:15:00 | 296.65 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-04-23 13:00:00 | 298.45 | 2025-04-25 10:15:00 | 296.65 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-05-07 09:30:00 | 282.00 | 2025-05-07 15:15:00 | 293.00 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest2 | 2025-05-09 10:45:00 | 290.10 | 2025-05-15 15:15:00 | 295.40 | STOP_HIT | 1.00 | 1.83% |
| BUY | retest2 | 2025-05-09 13:30:00 | 289.60 | 2025-05-15 15:15:00 | 295.40 | STOP_HIT | 1.00 | 2.00% |
| BUY | retest2 | 2025-05-09 14:30:00 | 289.40 | 2025-05-15 15:15:00 | 295.40 | STOP_HIT | 1.00 | 2.07% |
| BUY | retest2 | 2025-05-19 11:15:00 | 301.00 | 2025-05-21 14:15:00 | 331.10 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-04 13:45:00 | 334.50 | 2025-06-05 09:15:00 | 337.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-07-07 10:30:00 | 334.20 | 2025-07-10 09:15:00 | 340.40 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-07-07 15:15:00 | 333.15 | 2025-07-10 09:15:00 | 340.40 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-07-09 15:15:00 | 334.00 | 2025-07-10 09:15:00 | 340.40 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-07-15 13:45:00 | 354.20 | 2025-07-25 10:15:00 | 348.65 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-07-16 14:00:00 | 354.35 | 2025-07-25 10:15:00 | 348.65 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-07-17 15:00:00 | 353.85 | 2025-07-25 10:15:00 | 348.65 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-07-18 09:30:00 | 354.15 | 2025-07-25 10:15:00 | 348.65 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-07-21 09:45:00 | 354.60 | 2025-07-25 10:15:00 | 348.65 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-07-21 10:15:00 | 355.25 | 2025-07-25 10:15:00 | 348.65 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-07-21 14:45:00 | 354.95 | 2025-07-25 10:15:00 | 348.65 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-07-22 09:15:00 | 355.00 | 2025-07-25 10:15:00 | 348.65 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-07-22 12:30:00 | 353.75 | 2025-07-25 10:15:00 | 348.65 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-07-22 13:45:00 | 354.55 | 2025-07-25 10:15:00 | 348.65 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-07-23 09:45:00 | 353.95 | 2025-07-25 10:15:00 | 348.65 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-08-21 09:15:00 | 368.80 | 2025-08-26 09:15:00 | 361.25 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-08-28 09:15:00 | 365.70 | 2025-08-28 09:15:00 | 369.80 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-09-04 10:30:00 | 355.25 | 2025-09-09 11:15:00 | 354.25 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2025-09-12 10:30:00 | 359.75 | 2025-09-15 13:15:00 | 354.80 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-09-12 13:30:00 | 360.15 | 2025-09-15 13:15:00 | 354.80 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-09-12 14:45:00 | 359.95 | 2025-09-15 13:15:00 | 354.80 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-09-19 14:00:00 | 350.05 | 2025-09-22 12:15:00 | 354.60 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-09-19 15:00:00 | 349.00 | 2025-09-22 12:15:00 | 354.60 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-10-01 09:15:00 | 342.75 | 2025-10-06 10:15:00 | 348.45 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-10-13 14:30:00 | 360.40 | 2025-10-16 09:15:00 | 357.85 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-10-14 14:45:00 | 360.85 | 2025-10-17 10:15:00 | 356.45 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-10-15 09:15:00 | 361.00 | 2025-10-17 10:15:00 | 356.45 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-10-15 10:00:00 | 360.85 | 2025-10-17 10:15:00 | 356.45 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-10-15 15:15:00 | 363.25 | 2025-10-17 10:15:00 | 356.45 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-10-23 15:15:00 | 353.10 | 2025-10-28 09:15:00 | 359.70 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-10-24 10:00:00 | 354.05 | 2025-10-28 09:15:00 | 359.70 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-10-24 11:00:00 | 353.90 | 2025-10-28 09:15:00 | 359.70 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-10-27 15:15:00 | 351.15 | 2025-10-28 09:15:00 | 359.70 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-10-30 14:30:00 | 365.90 | 2025-10-31 15:15:00 | 356.10 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-10-31 09:30:00 | 364.85 | 2025-10-31 15:15:00 | 356.10 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-10-31 10:00:00 | 365.40 | 2025-10-31 15:15:00 | 356.10 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-10-31 11:30:00 | 364.45 | 2025-10-31 15:15:00 | 356.10 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2025-11-10 09:30:00 | 358.45 | 2025-11-10 11:15:00 | 360.50 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-11-13 15:00:00 | 353.15 | 2025-11-19 13:15:00 | 353.45 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-11-14 09:15:00 | 351.20 | 2025-11-19 13:15:00 | 353.45 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-11-18 09:15:00 | 349.15 | 2025-11-19 13:15:00 | 353.45 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-11-20 14:15:00 | 354.85 | 2025-11-21 09:15:00 | 351.95 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-11-20 15:15:00 | 355.00 | 2025-11-21 09:15:00 | 351.95 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-11-24 12:30:00 | 351.00 | 2025-11-24 15:15:00 | 353.95 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-11-24 14:30:00 | 350.60 | 2025-11-24 15:15:00 | 353.95 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-11-25 09:15:00 | 350.45 | 2025-11-25 11:15:00 | 354.65 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-11-28 14:45:00 | 359.10 | 2025-12-01 09:15:00 | 354.30 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-11-28 15:15:00 | 359.75 | 2025-12-01 09:15:00 | 354.30 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-12-02 10:15:00 | 349.05 | 2025-12-08 15:15:00 | 347.90 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2025-12-02 11:15:00 | 348.90 | 2025-12-15 09:15:00 | 342.95 | STOP_HIT | 1.00 | 1.71% |
| SELL | retest2 | 2025-12-02 12:15:00 | 348.25 | 2025-12-15 09:15:00 | 342.95 | STOP_HIT | 1.00 | 1.52% |
| SELL | retest2 | 2025-12-02 13:00:00 | 348.85 | 2025-12-15 09:15:00 | 342.95 | STOP_HIT | 1.00 | 1.69% |
| SELL | retest2 | 2025-12-08 09:15:00 | 341.70 | 2025-12-15 09:15:00 | 342.95 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-12-09 09:15:00 | 340.00 | 2025-12-15 09:15:00 | 342.95 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-12-11 09:15:00 | 335.20 | 2025-12-15 09:15:00 | 342.95 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-12-22 11:45:00 | 341.00 | 2025-12-23 15:15:00 | 343.75 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-12-22 14:00:00 | 340.65 | 2025-12-23 15:15:00 | 343.75 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-12-22 15:15:00 | 340.75 | 2025-12-23 15:15:00 | 343.75 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-12-23 10:15:00 | 340.10 | 2025-12-23 15:15:00 | 343.75 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2026-01-05 11:00:00 | 348.30 | 2026-01-06 09:15:00 | 342.80 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2026-01-20 09:30:00 | 324.55 | 2026-01-22 12:15:00 | 325.00 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2026-01-20 11:15:00 | 324.00 | 2026-01-22 12:15:00 | 325.00 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2026-01-21 10:00:00 | 323.35 | 2026-01-22 12:15:00 | 325.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2026-01-22 10:15:00 | 324.55 | 2026-01-22 12:15:00 | 325.00 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2026-01-28 10:00:00 | 336.00 | 2026-01-29 09:15:00 | 325.40 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2026-02-01 11:45:00 | 321.40 | 2026-02-03 13:15:00 | 324.15 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-02-18 13:00:00 | 332.00 | 2026-02-27 12:15:00 | 333.60 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2026-03-19 15:15:00 | 328.00 | 2026-03-25 13:15:00 | 324.65 | STOP_HIT | 1.00 | 1.02% |
| BUY | retest2 | 2026-03-27 11:00:00 | 329.30 | 2026-04-01 11:15:00 | 323.65 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-03-30 10:15:00 | 326.50 | 2026-04-01 11:15:00 | 323.65 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-03-30 11:45:00 | 325.00 | 2026-04-01 11:15:00 | 323.65 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2026-03-30 13:15:00 | 325.00 | 2026-04-01 11:15:00 | 323.65 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2026-04-06 09:15:00 | 314.75 | 2026-04-07 14:15:00 | 325.70 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2026-04-06 13:30:00 | 319.20 | 2026-04-07 14:15:00 | 325.70 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2026-04-06 15:00:00 | 318.35 | 2026-04-07 14:15:00 | 325.70 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2026-04-07 09:30:00 | 319.65 | 2026-04-07 14:15:00 | 325.70 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2026-04-09 11:15:00 | 325.85 | 2026-04-09 12:15:00 | 321.45 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2026-04-15 13:45:00 | 319.60 | 2026-04-16 10:15:00 | 322.20 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2026-04-15 15:00:00 | 318.75 | 2026-04-16 10:15:00 | 322.20 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-04-16 10:15:00 | 319.90 | 2026-04-16 10:15:00 | 322.20 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2026-04-21 13:15:00 | 321.45 | 2026-04-23 14:15:00 | 319.95 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2026-04-22 09:30:00 | 321.55 | 2026-04-23 14:15:00 | 319.95 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2026-04-22 11:00:00 | 326.90 | 2026-04-23 14:15:00 | 319.95 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2026-04-22 14:45:00 | 321.30 | 2026-04-23 14:15:00 | 319.95 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2026-05-06 11:00:00 | 303.00 | 2026-05-07 11:15:00 | 310.85 | STOP_HIT | 1.00 | -2.59% |
