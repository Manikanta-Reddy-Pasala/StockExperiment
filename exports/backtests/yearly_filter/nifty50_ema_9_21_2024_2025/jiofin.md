# JIOFIN (JIOFIN)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-11 15:15:00 (3717 bars)
- **Last close:** 239.98
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 142 |
| ALERT1 | 98 |
| ALERT2 | 96 |
| ALERT2_SKIP | 46 |
| ALERT3 | 262 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 108 |
| PARTIAL | 23 |
| TARGET_HIT | 7 |
| STOP_HIT | 108 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 137 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 60 / 77
- **Target hits / Stop hits / Partials:** 7 / 107 / 23
- **Avg / median % per leg:** 1.44% / -0.32%
- **Sum % (uncompounded):** 196.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 44 | 11 | 25.0% | 2 | 42 | 0 | -0.07% | -3.3% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 0 | 4 | 0 | 1.26% | 5.0% |
| BUY @ 3rd Alert (retest2) | 40 | 8 | 20.0% | 2 | 38 | 0 | -0.21% | -8.3% |
| SELL (all) | 93 | 49 | 52.7% | 5 | 65 | 23 | 2.15% | 199.9% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.83% | -1.7% |
| SELL @ 3rd Alert (retest2) | 91 | 48 | 52.7% | 5 | 63 | 23 | 2.22% | 201.6% |
| retest1 (combined) | 6 | 4 | 66.7% | 0 | 6 | 0 | 0.56% | 3.4% |
| retest2 (combined) | 131 | 56 | 42.7% | 7 | 101 | 23 | 1.48% | 193.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 13:15:00 | 351.10 | 347.56 | 347.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 15:15:00 | 353.40 | 349.48 | 348.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 12:15:00 | 353.80 | 354.33 | 352.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 13:00:00 | 353.80 | 354.33 | 352.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 352.95 | 354.06 | 352.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:30:00 | 352.95 | 354.06 | 352.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 354.90 | 354.23 | 352.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 356.90 | 354.38 | 353.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 12:15:00 | 361.00 | 362.69 | 362.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 12:15:00 | 361.00 | 362.69 | 362.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 15:15:00 | 360.00 | 361.56 | 362.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 346.65 | 346.26 | 349.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 13:45:00 | 346.50 | 346.26 | 349.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 356.15 | 347.75 | 349.09 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 11:15:00 | 355.50 | 350.51 | 350.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 13:15:00 | 356.85 | 352.63 | 351.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 349.95 | 353.28 | 352.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 349.95 | 353.28 | 352.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 349.95 | 353.28 | 352.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 348.60 | 353.28 | 352.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 335.55 | 349.73 | 350.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 325.35 | 344.86 | 348.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 346.65 | 341.16 | 344.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 10:15:00 | 346.65 | 341.16 | 344.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 346.65 | 341.16 | 344.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 346.65 | 341.16 | 344.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 343.65 | 341.66 | 344.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:30:00 | 344.55 | 341.66 | 344.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 344.25 | 342.17 | 344.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:00:00 | 344.25 | 342.17 | 344.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 345.90 | 342.92 | 344.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:30:00 | 346.80 | 342.92 | 344.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 344.40 | 343.22 | 344.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:45:00 | 345.60 | 343.22 | 344.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 347.05 | 343.98 | 344.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 349.45 | 343.98 | 344.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 349.60 | 345.11 | 345.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 14:15:00 | 352.95 | 351.07 | 349.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 11:15:00 | 352.50 | 352.56 | 350.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 12:00:00 | 352.50 | 352.56 | 350.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 350.30 | 352.06 | 350.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:30:00 | 349.25 | 352.06 | 350.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 350.95 | 351.84 | 350.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 351.95 | 351.84 | 350.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 352.85 | 351.49 | 351.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 15:15:00 | 358.70 | 359.74 | 359.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 15:15:00 | 358.70 | 359.74 | 359.78 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 09:15:00 | 366.30 | 361.05 | 360.37 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 09:15:00 | 355.90 | 360.82 | 361.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 349.00 | 354.27 | 355.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 09:15:00 | 361.05 | 355.38 | 356.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 09:15:00 | 361.05 | 355.38 | 356.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 361.05 | 355.38 | 356.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:00:00 | 361.05 | 355.38 | 356.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 358.10 | 355.93 | 356.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 11:30:00 | 357.25 | 356.16 | 356.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 13:15:00 | 357.85 | 356.54 | 356.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2024-06-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 13:15:00 | 357.85 | 356.54 | 356.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 14:15:00 | 358.60 | 356.95 | 356.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 09:15:00 | 354.05 | 356.51 | 356.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 354.05 | 356.51 | 356.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 354.05 | 356.51 | 356.49 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 10:15:00 | 354.10 | 356.02 | 356.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-01 12:15:00 | 353.55 | 355.21 | 355.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-02 09:15:00 | 354.70 | 354.24 | 355.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 09:15:00 | 354.70 | 354.24 | 355.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 354.70 | 354.24 | 355.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:45:00 | 355.20 | 354.24 | 355.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 352.55 | 352.35 | 353.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 12:15:00 | 351.70 | 352.34 | 353.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 13:15:00 | 351.95 | 352.43 | 352.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 09:15:00 | 351.70 | 351.46 | 351.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 10:00:00 | 350.95 | 350.27 | 350.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 351.45 | 350.50 | 350.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:45:00 | 351.35 | 350.50 | 350.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 351.45 | 350.69 | 351.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 12:00:00 | 351.45 | 350.69 | 351.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 12:15:00 | 351.40 | 350.83 | 351.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 13:00:00 | 351.40 | 350.83 | 351.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 351.95 | 351.06 | 351.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:00:00 | 351.95 | 351.06 | 351.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 347.90 | 350.05 | 350.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 10:15:00 | 346.35 | 350.05 | 350.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 11:00:00 | 347.40 | 349.52 | 350.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 13:45:00 | 347.15 | 348.47 | 349.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 09:15:00 | 351.95 | 348.85 | 348.97 | SL hit (close>static) qty=1.00 sl=350.90 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 10:15:00 | 352.40 | 349.56 | 349.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 13:15:00 | 352.75 | 351.78 | 350.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 09:15:00 | 348.45 | 352.16 | 351.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 348.45 | 352.16 | 351.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 348.45 | 352.16 | 351.36 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2024-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 11:15:00 | 347.30 | 350.50 | 350.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 12:15:00 | 345.00 | 349.40 | 350.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 13:15:00 | 340.55 | 337.53 | 340.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 13:15:00 | 340.55 | 337.53 | 340.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 13:15:00 | 340.55 | 337.53 | 340.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 13:30:00 | 340.40 | 337.53 | 340.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 14:15:00 | 338.15 | 337.66 | 340.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 331.30 | 339.18 | 339.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 10:45:00 | 337.45 | 336.75 | 338.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 320.58 | 325.86 | 327.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-06 09:15:00 | 320.65 | 319.69 | 322.89 | SL hit (close>ema200) qty=0.50 sl=319.69 alert=retest2 |

### Cycle 13 — BUY (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 15:15:00 | 331.00 | 321.93 | 320.89 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 15:15:00 | 323.00 | 326.47 | 326.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 14:15:00 | 320.15 | 323.43 | 324.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 13:15:00 | 323.70 | 322.05 | 323.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 13:15:00 | 323.70 | 322.05 | 323.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 323.70 | 322.05 | 323.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:00:00 | 323.70 | 322.05 | 323.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 328.50 | 323.34 | 323.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 15:00:00 | 328.50 | 323.34 | 323.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2024-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 15:15:00 | 327.90 | 324.25 | 324.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 333.70 | 326.14 | 325.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 12:15:00 | 331.95 | 333.57 | 331.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 12:15:00 | 331.95 | 333.57 | 331.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 12:15:00 | 331.95 | 333.57 | 331.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 12:45:00 | 332.15 | 333.57 | 331.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 13:15:00 | 330.95 | 333.05 | 331.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 09:15:00 | 332.10 | 332.32 | 331.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 10:00:00 | 332.20 | 332.30 | 331.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 10:30:00 | 332.20 | 332.28 | 331.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-22 14:15:00 | 329.60 | 331.22 | 331.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 14:15:00 | 329.60 | 331.22 | 331.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 13:15:00 | 328.10 | 329.98 | 330.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 09:15:00 | 329.95 | 329.14 | 330.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 09:15:00 | 329.95 | 329.14 | 330.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 329.95 | 329.14 | 330.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 10:45:00 | 324.25 | 328.16 | 329.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 10:45:00 | 328.95 | 325.80 | 325.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-30 09:15:00 | 329.55 | 326.46 | 326.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 329.55 | 326.46 | 326.10 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 13:15:00 | 323.70 | 325.76 | 325.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 14:15:00 | 321.30 | 324.86 | 325.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 09:15:00 | 325.00 | 324.50 | 325.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 09:15:00 | 325.00 | 324.50 | 325.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 325.00 | 324.50 | 325.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:45:00 | 327.90 | 324.50 | 325.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 326.10 | 324.82 | 325.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 10:45:00 | 325.70 | 324.82 | 325.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 326.30 | 325.11 | 325.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:00:00 | 326.30 | 325.11 | 325.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2024-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 13:15:00 | 329.00 | 326.11 | 325.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 14:15:00 | 345.40 | 329.96 | 327.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 12:15:00 | 345.25 | 345.28 | 340.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-04 12:45:00 | 345.25 | 345.28 | 340.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 15:15:00 | 344.35 | 346.30 | 344.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 09:15:00 | 343.20 | 346.30 | 344.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 338.75 | 344.79 | 343.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 338.75 | 344.79 | 343.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 338.15 | 343.46 | 343.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:30:00 | 337.70 | 343.46 | 343.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-09-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 11:15:00 | 338.40 | 342.45 | 342.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 334.20 | 338.39 | 340.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 341.15 | 337.15 | 338.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 14:15:00 | 341.15 | 337.15 | 338.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 341.15 | 337.15 | 338.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 15:00:00 | 341.15 | 337.15 | 338.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 344.10 | 338.54 | 339.32 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2024-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 09:15:00 | 347.55 | 340.34 | 340.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 10:15:00 | 353.05 | 342.88 | 341.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 12:15:00 | 350.45 | 350.90 | 347.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 13:00:00 | 350.45 | 350.90 | 347.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 344.90 | 349.18 | 347.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 344.90 | 349.18 | 347.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 344.80 | 348.30 | 347.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:15:00 | 346.85 | 348.30 | 347.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 11:15:00 | 343.20 | 346.49 | 346.44 | SL hit (close<static) qty=1.00 sl=343.30 alert=retest2 |

### Cycle 22 — SELL (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 12:15:00 | 343.00 | 345.79 | 346.13 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 15:15:00 | 350.50 | 346.79 | 346.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 14:15:00 | 353.40 | 349.00 | 347.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 11:15:00 | 348.35 | 349.64 | 348.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 11:15:00 | 348.35 | 349.64 | 348.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 348.35 | 349.64 | 348.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:45:00 | 348.45 | 349.64 | 348.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 349.90 | 349.69 | 348.74 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 10:15:00 | 345.25 | 347.88 | 348.20 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 15:15:00 | 350.60 | 348.43 | 348.25 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 13:15:00 | 346.55 | 347.90 | 348.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 10:15:00 | 343.10 | 346.29 | 347.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 15:15:00 | 348.55 | 344.94 | 345.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 15:15:00 | 348.55 | 344.94 | 345.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 348.55 | 344.94 | 345.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 352.90 | 344.94 | 345.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 09:15:00 | 355.00 | 346.95 | 346.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 09:15:00 | 356.95 | 352.30 | 349.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 15:15:00 | 353.40 | 354.14 | 352.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 09:15:00 | 354.20 | 354.14 | 352.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 352.25 | 353.76 | 352.15 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 10:15:00 | 350.80 | 351.56 | 351.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 11:15:00 | 349.70 | 351.19 | 351.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 14:15:00 | 353.00 | 351.26 | 351.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 14:15:00 | 353.00 | 351.26 | 351.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 353.00 | 351.26 | 351.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 15:00:00 | 353.00 | 351.26 | 351.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2024-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 15:15:00 | 352.50 | 351.51 | 351.45 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 09:15:00 | 350.50 | 351.31 | 351.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 15:15:00 | 349.10 | 350.12 | 350.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 09:15:00 | 352.05 | 350.50 | 350.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 09:15:00 | 352.05 | 350.50 | 350.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 352.05 | 350.50 | 350.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:00:00 | 352.05 | 350.50 | 350.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 10:15:00 | 359.30 | 352.26 | 351.57 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 14:15:00 | 350.10 | 352.96 | 353.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 10:15:00 | 347.90 | 351.23 | 352.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 13:15:00 | 351.00 | 350.25 | 351.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 13:15:00 | 351.00 | 350.25 | 351.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 351.00 | 350.25 | 351.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 13:45:00 | 351.35 | 350.25 | 351.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 351.50 | 350.50 | 351.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 15:00:00 | 351.50 | 350.50 | 351.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 350.20 | 350.44 | 351.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 345.95 | 350.44 | 351.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 15:15:00 | 344.00 | 340.83 | 340.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 344.00 | 340.83 | 340.69 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 14:15:00 | 342.20 | 342.97 | 343.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 15:15:00 | 340.95 | 342.57 | 342.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 09:15:00 | 330.30 | 330.17 | 331.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 09:15:00 | 330.30 | 330.17 | 331.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 330.30 | 330.17 | 331.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:00:00 | 328.55 | 329.76 | 331.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:00:00 | 328.70 | 329.34 | 330.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 15:15:00 | 312.12 | 319.71 | 324.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 15:15:00 | 312.26 | 319.71 | 324.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 09:15:00 | 319.95 | 319.76 | 323.85 | SL hit (close>ema200) qty=0.50 sl=319.76 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 13:15:00 | 318.15 | 315.86 | 315.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 11:15:00 | 325.65 | 318.29 | 317.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 12:15:00 | 321.55 | 322.47 | 320.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 13:00:00 | 321.55 | 322.47 | 320.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 13:15:00 | 320.70 | 322.12 | 320.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 09:30:00 | 322.25 | 321.29 | 320.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 11:45:00 | 322.40 | 321.47 | 320.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 15:00:00 | 322.65 | 321.68 | 321.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:00:00 | 325.90 | 322.56 | 321.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 320.40 | 322.76 | 321.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 320.40 | 322.76 | 321.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 320.45 | 322.30 | 321.71 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-04 11:15:00 | 320.05 | 321.85 | 321.56 | SL hit (close<static) qty=1.00 sl=320.10 alert=retest2 |

### Cycle 36 — SELL (started 2024-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 13:15:00 | 320.30 | 321.32 | 321.36 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 15:15:00 | 321.65 | 321.19 | 321.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 324.10 | 321.77 | 321.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 323.65 | 324.61 | 323.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 323.65 | 324.61 | 323.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 323.65 | 324.61 | 323.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 323.65 | 324.61 | 323.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 322.85 | 324.26 | 323.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 322.70 | 324.26 | 323.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 322.75 | 323.96 | 323.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:30:00 | 323.00 | 323.96 | 323.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 322.85 | 323.74 | 323.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:00:00 | 322.85 | 323.74 | 323.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 322.75 | 323.54 | 323.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:45:00 | 322.65 | 323.54 | 323.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 15:15:00 | 321.25 | 322.84 | 322.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 319.65 | 322.20 | 322.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 314.65 | 314.54 | 316.82 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 14:45:00 | 311.55 | 313.13 | 315.26 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 316.75 | 305.90 | 308.86 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 316.75 | 305.90 | 308.86 | SL hit (close>ema400) qty=1.00 sl=308.86 alert=retest1 |

### Cycle 39 — BUY (started 2024-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 12:15:00 | 314.95 | 311.05 | 310.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-14 14:15:00 | 318.25 | 313.31 | 311.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 319.30 | 320.26 | 318.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 15:00:00 | 319.30 | 320.26 | 318.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 317.85 | 319.78 | 317.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:15:00 | 315.50 | 319.78 | 317.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 314.20 | 318.67 | 317.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:30:00 | 315.90 | 318.67 | 317.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 314.40 | 317.81 | 317.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:30:00 | 313.60 | 317.81 | 317.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 12:15:00 | 314.75 | 316.80 | 316.94 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 320.05 | 316.81 | 316.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 14:15:00 | 323.45 | 319.25 | 317.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 10:15:00 | 327.90 | 328.40 | 325.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 10:45:00 | 327.85 | 328.40 | 325.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 325.65 | 327.17 | 325.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 325.65 | 327.17 | 325.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 325.65 | 326.86 | 325.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:15:00 | 330.95 | 326.86 | 325.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 09:15:00 | 336.95 | 339.43 | 339.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 336.95 | 339.43 | 339.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 15:15:00 | 336.30 | 337.82 | 338.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 10:15:00 | 336.70 | 335.67 | 336.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 10:15:00 | 336.70 | 335.67 | 336.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 336.70 | 335.67 | 336.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 11:00:00 | 336.70 | 335.67 | 336.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 11:15:00 | 339.40 | 336.42 | 336.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 11:45:00 | 340.90 | 336.42 | 336.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 12:15:00 | 339.05 | 336.94 | 337.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 12:30:00 | 338.85 | 336.94 | 337.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 13:15:00 | 341.85 | 337.92 | 337.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 14:15:00 | 342.75 | 338.89 | 338.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 11:15:00 | 338.85 | 340.05 | 339.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 11:15:00 | 338.85 | 340.05 | 339.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 338.85 | 340.05 | 339.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 12:00:00 | 338.85 | 340.05 | 339.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 339.40 | 339.92 | 339.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 13:15:00 | 338.45 | 339.92 | 339.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 13:15:00 | 337.95 | 339.53 | 338.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 13:30:00 | 338.75 | 339.53 | 338.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 14:15:00 | 338.20 | 339.26 | 338.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 14:45:00 | 337.55 | 339.26 | 338.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 338.35 | 339.08 | 338.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:15:00 | 335.80 | 339.08 | 338.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 333.00 | 337.86 | 338.31 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 09:15:00 | 340.65 | 338.59 | 338.34 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 12:15:00 | 337.85 | 338.87 | 338.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 336.75 | 338.44 | 338.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 12:15:00 | 305.40 | 304.86 | 306.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-26 13:00:00 | 305.40 | 304.86 | 306.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 306.65 | 305.46 | 306.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 11:45:00 | 305.50 | 305.57 | 306.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 14:45:00 | 305.35 | 305.56 | 306.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 14:15:00 | 308.15 | 304.83 | 305.38 | SL hit (close>static) qty=1.00 sl=307.35 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 14:15:00 | 304.35 | 303.16 | 303.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 12:15:00 | 307.60 | 305.36 | 304.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 304.40 | 306.01 | 305.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 304.40 | 306.01 | 305.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 304.40 | 306.01 | 305.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 304.75 | 306.01 | 305.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 299.00 | 304.61 | 304.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 297.55 | 301.73 | 303.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 301.00 | 300.10 | 301.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 10:00:00 | 301.00 | 300.10 | 301.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 297.80 | 299.40 | 300.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 12:00:00 | 296.50 | 298.82 | 300.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 12:30:00 | 296.95 | 298.45 | 299.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 13:00:00 | 296.95 | 298.45 | 299.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 296.40 | 298.40 | 299.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 13:15:00 | 282.10 | 287.38 | 291.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 13:15:00 | 282.10 | 287.38 | 291.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 14:15:00 | 281.68 | 286.01 | 290.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 14:15:00 | 281.58 | 286.01 | 290.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 13:15:00 | 266.85 | 274.54 | 282.19 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 49 — BUY (started 2025-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 12:15:00 | 277.75 | 275.62 | 275.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 279.15 | 276.89 | 276.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 276.55 | 276.82 | 276.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 10:15:00 | 276.55 | 276.82 | 276.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 276.55 | 276.82 | 276.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:00:00 | 276.55 | 276.82 | 276.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 278.00 | 277.06 | 276.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 14:15:00 | 278.75 | 277.46 | 276.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 09:15:00 | 274.35 | 277.24 | 276.80 | SL hit (close<static) qty=1.00 sl=276.35 alert=retest2 |

### Cycle 50 — SELL (started 2025-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 14:15:00 | 275.85 | 276.48 | 276.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 09:15:00 | 273.45 | 275.74 | 276.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 263.10 | 261.61 | 266.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 263.10 | 261.61 | 266.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 266.95 | 262.97 | 265.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 266.95 | 262.97 | 265.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 263.45 | 263.06 | 265.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 11:15:00 | 262.85 | 263.06 | 265.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 09:15:00 | 249.71 | 257.08 | 261.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-27 09:15:00 | 236.57 | 245.86 | 252.78 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 11:15:00 | 241.70 | 238.98 | 238.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 12:15:00 | 243.30 | 239.85 | 239.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 240.00 | 241.65 | 240.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 240.00 | 241.65 | 240.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 240.00 | 241.65 | 240.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 240.00 | 241.65 | 240.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 242.71 | 241.86 | 240.78 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 237.77 | 239.94 | 240.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 12:15:00 | 236.00 | 239.15 | 239.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 237.38 | 237.00 | 238.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 237.38 | 237.00 | 238.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 237.38 | 237.00 | 238.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 10:30:00 | 236.00 | 236.80 | 238.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 12:15:00 | 241.61 | 237.88 | 238.45 | SL hit (close>static) qty=1.00 sl=239.10 alert=retest2 |

### Cycle 53 — BUY (started 2025-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 13:15:00 | 244.95 | 239.29 | 239.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 253.50 | 243.82 | 241.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 15:15:00 | 249.40 | 249.43 | 245.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 09:15:00 | 249.20 | 249.43 | 245.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 250.20 | 250.54 | 248.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:15:00 | 247.95 | 250.54 | 248.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 251.53 | 250.74 | 248.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 252.33 | 250.74 | 248.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 245.29 | 249.01 | 248.76 | SL hit (close<static) qty=1.00 sl=246.13 alert=retest2 |

### Cycle 54 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 246.04 | 248.41 | 248.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 243.09 | 247.35 | 248.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 231.50 | 230.49 | 234.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 09:45:00 | 231.34 | 230.49 | 234.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 226.45 | 221.52 | 222.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 226.45 | 221.52 | 222.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 226.08 | 222.43 | 222.94 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 226.64 | 223.28 | 223.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 13:15:00 | 227.75 | 224.78 | 223.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 13:15:00 | 233.51 | 233.64 | 231.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 13:45:00 | 233.43 | 233.64 | 231.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 230.28 | 232.98 | 231.55 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 13:15:00 | 228.78 | 230.86 | 230.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 12:15:00 | 227.70 | 229.27 | 229.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 204.10 | 203.59 | 209.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 09:45:00 | 203.87 | 203.59 | 209.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 213.81 | 207.20 | 208.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:30:00 | 214.35 | 207.20 | 208.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 216.15 | 208.99 | 209.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:00:00 | 216.15 | 208.99 | 209.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 216.70 | 210.53 | 209.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 221.51 | 215.73 | 212.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 221.00 | 221.37 | 219.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 221.00 | 221.37 | 219.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 221.00 | 221.37 | 219.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 221.00 | 221.37 | 219.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 219.87 | 220.89 | 219.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 13:00:00 | 219.87 | 220.89 | 219.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 218.68 | 220.45 | 219.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 13:45:00 | 218.86 | 220.45 | 219.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 216.97 | 219.75 | 219.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 15:00:00 | 216.97 | 219.75 | 219.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 217.53 | 218.70 | 218.83 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 13:15:00 | 219.79 | 218.18 | 218.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 14:15:00 | 225.70 | 219.69 | 218.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 14:15:00 | 222.07 | 222.88 | 221.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 14:15:00 | 222.07 | 222.88 | 221.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 222.07 | 222.88 | 221.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 15:00:00 | 222.07 | 222.88 | 221.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 222.84 | 222.91 | 221.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 222.25 | 222.91 | 221.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 222.55 | 222.84 | 221.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:30:00 | 221.80 | 222.84 | 221.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 222.13 | 222.70 | 221.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:45:00 | 221.95 | 222.70 | 221.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 220.04 | 222.16 | 221.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 13:00:00 | 220.04 | 222.16 | 221.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 13:15:00 | 220.10 | 221.75 | 221.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 13:45:00 | 220.00 | 221.75 | 221.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 14:15:00 | 219.24 | 221.25 | 221.27 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 222.88 | 221.31 | 221.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 225.47 | 222.46 | 221.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 15:15:00 | 227.80 | 227.90 | 225.69 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 09:15:00 | 232.20 | 227.90 | 225.69 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 11:15:00 | 229.85 | 231.13 | 229.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 11:45:00 | 229.40 | 231.13 | 229.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 12:15:00 | 229.50 | 230.80 | 229.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 12:30:00 | 229.79 | 230.80 | 229.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 229.00 | 230.44 | 229.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-21 13:15:00 | 229.00 | 230.44 | 229.36 | SL hit (close<ema400) qty=1.00 sl=229.36 alert=retest1 |

### Cycle 62 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 11:15:00 | 227.97 | 230.45 | 230.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 10:15:00 | 224.75 | 227.78 | 229.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 225.71 | 224.02 | 225.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 225.71 | 224.02 | 225.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 225.71 | 224.02 | 225.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 225.71 | 224.02 | 225.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 225.20 | 224.25 | 225.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 228.41 | 224.25 | 225.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 228.09 | 225.02 | 225.71 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 230.50 | 227.00 | 226.54 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 223.65 | 228.80 | 228.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 11:15:00 | 223.09 | 227.09 | 228.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 214.85 | 214.73 | 219.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 219.60 | 214.73 | 219.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 218.80 | 215.54 | 219.16 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 225.49 | 220.94 | 220.54 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 10:15:00 | 219.55 | 220.26 | 220.27 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 11:15:00 | 221.06 | 220.42 | 220.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 226.78 | 221.96 | 221.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 238.99 | 240.34 | 236.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 09:30:00 | 240.20 | 240.34 | 236.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 253.26 | 256.97 | 254.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 253.26 | 256.97 | 254.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 251.50 | 255.88 | 254.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:30:00 | 251.35 | 255.88 | 254.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 13:15:00 | 253.35 | 254.54 | 254.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 14:00:00 | 253.35 | 254.54 | 254.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 253.43 | 254.32 | 253.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 14:30:00 | 253.29 | 254.32 | 253.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 15:15:00 | 252.25 | 253.90 | 253.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 09:15:00 | 254.69 | 253.90 | 253.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 09:15:00 | 254.85 | 259.35 | 259.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 254.85 | 259.35 | 259.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 13:15:00 | 253.90 | 256.81 | 258.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 12:15:00 | 254.15 | 253.82 | 255.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 12:45:00 | 254.05 | 253.82 | 255.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 255.75 | 254.20 | 255.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 255.75 | 254.20 | 255.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 255.75 | 254.51 | 255.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:45:00 | 255.70 | 254.51 | 255.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 256.50 | 254.91 | 255.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 256.95 | 254.91 | 255.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 255.40 | 255.01 | 255.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 10:30:00 | 254.90 | 255.04 | 255.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 11:30:00 | 254.85 | 255.23 | 255.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:15:00 | 254.85 | 255.35 | 255.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 10:15:00 | 242.16 | 250.14 | 252.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 10:15:00 | 242.11 | 250.14 | 252.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 10:15:00 | 242.11 | 250.14 | 252.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 14:15:00 | 248.50 | 248.39 | 251.04 | SL hit (close>ema200) qty=0.50 sl=248.39 alert=retest2 |

### Cycle 69 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 260.50 | 252.84 | 252.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 261.75 | 256.92 | 254.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 13:15:00 | 265.90 | 266.76 | 263.84 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 15:00:00 | 267.80 | 266.97 | 264.20 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 10:00:00 | 267.40 | 267.20 | 264.79 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 10:30:00 | 268.60 | 267.26 | 265.04 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 273.65 | 276.22 | 274.30 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-20 09:15:00 | 273.65 | 276.22 | 274.30 | SL hit (close<ema400) qty=1.00 sl=274.30 alert=retest1 |

### Cycle 70 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 271.60 | 273.73 | 273.89 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 14:15:00 | 274.20 | 273.98 | 273.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 09:15:00 | 275.85 | 274.44 | 274.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 13:15:00 | 273.75 | 274.69 | 274.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 13:15:00 | 273.75 | 274.69 | 274.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 273.75 | 274.69 | 274.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:00:00 | 273.75 | 274.69 | 274.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 274.60 | 274.67 | 274.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:30:00 | 273.45 | 274.67 | 274.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 274.65 | 274.67 | 274.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 09:15:00 | 276.60 | 274.67 | 274.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 286.30 | 287.29 | 287.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 14:15:00 | 286.30 | 287.29 | 287.35 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 11:15:00 | 288.70 | 287.33 | 287.32 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 14:15:00 | 286.25 | 287.25 | 287.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 10:15:00 | 284.85 | 286.47 | 286.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 11:15:00 | 284.90 | 284.58 | 285.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 12:00:00 | 284.90 | 284.58 | 285.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 286.15 | 284.89 | 285.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:45:00 | 285.60 | 284.89 | 285.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 288.40 | 285.59 | 285.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:00:00 | 288.40 | 285.59 | 285.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 14:15:00 | 290.10 | 286.49 | 286.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 11:15:00 | 292.20 | 289.50 | 287.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 14:15:00 | 302.95 | 303.08 | 300.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 14:45:00 | 302.70 | 303.08 | 300.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 301.00 | 302.54 | 300.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:30:00 | 300.05 | 302.54 | 300.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 300.75 | 302.18 | 300.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:30:00 | 300.70 | 302.18 | 300.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 305.30 | 302.81 | 300.81 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 295.15 | 300.30 | 300.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 292.60 | 297.51 | 299.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 294.60 | 293.88 | 295.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 294.60 | 293.88 | 295.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 293.65 | 293.89 | 295.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:45:00 | 290.80 | 292.96 | 294.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 13:15:00 | 290.90 | 292.96 | 294.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 14:30:00 | 290.50 | 292.16 | 293.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:30:00 | 291.00 | 291.42 | 292.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 288.30 | 286.20 | 288.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 286.70 | 286.20 | 288.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 291.85 | 287.33 | 288.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 291.85 | 287.33 | 288.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 292.60 | 288.38 | 288.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 290.55 | 288.38 | 288.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 13:15:00 | 293.25 | 289.94 | 289.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2025-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 13:15:00 | 293.25 | 289.94 | 289.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 294.25 | 290.80 | 289.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 291.25 | 291.32 | 290.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 291.25 | 291.32 | 290.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 291.25 | 291.32 | 290.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:30:00 | 293.15 | 291.85 | 290.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 12:00:00 | 293.20 | 291.85 | 290.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-27 10:15:00 | 322.46 | 313.39 | 307.93 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 12:15:00 | 324.45 | 325.50 | 325.61 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 10:15:00 | 328.40 | 326.10 | 325.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 331.40 | 328.52 | 327.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 14:15:00 | 330.40 | 330.95 | 329.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 14:15:00 | 330.40 | 330.95 | 329.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 330.40 | 330.95 | 329.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:45:00 | 329.75 | 330.95 | 329.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 329.70 | 330.70 | 329.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:15:00 | 330.40 | 330.70 | 329.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 332.50 | 331.06 | 329.69 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 326.65 | 330.00 | 330.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 15:15:00 | 325.40 | 327.40 | 328.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 10:15:00 | 321.40 | 321.28 | 323.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 11:00:00 | 321.40 | 321.28 | 323.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 319.75 | 320.61 | 322.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:45:00 | 321.10 | 320.61 | 322.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 319.30 | 319.80 | 320.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:30:00 | 320.50 | 319.80 | 320.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 315.60 | 318.58 | 319.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:15:00 | 314.55 | 318.58 | 319.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:00:00 | 314.60 | 316.89 | 317.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 318.05 | 315.14 | 315.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 318.05 | 315.14 | 315.02 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 311.50 | 315.29 | 315.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 311.25 | 313.23 | 314.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 10:15:00 | 313.45 | 312.62 | 313.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 10:15:00 | 313.45 | 312.62 | 313.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 313.45 | 312.62 | 313.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:15:00 | 315.00 | 312.62 | 313.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 314.05 | 312.91 | 313.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:45:00 | 312.55 | 312.67 | 313.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 10:00:00 | 312.75 | 310.66 | 312.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 11:15:00 | 317.05 | 312.65 | 312.73 | SL hit (close>static) qty=1.00 sl=315.85 alert=retest2 |

### Cycle 83 — BUY (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 12:15:00 | 317.50 | 313.62 | 313.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 13:15:00 | 318.95 | 314.68 | 313.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 15:15:00 | 320.00 | 320.48 | 318.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 09:15:00 | 327.80 | 320.48 | 318.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 328.75 | 322.13 | 319.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 11:45:00 | 332.15 | 328.75 | 325.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 334.10 | 329.36 | 326.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 10:45:00 | 331.80 | 330.55 | 327.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 12:15:00 | 332.10 | 330.67 | 328.05 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 329.30 | 332.21 | 331.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 329.30 | 332.21 | 331.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 327.50 | 331.27 | 330.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 327.50 | 331.27 | 330.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-06 11:15:00 | 327.95 | 330.60 | 330.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 327.95 | 330.60 | 330.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 321.90 | 327.16 | 328.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 324.70 | 323.22 | 325.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 324.70 | 323.22 | 325.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 327.40 | 324.06 | 326.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 324.05 | 324.06 | 326.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 14:15:00 | 328.60 | 324.52 | 324.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 328.60 | 324.52 | 324.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 329.95 | 326.09 | 325.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 14:15:00 | 327.10 | 327.64 | 326.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 15:00:00 | 327.10 | 327.64 | 326.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 327.15 | 329.51 | 328.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 327.15 | 329.51 | 328.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 326.95 | 329.00 | 328.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 326.95 | 329.00 | 328.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 328.55 | 328.56 | 328.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 331.25 | 328.10 | 328.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 12:15:00 | 328.95 | 328.76 | 328.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 13:00:00 | 329.75 | 328.96 | 328.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 09:30:00 | 329.10 | 330.26 | 329.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 329.85 | 330.18 | 329.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:00:00 | 329.85 | 330.18 | 329.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 329.90 | 330.12 | 329.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:30:00 | 329.65 | 330.12 | 329.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 329.25 | 329.95 | 329.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:00:00 | 329.25 | 329.95 | 329.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 329.35 | 329.83 | 329.67 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-20 14:15:00 | 328.10 | 329.48 | 329.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 14:15:00 | 328.10 | 329.48 | 329.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 10:15:00 | 327.10 | 328.66 | 329.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 13:15:00 | 321.75 | 321.08 | 323.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 14:00:00 | 321.75 | 321.08 | 323.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 311.50 | 312.93 | 314.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 15:15:00 | 311.40 | 312.93 | 314.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 10:15:00 | 311.35 | 312.62 | 313.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 12:45:00 | 311.20 | 312.08 | 312.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 12:15:00 | 311.40 | 312.47 | 312.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 311.85 | 311.08 | 311.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 311.85 | 311.08 | 311.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 310.90 | 311.04 | 311.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 11:30:00 | 310.30 | 310.80 | 311.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 14:00:00 | 310.20 | 310.57 | 311.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 13:15:00 | 312.40 | 311.59 | 311.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 13:15:00 | 312.40 | 311.59 | 311.49 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 15:15:00 | 310.85 | 311.34 | 311.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 10:15:00 | 310.50 | 311.05 | 311.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 312.50 | 309.92 | 310.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 312.50 | 309.92 | 310.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 312.50 | 309.92 | 310.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:45:00 | 312.80 | 309.92 | 310.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 312.25 | 310.39 | 310.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:00:00 | 312.25 | 310.39 | 310.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 11:15:00 | 312.80 | 310.87 | 310.77 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 15:15:00 | 310.30 | 310.74 | 310.76 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 09:15:00 | 311.35 | 310.86 | 310.81 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 10:15:00 | 310.40 | 310.85 | 310.91 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 314.80 | 311.64 | 311.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 315.80 | 313.76 | 312.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 12:15:00 | 316.50 | 316.68 | 315.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 13:00:00 | 316.50 | 316.68 | 315.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 315.50 | 316.44 | 315.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 315.50 | 316.44 | 315.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 315.60 | 316.27 | 315.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 316.65 | 316.20 | 315.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 15:00:00 | 317.30 | 317.15 | 316.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 14:15:00 | 313.90 | 316.35 | 316.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 313.90 | 316.35 | 316.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 311.80 | 315.06 | 315.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 10:15:00 | 298.60 | 298.37 | 301.90 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-29 11:30:00 | 297.85 | 298.19 | 301.50 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 297.80 | 295.27 | 296.70 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-01 11:15:00 | 297.80 | 295.27 | 296.70 | SL hit (close>ema400) qty=1.00 sl=296.70 alert=retest1 |

### Cycle 95 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 300.75 | 297.86 | 297.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 10:15:00 | 301.00 | 298.96 | 298.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 308.90 | 309.55 | 306.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:15:00 | 308.05 | 309.55 | 306.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 306.05 | 308.85 | 306.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 306.05 | 308.85 | 306.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 305.70 | 308.22 | 306.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:00:00 | 305.70 | 308.22 | 306.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 306.15 | 307.51 | 306.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:00:00 | 306.15 | 307.51 | 306.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 304.40 | 306.88 | 306.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:45:00 | 304.60 | 306.88 | 306.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 307.55 | 306.62 | 306.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:45:00 | 306.45 | 306.62 | 306.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 306.50 | 306.80 | 306.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 13:30:00 | 306.60 | 306.80 | 306.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 307.45 | 306.93 | 306.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:45:00 | 306.65 | 306.93 | 306.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 309.15 | 307.40 | 306.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 11:00:00 | 309.85 | 307.89 | 307.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 12:30:00 | 309.60 | 308.55 | 307.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 306.20 | 308.09 | 307.69 | SL hit (close<static) qty=1.00 sl=306.65 alert=retest2 |

### Cycle 96 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 305.80 | 307.17 | 307.32 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 15:15:00 | 308.65 | 307.60 | 307.46 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 306.30 | 307.33 | 307.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 305.75 | 307.02 | 307.23 | Break + close below crossover candle low |

### Cycle 99 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 312.10 | 307.48 | 307.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 14:15:00 | 312.95 | 310.60 | 309.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 14:15:00 | 311.70 | 311.80 | 310.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 15:00:00 | 311.70 | 311.80 | 310.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 309.10 | 311.31 | 310.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 308.95 | 311.31 | 310.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 309.50 | 310.95 | 310.47 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 308.15 | 310.18 | 310.19 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 314.70 | 310.62 | 310.32 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 308.65 | 310.80 | 311.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 10:15:00 | 308.45 | 309.89 | 310.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 307.75 | 307.67 | 308.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 10:00:00 | 307.75 | 307.67 | 308.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 305.85 | 306.24 | 307.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 307.45 | 306.24 | 307.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 307.40 | 306.25 | 307.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:45:00 | 307.25 | 306.25 | 307.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 307.20 | 306.44 | 307.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:15:00 | 306.45 | 306.44 | 307.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 308.40 | 306.83 | 307.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 308.40 | 306.83 | 307.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 308.40 | 307.14 | 307.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 309.50 | 307.14 | 307.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 309.25 | 307.77 | 307.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 09:15:00 | 311.00 | 309.51 | 308.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 13:15:00 | 309.80 | 310.02 | 309.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 13:15:00 | 309.80 | 310.02 | 309.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 309.80 | 310.02 | 309.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:30:00 | 309.55 | 310.02 | 309.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 310.00 | 310.02 | 309.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:45:00 | 309.45 | 310.02 | 309.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 310.10 | 309.98 | 309.41 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 306.60 | 308.82 | 309.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 11:15:00 | 306.00 | 307.62 | 308.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 298.70 | 298.51 | 300.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:00:00 | 298.70 | 298.51 | 300.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 300.95 | 299.32 | 300.49 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 302.35 | 301.04 | 301.02 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 298.65 | 300.78 | 300.93 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 11:15:00 | 303.65 | 301.28 | 301.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 13:15:00 | 304.80 | 302.47 | 301.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 311.05 | 311.51 | 308.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 14:00:00 | 311.05 | 311.51 | 308.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 312.75 | 311.60 | 309.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:15:00 | 314.30 | 311.60 | 309.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:45:00 | 315.35 | 312.39 | 309.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:00:00 | 314.30 | 312.77 | 310.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 13:00:00 | 313.90 | 313.00 | 310.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 312.35 | 313.36 | 312.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:30:00 | 312.30 | 313.36 | 312.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 312.10 | 313.11 | 312.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:45:00 | 311.70 | 313.11 | 312.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 312.45 | 312.98 | 312.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:15:00 | 310.40 | 312.98 | 312.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 308.40 | 312.06 | 311.92 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 308.40 | 312.06 | 311.92 | SL hit (close<static) qty=1.00 sl=309.30 alert=retest2 |

### Cycle 108 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 308.65 | 311.38 | 311.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 306.35 | 309.10 | 310.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 307.40 | 305.92 | 307.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 307.40 | 305.92 | 307.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 307.40 | 305.92 | 307.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 306.20 | 305.92 | 307.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 307.45 | 306.23 | 307.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:45:00 | 307.80 | 306.23 | 307.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 307.70 | 306.52 | 307.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:30:00 | 308.05 | 306.52 | 307.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 12:15:00 | 308.15 | 306.85 | 307.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:30:00 | 308.30 | 306.85 | 307.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 304.40 | 302.12 | 302.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 304.40 | 302.12 | 302.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 305.15 | 302.73 | 303.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 305.25 | 302.73 | 303.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 308.15 | 304.31 | 303.82 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 304.90 | 305.84 | 305.91 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 13:15:00 | 306.25 | 305.82 | 305.80 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 14:15:00 | 305.10 | 305.68 | 305.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 301.45 | 304.76 | 305.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 11:15:00 | 302.70 | 302.14 | 303.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 12:00:00 | 302.70 | 302.14 | 303.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 301.55 | 302.02 | 302.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:30:00 | 300.90 | 301.86 | 302.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:30:00 | 300.70 | 302.02 | 302.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 10:15:00 | 301.30 | 302.02 | 302.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 14:15:00 | 303.85 | 302.69 | 302.76 | SL hit (close>static) qty=1.00 sl=303.45 alert=retest2 |

### Cycle 113 — BUY (started 2025-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 15:15:00 | 303.85 | 302.92 | 302.86 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 300.90 | 302.52 | 302.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 298.05 | 301.29 | 302.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 293.35 | 292.42 | 294.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 10:00:00 | 293.35 | 292.42 | 294.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 294.70 | 292.88 | 294.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:30:00 | 295.85 | 292.88 | 294.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 296.45 | 293.59 | 294.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 296.45 | 293.59 | 294.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 298.30 | 295.43 | 295.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 300.80 | 297.00 | 296.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 298.20 | 299.32 | 298.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 298.20 | 299.32 | 298.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 298.20 | 299.32 | 298.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:30:00 | 297.75 | 299.32 | 298.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 298.50 | 299.16 | 298.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:30:00 | 298.25 | 299.16 | 298.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 299.50 | 299.22 | 298.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:30:00 | 298.10 | 299.22 | 298.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 295.95 | 298.75 | 298.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 295.95 | 298.75 | 298.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 295.20 | 298.04 | 298.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 294.85 | 297.40 | 297.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 294.90 | 293.03 | 293.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 294.90 | 293.03 | 293.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 294.90 | 293.03 | 293.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 296.50 | 293.03 | 293.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 295.10 | 293.45 | 293.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 295.80 | 293.45 | 293.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 12:15:00 | 296.35 | 294.37 | 294.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 297.10 | 294.92 | 294.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 298.60 | 298.87 | 297.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 14:45:00 | 298.80 | 298.87 | 297.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 298.90 | 299.78 | 298.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:00:00 | 298.90 | 299.78 | 298.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 298.60 | 299.54 | 298.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:30:00 | 298.55 | 299.54 | 298.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 298.60 | 299.35 | 298.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 299.55 | 299.28 | 298.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 12:15:00 | 297.20 | 298.68 | 298.65 | SL hit (close<static) qty=1.00 sl=298.05 alert=retest2 |

### Cycle 118 — SELL (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 13:15:00 | 297.30 | 298.41 | 298.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 295.55 | 297.47 | 298.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 293.20 | 293.13 | 294.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:15:00 | 294.55 | 293.13 | 294.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 294.20 | 293.35 | 294.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 294.10 | 293.35 | 294.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 295.55 | 293.79 | 294.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 295.55 | 293.79 | 294.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 296.00 | 294.23 | 294.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:45:00 | 295.55 | 294.23 | 294.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 295.00 | 294.96 | 294.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 295.85 | 294.96 | 294.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 294.95 | 294.95 | 294.99 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 10:15:00 | 295.50 | 295.06 | 295.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 12:15:00 | 295.65 | 295.19 | 295.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 299.50 | 300.45 | 299.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 13:15:00 | 299.50 | 300.45 | 299.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 299.50 | 300.45 | 299.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 299.50 | 300.45 | 299.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 300.40 | 300.44 | 299.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:30:00 | 300.90 | 300.45 | 299.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 11:15:00 | 294.65 | 299.15 | 299.06 | SL hit (close<static) qty=1.00 sl=299.00 alert=retest2 |

### Cycle 120 — SELL (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 12:15:00 | 295.65 | 298.45 | 298.75 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 304.30 | 299.71 | 299.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 11:15:00 | 304.75 | 300.72 | 299.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 301.25 | 302.28 | 301.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 301.25 | 302.28 | 301.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 301.25 | 302.28 | 301.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 301.25 | 302.28 | 301.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 296.50 | 301.13 | 300.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 296.50 | 301.13 | 300.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 296.55 | 300.21 | 300.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 294.35 | 298.27 | 299.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 287.80 | 287.44 | 291.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 13:00:00 | 287.80 | 287.44 | 291.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 285.80 | 287.35 | 290.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:45:00 | 284.30 | 286.77 | 289.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:30:00 | 284.15 | 286.15 | 288.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:30:00 | 284.10 | 285.73 | 288.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:45:00 | 283.95 | 285.19 | 288.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 286.85 | 285.36 | 287.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 286.85 | 285.36 | 287.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 286.75 | 285.64 | 287.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:30:00 | 287.20 | 285.64 | 287.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 287.65 | 286.04 | 287.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:30:00 | 287.35 | 286.04 | 287.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 285.90 | 286.01 | 287.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:15:00 | 285.50 | 286.01 | 287.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:30:00 | 284.90 | 286.08 | 286.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 271.22 | 276.08 | 279.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 270.65 | 276.08 | 279.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 270.08 | 274.15 | 278.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 269.94 | 274.15 | 278.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 269.89 | 274.15 | 278.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 269.75 | 274.15 | 278.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 266.75 | 265.70 | 269.42 | SL hit (close>ema200) qty=0.50 sl=265.70 alert=retest2 |

### Cycle 123 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 263.60 | 250.37 | 249.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 264.90 | 253.27 | 250.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 266.90 | 268.59 | 265.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 10:00:00 | 266.90 | 268.59 | 265.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 268.85 | 270.53 | 269.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:45:00 | 269.00 | 270.53 | 269.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 270.05 | 270.44 | 269.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:15:00 | 270.65 | 270.44 | 269.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 13:15:00 | 270.65 | 270.25 | 269.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 267.85 | 269.76 | 269.72 | SL hit (close<static) qty=1.00 sl=268.65 alert=retest2 |

### Cycle 124 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 268.00 | 269.41 | 269.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 267.50 | 269.03 | 269.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 262.60 | 262.20 | 264.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 14:45:00 | 262.60 | 262.20 | 264.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 262.55 | 262.30 | 263.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 263.20 | 262.30 | 263.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 262.20 | 262.47 | 263.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 13:30:00 | 263.35 | 262.47 | 263.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 262.65 | 262.39 | 263.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:30:00 | 264.15 | 262.39 | 263.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 263.25 | 262.57 | 263.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:00:00 | 263.25 | 262.57 | 263.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 263.40 | 262.73 | 263.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 12:00:00 | 263.40 | 262.73 | 263.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 12:15:00 | 263.80 | 262.95 | 263.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 12:30:00 | 263.90 | 262.95 | 263.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 264.40 | 263.35 | 263.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 264.40 | 263.35 | 263.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 263.90 | 263.46 | 263.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 263.25 | 263.46 | 263.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 257.80 | 258.96 | 260.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:15:00 | 257.15 | 258.96 | 260.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 255.35 | 258.14 | 259.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 09:30:00 | 257.10 | 256.23 | 257.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:45:00 | 257.20 | 257.12 | 257.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 256.35 | 256.97 | 257.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 14:15:00 | 255.50 | 256.85 | 257.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 09:45:00 | 255.70 | 256.36 | 256.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 10:15:00 | 258.05 | 256.70 | 256.95 | SL hit (close>static) qty=1.00 sl=257.50 alert=retest2 |

### Cycle 125 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 241.80 | 236.66 | 236.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 13:15:00 | 242.95 | 239.48 | 238.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 237.65 | 239.90 | 238.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 237.65 | 239.90 | 238.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 237.65 | 239.90 | 238.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 237.65 | 239.90 | 238.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 237.55 | 239.43 | 238.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:00:00 | 237.55 | 239.43 | 238.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 238.60 | 239.27 | 238.70 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2026-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 13:15:00 | 235.40 | 237.86 | 238.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 232.40 | 235.76 | 236.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 234.80 | 234.03 | 235.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 234.80 | 234.03 | 235.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 234.80 | 234.03 | 235.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 235.15 | 234.03 | 235.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 236.90 | 234.52 | 235.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:30:00 | 236.25 | 234.52 | 235.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 236.40 | 234.90 | 235.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 235.20 | 234.90 | 235.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 234.85 | 235.20 | 235.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 238.05 | 236.09 | 236.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 238.05 | 236.09 | 236.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 243.30 | 238.02 | 236.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 242.75 | 244.75 | 241.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 242.75 | 244.75 | 241.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 242.75 | 244.75 | 241.68 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 14:15:00 | 239.15 | 241.26 | 241.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 230.70 | 238.83 | 240.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 230.75 | 229.81 | 233.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 230.75 | 229.81 | 233.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 233.60 | 230.57 | 233.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 233.25 | 230.57 | 233.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 232.90 | 231.03 | 233.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 233.55 | 231.03 | 233.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 232.60 | 231.35 | 233.16 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 240.35 | 234.58 | 234.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 241.00 | 235.86 | 234.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 233.00 | 236.32 | 235.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 233.00 | 236.32 | 235.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 233.00 | 236.32 | 235.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 233.00 | 236.32 | 235.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 233.10 | 235.67 | 235.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 232.90 | 235.67 | 235.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 233.25 | 235.19 | 235.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 229.35 | 232.75 | 233.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 229.25 | 227.62 | 230.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 229.25 | 227.62 | 230.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 229.25 | 227.62 | 230.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 228.61 | 227.62 | 230.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 226.32 | 230.46 | 230.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 14:30:00 | 227.74 | 228.07 | 228.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 226.97 | 228.45 | 229.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 228.54 | 228.46 | 229.03 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 233.11 | 229.41 | 229.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 233.11 | 229.41 | 229.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 235.70 | 231.31 | 230.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 242.70 | 243.43 | 239.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 242.70 | 243.43 | 239.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 238.37 | 241.99 | 239.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:00:00 | 238.37 | 241.99 | 239.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 238.97 | 241.38 | 239.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 238.92 | 241.38 | 239.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 237.22 | 241.70 | 241.10 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 11:15:00 | 237.98 | 240.36 | 240.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 12:15:00 | 237.42 | 239.77 | 240.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 243.60 | 239.73 | 239.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 243.60 | 239.73 | 239.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 243.60 | 239.73 | 239.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:45:00 | 243.75 | 239.73 | 239.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 242.45 | 240.28 | 240.22 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 13:15:00 | 239.82 | 240.94 | 241.01 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 15:15:00 | 241.51 | 241.11 | 241.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 243.03 | 241.49 | 241.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 237.19 | 241.67 | 241.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 09:15:00 | 237.19 | 241.67 | 241.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 237.19 | 241.67 | 241.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:45:00 | 236.49 | 241.67 | 241.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2026-04-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 10:15:00 | 238.06 | 240.95 | 241.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 15:15:00 | 236.60 | 238.75 | 239.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 10:15:00 | 236.08 | 235.69 | 237.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-22 10:45:00 | 236.31 | 235.69 | 237.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 239.71 | 236.50 | 237.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 12:00:00 | 239.71 | 236.50 | 237.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 240.00 | 237.20 | 237.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 12:45:00 | 240.00 | 237.20 | 237.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 245.88 | 239.45 | 238.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 12:15:00 | 249.23 | 243.11 | 240.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 10:15:00 | 244.20 | 245.93 | 243.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-24 11:00:00 | 244.20 | 245.93 | 243.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 251.42 | 252.87 | 251.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 251.42 | 252.87 | 251.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 250.62 | 252.42 | 251.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 250.62 | 252.42 | 251.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 250.70 | 252.07 | 251.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 245.34 | 252.07 | 251.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 243.56 | 249.06 | 249.76 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 251.55 | 249.32 | 249.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 253.70 | 250.97 | 250.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 250.44 | 250.87 | 250.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 10:00:00 | 250.44 | 250.87 | 250.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 248.33 | 250.36 | 249.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:45:00 | 247.61 | 250.36 | 249.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 249.15 | 250.12 | 249.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 13:00:00 | 249.65 | 250.02 | 249.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 13:15:00 | 248.17 | 249.65 | 249.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 13:15:00 | 248.17 | 249.65 | 249.66 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 251.12 | 249.67 | 249.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 15:15:00 | 252.70 | 251.09 | 250.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 10:15:00 | 251.06 | 251.16 | 250.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 11:00:00 | 251.06 | 251.16 | 250.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 252.29 | 251.39 | 250.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:30:00 | 250.65 | 251.39 | 250.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 251.40 | 251.49 | 250.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 13:45:00 | 251.42 | 251.49 | 250.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 250.92 | 251.38 | 250.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 14:45:00 | 250.57 | 251.38 | 250.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 251.15 | 251.33 | 250.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:15:00 | 249.89 | 251.33 | 250.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 250.30 | 251.12 | 250.87 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 11:15:00 | 249.80 | 250.67 | 250.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 12:15:00 | 248.91 | 250.32 | 250.53 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-17 09:15:00 | 356.90 | 2024-05-27 12:15:00 | 361.00 | STOP_HIT | 1.00 | 1.15% |
| BUY | retest2 | 2024-06-11 09:15:00 | 351.95 | 2024-06-19 15:15:00 | 358.70 | STOP_HIT | 1.00 | 1.92% |
| BUY | retest2 | 2024-06-12 09:15:00 | 352.85 | 2024-06-19 15:15:00 | 358.70 | STOP_HIT | 1.00 | 1.66% |
| SELL | retest2 | 2024-06-28 11:30:00 | 357.25 | 2024-06-28 13:15:00 | 357.85 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2024-07-03 12:15:00 | 351.70 | 2024-07-12 09:15:00 | 351.95 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2024-07-04 13:15:00 | 351.95 | 2024-07-12 09:15:00 | 351.95 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2024-07-08 09:15:00 | 351.70 | 2024-07-12 09:15:00 | 351.95 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2024-07-09 10:00:00 | 350.95 | 2024-07-12 10:15:00 | 352.40 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2024-07-10 10:15:00 | 346.35 | 2024-07-12 10:15:00 | 352.40 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-07-10 11:00:00 | 347.40 | 2024-07-12 10:15:00 | 352.40 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-07-10 13:45:00 | 347.15 | 2024-07-12 10:15:00 | 352.40 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-07-23 12:15:00 | 331.30 | 2024-08-05 09:15:00 | 320.58 | PARTIAL | 0.50 | 3.24% |
| SELL | retest2 | 2024-07-23 12:15:00 | 331.30 | 2024-08-06 09:15:00 | 320.65 | STOP_HIT | 0.50 | 3.21% |
| SELL | retest2 | 2024-07-24 10:45:00 | 337.45 | 2024-08-06 13:15:00 | 314.74 | PARTIAL | 0.50 | 6.73% |
| SELL | retest2 | 2024-07-24 10:45:00 | 337.45 | 2024-08-07 11:15:00 | 319.30 | STOP_HIT | 0.50 | 5.38% |
| BUY | retest2 | 2024-08-22 09:15:00 | 332.10 | 2024-08-22 14:15:00 | 329.60 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-08-22 10:00:00 | 332.20 | 2024-08-22 14:15:00 | 329.60 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-08-22 10:30:00 | 332.20 | 2024-08-22 14:15:00 | 329.60 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-08-26 10:45:00 | 324.25 | 2024-08-30 09:15:00 | 329.55 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-08-29 10:45:00 | 328.95 | 2024-08-30 09:15:00 | 329.55 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2024-09-12 09:15:00 | 346.85 | 2024-09-12 11:15:00 | 343.20 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-10-03 09:15:00 | 345.95 | 2024-10-08 15:15:00 | 344.00 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest2 | 2024-10-21 12:00:00 | 328.55 | 2024-10-22 15:15:00 | 312.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 14:00:00 | 328.70 | 2024-10-22 15:15:00 | 312.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:00:00 | 328.55 | 2024-10-23 09:15:00 | 319.95 | STOP_HIT | 0.50 | 2.62% |
| SELL | retest2 | 2024-10-21 14:00:00 | 328.70 | 2024-10-23 09:15:00 | 319.95 | STOP_HIT | 0.50 | 2.66% |
| BUY | retest2 | 2024-10-31 09:30:00 | 322.25 | 2024-11-04 11:15:00 | 320.05 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-10-31 11:45:00 | 322.40 | 2024-11-04 11:15:00 | 320.05 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-10-31 15:00:00 | 322.65 | 2024-11-04 11:15:00 | 320.05 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-11-01 18:00:00 | 325.90 | 2024-11-04 11:15:00 | 320.05 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest1 | 2024-11-12 14:45:00 | 311.55 | 2024-11-14 09:15:00 | 316.75 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2024-11-29 09:15:00 | 330.95 | 2024-12-09 09:15:00 | 336.95 | STOP_HIT | 1.00 | 1.81% |
| SELL | retest2 | 2024-12-27 11:45:00 | 305.50 | 2024-12-30 14:15:00 | 308.15 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-12-27 14:45:00 | 305.35 | 2024-12-30 14:15:00 | 308.15 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2024-12-30 15:15:00 | 304.80 | 2025-01-01 14:15:00 | 304.35 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-01-08 12:00:00 | 296.50 | 2025-01-10 13:15:00 | 282.10 | PARTIAL | 0.50 | 4.86% |
| SELL | retest2 | 2025-01-08 12:30:00 | 296.95 | 2025-01-10 13:15:00 | 282.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 13:00:00 | 296.95 | 2025-01-10 14:15:00 | 281.68 | PARTIAL | 0.50 | 5.14% |
| SELL | retest2 | 2025-01-09 09:15:00 | 296.40 | 2025-01-10 14:15:00 | 281.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 12:00:00 | 296.50 | 2025-01-13 13:15:00 | 266.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-08 12:30:00 | 296.95 | 2025-01-13 13:15:00 | 267.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-08 13:00:00 | 296.95 | 2025-01-13 13:15:00 | 267.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-09 09:15:00 | 296.40 | 2025-01-13 13:15:00 | 266.76 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-01-17 14:15:00 | 278.75 | 2025-01-20 09:15:00 | 274.35 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-01-23 11:15:00 | 262.85 | 2025-01-24 09:15:00 | 249.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 11:15:00 | 262.85 | 2025-01-27 09:15:00 | 236.57 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-04 10:30:00 | 236.00 | 2025-02-04 12:15:00 | 241.61 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2025-02-07 11:15:00 | 252.33 | 2025-02-10 09:15:00 | 245.29 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest1 | 2025-03-20 09:15:00 | 232.20 | 2025-03-21 13:15:00 | 229.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-03-24 09:15:00 | 232.30 | 2025-03-25 10:15:00 | 228.35 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-04-28 09:15:00 | 254.69 | 2025-05-06 09:15:00 | 254.85 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-05-08 10:30:00 | 254.90 | 2025-05-09 10:15:00 | 242.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 11:30:00 | 254.85 | 2025-05-09 10:15:00 | 242.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 13:15:00 | 254.85 | 2025-05-09 10:15:00 | 242.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 10:30:00 | 254.90 | 2025-05-09 14:15:00 | 248.50 | STOP_HIT | 0.50 | 2.51% |
| SELL | retest2 | 2025-05-08 11:30:00 | 254.85 | 2025-05-09 14:15:00 | 248.50 | STOP_HIT | 0.50 | 2.49% |
| SELL | retest2 | 2025-05-08 13:15:00 | 254.85 | 2025-05-09 14:15:00 | 248.50 | STOP_HIT | 0.50 | 2.49% |
| BUY | retest1 | 2025-05-14 15:00:00 | 267.80 | 2025-05-20 09:15:00 | 273.65 | STOP_HIT | 1.00 | 2.18% |
| BUY | retest1 | 2025-05-15 10:00:00 | 267.40 | 2025-05-20 09:15:00 | 273.65 | STOP_HIT | 1.00 | 2.34% |
| BUY | retest1 | 2025-05-15 10:30:00 | 268.60 | 2025-05-20 09:15:00 | 273.65 | STOP_HIT | 1.00 | 1.88% |
| BUY | retest2 | 2025-05-23 09:15:00 | 276.60 | 2025-05-30 14:15:00 | 286.30 | STOP_HIT | 1.00 | 3.51% |
| SELL | retest2 | 2025-06-17 12:45:00 | 290.80 | 2025-06-20 13:15:00 | 293.25 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-06-17 13:15:00 | 290.90 | 2025-06-20 13:15:00 | 293.25 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-06-17 14:30:00 | 290.50 | 2025-06-20 13:15:00 | 293.25 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-06-18 10:30:00 | 291.00 | 2025-06-20 13:15:00 | 293.25 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-06-20 12:15:00 | 290.55 | 2025-06-20 13:15:00 | 293.25 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-06-23 11:30:00 | 293.15 | 2025-06-27 10:15:00 | 322.46 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-23 12:00:00 | 293.20 | 2025-06-27 10:15:00 | 322.52 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-18 10:15:00 | 314.55 | 2025-07-24 10:15:00 | 318.05 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-07-22 10:00:00 | 314.60 | 2025-07-24 10:15:00 | 318.05 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-07-28 12:45:00 | 312.55 | 2025-07-29 11:15:00 | 317.05 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-07-29 10:00:00 | 312.75 | 2025-07-29 11:15:00 | 317.05 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-08-01 11:45:00 | 332.15 | 2025-08-06 11:15:00 | 327.95 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-08-04 09:15:00 | 334.10 | 2025-08-06 11:15:00 | 327.95 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-08-04 10:45:00 | 331.80 | 2025-08-06 11:15:00 | 327.95 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-08-04 12:15:00 | 332.10 | 2025-08-06 11:15:00 | 327.95 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-08-08 09:15:00 | 324.05 | 2025-08-11 14:15:00 | 328.60 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-08-18 09:15:00 | 331.25 | 2025-08-20 14:15:00 | 328.10 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-08-18 12:15:00 | 328.95 | 2025-08-20 14:15:00 | 328.10 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-08-18 13:00:00 | 329.75 | 2025-08-20 14:15:00 | 328.10 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-08-20 09:30:00 | 329.10 | 2025-08-20 14:15:00 | 328.10 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-08-29 15:15:00 | 311.40 | 2025-09-08 13:15:00 | 312.40 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-09-03 10:15:00 | 311.35 | 2025-09-08 13:15:00 | 312.40 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-09-03 12:45:00 | 311.20 | 2025-09-08 13:15:00 | 312.40 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-09-04 12:15:00 | 311.40 | 2025-09-08 13:15:00 | 312.40 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-09-05 11:30:00 | 310.30 | 2025-09-08 13:15:00 | 312.40 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-09-05 14:00:00 | 310.20 | 2025-09-08 13:15:00 | 312.40 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-09-18 09:15:00 | 316.65 | 2025-09-22 14:15:00 | 313.90 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-09-19 15:00:00 | 317.30 | 2025-09-22 14:15:00 | 313.90 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest1 | 2025-09-29 11:30:00 | 297.85 | 2025-10-01 11:15:00 | 297.80 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2025-10-10 11:00:00 | 309.85 | 2025-10-13 09:15:00 | 306.20 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-10-10 12:30:00 | 309.60 | 2025-10-13 09:15:00 | 306.20 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-11-14 10:15:00 | 314.30 | 2025-11-18 09:15:00 | 308.40 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-11-14 10:45:00 | 315.35 | 2025-11-18 09:15:00 | 308.40 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-11-14 12:00:00 | 314.30 | 2025-11-18 09:15:00 | 308.40 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-11-14 13:00:00 | 313.90 | 2025-11-18 09:15:00 | 308.40 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-12-04 13:30:00 | 300.90 | 2025-12-05 14:15:00 | 303.85 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-12-05 09:30:00 | 300.70 | 2025-12-05 14:15:00 | 303.85 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-12-05 10:15:00 | 301.30 | 2025-12-05 14:15:00 | 303.85 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-12-26 09:15:00 | 299.55 | 2025-12-26 12:15:00 | 297.20 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2026-01-06 09:30:00 | 300.90 | 2026-01-06 11:15:00 | 294.65 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2026-01-13 10:45:00 | 284.30 | 2026-01-20 09:15:00 | 271.22 | PARTIAL | 0.50 | 4.60% |
| SELL | retest2 | 2026-01-13 11:30:00 | 284.15 | 2026-01-20 09:15:00 | 270.65 | PARTIAL | 0.50 | 4.75% |
| SELL | retest2 | 2026-01-13 12:30:00 | 284.10 | 2026-01-20 11:15:00 | 270.08 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2026-01-13 13:45:00 | 283.95 | 2026-01-20 11:15:00 | 269.94 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2026-01-14 14:15:00 | 285.50 | 2026-01-20 11:15:00 | 269.89 | PARTIAL | 0.50 | 5.47% |
| SELL | retest2 | 2026-01-16 09:30:00 | 284.90 | 2026-01-20 11:15:00 | 269.75 | PARTIAL | 0.50 | 5.32% |
| SELL | retest2 | 2026-01-13 10:45:00 | 284.30 | 2026-01-22 09:15:00 | 266.75 | STOP_HIT | 0.50 | 6.17% |
| SELL | retest2 | 2026-01-13 11:30:00 | 284.15 | 2026-01-22 09:15:00 | 266.75 | STOP_HIT | 0.50 | 6.12% |
| SELL | retest2 | 2026-01-13 12:30:00 | 284.10 | 2026-01-22 09:15:00 | 266.75 | STOP_HIT | 0.50 | 6.11% |
| SELL | retest2 | 2026-01-13 13:45:00 | 283.95 | 2026-01-22 09:15:00 | 266.75 | STOP_HIT | 0.50 | 6.06% |
| SELL | retest2 | 2026-01-14 14:15:00 | 285.50 | 2026-01-22 09:15:00 | 266.75 | STOP_HIT | 0.50 | 6.57% |
| SELL | retest2 | 2026-01-16 09:30:00 | 284.90 | 2026-01-22 09:15:00 | 266.75 | STOP_HIT | 0.50 | 6.37% |
| BUY | retest2 | 2026-02-11 11:15:00 | 270.65 | 2026-02-12 09:15:00 | 267.85 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2026-02-11 13:15:00 | 270.65 | 2026-02-12 09:15:00 | 267.85 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-02-23 10:15:00 | 257.15 | 2026-02-27 10:15:00 | 258.05 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2026-02-24 09:15:00 | 255.35 | 2026-02-27 10:15:00 | 258.05 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-02-25 09:30:00 | 257.10 | 2026-02-27 13:15:00 | 257.60 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2026-02-25 12:45:00 | 257.20 | 2026-03-02 13:15:00 | 244.29 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2026-02-26 14:15:00 | 255.50 | 2026-03-02 13:15:00 | 244.25 | PARTIAL | 0.50 | 4.41% |
| SELL | retest2 | 2026-02-27 09:45:00 | 255.70 | 2026-03-02 13:15:00 | 244.34 | PARTIAL | 0.50 | 4.44% |
| SELL | retest2 | 2026-02-27 10:45:00 | 255.40 | 2026-03-04 09:15:00 | 242.58 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2026-02-27 15:00:00 | 255.00 | 2026-03-04 09:15:00 | 242.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 12:45:00 | 257.20 | 2026-03-05 14:15:00 | 242.50 | STOP_HIT | 0.50 | 5.72% |
| SELL | retest2 | 2026-02-26 14:15:00 | 255.50 | 2026-03-05 14:15:00 | 242.50 | STOP_HIT | 0.50 | 5.09% |
| SELL | retest2 | 2026-02-27 09:45:00 | 255.70 | 2026-03-05 14:15:00 | 242.50 | STOP_HIT | 0.50 | 5.16% |
| SELL | retest2 | 2026-02-27 10:45:00 | 255.40 | 2026-03-05 14:15:00 | 242.50 | STOP_HIT | 0.50 | 5.05% |
| SELL | retest2 | 2026-02-27 15:00:00 | 255.00 | 2026-03-05 14:15:00 | 242.50 | STOP_HIT | 0.50 | 4.90% |
| SELL | retest2 | 2026-03-10 11:15:00 | 232.30 | 2026-03-11 09:15:00 | 238.90 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2026-03-10 11:45:00 | 232.25 | 2026-03-11 09:15:00 | 238.90 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2026-03-17 11:15:00 | 235.20 | 2026-03-17 13:15:00 | 238.05 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-03-17 12:15:00 | 234.85 | 2026-03-17 13:15:00 | 238.05 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-04-01 10:15:00 | 228.61 | 2026-04-06 12:15:00 | 233.11 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-04-02 09:15:00 | 226.32 | 2026-04-06 12:15:00 | 233.11 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2026-04-02 14:30:00 | 227.74 | 2026-04-06 12:15:00 | 233.11 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2026-04-06 09:15:00 | 226.97 | 2026-04-06 12:15:00 | 233.11 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2026-05-05 13:00:00 | 249.65 | 2026-05-05 13:15:00 | 248.17 | STOP_HIT | 1.00 | -0.59% |
