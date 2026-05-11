# Birlasoft Ltd. (BSOFT)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 362.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 223 |
| ALERT1 | 154 |
| ALERT2 | 149 |
| ALERT2_SKIP | 75 |
| ALERT3 | 417 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 157 |
| PARTIAL | 18 |
| TARGET_HIT | 8 |
| STOP_HIT | 155 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 181 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 65 / 116
- **Target hits / Stop hits / Partials:** 8 / 155 / 18
- **Avg / median % per leg:** 0.36% / -0.77%
- **Sum % (uncompounded):** 65.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 63 | 17 | 27.0% | 6 | 56 | 1 | 0.38% | 23.7% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 1 | 5 | 1 | 1.64% | 11.5% |
| BUY @ 3rd Alert (retest2) | 56 | 15 | 26.8% | 5 | 51 | 0 | 0.22% | 12.2% |
| SELL (all) | 118 | 48 | 40.7% | 2 | 99 | 17 | 0.35% | 41.8% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.24% | 1.2% |
| SELL @ 3rd Alert (retest2) | 117 | 47 | 40.2% | 2 | 98 | 17 | 0.35% | 40.6% |
| retest1 (combined) | 8 | 3 | 37.5% | 1 | 6 | 1 | 1.59% | 12.8% |
| retest2 (combined) | 173 | 62 | 35.8% | 7 | 149 | 17 | 0.30% | 52.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 12:15:00 | 330.30 | 333.49 | 333.55 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 13:15:00 | 338.25 | 333.61 | 333.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 09:15:00 | 339.50 | 335.49 | 334.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-26 15:15:00 | 338.70 | 338.87 | 336.87 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-29 09:15:00 | 340.10 | 338.87 | 336.87 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 09:15:00 | 337.90 | 338.67 | 336.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-29 09:45:00 | 337.40 | 338.67 | 336.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 10:15:00 | 337.60 | 338.46 | 337.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-29 11:15:00 | 338.60 | 338.46 | 337.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-30 10:15:00 | 337.45 | 338.86 | 338.07 | SL hit (close<ema400) qty=1.00 sl=338.07 alert=retest1 |

### Cycle 3 — SELL (started 2023-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 14:15:00 | 336.20 | 337.52 | 337.62 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 09:15:00 | 342.40 | 338.26 | 337.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 09:15:00 | 346.55 | 342.52 | 340.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-01 15:15:00 | 343.15 | 344.45 | 342.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 15:15:00 | 343.15 | 344.45 | 342.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 15:15:00 | 343.15 | 344.45 | 342.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 09:15:00 | 349.40 | 344.45 | 342.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-06 09:15:00 | 339.85 | 346.29 | 346.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 09:15:00 | 339.85 | 346.29 | 346.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-06 10:15:00 | 333.80 | 343.79 | 345.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-06 13:15:00 | 343.20 | 342.00 | 343.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-06 13:15:00 | 343.20 | 342.00 | 343.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 13:15:00 | 343.20 | 342.00 | 343.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-06 14:00:00 | 343.20 | 342.00 | 343.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 14:15:00 | 343.85 | 342.37 | 343.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-07 10:15:00 | 341.90 | 342.50 | 343.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-07 12:00:00 | 341.80 | 342.66 | 343.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-07 13:00:00 | 341.75 | 342.48 | 343.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-07 13:30:00 | 341.70 | 342.27 | 343.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 338.20 | 332.90 | 335.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 09:30:00 | 339.00 | 332.90 | 335.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 10:15:00 | 337.90 | 333.90 | 335.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 10:45:00 | 338.00 | 333.90 | 335.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 11:15:00 | 339.00 | 334.92 | 335.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 11:45:00 | 339.40 | 334.92 | 335.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-06-12 14:15:00 | 337.95 | 336.53 | 336.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2023-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 14:15:00 | 337.95 | 336.53 | 336.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 09:15:00 | 339.75 | 337.30 | 336.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-13 11:15:00 | 337.20 | 337.36 | 336.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-13 12:00:00 | 337.20 | 337.36 | 336.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 12:15:00 | 335.85 | 337.06 | 336.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-13 12:30:00 | 335.45 | 337.06 | 336.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 13:15:00 | 336.45 | 336.94 | 336.82 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-13 14:15:00 | 335.95 | 336.74 | 336.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-14 09:15:00 | 334.55 | 336.26 | 336.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-14 10:15:00 | 336.95 | 336.40 | 336.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-14 10:15:00 | 336.95 | 336.40 | 336.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 10:15:00 | 336.95 | 336.40 | 336.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-14 11:00:00 | 336.95 | 336.40 | 336.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 11:15:00 | 337.30 | 336.58 | 336.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-14 12:00:00 | 337.30 | 336.58 | 336.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2023-06-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 12:15:00 | 337.95 | 336.85 | 336.75 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-06-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 12:15:00 | 335.00 | 336.73 | 336.89 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 09:15:00 | 342.25 | 337.17 | 336.96 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 12:15:00 | 336.00 | 337.41 | 337.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-19 13:15:00 | 335.00 | 336.93 | 337.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 11:15:00 | 336.75 | 336.37 | 336.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 11:15:00 | 336.75 | 336.37 | 336.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 11:15:00 | 336.75 | 336.37 | 336.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-20 11:45:00 | 337.85 | 336.37 | 336.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 12:15:00 | 337.05 | 336.50 | 336.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-20 13:00:00 | 337.05 | 336.50 | 336.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 13:15:00 | 335.60 | 336.32 | 336.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-20 13:30:00 | 336.80 | 336.32 | 336.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 14:15:00 | 337.40 | 336.54 | 336.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-20 15:00:00 | 337.40 | 336.54 | 336.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 15:15:00 | 337.80 | 336.79 | 336.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 09:15:00 | 341.75 | 336.79 | 336.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2023-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 09:15:00 | 340.60 | 337.55 | 337.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-22 14:15:00 | 343.50 | 340.88 | 339.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-23 09:15:00 | 341.25 | 341.59 | 340.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-23 09:15:00 | 341.25 | 341.59 | 340.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 09:15:00 | 341.25 | 341.59 | 340.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 10:00:00 | 341.25 | 341.59 | 340.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 15:15:00 | 341.00 | 343.95 | 342.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-26 09:15:00 | 340.40 | 343.95 | 342.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 335.75 | 342.31 | 341.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-26 10:00:00 | 335.75 | 342.31 | 341.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 10:15:00 | 338.30 | 341.51 | 341.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-26 11:15:00 | 338.95 | 341.51 | 341.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-26 11:15:00 | 339.15 | 341.04 | 341.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2023-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-26 11:15:00 | 339.15 | 341.04 | 341.18 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 09:15:00 | 352.60 | 342.61 | 341.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 10:15:00 | 354.75 | 345.04 | 342.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 13:15:00 | 350.70 | 350.95 | 348.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-28 14:00:00 | 350.70 | 350.95 | 348.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 11:15:00 | 354.90 | 356.50 | 354.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 11:45:00 | 354.30 | 356.50 | 354.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 14:15:00 | 354.45 | 356.10 | 354.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 15:00:00 | 354.45 | 356.10 | 354.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 15:15:00 | 354.65 | 355.81 | 354.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 09:15:00 | 352.95 | 355.81 | 354.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 352.80 | 355.21 | 354.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 09:30:00 | 352.40 | 355.21 | 354.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 10:15:00 | 352.60 | 354.69 | 354.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 10:30:00 | 353.15 | 354.69 | 354.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2023-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 12:15:00 | 351.90 | 353.80 | 354.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-06 11:15:00 | 350.05 | 352.26 | 352.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-06 14:15:00 | 353.30 | 352.40 | 352.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-06 14:15:00 | 353.30 | 352.40 | 352.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 14:15:00 | 353.30 | 352.40 | 352.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-06 15:00:00 | 353.30 | 352.40 | 352.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 15:15:00 | 355.25 | 352.97 | 353.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-07 09:15:00 | 357.25 | 352.97 | 353.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2023-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-07 09:15:00 | 359.25 | 354.23 | 353.61 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 09:15:00 | 349.35 | 353.74 | 353.84 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 14:15:00 | 355.15 | 352.09 | 351.79 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 11:15:00 | 349.55 | 351.40 | 351.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-12 15:15:00 | 348.90 | 350.19 | 350.89 | Break + close below crossover candle low |

### Cycle 20 — BUY (started 2023-07-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 09:15:00 | 361.20 | 352.39 | 351.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-13 10:15:00 | 365.05 | 354.92 | 353.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 11:15:00 | 383.80 | 384.39 | 378.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-18 12:00:00 | 383.80 | 384.39 | 378.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 392.30 | 391.75 | 389.73 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 14:15:00 | 386.00 | 388.49 | 388.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-24 09:15:00 | 381.15 | 386.63 | 387.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 13:15:00 | 385.65 | 385.19 | 386.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-24 13:30:00 | 385.80 | 385.19 | 386.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 10:15:00 | 383.50 | 380.34 | 382.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 11:00:00 | 383.50 | 380.34 | 382.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 11:15:00 | 382.95 | 380.86 | 382.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-26 12:15:00 | 381.40 | 380.86 | 382.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-26 12:45:00 | 380.85 | 380.73 | 382.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-26 14:30:00 | 381.25 | 380.56 | 381.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-27 09:15:00 | 384.80 | 381.01 | 381.72 | SL hit (close>static) qty=1.00 sl=383.50 alert=retest2 |

### Cycle 22 — BUY (started 2023-07-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 11:15:00 | 385.40 | 382.25 | 382.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-27 12:15:00 | 402.00 | 386.20 | 383.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 09:15:00 | 430.30 | 430.63 | 423.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 10:15:00 | 424.30 | 429.37 | 423.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 424.30 | 429.37 | 423.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 10:45:00 | 423.05 | 429.37 | 423.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 423.55 | 428.20 | 423.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:30:00 | 422.90 | 428.20 | 423.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 12:15:00 | 424.40 | 427.44 | 423.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 13:15:00 | 421.60 | 427.44 | 423.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 13:15:00 | 418.90 | 425.73 | 423.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 14:00:00 | 418.90 | 425.73 | 423.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 14:15:00 | 425.55 | 425.70 | 423.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 09:15:00 | 429.65 | 425.56 | 423.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 10:15:00 | 427.40 | 425.71 | 423.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 10:45:00 | 429.95 | 426.71 | 424.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-10 15:15:00 | 445.75 | 447.93 | 447.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2023-08-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 15:15:00 | 445.75 | 447.93 | 447.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 09:15:00 | 444.10 | 447.16 | 447.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-11 14:15:00 | 446.55 | 445.26 | 446.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 14:15:00 | 446.55 | 445.26 | 446.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 14:15:00 | 446.55 | 445.26 | 446.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-11 15:00:00 | 446.55 | 445.26 | 446.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 15:15:00 | 445.00 | 445.21 | 446.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-14 09:15:00 | 441.50 | 445.21 | 446.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-14 09:15:00 | 446.80 | 445.53 | 446.21 | SL hit (close>static) qty=1.00 sl=446.65 alert=retest2 |

### Cycle 24 — BUY (started 2023-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-14 15:15:00 | 448.00 | 446.45 | 446.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-16 09:15:00 | 449.20 | 447.00 | 446.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-18 09:15:00 | 457.60 | 462.24 | 457.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 09:15:00 | 457.60 | 462.24 | 457.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 09:15:00 | 457.60 | 462.24 | 457.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 10:00:00 | 457.60 | 462.24 | 457.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 10:15:00 | 459.70 | 461.73 | 457.27 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-21 09:15:00 | 453.20 | 455.25 | 455.35 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 10:15:00 | 457.00 | 455.60 | 455.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 14:15:00 | 462.40 | 457.66 | 456.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 12:15:00 | 466.20 | 466.26 | 463.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-23 13:00:00 | 466.20 | 466.26 | 463.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 09:15:00 | 528.85 | 526.19 | 522.15 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-09-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 09:15:00 | 518.65 | 521.12 | 521.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-07 10:15:00 | 517.25 | 520.35 | 520.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-08 09:15:00 | 516.90 | 515.75 | 517.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 09:15:00 | 516.90 | 515.75 | 517.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 09:15:00 | 516.90 | 515.75 | 517.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-08 09:30:00 | 520.15 | 515.75 | 517.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 10:15:00 | 517.65 | 516.13 | 517.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-08 11:00:00 | 517.65 | 516.13 | 517.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 11:15:00 | 517.00 | 516.31 | 517.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-08 12:15:00 | 518.25 | 516.31 | 517.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 12:15:00 | 517.30 | 516.50 | 517.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-08 12:45:00 | 518.05 | 516.50 | 517.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 13:15:00 | 518.00 | 516.80 | 517.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-08 14:30:00 | 516.50 | 517.10 | 517.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-11 09:15:00 | 514.90 | 517.12 | 517.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-12 09:30:00 | 510.85 | 510.06 | 513.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-13 10:15:00 | 490.67 | 498.03 | 504.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-13 10:15:00 | 489.15 | 498.03 | 504.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-09-14 09:15:00 | 506.25 | 497.75 | 501.12 | SL hit (close>ema200) qty=0.50 sl=497.75 alert=retest2 |

### Cycle 28 — BUY (started 2023-09-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 15:15:00 | 505.25 | 502.87 | 502.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 09:15:00 | 507.50 | 503.80 | 503.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 14:15:00 | 502.15 | 504.56 | 503.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 14:15:00 | 502.15 | 504.56 | 503.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 14:15:00 | 502.15 | 504.56 | 503.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-15 15:00:00 | 502.15 | 504.56 | 503.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 15:15:00 | 505.00 | 504.65 | 503.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 09:15:00 | 498.30 | 504.65 | 503.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2023-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 09:15:00 | 495.75 | 502.87 | 503.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 13:15:00 | 492.50 | 497.26 | 500.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 10:15:00 | 491.60 | 487.99 | 491.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 10:15:00 | 491.60 | 487.99 | 491.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 10:15:00 | 491.60 | 487.99 | 491.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 11:00:00 | 491.60 | 487.99 | 491.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 11:15:00 | 492.45 | 488.88 | 491.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 11:45:00 | 494.45 | 488.88 | 491.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 12:15:00 | 492.90 | 489.68 | 491.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 12:45:00 | 492.65 | 489.68 | 491.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 13:15:00 | 492.70 | 490.29 | 491.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 13:45:00 | 494.50 | 490.29 | 491.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 15:15:00 | 491.95 | 491.05 | 492.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 09:15:00 | 486.00 | 491.05 | 492.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 13:45:00 | 488.45 | 488.34 | 489.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-25 09:45:00 | 488.15 | 487.86 | 489.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-25 11:15:00 | 493.65 | 489.74 | 489.98 | SL hit (close>static) qty=1.00 sl=493.00 alert=retest2 |

### Cycle 30 — BUY (started 2023-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 10:15:00 | 490.80 | 490.03 | 489.99 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 11:15:00 | 489.30 | 489.88 | 489.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-26 12:15:00 | 485.10 | 488.93 | 489.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-27 09:15:00 | 487.10 | 486.60 | 488.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-27 09:15:00 | 487.10 | 486.60 | 488.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 487.10 | 486.60 | 488.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 09:45:00 | 487.35 | 486.60 | 488.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 10:15:00 | 486.30 | 486.54 | 487.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 10:45:00 | 488.30 | 486.54 | 487.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 11:15:00 | 485.05 | 486.24 | 487.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 11:45:00 | 485.55 | 486.24 | 487.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 12:15:00 | 489.65 | 486.92 | 487.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 13:00:00 | 489.65 | 486.92 | 487.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 13:15:00 | 490.90 | 487.72 | 488.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 13:45:00 | 491.40 | 487.72 | 488.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2023-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 15:15:00 | 491.50 | 488.90 | 488.57 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-09-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 11:15:00 | 483.20 | 487.59 | 488.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 13:15:00 | 479.80 | 485.20 | 486.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 11:15:00 | 481.30 | 480.54 | 483.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-29 11:45:00 | 481.65 | 480.54 | 483.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 12:15:00 | 487.00 | 481.83 | 483.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 12:30:00 | 485.25 | 481.83 | 483.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 13:15:00 | 487.45 | 482.95 | 484.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 14:00:00 | 487.45 | 482.95 | 484.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 15:15:00 | 484.00 | 483.11 | 483.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-03 09:15:00 | 487.00 | 483.11 | 483.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 488.30 | 484.14 | 484.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-03 09:45:00 | 489.75 | 484.14 | 484.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2023-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 10:15:00 | 493.90 | 486.10 | 485.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 11:15:00 | 501.65 | 489.21 | 486.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 09:15:00 | 492.20 | 493.32 | 490.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 09:15:00 | 492.20 | 493.32 | 490.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 492.20 | 493.32 | 490.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 09:15:00 | 504.75 | 491.13 | 490.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-10-17 14:15:00 | 555.23 | 548.80 | 545.52 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2023-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 11:15:00 | 544.25 | 548.42 | 548.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 12:15:00 | 541.50 | 547.03 | 548.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 14:15:00 | 549.55 | 546.66 | 547.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 14:15:00 | 549.55 | 546.66 | 547.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 14:15:00 | 549.55 | 546.66 | 547.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 15:00:00 | 549.55 | 546.66 | 547.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 15:15:00 | 546.40 | 546.61 | 547.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 09:15:00 | 542.00 | 546.61 | 547.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 514.90 | 524.56 | 530.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-26 12:15:00 | 521.25 | 521.12 | 527.48 | SL hit (close>ema200) qty=0.50 sl=521.12 alert=retest2 |

### Cycle 36 — BUY (started 2023-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 11:15:00 | 539.25 | 530.77 | 529.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-27 13:15:00 | 539.90 | 533.75 | 531.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-30 12:15:00 | 537.30 | 537.74 | 535.03 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-30 14:15:00 | 539.90 | 537.92 | 535.36 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-01 09:15:00 | 566.89 | 547.62 | 542.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2023-11-06 09:15:00 | 593.89 | 587.75 | 579.21 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 37 — SELL (started 2023-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 13:15:00 | 582.10 | 585.96 | 586.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-08 14:15:00 | 581.15 | 585.00 | 585.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-09 09:15:00 | 586.00 | 584.72 | 585.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 09:15:00 | 586.00 | 584.72 | 585.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 586.00 | 584.72 | 585.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-09 10:00:00 | 586.00 | 584.72 | 585.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 10:15:00 | 586.25 | 585.03 | 585.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-09 11:00:00 | 586.25 | 585.03 | 585.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 11:15:00 | 584.00 | 584.82 | 585.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-09 12:15:00 | 583.40 | 584.82 | 585.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-09 14:15:00 | 583.35 | 584.46 | 585.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-09 15:00:00 | 582.95 | 584.16 | 584.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-10 09:45:00 | 581.20 | 584.00 | 584.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 10:15:00 | 582.40 | 583.68 | 584.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 10:45:00 | 582.10 | 583.68 | 584.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 12:15:00 | 582.55 | 583.34 | 584.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 12:45:00 | 583.90 | 583.34 | 584.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 15:15:00 | 583.95 | 583.38 | 584.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-12 18:15:00 | 589.55 | 583.38 | 584.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-11-12 18:15:00 | 591.65 | 585.03 | 584.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2023-11-12 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-12 18:15:00 | 591.65 | 585.03 | 584.71 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 13:15:00 | 583.60 | 584.53 | 584.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-13 15:15:00 | 583.25 | 584.09 | 584.39 | Break + close below crossover candle low |

### Cycle 40 — BUY (started 2023-11-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 09:15:00 | 607.20 | 588.71 | 586.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 14:15:00 | 611.30 | 600.39 | 593.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 13:15:00 | 620.65 | 622.96 | 617.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-20 13:45:00 | 620.95 | 622.96 | 617.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 10:15:00 | 615.00 | 620.85 | 618.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-21 10:45:00 | 616.00 | 620.85 | 618.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 11:15:00 | 616.00 | 619.88 | 618.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-21 11:30:00 | 614.50 | 619.88 | 618.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 13:15:00 | 618.00 | 619.60 | 618.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-21 14:00:00 | 618.00 | 619.60 | 618.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 14:15:00 | 615.50 | 618.78 | 618.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-21 15:00:00 | 615.50 | 618.78 | 618.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 15:15:00 | 615.55 | 618.14 | 617.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 09:15:00 | 614.25 | 618.14 | 617.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 12:15:00 | 619.95 | 620.25 | 619.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 13:00:00 | 619.95 | 620.25 | 619.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 13:15:00 | 620.95 | 620.39 | 619.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-22 14:45:00 | 624.15 | 621.21 | 619.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-23 13:45:00 | 624.70 | 623.21 | 621.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-23 14:15:00 | 623.80 | 623.21 | 621.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-24 09:15:00 | 613.75 | 621.17 | 621.10 | SL hit (close<static) qty=1.00 sl=619.00 alert=retest2 |

### Cycle 41 — SELL (started 2023-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 10:15:00 | 617.85 | 620.50 | 620.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 09:15:00 | 609.35 | 614.97 | 617.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 14:15:00 | 611.75 | 609.13 | 613.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-28 14:45:00 | 611.35 | 609.13 | 613.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 15:15:00 | 611.90 | 609.69 | 613.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 09:15:00 | 634.45 | 609.69 | 613.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 09:15:00 | 630.80 | 613.91 | 614.61 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2023-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 10:15:00 | 631.00 | 617.33 | 616.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 09:15:00 | 633.05 | 629.56 | 625.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 15:15:00 | 633.00 | 633.30 | 629.64 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 09:15:00 | 637.50 | 633.30 | 629.64 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 12:00:00 | 636.20 | 634.70 | 631.27 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 13:15:00 | 636.85 | 634.83 | 631.64 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 14:30:00 | 637.30 | 635.09 | 632.33 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 09:15:00 | 633.55 | 634.77 | 632.66 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-12-05 10:15:00 | 632.65 | 634.35 | 632.66 | SL hit (close<ema400) qty=1.00 sl=632.66 alert=retest1 |

### Cycle 43 — SELL (started 2023-12-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-28 12:15:00 | 723.25 | 729.93 | 730.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-28 14:15:00 | 721.30 | 727.29 | 729.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-03 11:15:00 | 689.00 | 688.03 | 697.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-03 11:45:00 | 686.45 | 688.03 | 697.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 09:15:00 | 694.65 | 688.53 | 694.02 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2024-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 10:15:00 | 707.00 | 695.70 | 694.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-05 14:15:00 | 717.20 | 702.92 | 698.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 14:15:00 | 708.90 | 711.83 | 706.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-08 15:00:00 | 708.90 | 711.83 | 706.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 10:15:00 | 775.90 | 782.13 | 773.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 10:45:00 | 774.50 | 782.13 | 773.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 11:15:00 | 768.70 | 779.44 | 773.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 12:00:00 | 768.70 | 779.44 | 773.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 12:15:00 | 764.35 | 776.42 | 772.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 13:00:00 | 764.35 | 776.42 | 772.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 14:15:00 | 780.70 | 776.86 | 773.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 14:30:00 | 775.55 | 776.86 | 773.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 788.45 | 779.45 | 775.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 11:30:00 | 796.00 | 784.41 | 778.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-19 09:15:00 | 805.40 | 782.66 | 782.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-01 12:15:00 | 827.20 | 837.73 | 838.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2024-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 12:15:00 | 827.20 | 837.73 | 838.29 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2024-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 11:15:00 | 845.55 | 838.76 | 838.01 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 14:15:00 | 834.90 | 839.12 | 839.64 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 09:15:00 | 854.90 | 841.46 | 840.56 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2024-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 11:15:00 | 830.60 | 844.29 | 844.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-07 12:15:00 | 827.40 | 840.91 | 842.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-08 09:15:00 | 839.85 | 837.43 | 840.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 09:15:00 | 839.85 | 837.43 | 840.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 09:15:00 | 839.85 | 837.43 | 840.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-08 11:00:00 | 835.70 | 837.09 | 839.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-08 13:45:00 | 837.45 | 837.83 | 839.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-08 14:45:00 | 836.35 | 837.90 | 839.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-08 15:15:00 | 837.50 | 837.90 | 839.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 836.75 | 837.60 | 839.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-09 10:30:00 | 827.00 | 834.53 | 837.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-12 10:00:00 | 827.25 | 830.70 | 833.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-12 12:00:00 | 826.95 | 829.25 | 832.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-12 12:45:00 | 828.65 | 829.11 | 832.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 14:15:00 | 825.25 | 828.47 | 831.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-12 14:30:00 | 827.25 | 828.47 | 831.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 12:15:00 | 825.90 | 823.33 | 827.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 13:00:00 | 825.90 | 823.33 | 827.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 13:15:00 | 830.15 | 824.70 | 827.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 13:30:00 | 830.80 | 824.70 | 827.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 14:15:00 | 844.75 | 828.71 | 829.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 15:00:00 | 844.75 | 828.71 | 829.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-02-13 15:15:00 | 842.05 | 831.38 | 830.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2024-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 15:15:00 | 842.05 | 831.38 | 830.33 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2024-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 10:15:00 | 822.65 | 829.13 | 829.46 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 15:15:00 | 838.00 | 829.68 | 829.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 843.95 | 832.53 | 830.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 11:15:00 | 837.95 | 839.90 | 836.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 11:15:00 | 837.95 | 839.90 | 836.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 11:15:00 | 837.95 | 839.90 | 836.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 11:30:00 | 835.85 | 839.90 | 836.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 12:15:00 | 837.15 | 839.35 | 836.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 12:45:00 | 836.35 | 839.35 | 836.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 13:15:00 | 842.50 | 839.98 | 837.31 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2024-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 12:15:00 | 831.45 | 836.35 | 836.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 09:15:00 | 820.15 | 831.29 | 834.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 09:15:00 | 808.75 | 805.65 | 813.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 09:15:00 | 808.75 | 805.65 | 813.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 808.75 | 805.65 | 813.02 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2024-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 09:15:00 | 832.10 | 816.53 | 815.34 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-23 15:15:00 | 810.00 | 815.17 | 815.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 09:15:00 | 794.50 | 811.04 | 813.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 10:15:00 | 791.35 | 788.87 | 797.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-27 10:45:00 | 791.95 | 788.87 | 797.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 12:15:00 | 795.50 | 790.55 | 796.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 12:45:00 | 795.90 | 790.55 | 796.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 14:15:00 | 799.55 | 792.77 | 796.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 15:00:00 | 799.55 | 792.77 | 796.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 15:15:00 | 798.00 | 793.82 | 796.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-28 09:15:00 | 794.55 | 793.82 | 796.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 14:15:00 | 780.50 | 773.46 | 779.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 15:00:00 | 780.50 | 773.46 | 779.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 778.00 | 774.37 | 779.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:15:00 | 787.00 | 774.37 | 779.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 782.20 | 775.94 | 779.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 10:30:00 | 778.00 | 776.21 | 779.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-02 11:15:00 | 787.05 | 780.03 | 779.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2024-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-02 11:15:00 | 787.05 | 780.03 | 779.64 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 09:15:00 | 757.00 | 779.02 | 780.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 10:15:00 | 755.60 | 774.34 | 778.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 13:15:00 | 757.45 | 754.48 | 762.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-06 14:00:00 | 757.45 | 754.48 | 762.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 14:15:00 | 768.50 | 757.29 | 763.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 15:00:00 | 768.50 | 757.29 | 763.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 15:15:00 | 771.70 | 760.17 | 763.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 09:15:00 | 771.95 | 760.17 | 763.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2024-03-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 11:15:00 | 771.75 | 766.72 | 766.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-11 12:15:00 | 778.90 | 771.38 | 769.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-12 10:15:00 | 772.30 | 773.72 | 771.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 10:15:00 | 772.30 | 773.72 | 771.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 10:15:00 | 772.30 | 773.72 | 771.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 10:45:00 | 772.50 | 773.72 | 771.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 11:15:00 | 764.10 | 771.80 | 770.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 12:00:00 | 764.10 | 771.80 | 770.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 12:15:00 | 765.00 | 770.44 | 770.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 12:45:00 | 762.00 | 770.44 | 770.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2024-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 13:15:00 | 763.75 | 769.10 | 769.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 14:15:00 | 759.95 | 767.27 | 768.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 09:15:00 | 745.40 | 741.61 | 752.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 10:00:00 | 745.40 | 741.61 | 752.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 10:15:00 | 753.40 | 743.97 | 752.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 11:00:00 | 753.40 | 743.97 | 752.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 11:15:00 | 756.50 | 746.47 | 752.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 12:00:00 | 756.50 | 746.47 | 752.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 12:15:00 | 758.25 | 748.83 | 753.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 12:30:00 | 755.40 | 748.83 | 753.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 13:15:00 | 748.45 | 748.23 | 751.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 13:45:00 | 746.90 | 748.23 | 751.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 14:15:00 | 755.00 | 749.59 | 751.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 14:45:00 | 756.20 | 749.59 | 751.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 15:15:00 | 754.60 | 750.59 | 751.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 09:15:00 | 749.65 | 750.59 | 751.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-21 14:15:00 | 739.40 | 729.93 | 729.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2024-03-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 14:15:00 | 739.40 | 729.93 | 729.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-26 10:15:00 | 743.05 | 734.23 | 732.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-28 09:15:00 | 746.80 | 756.63 | 749.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 09:15:00 | 746.80 | 756.63 | 749.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 746.80 | 756.63 | 749.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 10:00:00 | 746.80 | 756.63 | 749.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 10:15:00 | 748.15 | 754.94 | 749.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 11:15:00 | 749.15 | 754.94 | 749.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 13:00:00 | 749.50 | 752.78 | 749.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-28 14:15:00 | 741.80 | 749.57 | 748.57 | SL hit (close<static) qty=1.00 sl=745.25 alert=retest2 |

### Cycle 61 — SELL (started 2024-04-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 13:15:00 | 749.50 | 768.03 | 769.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-08 09:15:00 | 743.60 | 757.79 | 763.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 10:15:00 | 741.25 | 734.28 | 740.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-10 10:15:00 | 741.25 | 734.28 | 740.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 10:15:00 | 741.25 | 734.28 | 740.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 11:00:00 | 741.25 | 734.28 | 740.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 11:15:00 | 742.40 | 735.90 | 740.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 12:00:00 | 742.40 | 735.90 | 740.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 12:15:00 | 741.90 | 737.10 | 741.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 12:45:00 | 743.45 | 737.10 | 741.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 13:15:00 | 738.75 | 737.43 | 740.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 13:30:00 | 743.20 | 737.43 | 740.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 15:15:00 | 738.55 | 737.74 | 740.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:15:00 | 743.00 | 737.74 | 740.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 746.40 | 739.47 | 740.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 10:00:00 | 746.40 | 739.47 | 740.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 10:15:00 | 745.75 | 740.73 | 741.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 10:30:00 | 748.25 | 740.73 | 741.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2024-04-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-12 11:15:00 | 747.00 | 741.98 | 741.87 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-04-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 12:15:00 | 738.45 | 741.28 | 741.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 13:15:00 | 736.80 | 740.38 | 741.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 13:15:00 | 711.00 | 709.99 | 718.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 14:00:00 | 711.00 | 709.99 | 718.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 711.50 | 709.73 | 716.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 09:45:00 | 719.15 | 709.73 | 716.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 12:15:00 | 710.55 | 710.33 | 715.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 13:15:00 | 707.25 | 710.33 | 715.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-22 11:15:00 | 671.89 | 679.81 | 690.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-24 09:15:00 | 662.65 | 658.40 | 667.91 | SL hit (close>ema200) qty=0.50 sl=658.40 alert=retest2 |

### Cycle 64 — BUY (started 2024-04-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 09:15:00 | 688.90 | 666.93 | 666.14 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 10:15:00 | 661.40 | 670.42 | 671.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 12:15:00 | 658.25 | 666.36 | 668.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-07 13:15:00 | 612.60 | 611.60 | 622.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-07 13:45:00 | 610.50 | 611.60 | 622.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 11:15:00 | 619.15 | 613.88 | 619.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 12:00:00 | 619.15 | 613.88 | 619.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 620.00 | 615.10 | 619.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 13:00:00 | 620.00 | 615.10 | 619.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 13:15:00 | 616.20 | 615.32 | 619.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 14:15:00 | 613.95 | 615.32 | 619.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 09:15:00 | 610.70 | 615.83 | 618.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 11:15:00 | 583.25 | 597.07 | 606.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 11:15:00 | 580.16 | 597.07 | 606.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-13 14:15:00 | 585.00 | 580.47 | 588.88 | SL hit (close>ema200) qty=0.50 sl=580.47 alert=retest2 |

### Cycle 66 — BUY (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 13:15:00 | 608.05 | 594.55 | 592.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 610.90 | 597.82 | 594.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 09:15:00 | 612.60 | 612.70 | 608.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 10:00:00 | 612.60 | 612.70 | 608.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 15:15:00 | 612.00 | 612.54 | 610.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-18 09:15:00 | 613.25 | 612.54 | 610.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-18 11:30:00 | 613.85 | 612.68 | 610.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-21 09:15:00 | 602.95 | 611.26 | 610.38 | SL hit (close<static) qty=1.00 sl=609.00 alert=retest2 |

### Cycle 67 — SELL (started 2024-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 10:15:00 | 599.00 | 608.81 | 609.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 11:15:00 | 595.45 | 606.14 | 608.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 10:15:00 | 602.20 | 600.08 | 603.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 10:15:00 | 602.20 | 600.08 | 603.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 602.20 | 600.08 | 603.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:00:00 | 602.20 | 600.08 | 603.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 602.95 | 600.66 | 603.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:30:00 | 604.90 | 600.66 | 603.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 601.85 | 600.90 | 603.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 13:00:00 | 601.85 | 600.90 | 603.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 598.65 | 600.45 | 602.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-22 14:45:00 | 597.00 | 600.62 | 602.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 09:15:00 | 612.45 | 603.21 | 603.52 | SL hit (close>static) qty=1.00 sl=604.50 alert=retest2 |

### Cycle 68 — BUY (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 10:15:00 | 614.90 | 605.54 | 604.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 11:15:00 | 620.80 | 608.60 | 606.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 14:15:00 | 618.20 | 620.04 | 615.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-24 15:00:00 | 618.20 | 620.04 | 615.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 619.80 | 628.40 | 625.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:00:00 | 619.80 | 628.40 | 625.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 620.00 | 626.72 | 624.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:30:00 | 620.55 | 626.72 | 624.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 09:15:00 | 620.45 | 623.03 | 623.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 13:15:00 | 612.60 | 616.68 | 619.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 15:15:00 | 616.00 | 615.47 | 618.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 09:15:00 | 612.30 | 615.47 | 618.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 604.55 | 613.28 | 616.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 12:45:00 | 603.55 | 609.37 | 614.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 591.50 | 612.65 | 612.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:30:00 | 601.40 | 606.06 | 609.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 573.37 | 601.49 | 606.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 571.33 | 601.49 | 606.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 13:45:00 | 602.95 | 600.70 | 605.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 612.20 | 602.72 | 605.57 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-05 09:15:00 | 612.20 | 602.72 | 605.57 | SL hit (close>ema200) qty=0.50 sl=602.72 alert=retest2 |

### Cycle 70 — BUY (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 11:15:00 | 630.60 | 611.90 | 609.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 15:15:00 | 631.10 | 622.93 | 616.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 09:15:00 | 666.65 | 671.05 | 656.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 10:00:00 | 666.65 | 671.05 | 656.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 12:15:00 | 674.10 | 676.07 | 672.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 12:45:00 | 674.55 | 676.07 | 672.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 672.05 | 674.91 | 672.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 15:00:00 | 672.05 | 674.91 | 672.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 15:15:00 | 673.00 | 674.53 | 672.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 09:15:00 | 681.30 | 674.53 | 672.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-18 09:15:00 | 673.85 | 678.86 | 679.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2024-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 09:15:00 | 673.85 | 678.86 | 679.34 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 13:15:00 | 682.30 | 679.66 | 679.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 15:15:00 | 687.00 | 681.70 | 680.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-20 11:15:00 | 689.15 | 689.67 | 686.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-20 12:00:00 | 689.15 | 689.67 | 686.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 12:15:00 | 686.05 | 688.95 | 686.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 12:30:00 | 686.80 | 688.95 | 686.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 13:15:00 | 687.00 | 688.56 | 686.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 13:45:00 | 684.90 | 688.56 | 686.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 15:15:00 | 687.45 | 688.21 | 686.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 09:15:00 | 694.00 | 688.21 | 686.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 14:15:00 | 682.00 | 690.02 | 688.75 | SL hit (close<static) qty=1.00 sl=685.30 alert=retest2 |

### Cycle 73 — SELL (started 2024-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 15:15:00 | 679.15 | 687.85 | 687.87 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 09:15:00 | 691.25 | 688.53 | 688.18 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 14:15:00 | 685.20 | 688.37 | 688.46 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 14:15:00 | 697.20 | 689.58 | 688.72 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 15:15:00 | 683.00 | 688.71 | 689.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 09:15:00 | 681.70 | 687.31 | 688.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 11:15:00 | 691.25 | 687.17 | 688.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 11:15:00 | 691.25 | 687.17 | 688.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 691.25 | 687.17 | 688.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 12:00:00 | 691.25 | 687.17 | 688.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 686.65 | 687.07 | 688.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 12:30:00 | 692.10 | 687.07 | 688.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 688.15 | 687.28 | 688.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:45:00 | 689.30 | 687.28 | 688.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 687.80 | 687.39 | 688.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:45:00 | 688.65 | 687.39 | 688.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 686.60 | 687.23 | 687.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:15:00 | 691.45 | 687.23 | 687.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2024-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 09:15:00 | 694.00 | 688.58 | 688.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 10:15:00 | 695.45 | 689.96 | 689.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 14:15:00 | 688.80 | 691.01 | 690.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 14:15:00 | 688.80 | 691.01 | 690.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 688.80 | 691.01 | 690.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 15:00:00 | 688.80 | 691.01 | 690.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 691.00 | 691.01 | 690.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 09:15:00 | 698.85 | 691.01 | 690.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 11:15:00 | 707.30 | 713.27 | 714.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 707.30 | 713.27 | 714.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 13:15:00 | 704.75 | 710.60 | 712.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 09:15:00 | 696.10 | 694.74 | 699.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 09:15:00 | 696.10 | 694.74 | 699.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 696.10 | 694.74 | 699.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 11:45:00 | 694.95 | 695.05 | 699.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 09:15:00 | 714.40 | 699.16 | 699.40 | SL hit (close>static) qty=1.00 sl=704.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 10:15:00 | 719.80 | 703.29 | 701.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 11:15:00 | 726.55 | 707.94 | 703.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 11:15:00 | 727.90 | 732.91 | 726.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 12:00:00 | 727.90 | 732.91 | 726.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 730.00 | 732.33 | 726.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 09:45:00 | 731.65 | 730.23 | 727.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 11:00:00 | 734.60 | 731.11 | 727.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 14:15:00 | 722.60 | 736.84 | 735.59 | SL hit (close<static) qty=1.00 sl=726.55 alert=retest2 |

### Cycle 81 — SELL (started 2024-07-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 15:15:00 | 724.00 | 734.27 | 734.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 09:15:00 | 710.45 | 729.51 | 732.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 13:15:00 | 721.75 | 710.73 | 716.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 13:15:00 | 721.75 | 710.73 | 716.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 721.75 | 710.73 | 716.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:00:00 | 721.75 | 710.73 | 716.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 723.45 | 713.27 | 717.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:45:00 | 728.65 | 713.27 | 717.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 720.95 | 716.70 | 717.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 11:00:00 | 720.95 | 716.70 | 717.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 11:15:00 | 714.80 | 716.32 | 717.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 12:15:00 | 713.20 | 716.32 | 717.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 09:30:00 | 713.55 | 715.46 | 716.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 09:15:00 | 721.85 | 711.95 | 713.69 | SL hit (close>static) qty=1.00 sl=721.10 alert=retest2 |

### Cycle 82 — BUY (started 2024-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 11:15:00 | 727.50 | 716.70 | 715.65 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 10:15:00 | 713.35 | 718.99 | 719.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 11:15:00 | 712.00 | 717.59 | 718.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 602.00 | 587.74 | 609.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 10:00:00 | 602.00 | 587.74 | 609.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 579.65 | 575.77 | 581.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:15:00 | 576.70 | 575.77 | 581.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 12:00:00 | 577.35 | 576.24 | 580.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 572.45 | 576.73 | 579.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 12:15:00 | 586.40 | 580.63 | 580.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2024-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 12:15:00 | 586.40 | 580.63 | 580.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 13:15:00 | 589.30 | 582.36 | 581.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 09:15:00 | 583.80 | 584.06 | 582.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 09:15:00 | 583.80 | 584.06 | 582.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 583.80 | 584.06 | 582.50 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2024-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 13:15:00 | 572.80 | 580.42 | 581.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 14:15:00 | 567.40 | 577.82 | 579.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 13:15:00 | 566.70 | 566.68 | 572.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 14:00:00 | 566.70 | 566.68 | 572.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 589.05 | 571.16 | 572.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:30:00 | 587.50 | 571.16 | 572.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 10:15:00 | 597.35 | 576.40 | 575.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 11:15:00 | 599.90 | 581.10 | 577.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 09:15:00 | 617.15 | 617.93 | 605.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 09:15:00 | 608.95 | 613.99 | 609.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 608.95 | 613.99 | 609.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 10:45:00 | 613.50 | 613.18 | 609.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 09:15:00 | 613.55 | 611.16 | 609.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-22 13:15:00 | 605.50 | 608.71 | 609.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 13:15:00 | 605.50 | 608.71 | 609.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 14:15:00 | 601.20 | 607.21 | 608.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 15:15:00 | 607.50 | 601.60 | 603.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 15:15:00 | 607.50 | 601.60 | 603.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 607.50 | 601.60 | 603.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 621.00 | 601.60 | 603.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 626.50 | 606.58 | 605.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 10:15:00 | 633.25 | 611.91 | 608.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 10:15:00 | 673.20 | 674.70 | 664.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-30 10:45:00 | 671.55 | 674.70 | 664.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 667.10 | 671.33 | 666.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 11:30:00 | 672.90 | 669.91 | 667.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 10:15:00 | 661.15 | 667.26 | 667.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 10:15:00 | 661.15 | 667.26 | 667.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 09:15:00 | 656.85 | 659.62 | 662.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 11:15:00 | 661.20 | 659.81 | 661.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 11:15:00 | 661.20 | 659.81 | 661.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 661.20 | 659.81 | 661.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 12:45:00 | 656.40 | 659.47 | 661.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 14:15:00 | 623.58 | 636.90 | 647.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-10 12:15:00 | 640.35 | 635.04 | 641.92 | SL hit (close>ema200) qty=0.50 sl=635.04 alert=retest2 |

### Cycle 90 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 645.70 | 639.02 | 638.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 10:15:00 | 660.80 | 643.37 | 640.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 09:15:00 | 644.55 | 653.29 | 648.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 09:15:00 | 644.55 | 653.29 | 648.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 644.55 | 653.29 | 648.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 10:00:00 | 644.55 | 653.29 | 648.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 652.75 | 653.18 | 648.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 11:15:00 | 653.40 | 653.18 | 648.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-16 14:15:00 | 633.55 | 646.48 | 646.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 14:15:00 | 633.55 | 646.48 | 646.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 09:15:00 | 628.00 | 640.84 | 643.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 11:15:00 | 644.40 | 640.08 | 642.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 11:15:00 | 644.40 | 640.08 | 642.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 644.40 | 640.08 | 642.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:45:00 | 643.90 | 640.08 | 642.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 12:15:00 | 644.20 | 640.91 | 643.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 12:45:00 | 643.85 | 640.91 | 643.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 13:15:00 | 647.20 | 642.16 | 643.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 13:30:00 | 650.00 | 642.16 | 643.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 647.35 | 643.20 | 643.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 15:00:00 | 647.35 | 643.20 | 643.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 627.50 | 633.20 | 637.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 10:15:00 | 622.85 | 633.20 | 637.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 10:15:00 | 624.95 | 624.96 | 629.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 11:00:00 | 622.80 | 624.53 | 629.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 09:45:00 | 624.80 | 627.50 | 628.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 633.10 | 628.62 | 629.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 10:45:00 | 634.65 | 628.62 | 629.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-23 11:15:00 | 635.25 | 629.94 | 629.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 11:15:00 | 635.25 | 629.94 | 629.84 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 09:15:00 | 629.00 | 630.01 | 630.05 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 10:15:00 | 631.25 | 630.26 | 630.16 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 11:15:00 | 629.00 | 630.01 | 630.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 12:15:00 | 627.95 | 629.60 | 629.86 | Break + close below crossover candle low |

### Cycle 96 — BUY (started 2024-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 13:15:00 | 632.90 | 630.26 | 630.14 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 10:15:00 | 628.20 | 629.98 | 630.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 11:15:00 | 621.60 | 628.31 | 629.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 09:15:00 | 622.80 | 618.52 | 621.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 09:15:00 | 622.80 | 618.52 | 621.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 622.80 | 618.52 | 621.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 10:45:00 | 620.95 | 618.96 | 621.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 09:15:00 | 589.90 | 596.21 | 601.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-04 10:15:00 | 592.50 | 591.25 | 595.60 | SL hit (close>ema200) qty=0.50 sl=591.25 alert=retest2 |

### Cycle 98 — BUY (started 2024-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 12:15:00 | 588.00 | 580.35 | 579.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 13:15:00 | 592.10 | 582.70 | 580.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 585.25 | 587.22 | 584.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 12:15:00 | 585.25 | 587.22 | 584.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 585.25 | 587.22 | 584.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 12:45:00 | 584.20 | 587.22 | 584.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 587.70 | 587.32 | 584.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 09:15:00 | 591.85 | 586.47 | 584.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:30:00 | 589.55 | 593.56 | 590.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 11:45:00 | 590.30 | 591.84 | 590.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 10:15:00 | 588.60 | 592.36 | 592.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 10:15:00 | 588.60 | 592.36 | 592.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 11:15:00 | 587.40 | 591.37 | 591.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 12:15:00 | 592.65 | 591.62 | 591.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 12:15:00 | 592.65 | 591.62 | 591.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 12:15:00 | 592.65 | 591.62 | 591.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 13:00:00 | 592.65 | 591.62 | 591.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2024-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 13:15:00 | 597.00 | 592.70 | 592.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 14:15:00 | 599.70 | 594.10 | 593.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 13:15:00 | 594.65 | 599.23 | 596.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 13:15:00 | 594.65 | 599.23 | 596.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 594.65 | 599.23 | 596.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 14:00:00 | 594.65 | 599.23 | 596.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 594.05 | 598.20 | 596.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 15:15:00 | 591.00 | 598.20 | 596.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 15:15:00 | 591.00 | 596.76 | 595.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-18 09:15:00 | 583.85 | 596.76 | 595.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 09:15:00 | 579.30 | 593.26 | 594.41 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-10-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 10:15:00 | 602.70 | 595.00 | 593.95 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-10-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 11:15:00 | 590.55 | 593.84 | 594.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 12:15:00 | 588.50 | 592.77 | 593.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 586.80 | 585.98 | 589.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 10:00:00 | 586.80 | 585.98 | 589.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 597.60 | 588.30 | 590.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:00:00 | 597.60 | 588.30 | 590.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 597.00 | 590.04 | 591.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:30:00 | 598.85 | 590.04 | 591.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2024-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 12:15:00 | 599.00 | 591.84 | 591.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-23 13:15:00 | 600.00 | 593.47 | 592.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-24 09:15:00 | 578.30 | 592.59 | 592.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 09:15:00 | 578.30 | 592.59 | 592.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 578.30 | 592.59 | 592.58 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2024-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 10:15:00 | 579.00 | 589.87 | 591.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 12:15:00 | 574.85 | 585.01 | 588.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-25 11:15:00 | 574.50 | 574.28 | 580.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-25 12:00:00 | 574.50 | 574.28 | 580.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 575.50 | 572.81 | 576.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:45:00 | 576.05 | 572.81 | 576.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 13:15:00 | 577.00 | 574.00 | 576.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 14:00:00 | 577.00 | 574.00 | 576.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 575.10 | 574.22 | 576.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 14:30:00 | 577.15 | 574.22 | 576.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 570.70 | 573.72 | 575.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 10:30:00 | 566.00 | 571.90 | 574.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 11:45:00 | 566.10 | 570.92 | 573.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-29 14:15:00 | 583.70 | 573.86 | 574.55 | SL hit (close>static) qty=1.00 sl=576.45 alert=retest2 |

### Cycle 106 — BUY (started 2024-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 15:15:00 | 583.00 | 575.69 | 575.32 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 09:15:00 | 555.45 | 572.89 | 574.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 10:15:00 | 549.50 | 568.21 | 572.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 557.50 | 556.92 | 563.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-01 18:00:00 | 557.50 | 556.92 | 563.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 551.90 | 548.85 | 554.05 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 571.60 | 557.33 | 556.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 10:15:00 | 575.90 | 561.04 | 557.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 10:15:00 | 573.00 | 573.60 | 567.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:30:00 | 573.00 | 573.60 | 567.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 13:15:00 | 572.05 | 575.42 | 572.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 14:00:00 | 572.05 | 575.42 | 572.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 567.95 | 573.92 | 572.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 15:00:00 | 567.95 | 573.92 | 572.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 565.70 | 572.28 | 571.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:15:00 | 563.25 | 572.28 | 571.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 573.90 | 572.04 | 571.43 | EMA400 retest candle locked (from upside) |

### Cycle 109 — SELL (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 13:15:00 | 567.35 | 570.55 | 570.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 14:15:00 | 564.25 | 569.09 | 570.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 11:15:00 | 558.60 | 556.00 | 560.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 12:00:00 | 558.60 | 556.00 | 560.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 14:15:00 | 560.40 | 557.26 | 559.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 15:00:00 | 560.40 | 557.26 | 559.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 15:15:00 | 560.50 | 557.91 | 559.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 09:15:00 | 543.40 | 557.91 | 559.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 560.15 | 552.36 | 554.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 560.15 | 552.36 | 554.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 560.00 | 553.89 | 555.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:45:00 | 560.95 | 553.89 | 555.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 545.95 | 552.66 | 554.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 12:45:00 | 543.15 | 549.09 | 552.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 13:30:00 | 543.25 | 548.28 | 551.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 12:15:00 | 558.90 | 552.37 | 552.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2024-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 12:15:00 | 558.90 | 552.37 | 552.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 13:15:00 | 560.25 | 553.95 | 552.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 10:15:00 | 591.45 | 596.56 | 589.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 11:00:00 | 591.45 | 596.56 | 589.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 591.80 | 594.80 | 590.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 591.80 | 594.80 | 590.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 591.05 | 593.65 | 590.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 09:45:00 | 599.25 | 593.15 | 591.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 10:15:00 | 605.05 | 607.24 | 607.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2024-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 10:15:00 | 605.05 | 607.24 | 607.51 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 11:15:00 | 609.75 | 607.74 | 607.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 13:15:00 | 614.50 | 609.38 | 608.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 11:15:00 | 612.85 | 613.22 | 611.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-12 12:00:00 | 612.85 | 613.22 | 611.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 13:15:00 | 611.65 | 612.94 | 611.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 14:00:00 | 611.65 | 612.94 | 611.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 14:15:00 | 608.85 | 612.12 | 611.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 15:00:00 | 608.85 | 612.12 | 611.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 610.50 | 611.80 | 610.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:15:00 | 605.30 | 611.80 | 610.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 599.50 | 609.34 | 609.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 10:15:00 | 597.60 | 606.99 | 608.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 12:15:00 | 606.20 | 605.98 | 607.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 13:00:00 | 606.20 | 605.98 | 607.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 606.60 | 605.31 | 606.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 605.50 | 605.31 | 606.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 607.25 | 605.70 | 606.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:00:00 | 607.25 | 605.70 | 606.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 606.95 | 605.95 | 606.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:30:00 | 607.00 | 605.95 | 606.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 608.00 | 606.36 | 607.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 12:30:00 | 608.70 | 606.36 | 607.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 608.30 | 606.75 | 607.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 14:00:00 | 608.30 | 606.75 | 607.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 603.75 | 606.15 | 606.88 | EMA400 retest candle locked (from downside) |

### Cycle 114 — BUY (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 11:15:00 | 608.95 | 607.39 | 607.24 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 09:15:00 | 600.75 | 606.38 | 606.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 12:15:00 | 599.00 | 603.80 | 605.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 10:15:00 | 594.00 | 593.75 | 597.38 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 11:30:00 | 590.70 | 592.17 | 596.33 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 583.35 | 576.38 | 581.62 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-24 09:15:00 | 583.35 | 576.38 | 581.62 | SL hit (close>ema400) qty=1.00 sl=581.62 alert=retest1 |

### Cycle 116 — BUY (started 2025-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 10:15:00 | 555.00 | 545.98 | 545.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-13 09:15:00 | 559.45 | 551.86 | 548.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 11:15:00 | 547.00 | 551.96 | 549.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-13 11:15:00 | 547.00 | 551.96 | 549.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 11:15:00 | 547.00 | 551.96 | 549.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 12:00:00 | 547.00 | 551.96 | 549.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 12:15:00 | 534.10 | 548.38 | 548.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 13:00:00 | 534.10 | 548.38 | 548.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2025-01-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 13:15:00 | 529.35 | 544.58 | 546.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 14:15:00 | 525.20 | 540.70 | 544.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 15:15:00 | 534.75 | 533.99 | 537.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 09:15:00 | 532.30 | 533.99 | 537.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 531.15 | 533.42 | 537.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 14:15:00 | 527.40 | 533.03 | 536.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 15:00:00 | 527.45 | 531.92 | 535.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 14:30:00 | 527.30 | 531.34 | 532.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 09:15:00 | 548.35 | 532.04 | 531.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 548.35 | 532.04 | 531.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 10:15:00 | 556.70 | 536.97 | 533.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 546.70 | 549.37 | 542.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 546.70 | 549.37 | 542.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 546.70 | 549.37 | 542.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 546.70 | 549.37 | 542.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 545.75 | 549.14 | 545.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:30:00 | 544.65 | 549.14 | 545.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 15:15:00 | 544.60 | 548.23 | 545.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 09:15:00 | 534.70 | 548.23 | 545.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 528.75 | 544.34 | 543.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 10:00:00 | 528.75 | 544.34 | 543.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 10:15:00 | 530.95 | 541.66 | 542.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 12:15:00 | 524.45 | 537.08 | 540.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 527.15 | 524.95 | 531.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 527.15 | 524.95 | 531.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 528.10 | 523.80 | 528.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:45:00 | 531.20 | 523.80 | 528.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 528.20 | 524.68 | 528.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 528.20 | 524.68 | 528.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 528.10 | 525.36 | 528.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:30:00 | 528.75 | 525.36 | 528.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 532.20 | 526.73 | 528.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 13:00:00 | 532.20 | 526.73 | 528.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 13:15:00 | 536.00 | 528.58 | 529.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 14:00:00 | 536.00 | 528.58 | 529.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 536.45 | 531.03 | 530.30 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 523.30 | 529.73 | 530.00 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2025-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 15:15:00 | 532.00 | 530.30 | 530.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 10:15:00 | 534.20 | 531.34 | 530.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 531.10 | 534.35 | 533.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 531.10 | 534.35 | 533.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 531.10 | 534.35 | 533.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 531.05 | 534.35 | 533.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 537.00 | 534.88 | 533.53 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2025-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 12:15:00 | 529.65 | 532.77 | 533.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 10:15:00 | 526.95 | 530.40 | 531.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 534.40 | 528.99 | 530.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 09:15:00 | 534.40 | 528.99 | 530.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 534.40 | 528.99 | 530.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:45:00 | 534.20 | 528.99 | 530.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 538.00 | 530.79 | 530.79 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 14:15:00 | 529.00 | 531.82 | 532.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 525.30 | 530.23 | 531.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 10:15:00 | 532.35 | 530.65 | 531.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 10:15:00 | 532.35 | 530.65 | 531.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 532.35 | 530.65 | 531.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:00:00 | 532.35 | 530.65 | 531.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 526.45 | 529.81 | 530.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:30:00 | 533.00 | 529.81 | 530.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 531.80 | 530.02 | 530.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 14:00:00 | 531.80 | 530.02 | 530.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 534.30 | 530.87 | 531.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 15:00:00 | 534.30 | 530.87 | 531.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2025-02-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 15:15:00 | 534.50 | 531.60 | 531.44 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 527.35 | 530.75 | 531.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 10:15:00 | 526.25 | 529.85 | 530.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 14:15:00 | 525.85 | 525.79 | 528.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 14:15:00 | 525.85 | 525.79 | 528.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 14:15:00 | 525.85 | 525.79 | 528.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 15:00:00 | 525.85 | 525.79 | 528.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 514.95 | 523.33 | 526.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 11:00:00 | 513.45 | 521.36 | 525.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 487.78 | 507.43 | 515.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-12 14:15:00 | 503.00 | 501.61 | 509.13 | SL hit (close>ema200) qty=0.50 sl=501.61 alert=retest2 |

### Cycle 128 — BUY (started 2025-02-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 12:15:00 | 480.40 | 475.24 | 474.75 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 463.20 | 473.26 | 474.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 14:15:00 | 459.30 | 464.46 | 468.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 13:15:00 | 419.40 | 418.24 | 425.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 14:00:00 | 419.40 | 418.24 | 425.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 430.95 | 420.71 | 424.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:30:00 | 435.40 | 420.71 | 424.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 434.35 | 423.44 | 425.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:00:00 | 434.35 | 423.44 | 425.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2025-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 13:15:00 | 432.10 | 427.85 | 427.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 433.70 | 429.02 | 427.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 434.35 | 437.37 | 434.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 11:15:00 | 434.35 | 437.37 | 434.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 434.35 | 437.37 | 434.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:00:00 | 434.35 | 437.37 | 434.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 434.45 | 436.78 | 434.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 14:15:00 | 437.00 | 436.55 | 434.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 09:15:00 | 427.60 | 434.16 | 434.16 | SL hit (close<static) qty=1.00 sl=433.10 alert=retest2 |

### Cycle 131 — SELL (started 2025-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 10:15:00 | 425.10 | 432.35 | 433.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 11:15:00 | 417.50 | 429.38 | 431.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 402.55 | 401.30 | 411.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 09:15:00 | 394.65 | 401.30 | 411.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 393.75 | 389.90 | 393.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 12:00:00 | 393.75 | 389.90 | 393.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 388.10 | 389.54 | 393.32 | EMA400 retest candle locked (from downside) |

### Cycle 132 — BUY (started 2025-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 12:15:00 | 398.35 | 394.04 | 393.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 09:15:00 | 405.05 | 397.49 | 396.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 10:15:00 | 405.60 | 406.05 | 402.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 11:00:00 | 405.60 | 406.05 | 402.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 403.85 | 405.32 | 403.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 13:45:00 | 403.05 | 405.32 | 403.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 406.55 | 408.89 | 406.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:45:00 | 406.95 | 408.89 | 406.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 406.65 | 408.44 | 406.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 403.55 | 408.44 | 406.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 404.40 | 407.64 | 406.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:00:00 | 404.40 | 407.64 | 406.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 406.75 | 407.46 | 406.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:30:00 | 405.60 | 407.46 | 406.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 402.85 | 406.54 | 406.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 15:00:00 | 402.85 | 406.54 | 406.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 15:15:00 | 400.90 | 405.41 | 405.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 11:15:00 | 398.70 | 402.77 | 404.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 396.75 | 394.46 | 397.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 15:00:00 | 396.75 | 394.46 | 397.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 396.05 | 394.94 | 397.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 12:15:00 | 394.25 | 395.46 | 397.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 09:15:00 | 374.54 | 384.58 | 387.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-07 09:15:00 | 354.82 | 364.37 | 374.73 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 134 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 360.10 | 357.12 | 356.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 14:15:00 | 361.00 | 358.14 | 357.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 364.15 | 369.67 | 367.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 364.15 | 369.67 | 367.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 364.15 | 369.67 | 367.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:00:00 | 364.15 | 369.67 | 367.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 365.50 | 368.84 | 367.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:30:00 | 366.85 | 368.23 | 367.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:15:00 | 366.80 | 368.23 | 367.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-25 09:15:00 | 403.54 | 395.06 | 392.59 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2025-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 13:15:00 | 389.45 | 391.12 | 391.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 14:15:00 | 386.20 | 390.13 | 390.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 11:15:00 | 388.55 | 388.27 | 389.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 11:15:00 | 388.55 | 388.27 | 389.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 388.55 | 388.27 | 389.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:30:00 | 388.50 | 388.27 | 389.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 390.45 | 388.71 | 389.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 13:00:00 | 390.45 | 388.71 | 389.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 13:15:00 | 390.30 | 389.03 | 389.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 13:45:00 | 390.85 | 389.03 | 389.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 389.55 | 389.13 | 389.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 14:45:00 | 390.00 | 389.13 | 389.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 390.60 | 389.42 | 389.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 394.70 | 389.42 | 389.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 392.90 | 390.12 | 390.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 14:15:00 | 397.75 | 393.85 | 392.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 11:15:00 | 394.20 | 394.91 | 393.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 11:15:00 | 394.20 | 394.91 | 393.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 11:15:00 | 394.20 | 394.91 | 393.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 11:45:00 | 394.55 | 394.91 | 393.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 392.60 | 394.45 | 393.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 13:00:00 | 392.60 | 394.45 | 393.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 13:15:00 | 392.00 | 393.96 | 393.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 14:00:00 | 392.00 | 393.96 | 393.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 14:15:00 | 390.15 | 393.20 | 392.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 15:00:00 | 390.15 | 393.20 | 392.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2025-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 15:15:00 | 388.00 | 392.16 | 392.36 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 09:15:00 | 404.95 | 394.72 | 393.51 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 10:15:00 | 393.00 | 396.10 | 396.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 11:15:00 | 391.15 | 395.11 | 395.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 388.95 | 388.38 | 391.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 10:45:00 | 386.40 | 388.38 | 391.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 390.35 | 388.87 | 390.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 390.35 | 388.87 | 390.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 392.80 | 389.78 | 390.86 | EMA400 retest candle locked (from downside) |

### Cycle 140 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 11:15:00 | 395.85 | 391.62 | 391.55 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 13:15:00 | 387.45 | 390.83 | 391.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 14:15:00 | 386.70 | 390.01 | 390.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 15:15:00 | 385.35 | 384.65 | 386.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-12 09:15:00 | 402.20 | 384.65 | 386.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 142 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 407.55 | 389.23 | 388.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 410.40 | 396.31 | 392.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 13:15:00 | 428.70 | 429.64 | 425.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 14:00:00 | 428.70 | 429.64 | 425.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 432.90 | 430.27 | 426.87 | EMA400 retest candle locked (from upside) |

### Cycle 143 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 423.45 | 427.04 | 427.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 421.05 | 425.07 | 426.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 14:15:00 | 421.55 | 421.40 | 423.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 15:00:00 | 421.55 | 421.40 | 423.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 428.70 | 422.88 | 423.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 428.70 | 422.88 | 423.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 427.70 | 423.85 | 423.97 | EMA400 retest candle locked (from downside) |

### Cycle 144 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 426.30 | 424.34 | 424.18 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 09:15:00 | 422.05 | 424.19 | 424.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 12:15:00 | 416.00 | 421.39 | 422.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 09:15:00 | 418.55 | 417.55 | 419.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 09:15:00 | 418.55 | 417.55 | 419.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 418.55 | 417.55 | 419.33 | EMA400 retest candle locked (from downside) |

### Cycle 146 — BUY (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 15:15:00 | 423.00 | 420.01 | 419.78 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 401.85 | 416.68 | 418.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 10:15:00 | 400.25 | 413.39 | 416.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 14:15:00 | 399.25 | 399.14 | 404.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 15:00:00 | 399.25 | 399.14 | 404.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 400.65 | 397.57 | 400.21 | EMA400 retest candle locked (from downside) |

### Cycle 148 — BUY (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 14:15:00 | 403.00 | 401.75 | 401.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 11:15:00 | 405.85 | 402.90 | 402.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 424.40 | 428.93 | 425.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 424.40 | 428.93 | 425.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 424.40 | 428.93 | 425.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:00:00 | 424.40 | 428.93 | 425.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 424.20 | 427.99 | 425.47 | EMA400 retest candle locked (from upside) |

### Cycle 149 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 420.15 | 424.27 | 424.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 15:15:00 | 417.00 | 422.82 | 423.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 416.20 | 415.47 | 418.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:00:00 | 416.20 | 415.47 | 418.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 418.00 | 415.97 | 418.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:30:00 | 417.85 | 415.97 | 418.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 417.55 | 416.29 | 418.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:30:00 | 418.10 | 416.29 | 418.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 422.95 | 417.62 | 418.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 422.95 | 417.62 | 418.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 423.00 | 418.70 | 419.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 420.50 | 418.70 | 419.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 423.70 | 419.84 | 419.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 12:15:00 | 424.25 | 421.31 | 420.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 14:15:00 | 422.40 | 422.50 | 421.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 15:00:00 | 422.40 | 422.50 | 421.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 421.30 | 422.73 | 421.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:45:00 | 421.25 | 422.73 | 421.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 423.65 | 422.92 | 421.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 13:00:00 | 425.20 | 423.37 | 422.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 09:15:00 | 417.40 | 422.47 | 422.12 | SL hit (close<static) qty=1.00 sl=419.65 alert=retest2 |

### Cycle 151 — SELL (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 10:15:00 | 415.90 | 421.15 | 421.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 11:15:00 | 414.75 | 419.87 | 420.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 416.30 | 415.89 | 418.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 416.30 | 415.89 | 418.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 416.30 | 415.89 | 418.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 414.80 | 415.89 | 418.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 420.05 | 416.73 | 418.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:45:00 | 419.40 | 416.73 | 418.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 422.40 | 417.86 | 418.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:30:00 | 422.40 | 417.86 | 418.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 421.25 | 419.05 | 419.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:30:00 | 421.15 | 419.05 | 419.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 420.45 | 419.33 | 419.29 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 09:15:00 | 416.50 | 419.16 | 419.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-23 10:15:00 | 414.90 | 418.31 | 418.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 13:15:00 | 418.30 | 418.04 | 418.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 13:15:00 | 418.30 | 418.04 | 418.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 418.30 | 418.04 | 418.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:45:00 | 418.00 | 418.04 | 418.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 420.00 | 418.43 | 418.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 15:00:00 | 420.00 | 418.43 | 418.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 418.50 | 418.45 | 418.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 422.80 | 418.45 | 418.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 426.50 | 420.06 | 419.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 434.25 | 424.95 | 422.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 11:15:00 | 442.15 | 443.07 | 438.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 12:00:00 | 442.15 | 443.07 | 438.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 436.20 | 441.70 | 438.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:00:00 | 436.20 | 441.70 | 438.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 435.70 | 440.50 | 437.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:30:00 | 433.80 | 440.50 | 437.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — SELL (started 2025-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 10:15:00 | 434.45 | 436.41 | 436.52 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 14:15:00 | 438.55 | 436.01 | 435.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 09:15:00 | 442.70 | 437.80 | 436.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 12:15:00 | 441.20 | 441.69 | 439.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-03 13:00:00 | 441.20 | 441.69 | 439.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 438.55 | 440.98 | 439.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 15:00:00 | 438.55 | 440.98 | 439.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 438.50 | 440.49 | 439.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 439.80 | 440.49 | 439.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 437.45 | 439.68 | 439.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:00:00 | 437.45 | 439.68 | 439.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — SELL (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 11:15:00 | 436.10 | 438.97 | 439.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 14:15:00 | 432.65 | 434.50 | 436.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 15:15:00 | 435.45 | 434.69 | 436.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 09:15:00 | 435.90 | 434.69 | 436.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 434.75 | 434.70 | 435.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:45:00 | 434.60 | 434.70 | 435.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 435.20 | 434.25 | 435.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:00:00 | 435.20 | 434.25 | 435.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 432.90 | 433.98 | 435.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 432.55 | 433.92 | 434.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:45:00 | 431.90 | 433.54 | 434.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 11:00:00 | 432.55 | 433.34 | 434.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 11:45:00 | 432.10 | 433.42 | 434.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 421.55 | 419.66 | 422.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 422.35 | 419.66 | 422.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 427.40 | 421.21 | 422.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 427.40 | 421.21 | 422.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 428.30 | 422.63 | 423.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:00:00 | 428.30 | 422.63 | 423.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-15 12:15:00 | 428.75 | 423.85 | 423.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 428.75 | 423.85 | 423.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 13:15:00 | 430.40 | 425.16 | 424.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 14:15:00 | 433.00 | 433.11 | 430.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 14:45:00 | 433.20 | 433.11 | 430.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 427.85 | 431.76 | 430.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 427.85 | 431.76 | 430.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 425.05 | 430.42 | 430.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 422.85 | 430.42 | 430.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 424.80 | 429.29 | 429.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 12:15:00 | 422.35 | 427.91 | 428.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 15:15:00 | 405.65 | 405.64 | 410.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 09:15:00 | 405.00 | 405.64 | 410.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 392.30 | 390.21 | 394.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:45:00 | 395.10 | 390.21 | 394.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 397.10 | 391.59 | 394.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:00:00 | 397.10 | 391.59 | 394.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 398.75 | 393.02 | 395.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:00:00 | 398.75 | 393.02 | 395.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 406.40 | 397.23 | 396.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 14:15:00 | 409.55 | 399.69 | 398.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 395.40 | 408.58 | 405.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 395.40 | 408.58 | 405.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 395.40 | 408.58 | 405.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:00:00 | 395.40 | 408.58 | 405.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 399.55 | 406.77 | 405.06 | EMA400 retest candle locked (from upside) |

### Cycle 161 — SELL (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 12:15:00 | 399.85 | 403.92 | 403.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 391.80 | 401.05 | 402.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 09:15:00 | 400.40 | 399.15 | 401.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-01 09:45:00 | 399.50 | 399.15 | 401.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 400.20 | 399.36 | 401.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 15:15:00 | 399.70 | 400.18 | 401.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 12:15:00 | 402.65 | 400.47 | 400.84 | SL hit (close>static) qty=1.00 sl=402.00 alert=retest2 |

### Cycle 162 — BUY (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 13:15:00 | 404.20 | 401.21 | 401.15 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 10:15:00 | 397.50 | 400.50 | 400.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 12:15:00 | 396.00 | 399.29 | 400.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 374.65 | 373.69 | 378.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 377.15 | 374.17 | 376.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 377.15 | 374.17 | 376.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:00:00 | 377.15 | 374.17 | 376.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 374.80 | 374.30 | 376.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 12:30:00 | 374.15 | 374.17 | 375.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 13:00:00 | 373.70 | 374.17 | 375.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 09:30:00 | 373.40 | 371.91 | 373.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 11:45:00 | 373.85 | 372.86 | 373.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 371.10 | 372.51 | 373.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 13:30:00 | 370.45 | 372.09 | 372.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 10:45:00 | 370.85 | 371.44 | 372.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 11:45:00 | 370.60 | 370.85 | 371.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 15:15:00 | 373.35 | 371.88 | 371.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — BUY (started 2025-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 15:15:00 | 373.35 | 371.88 | 371.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 374.25 | 372.35 | 371.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 14:15:00 | 381.65 | 382.82 | 379.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 15:00:00 | 381.65 | 382.82 | 379.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 378.65 | 381.80 | 379.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 378.35 | 381.80 | 379.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 378.05 | 381.05 | 379.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 11:15:00 | 378.90 | 381.05 | 379.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 12:15:00 | 376.60 | 379.55 | 379.13 | SL hit (close<static) qty=1.00 sl=377.15 alert=retest2 |

### Cycle 165 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 375.35 | 378.25 | 378.58 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 382.60 | 378.74 | 378.73 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 374.50 | 378.93 | 379.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 11:15:00 | 373.65 | 377.87 | 378.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 13:15:00 | 371.90 | 371.24 | 373.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 13:45:00 | 371.90 | 371.24 | 373.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 371.95 | 370.65 | 372.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:00:00 | 370.95 | 370.71 | 372.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:45:00 | 370.20 | 370.71 | 372.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 373.80 | 371.15 | 371.86 | SL hit (close>static) qty=1.00 sl=373.40 alert=retest2 |

### Cycle 168 — BUY (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 12:15:00 | 375.45 | 372.52 | 372.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 377.25 | 374.56 | 373.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 374.25 | 375.40 | 374.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 374.25 | 375.40 | 374.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 374.25 | 375.40 | 374.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 374.25 | 375.40 | 374.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 377.65 | 375.85 | 374.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:30:00 | 376.50 | 375.85 | 374.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 376.15 | 376.10 | 374.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:30:00 | 374.35 | 376.10 | 374.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 374.50 | 375.78 | 374.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:45:00 | 374.70 | 375.78 | 374.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 374.95 | 375.61 | 374.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:30:00 | 374.10 | 375.61 | 374.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 374.00 | 375.29 | 374.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:00:00 | 374.00 | 375.29 | 374.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 374.45 | 375.12 | 374.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:30:00 | 374.15 | 375.12 | 374.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 376.70 | 375.44 | 374.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:45:00 | 376.40 | 375.44 | 374.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 374.65 | 375.77 | 375.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:00:00 | 374.65 | 375.77 | 375.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 374.50 | 375.52 | 375.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 374.50 | 375.52 | 375.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 374.35 | 375.28 | 375.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:15:00 | 374.05 | 375.28 | 375.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 372.05 | 374.41 | 374.71 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 10:15:00 | 377.00 | 374.95 | 374.83 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 11:15:00 | 373.70 | 374.83 | 374.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 13:15:00 | 372.85 | 374.31 | 374.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 376.90 | 374.57 | 374.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 376.90 | 374.57 | 374.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 376.90 | 374.57 | 374.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:30:00 | 380.30 | 374.57 | 374.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 375.25 | 374.71 | 374.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:30:00 | 376.20 | 374.71 | 374.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 374.50 | 374.66 | 374.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:15:00 | 375.70 | 374.66 | 374.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — BUY (started 2025-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 12:15:00 | 375.30 | 374.79 | 374.77 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 13:15:00 | 374.45 | 374.72 | 374.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 15:15:00 | 373.50 | 374.46 | 374.62 | Break + close below crossover candle low |

### Cycle 174 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 386.80 | 376.93 | 375.73 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 14:15:00 | 376.00 | 378.08 | 378.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 09:15:00 | 374.00 | 376.93 | 377.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 376.25 | 374.71 | 375.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 09:15:00 | 376.25 | 374.71 | 375.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 376.25 | 374.71 | 375.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:45:00 | 377.25 | 374.71 | 375.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 375.25 | 374.81 | 375.78 | EMA400 retest candle locked (from downside) |

### Cycle 176 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 378.40 | 376.34 | 376.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 12:15:00 | 380.70 | 377.67 | 376.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 15:15:00 | 381.00 | 381.46 | 379.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 15:15:00 | 381.00 | 381.46 | 379.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 381.00 | 381.46 | 379.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 388.85 | 381.46 | 379.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 15:15:00 | 381.40 | 383.08 | 383.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 15:15:00 | 381.40 | 383.08 | 383.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 09:15:00 | 375.20 | 381.50 | 382.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 14:15:00 | 361.30 | 360.66 | 364.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-26 15:00:00 | 361.30 | 360.66 | 364.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 351.40 | 351.57 | 354.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:45:00 | 347.00 | 350.74 | 354.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 09:15:00 | 347.85 | 349.45 | 351.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 11:15:00 | 355.75 | 351.44 | 351.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — BUY (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 11:15:00 | 355.75 | 351.44 | 351.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 12:15:00 | 356.60 | 352.47 | 351.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 354.85 | 355.06 | 353.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 10:00:00 | 354.85 | 355.06 | 353.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 355.20 | 355.04 | 353.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:30:00 | 354.95 | 355.04 | 353.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 359.70 | 356.17 | 354.69 | EMA400 retest candle locked (from upside) |

### Cycle 179 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 353.05 | 357.54 | 357.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 11:15:00 | 352.30 | 356.49 | 357.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 344.00 | 343.94 | 346.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-16 09:30:00 | 345.45 | 343.94 | 346.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 344.50 | 344.07 | 345.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:30:00 | 344.95 | 344.07 | 345.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 340.70 | 343.63 | 345.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 13:00:00 | 339.25 | 341.83 | 343.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 338.90 | 341.30 | 343.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 350.75 | 341.29 | 341.57 | SL hit (close>static) qty=1.00 sl=345.50 alert=retest2 |

### Cycle 180 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 352.00 | 343.43 | 342.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 380.00 | 350.75 | 345.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 14:15:00 | 386.00 | 386.60 | 378.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-27 15:00:00 | 386.00 | 386.60 | 378.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 377.00 | 384.87 | 379.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 377.00 | 384.87 | 379.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 377.40 | 383.38 | 379.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:30:00 | 376.50 | 383.38 | 379.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 377.45 | 379.73 | 378.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:30:00 | 377.30 | 379.73 | 378.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 376.75 | 379.13 | 378.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 376.20 | 379.13 | 378.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 372.90 | 377.89 | 377.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 13:15:00 | 371.85 | 373.40 | 374.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 14:15:00 | 372.05 | 370.86 | 372.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 14:15:00 | 372.05 | 370.86 | 372.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 372.05 | 370.86 | 372.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 372.05 | 370.86 | 372.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 373.10 | 371.31 | 372.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 374.40 | 371.31 | 372.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 371.50 | 371.35 | 372.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 10:15:00 | 369.65 | 371.35 | 372.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 360.70 | 369.17 | 369.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 10:15:00 | 371.50 | 370.01 | 369.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — BUY (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 10:15:00 | 371.50 | 370.01 | 369.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 13:15:00 | 376.35 | 372.75 | 371.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 09:15:00 | 374.70 | 374.72 | 372.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 10:00:00 | 374.70 | 374.72 | 372.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 374.25 | 375.98 | 374.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 15:00:00 | 374.25 | 375.98 | 374.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 374.80 | 375.74 | 374.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:15:00 | 369.30 | 375.74 | 374.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 371.70 | 374.93 | 374.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 13:00:00 | 379.90 | 375.72 | 374.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 15:15:00 | 387.40 | 389.84 | 390.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — SELL (started 2025-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 15:15:00 | 387.40 | 389.84 | 390.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 09:15:00 | 384.30 | 388.73 | 389.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 13:15:00 | 386.80 | 386.23 | 387.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 14:00:00 | 386.80 | 386.23 | 387.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 391.20 | 387.22 | 388.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 15:00:00 | 391.20 | 387.22 | 388.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 390.85 | 387.95 | 388.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 385.95 | 387.95 | 388.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 11:15:00 | 396.40 | 386.93 | 386.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — BUY (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 11:15:00 | 396.40 | 386.93 | 386.33 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 382.25 | 386.97 | 387.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 380.30 | 383.19 | 385.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 383.80 | 383.31 | 384.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 383.80 | 383.31 | 384.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 383.80 | 383.31 | 384.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:30:00 | 385.30 | 383.31 | 384.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 381.85 | 383.02 | 384.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:30:00 | 384.95 | 383.02 | 384.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 385.25 | 383.46 | 384.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 12:00:00 | 385.25 | 383.46 | 384.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 385.50 | 383.87 | 384.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 12:30:00 | 385.40 | 383.87 | 384.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 381.30 | 383.56 | 384.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:15:00 | 381.25 | 383.56 | 384.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 383.35 | 383.51 | 384.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:45:00 | 383.65 | 383.51 | 384.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 383.05 | 382.08 | 383.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:15:00 | 384.50 | 382.08 | 383.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 384.15 | 382.49 | 383.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:30:00 | 384.00 | 382.49 | 383.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 384.60 | 382.91 | 383.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 13:15:00 | 382.65 | 383.02 | 383.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 14:15:00 | 386.70 | 384.07 | 383.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 386.70 | 384.07 | 383.78 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 381.50 | 383.83 | 384.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 379.95 | 381.90 | 382.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 382.30 | 381.55 | 382.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 382.30 | 381.55 | 382.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 382.30 | 381.55 | 382.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 10:45:00 | 381.20 | 381.49 | 382.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:30:00 | 380.65 | 381.96 | 382.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 385.45 | 382.97 | 382.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — BUY (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 14:15:00 | 385.45 | 382.97 | 382.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 09:15:00 | 399.70 | 386.76 | 384.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 13:15:00 | 432.50 | 433.04 | 424.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-05 14:00:00 | 432.50 | 433.04 | 424.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 427.75 | 432.90 | 429.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:30:00 | 428.05 | 432.90 | 429.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 427.95 | 431.91 | 428.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:30:00 | 426.00 | 431.91 | 428.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 15:15:00 | 428.00 | 431.13 | 428.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:15:00 | 428.50 | 431.13 | 428.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 430.45 | 430.28 | 428.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:30:00 | 427.75 | 430.28 | 428.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 430.10 | 430.88 | 429.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 429.20 | 430.88 | 429.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 428.80 | 430.46 | 429.71 | EMA400 retest candle locked (from upside) |

### Cycle 189 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 426.35 | 428.81 | 429.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 13:15:00 | 423.80 | 427.24 | 428.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 426.70 | 425.83 | 427.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 426.70 | 425.83 | 427.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 426.70 | 425.83 | 427.25 | EMA400 retest candle locked (from downside) |

### Cycle 190 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 428.40 | 427.97 | 427.92 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 15:15:00 | 426.00 | 427.57 | 427.75 | EMA200 below EMA400 |

### Cycle 192 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 431.45 | 428.45 | 428.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 10:15:00 | 433.70 | 430.25 | 429.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 432.55 | 432.82 | 431.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 432.55 | 432.82 | 431.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 432.55 | 432.82 | 431.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 436.70 | 432.82 | 431.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 428.70 | 432.06 | 431.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:00:00 | 428.70 | 432.06 | 431.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 428.05 | 431.26 | 430.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 428.05 | 431.26 | 430.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 425.40 | 429.59 | 430.12 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 11:15:00 | 431.50 | 430.41 | 430.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 12:15:00 | 435.05 | 431.34 | 430.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 15:15:00 | 430.00 | 431.52 | 431.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 15:15:00 | 430.00 | 431.52 | 431.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 430.00 | 431.52 | 431.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 438.40 | 431.52 | 431.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 12:30:00 | 433.10 | 432.69 | 431.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 13:15:00 | 429.50 | 432.05 | 431.68 | SL hit (close<static) qty=1.00 sl=429.60 alert=retest2 |

### Cycle 195 — SELL (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 12:15:00 | 430.50 | 431.72 | 431.75 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 432.55 | 431.89 | 431.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 433.10 | 432.13 | 431.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 10:15:00 | 450.05 | 453.25 | 448.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 11:00:00 | 450.05 | 453.25 | 448.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 449.65 | 452.53 | 448.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:30:00 | 449.05 | 452.53 | 448.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 448.75 | 451.78 | 448.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:00:00 | 448.75 | 451.78 | 448.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 449.00 | 451.22 | 448.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 14:45:00 | 450.10 | 450.89 | 448.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 12:15:00 | 449.60 | 450.52 | 449.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 13:00:00 | 449.70 | 450.35 | 449.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 13:30:00 | 450.00 | 450.01 | 449.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 449.60 | 449.93 | 449.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:30:00 | 448.70 | 449.93 | 449.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 448.55 | 449.65 | 449.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 449.65 | 449.65 | 449.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 449.35 | 449.59 | 449.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-29 10:15:00 | 438.40 | 447.35 | 448.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 438.40 | 447.35 | 448.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 11:15:00 | 436.80 | 445.24 | 447.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 430.95 | 429.57 | 435.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:30:00 | 430.55 | 429.57 | 435.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 433.70 | 430.72 | 433.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:30:00 | 433.40 | 430.72 | 433.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 433.85 | 431.35 | 433.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:45:00 | 433.80 | 431.35 | 433.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 432.40 | 431.56 | 433.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 432.45 | 431.56 | 433.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 432.40 | 431.73 | 433.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 09:45:00 | 429.55 | 431.71 | 432.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 15:15:00 | 434.60 | 432.11 | 432.48 | SL hit (close>static) qty=1.00 sl=434.05 alert=retest2 |

### Cycle 198 — BUY (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 11:15:00 | 440.10 | 429.48 | 428.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 13:15:00 | 441.00 | 433.44 | 430.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 12:15:00 | 432.65 | 436.84 | 434.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 12:15:00 | 432.65 | 436.84 | 434.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 432.65 | 436.84 | 434.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:00:00 | 432.65 | 436.84 | 434.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 429.85 | 435.44 | 433.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:30:00 | 428.90 | 435.44 | 433.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 429.70 | 433.76 | 433.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:15:00 | 425.80 | 433.76 | 433.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — SELL (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 10:15:00 | 430.40 | 432.51 | 432.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 11:15:00 | 427.40 | 431.49 | 432.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 417.05 | 415.86 | 420.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 417.05 | 415.86 | 420.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 417.05 | 415.86 | 420.80 | EMA400 retest candle locked (from downside) |

### Cycle 200 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 435.50 | 422.47 | 421.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 10:15:00 | 438.85 | 425.74 | 423.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 426.35 | 430.66 | 427.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 09:15:00 | 426.35 | 430.66 | 427.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 426.35 | 430.66 | 427.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 424.55 | 430.66 | 427.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 425.35 | 429.59 | 427.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:30:00 | 424.00 | 429.59 | 427.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 426.00 | 427.16 | 426.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 418.00 | 427.16 | 426.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 410.55 | 423.84 | 425.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 405.95 | 420.26 | 423.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 410.20 | 406.82 | 411.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 15:15:00 | 410.20 | 406.82 | 411.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 410.20 | 406.82 | 411.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 415.60 | 406.82 | 411.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 415.85 | 408.63 | 411.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 416.80 | 408.63 | 411.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 414.20 | 409.74 | 411.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:45:00 | 412.50 | 410.44 | 411.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:00:00 | 412.55 | 410.86 | 412.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 15:15:00 | 414.40 | 412.98 | 412.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 414.40 | 412.98 | 412.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 422.40 | 414.86 | 413.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 12:15:00 | 414.00 | 415.88 | 414.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 12:15:00 | 414.00 | 415.88 | 414.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 414.00 | 415.88 | 414.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:00:00 | 414.00 | 415.88 | 414.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 408.65 | 414.43 | 414.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:45:00 | 408.85 | 414.43 | 414.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 407.15 | 412.98 | 413.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 15:15:00 | 404.80 | 411.34 | 412.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 400.20 | 399.42 | 404.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 09:15:00 | 402.65 | 399.42 | 404.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 402.75 | 400.09 | 404.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 406.75 | 400.09 | 404.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 403.40 | 401.30 | 404.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:45:00 | 403.90 | 401.30 | 404.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 398.55 | 400.75 | 403.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:45:00 | 399.55 | 400.75 | 403.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 402.25 | 400.42 | 403.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 14:45:00 | 402.10 | 400.42 | 403.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 401.60 | 400.66 | 402.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:15:00 | 423.60 | 400.66 | 402.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 408.20 | 402.16 | 403.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:15:00 | 406.10 | 402.16 | 403.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 11:15:00 | 409.80 | 404.88 | 404.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — BUY (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 11:15:00 | 409.80 | 404.88 | 404.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 417.40 | 409.39 | 406.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 412.25 | 414.86 | 411.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 412.25 | 414.86 | 411.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 412.25 | 414.86 | 411.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:30:00 | 412.55 | 414.86 | 411.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 412.05 | 414.30 | 411.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 11:00:00 | 412.05 | 414.30 | 411.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 414.45 | 414.33 | 411.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 414.45 | 414.33 | 411.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 425.40 | 416.54 | 413.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 15:00:00 | 427.45 | 420.08 | 415.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 437.75 | 420.75 | 417.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 15:15:00 | 432.00 | 440.46 | 441.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — SELL (started 2026-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 15:15:00 | 432.00 | 440.46 | 441.35 | EMA200 below EMA400 |

### Cycle 206 — BUY (started 2026-02-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 13:15:00 | 443.70 | 441.85 | 441.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 14:15:00 | 445.50 | 442.58 | 441.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 09:15:00 | 440.00 | 459.45 | 457.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 09:15:00 | 440.00 | 459.45 | 457.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 440.00 | 459.45 | 457.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:45:00 | 441.15 | 459.45 | 457.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 439.85 | 455.53 | 455.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 12:15:00 | 435.60 | 448.91 | 452.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-19 09:15:00 | 385.05 | 379.87 | 387.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-19 10:15:00 | 389.05 | 379.87 | 387.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 386.85 | 381.27 | 387.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:30:00 | 387.25 | 381.27 | 387.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 383.80 | 381.78 | 387.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 379.30 | 384.09 | 386.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:30:00 | 381.50 | 379.63 | 382.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 12:15:00 | 362.43 | 371.93 | 376.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 375.40 | 370.46 | 374.11 | SL hit (close>ema200) qty=0.50 sl=370.46 alert=retest2 |

### Cycle 208 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 384.80 | 376.70 | 375.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 13:15:00 | 386.65 | 381.19 | 378.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 11:15:00 | 381.40 | 384.51 | 381.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 11:15:00 | 381.40 | 384.51 | 381.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 381.40 | 384.51 | 381.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:00:00 | 381.40 | 384.51 | 381.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 383.85 | 384.38 | 381.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 14:45:00 | 386.90 | 385.42 | 382.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 09:45:00 | 387.20 | 386.19 | 383.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 11:15:00 | 380.30 | 384.91 | 383.22 | SL hit (close<static) qty=1.00 sl=381.05 alert=retest2 |

### Cycle 209 — SELL (started 2026-03-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 13:15:00 | 377.40 | 382.10 | 382.16 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2026-03-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 14:15:00 | 384.00 | 382.48 | 382.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-04 09:15:00 | 389.55 | 383.98 | 383.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 11:15:00 | 378.30 | 382.94 | 382.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 11:15:00 | 378.30 | 382.94 | 382.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 378.30 | 382.94 | 382.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 12:00:00 | 378.30 | 382.94 | 382.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — SELL (started 2026-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 12:15:00 | 381.15 | 382.58 | 382.59 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2026-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 13:15:00 | 384.90 | 383.05 | 382.80 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 11:15:00 | 381.65 | 382.70 | 382.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 13:15:00 | 379.00 | 381.75 | 382.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 15:15:00 | 371.00 | 370.66 | 374.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-10 09:15:00 | 379.65 | 370.66 | 374.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 383.30 | 373.19 | 374.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:45:00 | 381.85 | 373.19 | 374.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 383.70 | 375.29 | 375.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:30:00 | 384.00 | 375.29 | 375.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 214 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 383.55 | 376.94 | 376.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 12:15:00 | 387.90 | 379.13 | 377.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 383.95 | 384.60 | 381.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 12:45:00 | 383.90 | 384.60 | 381.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 381.90 | 383.92 | 381.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 381.90 | 383.92 | 381.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 382.05 | 383.54 | 381.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 379.20 | 383.54 | 381.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 373.40 | 381.51 | 381.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 374.70 | 381.51 | 381.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 215 — SELL (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 10:15:00 | 375.90 | 380.39 | 380.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 11:15:00 | 370.95 | 378.50 | 379.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 362.85 | 352.66 | 359.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 362.85 | 352.66 | 359.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 362.85 | 352.66 | 359.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 362.85 | 352.66 | 359.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 363.55 | 354.84 | 359.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:45:00 | 361.55 | 354.84 | 359.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 364.25 | 359.23 | 360.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 364.25 | 359.23 | 360.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 216 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 371.50 | 362.55 | 361.83 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 14:15:00 | 356.65 | 363.85 | 364.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 353.80 | 361.06 | 362.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 347.95 | 346.23 | 351.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:30:00 | 350.65 | 346.23 | 351.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 353.00 | 348.09 | 350.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 15:00:00 | 347.95 | 349.89 | 350.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 330.55 | 337.68 | 342.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 12:15:00 | 343.45 | 335.88 | 340.57 | SL hit (close>ema200) qty=0.50 sl=335.88 alert=retest2 |

### Cycle 218 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 349.90 | 342.48 | 341.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 10:15:00 | 353.85 | 347.69 | 345.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 14:15:00 | 365.00 | 365.25 | 359.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-06 15:00:00 | 365.00 | 365.25 | 359.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 369.15 | 371.02 | 368.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 369.15 | 371.02 | 368.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 369.65 | 370.75 | 368.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 15:15:00 | 374.70 | 371.57 | 369.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 14:15:00 | 365.95 | 369.68 | 369.65 | SL hit (close<static) qty=1.00 sl=368.55 alert=retest2 |

### Cycle 219 — SELL (started 2026-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-10 15:15:00 | 366.45 | 369.04 | 369.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 09:15:00 | 361.55 | 367.54 | 368.65 | Break + close below crossover candle low |

### Cycle 220 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 378.35 | 368.04 | 368.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 12:15:00 | 379.25 | 372.48 | 370.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 385.05 | 388.54 | 385.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 09:15:00 | 385.05 | 388.54 | 385.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 385.05 | 388.54 | 385.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:45:00 | 384.85 | 388.54 | 385.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 386.00 | 388.03 | 385.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 395.30 | 384.35 | 384.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 09:15:00 | 384.25 | 390.44 | 388.70 | SL hit (close<static) qty=1.00 sl=384.35 alert=retest2 |

### Cycle 221 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 385.30 | 387.55 | 387.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 13:15:00 | 382.60 | 385.10 | 386.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 14:15:00 | 385.15 | 385.11 | 386.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-23 15:00:00 | 385.15 | 385.11 | 386.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 385.90 | 385.27 | 386.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 09:15:00 | 379.25 | 385.27 | 386.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 376.55 | 370.03 | 369.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 222 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 376.55 | 370.03 | 369.15 | EMA200 above EMA400 |

### Cycle 223 — SELL (started 2026-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 12:15:00 | 366.65 | 369.72 | 370.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-07 14:15:00 | 359.95 | 367.29 | 368.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 12:15:00 | 366.35 | 365.92 | 367.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-08 13:00:00 | 366.35 | 365.92 | 367.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 365.60 | 365.86 | 367.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 14:15:00 | 364.75 | 365.86 | 367.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-29 09:15:00 | 340.10 | 2023-05-30 10:15:00 | 337.45 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2023-05-29 11:15:00 | 338.60 | 2023-05-30 13:15:00 | 336.40 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2023-06-02 09:15:00 | 349.40 | 2023-06-06 09:15:00 | 339.85 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2023-06-07 10:15:00 | 341.90 | 2023-06-12 14:15:00 | 337.95 | STOP_HIT | 1.00 | 1.16% |
| SELL | retest2 | 2023-06-07 12:00:00 | 341.80 | 2023-06-12 14:15:00 | 337.95 | STOP_HIT | 1.00 | 1.13% |
| SELL | retest2 | 2023-06-07 13:00:00 | 341.75 | 2023-06-12 14:15:00 | 337.95 | STOP_HIT | 1.00 | 1.11% |
| SELL | retest2 | 2023-06-07 13:30:00 | 341.70 | 2023-06-12 14:15:00 | 337.95 | STOP_HIT | 1.00 | 1.10% |
| BUY | retest2 | 2023-06-26 11:15:00 | 338.95 | 2023-06-26 11:15:00 | 339.15 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2023-07-26 12:15:00 | 381.40 | 2023-07-27 09:15:00 | 384.80 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2023-07-26 12:45:00 | 380.85 | 2023-07-27 09:15:00 | 384.80 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2023-07-26 14:30:00 | 381.25 | 2023-07-27 09:15:00 | 384.80 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2023-08-03 09:15:00 | 429.65 | 2023-08-10 15:15:00 | 445.75 | STOP_HIT | 1.00 | 3.75% |
| BUY | retest2 | 2023-08-03 10:15:00 | 427.40 | 2023-08-10 15:15:00 | 445.75 | STOP_HIT | 1.00 | 4.29% |
| BUY | retest2 | 2023-08-03 10:45:00 | 429.95 | 2023-08-10 15:15:00 | 445.75 | STOP_HIT | 1.00 | 3.67% |
| SELL | retest2 | 2023-08-14 09:15:00 | 441.50 | 2023-08-14 09:15:00 | 446.80 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2023-08-14 10:15:00 | 443.45 | 2023-08-14 13:15:00 | 447.30 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2023-09-08 14:30:00 | 516.50 | 2023-09-13 10:15:00 | 490.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-11 09:15:00 | 514.90 | 2023-09-13 10:15:00 | 489.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-08 14:30:00 | 516.50 | 2023-09-14 09:15:00 | 506.25 | STOP_HIT | 0.50 | 1.98% |
| SELL | retest2 | 2023-09-11 09:15:00 | 514.90 | 2023-09-14 09:15:00 | 506.25 | STOP_HIT | 0.50 | 1.68% |
| SELL | retest2 | 2023-09-12 09:30:00 | 510.85 | 2023-09-14 15:15:00 | 505.25 | STOP_HIT | 1.00 | 1.10% |
| SELL | retest2 | 2023-09-22 09:15:00 | 486.00 | 2023-09-25 11:15:00 | 493.65 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2023-09-22 13:45:00 | 488.45 | 2023-09-25 11:15:00 | 493.65 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2023-09-25 09:45:00 | 488.15 | 2023-09-25 11:15:00 | 493.65 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2023-09-25 13:45:00 | 488.80 | 2023-09-26 10:15:00 | 490.80 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2023-10-05 09:15:00 | 504.75 | 2023-10-17 14:15:00 | 555.23 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-10-23 09:15:00 | 542.00 | 2023-10-26 09:15:00 | 514.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-23 09:15:00 | 542.00 | 2023-10-26 12:15:00 | 521.25 | STOP_HIT | 0.50 | 3.83% |
| BUY | retest1 | 2023-10-30 14:15:00 | 539.90 | 2023-11-01 09:15:00 | 566.89 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-10-30 14:15:00 | 539.90 | 2023-11-06 09:15:00 | 593.89 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2023-11-08 09:15:00 | 587.75 | 2023-11-08 13:15:00 | 582.10 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2023-11-08 13:00:00 | 585.20 | 2023-11-08 13:15:00 | 582.10 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2023-11-09 12:15:00 | 583.40 | 2023-11-12 18:15:00 | 591.65 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2023-11-09 14:15:00 | 583.35 | 2023-11-12 18:15:00 | 591.65 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2023-11-09 15:00:00 | 582.95 | 2023-11-12 18:15:00 | 591.65 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2023-11-10 09:45:00 | 581.20 | 2023-11-12 18:15:00 | 591.65 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2023-11-22 14:45:00 | 624.15 | 2023-11-24 09:15:00 | 613.75 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2023-11-23 13:45:00 | 624.70 | 2023-11-24 09:15:00 | 613.75 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2023-11-23 14:15:00 | 623.80 | 2023-11-24 09:15:00 | 613.75 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest1 | 2023-12-04 09:15:00 | 637.50 | 2023-12-05 10:15:00 | 632.65 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest1 | 2023-12-04 12:00:00 | 636.20 | 2023-12-05 10:15:00 | 632.65 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2023-12-04 13:15:00 | 636.85 | 2023-12-05 10:15:00 | 632.65 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest1 | 2023-12-04 14:30:00 | 637.30 | 2023-12-05 10:15:00 | 632.65 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2023-12-05 15:00:00 | 637.55 | 2023-12-14 09:15:00 | 701.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-06 09:15:00 | 642.00 | 2023-12-14 10:15:00 | 706.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-17 11:30:00 | 796.00 | 2024-02-01 12:15:00 | 827.20 | STOP_HIT | 1.00 | 3.92% |
| BUY | retest2 | 2024-01-19 09:15:00 | 805.40 | 2024-02-01 12:15:00 | 827.20 | STOP_HIT | 1.00 | 2.71% |
| SELL | retest2 | 2024-02-08 11:00:00 | 835.70 | 2024-02-13 15:15:00 | 842.05 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-02-08 13:45:00 | 837.45 | 2024-02-13 15:15:00 | 842.05 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2024-02-08 14:45:00 | 836.35 | 2024-02-13 15:15:00 | 842.05 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2024-02-08 15:15:00 | 837.50 | 2024-02-13 15:15:00 | 842.05 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-02-09 10:30:00 | 827.00 | 2024-02-13 15:15:00 | 842.05 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-02-12 10:00:00 | 827.25 | 2024-02-13 15:15:00 | 842.05 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-02-12 12:00:00 | 826.95 | 2024-02-13 15:15:00 | 842.05 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-02-12 12:45:00 | 828.65 | 2024-02-13 15:15:00 | 842.05 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-03-01 10:30:00 | 778.00 | 2024-03-02 11:15:00 | 787.05 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-03-18 09:15:00 | 749.65 | 2024-03-21 14:15:00 | 739.40 | STOP_HIT | 1.00 | 1.37% |
| BUY | retest2 | 2024-03-28 11:15:00 | 749.15 | 2024-03-28 14:15:00 | 741.80 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-03-28 13:00:00 | 749.50 | 2024-03-28 14:15:00 | 741.80 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-04-01 09:15:00 | 752.25 | 2024-04-05 13:15:00 | 749.50 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2024-04-18 13:15:00 | 707.25 | 2024-04-22 11:15:00 | 671.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-18 13:15:00 | 707.25 | 2024-04-24 09:15:00 | 662.65 | STOP_HIT | 0.50 | 6.31% |
| SELL | retest2 | 2024-05-08 14:15:00 | 613.95 | 2024-05-10 11:15:00 | 583.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-09 09:15:00 | 610.70 | 2024-05-10 11:15:00 | 580.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-08 14:15:00 | 613.95 | 2024-05-13 14:15:00 | 585.00 | STOP_HIT | 0.50 | 4.72% |
| SELL | retest2 | 2024-05-09 09:15:00 | 610.70 | 2024-05-13 14:15:00 | 585.00 | STOP_HIT | 0.50 | 4.21% |
| BUY | retest2 | 2024-05-18 09:15:00 | 613.25 | 2024-05-21 09:15:00 | 602.95 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-05-18 11:30:00 | 613.85 | 2024-05-21 09:15:00 | 602.95 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-05-22 14:45:00 | 597.00 | 2024-05-23 09:15:00 | 612.45 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2024-05-31 12:45:00 | 603.55 | 2024-06-04 12:15:00 | 573.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 591.50 | 2024-06-04 12:15:00 | 571.33 | PARTIAL | 0.50 | 3.41% |
| SELL | retest2 | 2024-05-31 12:45:00 | 603.55 | 2024-06-05 09:15:00 | 612.20 | STOP_HIT | 0.50 | -1.43% |
| SELL | retest2 | 2024-06-04 09:15:00 | 591.50 | 2024-06-05 09:15:00 | 612.20 | STOP_HIT | 0.50 | -3.50% |
| SELL | retest2 | 2024-06-04 10:30:00 | 601.40 | 2024-06-05 10:15:00 | 625.25 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2024-06-04 13:45:00 | 602.95 | 2024-06-05 10:15:00 | 625.25 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest2 | 2024-06-13 09:15:00 | 681.30 | 2024-06-18 09:15:00 | 673.85 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-06-21 09:15:00 | 694.00 | 2024-06-21 14:15:00 | 682.00 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-07-01 09:15:00 | 698.85 | 2024-07-08 11:15:00 | 707.30 | STOP_HIT | 1.00 | 1.21% |
| SELL | retest2 | 2024-07-11 11:45:00 | 694.95 | 2024-07-12 09:15:00 | 714.40 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2024-07-18 09:45:00 | 731.65 | 2024-07-19 14:15:00 | 722.60 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-07-18 11:00:00 | 734.60 | 2024-07-19 14:15:00 | 722.60 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-07-24 12:15:00 | 713.20 | 2024-07-26 09:15:00 | 721.85 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-07-25 09:30:00 | 713.55 | 2024-07-26 09:15:00 | 721.85 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-08-09 10:15:00 | 576.70 | 2024-08-12 12:15:00 | 586.40 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-08-09 12:00:00 | 577.35 | 2024-08-12 12:15:00 | 586.40 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-08-12 09:15:00 | 572.45 | 2024-08-12 12:15:00 | 586.40 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2024-08-21 10:45:00 | 613.50 | 2024-08-22 13:15:00 | 605.50 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-08-22 09:15:00 | 613.55 | 2024-08-22 13:15:00 | 605.50 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-09-03 11:30:00 | 672.90 | 2024-09-04 10:15:00 | 661.15 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-09-06 12:45:00 | 656.40 | 2024-09-09 14:15:00 | 623.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-06 12:45:00 | 656.40 | 2024-09-10 12:15:00 | 640.35 | STOP_HIT | 0.50 | 2.45% |
| BUY | retest2 | 2024-09-16 11:15:00 | 653.40 | 2024-09-16 14:15:00 | 633.55 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2024-09-19 10:15:00 | 622.85 | 2024-09-23 11:15:00 | 635.25 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2024-09-20 10:15:00 | 624.95 | 2024-09-23 11:15:00 | 635.25 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-09-20 11:00:00 | 622.80 | 2024-09-23 11:15:00 | 635.25 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-09-23 09:45:00 | 624.80 | 2024-09-23 11:15:00 | 635.25 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2024-09-27 10:45:00 | 620.95 | 2024-10-03 09:15:00 | 589.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 10:45:00 | 620.95 | 2024-10-04 10:15:00 | 592.50 | STOP_HIT | 0.50 | 4.58% |
| BUY | retest2 | 2024-10-11 09:15:00 | 591.85 | 2024-10-16 10:15:00 | 588.60 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-10-14 09:30:00 | 589.55 | 2024-10-16 10:15:00 | 588.60 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2024-10-14 11:45:00 | 590.30 | 2024-10-16 10:15:00 | 588.60 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2024-10-29 10:30:00 | 566.00 | 2024-10-29 14:15:00 | 583.70 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2024-10-29 11:45:00 | 566.10 | 2024-10-29 14:15:00 | 583.70 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2024-11-21 12:45:00 | 543.15 | 2024-11-22 12:15:00 | 558.90 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2024-11-21 13:30:00 | 543.25 | 2024-11-22 12:15:00 | 558.90 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2024-12-02 09:45:00 | 599.25 | 2024-12-11 10:15:00 | 605.05 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest1 | 2024-12-20 11:30:00 | 590.70 | 2024-12-24 09:15:00 | 583.35 | STOP_HIT | 1.00 | 1.24% |
| SELL | retest2 | 2024-12-24 11:30:00 | 580.00 | 2025-01-06 09:15:00 | 551.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-24 11:30:00 | 580.00 | 2025-01-07 14:15:00 | 546.95 | STOP_HIT | 0.50 | 5.70% |
| SELL | retest2 | 2025-01-15 14:15:00 | 527.40 | 2025-01-23 09:15:00 | 548.35 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2025-01-15 15:00:00 | 527.45 | 2025-01-23 09:15:00 | 548.35 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2025-01-21 14:30:00 | 527.30 | 2025-01-23 09:15:00 | 548.35 | STOP_HIT | 1.00 | -3.99% |
| SELL | retest2 | 2025-02-11 11:00:00 | 513.45 | 2025-02-12 09:15:00 | 487.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 11:00:00 | 513.45 | 2025-02-12 14:15:00 | 503.00 | STOP_HIT | 0.50 | 2.04% |
| SELL | retest2 | 2025-02-13 10:45:00 | 512.50 | 2025-02-14 10:15:00 | 488.63 | PARTIAL | 0.50 | 4.66% |
| SELL | retest2 | 2025-02-13 11:45:00 | 514.35 | 2025-02-14 12:15:00 | 486.88 | PARTIAL | 0.50 | 5.34% |
| SELL | retest2 | 2025-02-13 10:45:00 | 512.50 | 2025-02-18 12:15:00 | 462.92 | TARGET_HIT | 0.50 | 9.68% |
| SELL | retest2 | 2025-02-13 11:45:00 | 514.35 | 2025-02-18 14:15:00 | 473.45 | STOP_HIT | 0.50 | 7.95% |
| BUY | retest2 | 2025-03-07 14:15:00 | 437.00 | 2025-03-10 09:15:00 | 427.60 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-03-28 12:15:00 | 394.25 | 2025-04-04 09:15:00 | 374.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 12:15:00 | 394.25 | 2025-04-07 09:15:00 | 354.82 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-04-17 11:30:00 | 366.85 | 2025-04-25 09:15:00 | 403.54 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-17 12:15:00 | 366.80 | 2025-04-25 09:15:00 | 403.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-18 13:00:00 | 425.20 | 2025-06-19 09:15:00 | 417.40 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-07-09 09:15:00 | 432.55 | 2025-07-15 12:15:00 | 428.75 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2025-07-09 09:45:00 | 431.90 | 2025-07-15 12:15:00 | 428.75 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2025-07-09 11:00:00 | 432.55 | 2025-07-15 12:15:00 | 428.75 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2025-07-09 11:45:00 | 432.10 | 2025-07-15 12:15:00 | 428.75 | STOP_HIT | 1.00 | 0.78% |
| SELL | retest2 | 2025-08-01 15:15:00 | 399.70 | 2025-08-04 12:15:00 | 402.65 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-08-12 12:30:00 | 374.15 | 2025-08-19 15:15:00 | 373.35 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2025-08-12 13:00:00 | 373.70 | 2025-08-19 15:15:00 | 373.35 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-08-14 09:30:00 | 373.40 | 2025-08-19 15:15:00 | 373.35 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-08-14 11:45:00 | 373.85 | 2025-08-19 15:15:00 | 373.35 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-08-14 13:30:00 | 370.45 | 2025-08-19 15:15:00 | 373.35 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-08-18 10:45:00 | 370.85 | 2025-08-19 15:15:00 | 373.35 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-08-18 11:45:00 | 370.60 | 2025-08-19 15:15:00 | 373.35 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-08-22 11:15:00 | 378.90 | 2025-08-22 12:15:00 | 376.60 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-08-29 13:00:00 | 370.95 | 2025-09-01 10:15:00 | 373.80 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-08-29 13:45:00 | 370.20 | 2025-09-01 10:15:00 | 373.80 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-09-18 09:15:00 | 388.85 | 2025-09-19 15:15:00 | 381.40 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-10-01 09:45:00 | 347.00 | 2025-10-06 11:15:00 | 355.75 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-10-03 09:15:00 | 347.85 | 2025-10-06 11:15:00 | 355.75 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-10-17 13:00:00 | 339.25 | 2025-10-21 13:15:00 | 350.75 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2025-10-20 09:15:00 | 338.90 | 2025-10-21 13:15:00 | 350.75 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2025-11-04 10:15:00 | 369.65 | 2025-11-07 10:15:00 | 371.50 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-11-07 09:15:00 | 360.70 | 2025-11-07 10:15:00 | 371.50 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2025-11-11 13:00:00 | 379.90 | 2025-11-14 15:15:00 | 387.40 | STOP_HIT | 1.00 | 1.97% |
| SELL | retest2 | 2025-11-18 09:15:00 | 385.95 | 2025-11-19 11:15:00 | 396.40 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-11-26 13:15:00 | 382.65 | 2025-11-26 14:15:00 | 386.70 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-12-01 10:45:00 | 381.20 | 2025-12-01 14:15:00 | 385.45 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-12-01 11:30:00 | 380.65 | 2025-12-01 14:15:00 | 385.45 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-12-18 09:15:00 | 438.40 | 2025-12-18 13:15:00 | 429.50 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-12-18 12:30:00 | 433.10 | 2025-12-18 13:15:00 | 429.50 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-12-18 15:00:00 | 434.25 | 2025-12-19 12:15:00 | 430.50 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-12-19 09:15:00 | 434.85 | 2025-12-19 12:15:00 | 430.50 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-12-24 14:45:00 | 450.10 | 2025-12-29 10:15:00 | 438.40 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2025-12-26 12:15:00 | 449.60 | 2025-12-29 10:15:00 | 438.40 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2025-12-26 13:00:00 | 449.70 | 2025-12-29 10:15:00 | 438.40 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-12-26 13:30:00 | 450.00 | 2025-12-29 10:15:00 | 438.40 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2026-01-02 09:45:00 | 429.55 | 2026-01-02 15:15:00 | 434.60 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2026-01-05 09:15:00 | 427.80 | 2026-01-07 11:15:00 | 440.10 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2026-01-07 10:15:00 | 429.45 | 2026-01-07 11:15:00 | 440.10 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2026-01-22 11:45:00 | 412.50 | 2026-01-22 15:15:00 | 414.40 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2026-01-22 13:00:00 | 412.55 | 2026-01-22 15:15:00 | 414.40 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2026-01-29 10:15:00 | 406.10 | 2026-01-29 11:15:00 | 409.80 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-02-01 15:00:00 | 427.45 | 2026-02-06 15:15:00 | 432.00 | STOP_HIT | 1.00 | 1.06% |
| BUY | retest2 | 2026-02-03 09:15:00 | 437.75 | 2026-02-06 15:15:00 | 432.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-02-20 09:15:00 | 379.30 | 2026-02-24 12:15:00 | 362.43 | PARTIAL | 0.50 | 4.45% |
| SELL | retest2 | 2026-02-20 09:15:00 | 379.30 | 2026-02-25 09:15:00 | 375.40 | STOP_HIT | 0.50 | 1.03% |
| SELL | retest2 | 2026-02-23 09:30:00 | 381.50 | 2026-02-26 10:15:00 | 384.80 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-02-25 09:30:00 | 379.20 | 2026-02-26 10:15:00 | 384.80 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2026-02-27 14:45:00 | 386.90 | 2026-03-02 11:15:00 | 380.30 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2026-03-02 09:45:00 | 387.20 | 2026-03-02 11:15:00 | 380.30 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-03-25 15:00:00 | 347.95 | 2026-03-30 09:15:00 | 330.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 15:00:00 | 347.95 | 2026-03-30 12:15:00 | 343.45 | STOP_HIT | 0.50 | 1.29% |
| SELL | retest2 | 2026-04-01 09:30:00 | 347.75 | 2026-04-01 11:15:00 | 349.90 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2026-04-01 10:45:00 | 348.10 | 2026-04-01 11:15:00 | 349.90 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2026-04-09 15:15:00 | 374.70 | 2026-04-10 14:15:00 | 365.95 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2026-04-21 09:15:00 | 395.30 | 2026-04-22 09:15:00 | 384.25 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2026-04-22 11:45:00 | 386.30 | 2026-04-23 09:15:00 | 385.30 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2026-04-24 09:15:00 | 379.25 | 2026-05-06 09:15:00 | 376.55 | STOP_HIT | 1.00 | 0.71% |
