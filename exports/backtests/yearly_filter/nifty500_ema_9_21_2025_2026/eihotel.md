# EIH Ltd. (EIHOTEL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 336.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 81 |
| ALERT1 | 49 |
| ALERT2 | 48 |
| ALERT2_SKIP | 29 |
| ALERT3 | 128 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 64 |
| PARTIAL | 7 |
| TARGET_HIT | 1 |
| STOP_HIT | 61 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 68 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 31 / 37
- **Target hits / Stop hits / Partials:** 1 / 60 / 7
- **Avg / median % per leg:** 0.40% / -0.24%
- **Sum % (uncompounded):** 27.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 6 | 19.4% | 1 | 30 | 0 | -0.38% | -11.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 31 | 6 | 19.4% | 1 | 30 | 0 | -0.38% | -11.8% |
| SELL (all) | 37 | 25 | 67.6% | 0 | 30 | 7 | 1.06% | 39.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 37 | 25 | 67.6% | 0 | 30 | 7 | 1.06% | 39.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 68 | 31 | 45.6% | 1 | 60 | 7 | 0.40% | 27.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 367.70 | 357.29 | 357.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 369.30 | 364.35 | 361.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 363.35 | 365.14 | 362.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 12:15:00 | 363.35 | 365.14 | 362.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 363.35 | 365.14 | 362.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:45:00 | 362.85 | 365.14 | 362.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 15:15:00 | 366.00 | 366.46 | 365.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:30:00 | 365.75 | 366.23 | 365.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 363.85 | 365.75 | 365.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:00:00 | 363.85 | 365.75 | 365.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 363.65 | 365.33 | 364.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 12:00:00 | 363.65 | 365.33 | 364.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 364.90 | 365.11 | 364.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:15:00 | 366.00 | 365.11 | 364.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 366.60 | 364.76 | 364.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:45:00 | 365.50 | 365.94 | 365.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 13:15:00 | 371.40 | 374.39 | 374.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 13:15:00 | 371.40 | 374.39 | 374.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 14:15:00 | 370.80 | 373.67 | 374.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 375.95 | 373.70 | 374.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 375.95 | 373.70 | 374.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 375.95 | 373.70 | 374.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:30:00 | 375.90 | 373.70 | 374.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 375.65 | 374.09 | 374.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:45:00 | 376.85 | 374.09 | 374.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 375.55 | 374.38 | 374.37 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 12:15:00 | 372.65 | 374.04 | 374.21 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 377.00 | 374.46 | 374.29 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 11:15:00 | 373.50 | 374.48 | 374.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 12:15:00 | 371.60 | 373.91 | 374.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 365.45 | 365.32 | 366.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 365.45 | 365.32 | 366.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 365.45 | 365.32 | 366.96 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 13:15:00 | 372.25 | 367.92 | 367.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 09:15:00 | 376.00 | 370.20 | 368.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 376.00 | 376.16 | 373.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 13:00:00 | 376.00 | 376.16 | 373.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 374.40 | 375.78 | 374.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:00:00 | 374.40 | 375.78 | 374.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 374.00 | 375.42 | 374.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 10:00:00 | 377.00 | 375.74 | 374.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 10:30:00 | 376.95 | 376.09 | 374.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 13:45:00 | 377.00 | 376.39 | 375.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 377.20 | 375.65 | 375.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 376.75 | 375.87 | 375.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:30:00 | 376.90 | 375.87 | 375.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 374.90 | 375.68 | 375.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:45:00 | 375.00 | 375.68 | 375.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 374.35 | 375.41 | 375.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:00:00 | 374.35 | 375.41 | 375.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-05 14:15:00 | 374.45 | 374.80 | 374.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 14:15:00 | 374.45 | 374.80 | 374.84 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 15:15:00 | 375.40 | 374.92 | 374.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 09:15:00 | 377.45 | 375.42 | 375.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 13:15:00 | 375.80 | 376.15 | 375.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 13:15:00 | 375.80 | 376.15 | 375.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 13:15:00 | 375.80 | 376.15 | 375.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 13:45:00 | 375.40 | 376.15 | 375.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 374.80 | 375.88 | 375.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 374.80 | 375.88 | 375.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 374.80 | 375.67 | 375.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 375.45 | 375.67 | 375.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 10:45:00 | 377.00 | 375.88 | 375.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 373.15 | 378.34 | 378.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 373.15 | 378.34 | 378.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 369.65 | 373.85 | 376.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 15:15:00 | 359.20 | 358.77 | 362.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-17 09:15:00 | 358.70 | 358.77 | 362.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 361.00 | 359.22 | 362.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 362.40 | 359.22 | 362.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 361.80 | 359.73 | 362.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:30:00 | 362.00 | 359.73 | 362.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 360.05 | 359.80 | 361.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:30:00 | 358.50 | 359.47 | 361.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 09:15:00 | 340.57 | 347.59 | 351.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 13:15:00 | 346.90 | 346.81 | 349.91 | SL hit (close>ema200) qty=0.50 sl=346.81 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 352.45 | 349.18 | 348.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 354.25 | 350.13 | 349.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 360.95 | 363.74 | 360.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 12:15:00 | 360.95 | 363.74 | 360.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 360.95 | 363.74 | 360.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:00:00 | 360.95 | 363.74 | 360.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 361.40 | 363.27 | 361.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:30:00 | 360.90 | 363.27 | 361.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 357.65 | 362.15 | 360.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 357.65 | 362.15 | 360.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 358.35 | 361.39 | 360.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 364.15 | 361.39 | 360.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 13:15:00 | 363.70 | 364.89 | 364.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 13:15:00 | 363.70 | 364.89 | 364.94 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 366.95 | 365.25 | 365.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 11:15:00 | 369.40 | 366.08 | 365.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 10:15:00 | 367.85 | 367.95 | 366.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 11:00:00 | 367.85 | 367.95 | 366.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 365.65 | 367.49 | 366.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:00:00 | 365.65 | 367.49 | 366.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 365.50 | 367.09 | 366.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:45:00 | 365.25 | 367.09 | 366.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 365.50 | 366.77 | 366.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 14:00:00 | 365.50 | 366.77 | 366.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 366.10 | 366.59 | 366.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 368.55 | 366.59 | 366.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 367.75 | 366.82 | 366.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 09:15:00 | 370.55 | 367.72 | 367.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 09:45:00 | 370.85 | 368.91 | 367.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 379.25 | 381.39 | 381.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 379.25 | 381.39 | 381.59 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 12:15:00 | 383.45 | 381.93 | 381.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 09:15:00 | 385.45 | 382.55 | 382.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 382.40 | 382.52 | 382.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 10:15:00 | 382.40 | 382.52 | 382.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 382.40 | 382.52 | 382.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 382.40 | 382.52 | 382.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 381.10 | 382.24 | 382.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:30:00 | 381.25 | 382.24 | 382.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 12:15:00 | 380.45 | 381.88 | 381.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 12:15:00 | 379.10 | 380.58 | 381.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 13:15:00 | 382.85 | 381.03 | 381.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 13:15:00 | 382.85 | 381.03 | 381.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 382.85 | 381.03 | 381.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:00:00 | 382.85 | 381.03 | 381.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 14:15:00 | 383.25 | 381.47 | 381.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 09:15:00 | 388.85 | 383.43 | 382.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 379.00 | 384.89 | 384.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 379.00 | 384.89 | 384.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 379.00 | 384.89 | 384.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 379.00 | 384.89 | 384.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 381.00 | 384.11 | 383.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 11:15:00 | 381.45 | 384.11 | 383.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 377.40 | 382.77 | 383.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 377.40 | 382.77 | 383.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 12:15:00 | 376.45 | 381.51 | 382.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 10:15:00 | 379.35 | 378.31 | 380.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 10:45:00 | 380.00 | 378.31 | 380.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 378.60 | 376.96 | 378.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 10:45:00 | 376.20 | 377.10 | 378.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 09:15:00 | 383.50 | 379.20 | 378.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 383.50 | 379.20 | 378.95 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 378.45 | 379.07 | 379.08 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 11:15:00 | 380.40 | 379.33 | 379.20 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 374.35 | 378.40 | 378.83 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 10:15:00 | 381.30 | 379.34 | 379.16 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 376.25 | 378.74 | 378.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 372.95 | 377.27 | 378.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 14:15:00 | 376.65 | 375.55 | 376.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 14:15:00 | 376.65 | 375.55 | 376.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 376.65 | 375.55 | 376.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 379.00 | 375.55 | 376.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 377.00 | 375.84 | 376.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 09:45:00 | 374.90 | 375.51 | 376.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 10:15:00 | 356.15 | 363.99 | 368.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 15:15:00 | 361.00 | 360.50 | 364.59 | SL hit (close>ema200) qty=0.50 sl=360.50 alert=retest2 |

### Cycle 25 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 363.55 | 361.23 | 361.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 13:15:00 | 364.00 | 361.78 | 361.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 13:15:00 | 393.80 | 394.26 | 384.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 14:00:00 | 393.80 | 394.26 | 384.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 13:15:00 | 386.35 | 389.83 | 386.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 13:45:00 | 386.60 | 389.83 | 386.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 384.80 | 388.83 | 386.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 15:00:00 | 384.80 | 388.83 | 386.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 387.65 | 388.59 | 386.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:15:00 | 389.30 | 388.59 | 386.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 386.05 | 388.08 | 386.71 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 15:15:00 | 385.55 | 386.33 | 386.41 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 09:15:00 | 394.85 | 388.04 | 387.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 10:15:00 | 398.90 | 390.21 | 388.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 14:15:00 | 406.15 | 406.82 | 402.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 15:00:00 | 406.15 | 406.82 | 402.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 406.20 | 408.43 | 405.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 403.75 | 408.43 | 405.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 403.60 | 407.47 | 405.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:45:00 | 404.10 | 407.47 | 405.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 402.00 | 406.37 | 405.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:00:00 | 402.00 | 406.37 | 405.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 402.10 | 405.18 | 405.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 14:00:00 | 402.10 | 405.18 | 405.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 15:15:00 | 402.40 | 404.53 | 404.73 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 10:15:00 | 405.90 | 404.98 | 404.91 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 12:15:00 | 403.30 | 404.67 | 404.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 401.95 | 403.99 | 404.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 404.65 | 403.40 | 404.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 404.65 | 403.40 | 404.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 404.65 | 403.40 | 404.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 404.65 | 403.40 | 404.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 405.30 | 403.78 | 404.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 407.35 | 403.78 | 404.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 400.75 | 401.29 | 402.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 14:15:00 | 396.55 | 400.69 | 401.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 11:15:00 | 403.45 | 400.73 | 401.30 | SL hit (close>static) qty=1.00 sl=402.70 alert=retest2 |

### Cycle 31 — BUY (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 12:15:00 | 405.55 | 401.70 | 401.69 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 13:15:00 | 399.80 | 401.32 | 401.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 14:15:00 | 397.95 | 400.64 | 401.19 | Break + close below crossover candle low |

### Cycle 33 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 415.55 | 403.20 | 402.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 10:15:00 | 418.40 | 413.75 | 410.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 13:15:00 | 411.50 | 413.54 | 411.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 411.50 | 413.54 | 411.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 411.50 | 413.54 | 411.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 12:30:00 | 415.35 | 413.93 | 412.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 13:15:00 | 407.60 | 413.85 | 413.72 | SL hit (close<static) qty=1.00 sl=409.70 alert=retest2 |

### Cycle 34 — SELL (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 14:15:00 | 408.90 | 412.86 | 413.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 12:15:00 | 405.50 | 409.47 | 411.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 12:15:00 | 401.00 | 398.30 | 400.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 12:15:00 | 401.00 | 398.30 | 400.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 401.00 | 398.30 | 400.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:45:00 | 401.10 | 398.30 | 400.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 400.00 | 398.64 | 400.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 14:45:00 | 399.00 | 399.03 | 400.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 15:15:00 | 403.00 | 399.83 | 400.80 | SL hit (close>static) qty=1.00 sl=401.30 alert=retest2 |

### Cycle 35 — BUY (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 13:15:00 | 402.25 | 401.45 | 401.35 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 09:15:00 | 397.50 | 400.68 | 401.03 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 15:15:00 | 403.05 | 401.35 | 401.14 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 09:15:00 | 397.80 | 400.83 | 401.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 13:15:00 | 395.40 | 398.48 | 399.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 389.90 | 389.07 | 391.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 10:00:00 | 389.90 | 389.07 | 391.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 364.50 | 363.78 | 366.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 368.00 | 363.78 | 366.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 371.70 | 365.36 | 366.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:00:00 | 371.70 | 365.36 | 366.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 373.60 | 367.01 | 367.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:30:00 | 370.30 | 367.01 | 367.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 11:15:00 | 380.20 | 369.65 | 368.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 12:15:00 | 404.00 | 376.52 | 371.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 384.00 | 384.87 | 378.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 11:00:00 | 384.00 | 384.87 | 378.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 380.85 | 382.44 | 379.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:45:00 | 381.35 | 382.44 | 379.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 384.35 | 382.43 | 380.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:30:00 | 380.75 | 382.43 | 380.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 390.45 | 391.10 | 388.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 11:15:00 | 390.65 | 391.10 | 388.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 11:45:00 | 390.90 | 391.05 | 388.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 13:00:00 | 390.75 | 390.99 | 388.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 14:30:00 | 390.65 | 390.89 | 388.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 392.95 | 391.25 | 389.44 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-10 14:15:00 | 384.95 | 388.80 | 388.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 14:15:00 | 384.95 | 388.80 | 388.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 11:15:00 | 382.05 | 386.56 | 387.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 380.75 | 378.90 | 381.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 380.75 | 378.90 | 381.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 380.75 | 378.90 | 381.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 381.00 | 378.90 | 381.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 382.10 | 379.54 | 381.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 12:30:00 | 380.45 | 380.28 | 381.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 14:15:00 | 386.45 | 381.77 | 381.93 | SL hit (close>static) qty=1.00 sl=384.10 alert=retest2 |

### Cycle 41 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 385.00 | 382.41 | 382.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 13:15:00 | 387.15 | 385.04 | 383.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 382.60 | 385.42 | 384.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 12:15:00 | 382.60 | 385.42 | 384.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 382.60 | 385.42 | 384.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 382.60 | 385.42 | 384.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 385.00 | 385.34 | 384.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 390.70 | 386.33 | 385.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:30:00 | 387.35 | 386.76 | 386.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 10:00:00 | 387.10 | 386.76 | 386.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 11:45:00 | 387.90 | 386.83 | 386.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 387.10 | 386.89 | 386.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 387.10 | 386.89 | 386.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 385.00 | 386.51 | 386.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 384.35 | 386.51 | 386.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 386.15 | 386.44 | 386.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:45:00 | 384.45 | 386.44 | 386.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-24 15:15:00 | 384.70 | 386.09 | 386.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 15:15:00 | 384.70 | 386.09 | 386.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 13:15:00 | 384.00 | 385.07 | 385.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 386.80 | 385.24 | 385.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 386.80 | 385.24 | 385.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 386.80 | 385.24 | 385.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:45:00 | 388.40 | 385.24 | 385.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 390.60 | 386.31 | 386.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 14:15:00 | 393.30 | 389.39 | 387.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 09:15:00 | 389.65 | 390.20 | 388.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 10:00:00 | 389.65 | 390.20 | 388.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 389.00 | 389.86 | 388.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:45:00 | 389.30 | 389.86 | 388.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 389.95 | 389.88 | 388.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 11:00:00 | 392.50 | 390.09 | 389.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 10:15:00 | 389.10 | 390.60 | 390.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 389.10 | 390.60 | 390.71 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 11:15:00 | 393.00 | 391.08 | 390.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 12:15:00 | 394.85 | 391.83 | 391.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 391.00 | 396.07 | 394.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 391.00 | 396.07 | 394.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 391.00 | 396.07 | 394.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:45:00 | 390.95 | 396.07 | 394.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 390.60 | 394.98 | 394.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 387.25 | 394.98 | 394.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 12:15:00 | 392.00 | 393.93 | 394.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 385.50 | 390.72 | 392.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 390.25 | 389.94 | 391.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 12:45:00 | 390.40 | 389.94 | 391.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 390.50 | 390.05 | 391.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 391.40 | 390.05 | 391.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 393.10 | 390.66 | 391.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 393.10 | 390.66 | 391.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 390.25 | 390.58 | 391.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 388.80 | 390.58 | 391.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 10:45:00 | 388.50 | 389.57 | 390.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 12:30:00 | 388.20 | 387.40 | 388.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 10:15:00 | 384.85 | 379.00 | 378.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 10:15:00 | 384.85 | 379.00 | 378.37 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 12:15:00 | 376.90 | 379.40 | 379.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 375.50 | 377.43 | 378.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 375.85 | 374.55 | 375.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 375.85 | 374.55 | 375.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 375.85 | 374.55 | 375.69 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 381.30 | 376.84 | 376.40 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 375.40 | 376.48 | 376.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 14:15:00 | 372.85 | 375.75 | 376.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 09:15:00 | 375.55 | 375.27 | 375.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 375.55 | 375.27 | 375.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 375.55 | 375.27 | 375.93 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 11:15:00 | 377.45 | 375.91 | 375.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 12:15:00 | 378.60 | 376.45 | 376.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 14:15:00 | 376.00 | 376.49 | 376.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 14:15:00 | 376.00 | 376.49 | 376.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 376.00 | 376.49 | 376.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 15:00:00 | 376.00 | 376.49 | 376.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 376.60 | 376.51 | 376.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 380.80 | 376.51 | 376.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 10:00:00 | 378.80 | 376.97 | 376.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 12:15:00 | 377.65 | 377.18 | 376.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 09:30:00 | 378.05 | 378.39 | 377.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 378.05 | 378.33 | 377.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 378.05 | 378.33 | 377.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 376.50 | 377.96 | 377.77 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-04 13:15:00 | 376.75 | 377.52 | 377.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 13:15:00 | 376.75 | 377.52 | 377.59 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 14:15:00 | 378.10 | 377.64 | 377.64 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 11:15:00 | 375.10 | 377.18 | 377.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 14:15:00 | 374.15 | 376.15 | 376.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 09:15:00 | 375.80 | 373.93 | 374.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 09:15:00 | 375.80 | 373.93 | 374.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 375.80 | 373.93 | 374.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:00:00 | 375.80 | 373.93 | 374.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 377.10 | 374.56 | 375.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 377.10 | 374.56 | 375.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 11:15:00 | 379.50 | 375.55 | 375.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 14:15:00 | 380.20 | 376.88 | 376.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 13:15:00 | 381.10 | 381.63 | 379.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 14:00:00 | 381.10 | 381.63 | 379.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 379.00 | 381.10 | 379.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 15:00:00 | 379.00 | 381.10 | 379.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 379.35 | 380.75 | 379.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 376.95 | 380.75 | 379.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 375.00 | 379.60 | 378.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:45:00 | 374.20 | 379.60 | 378.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 376.00 | 378.88 | 378.68 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 11:15:00 | 375.90 | 378.29 | 378.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 11:15:00 | 374.25 | 376.09 | 376.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 12:15:00 | 365.85 | 363.15 | 365.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 12:15:00 | 365.85 | 363.15 | 365.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 365.85 | 363.15 | 365.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:45:00 | 365.25 | 363.15 | 365.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 367.05 | 363.93 | 365.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:45:00 | 367.35 | 363.93 | 365.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 364.90 | 364.13 | 365.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:30:00 | 367.10 | 364.13 | 365.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 365.50 | 364.40 | 365.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 365.45 | 364.40 | 365.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 368.45 | 365.21 | 365.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:00:00 | 368.45 | 365.21 | 365.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 368.00 | 365.77 | 365.81 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 368.00 | 366.22 | 366.01 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-12-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 14:15:00 | 366.10 | 366.36 | 366.38 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 09:15:00 | 367.75 | 366.48 | 366.42 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 12:15:00 | 365.30 | 366.36 | 366.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 13:15:00 | 364.00 | 365.89 | 366.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 09:15:00 | 366.20 | 365.50 | 365.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 09:15:00 | 366.20 | 365.50 | 365.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 366.20 | 365.50 | 365.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 366.20 | 365.50 | 365.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 366.00 | 365.60 | 365.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:15:00 | 367.20 | 365.60 | 365.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 367.10 | 365.90 | 366.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 367.10 | 365.90 | 366.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 12:15:00 | 367.65 | 366.25 | 366.15 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 365.35 | 366.31 | 366.36 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 367.30 | 365.58 | 365.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 13:15:00 | 368.40 | 366.49 | 366.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 10:15:00 | 366.35 | 366.85 | 366.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 10:15:00 | 366.35 | 366.85 | 366.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 366.35 | 366.85 | 366.39 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 10:15:00 | 364.20 | 365.79 | 366.01 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 12:15:00 | 367.70 | 366.29 | 366.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 369.00 | 367.15 | 366.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 367.40 | 368.08 | 367.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 13:15:00 | 367.40 | 368.08 | 367.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 367.40 | 368.08 | 367.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 367.40 | 368.08 | 367.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 365.30 | 367.53 | 367.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:30:00 | 364.50 | 367.53 | 367.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 15:15:00 | 363.15 | 366.65 | 366.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 358.80 | 365.08 | 366.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 346.20 | 346.10 | 348.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 344.90 | 346.10 | 348.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 346.55 | 345.74 | 347.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 346.55 | 345.74 | 347.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 345.25 | 345.84 | 347.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:30:00 | 344.40 | 345.50 | 346.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 343.85 | 345.20 | 346.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 15:15:00 | 327.18 | 331.44 | 334.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 326.66 | 330.89 | 334.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 12:15:00 | 333.60 | 330.73 | 333.33 | SL hit (close>ema200) qty=0.50 sl=330.73 alert=retest2 |

### Cycle 67 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 317.00 | 314.09 | 313.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 13:15:00 | 320.45 | 316.54 | 315.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 328.05 | 329.69 | 325.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 11:00:00 | 328.05 | 329.69 | 325.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 327.40 | 329.51 | 327.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 327.40 | 329.51 | 327.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 348.15 | 333.24 | 329.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 12:15:00 | 350.60 | 333.24 | 329.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 09:15:00 | 352.25 | 342.00 | 335.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 13:30:00 | 350.35 | 348.79 | 342.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 338.45 | 342.91 | 343.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 338.45 | 342.91 | 343.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 12:15:00 | 337.05 | 340.92 | 342.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 331.30 | 330.82 | 332.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 331.30 | 330.82 | 332.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 333.95 | 331.16 | 332.45 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 335.50 | 333.06 | 333.01 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 331.15 | 332.94 | 333.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 330.10 | 332.37 | 332.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 15:15:00 | 321.50 | 321.29 | 323.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 09:15:00 | 320.80 | 321.29 | 323.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 318.50 | 320.74 | 323.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:00:00 | 317.60 | 320.11 | 322.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:45:00 | 317.35 | 319.53 | 322.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 315.95 | 318.27 | 320.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 09:15:00 | 316.75 | 318.60 | 319.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 315.70 | 317.36 | 318.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 13:15:00 | 315.00 | 317.36 | 318.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 15:00:00 | 311.60 | 316.48 | 317.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 301.72 | 314.81 | 316.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 301.48 | 314.81 | 316.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 300.91 | 314.81 | 316.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 14:15:00 | 312.40 | 312.36 | 314.66 | SL hit (close>ema200) qty=0.50 sl=312.36 alert=retest2 |

### Cycle 71 — BUY (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 11:15:00 | 317.75 | 312.71 | 312.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 12:15:00 | 328.25 | 315.81 | 313.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 15:15:00 | 328.50 | 328.92 | 323.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-09 09:15:00 | 316.40 | 328.92 | 323.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 316.40 | 326.41 | 323.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:00:00 | 316.40 | 326.41 | 323.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 314.55 | 324.04 | 322.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 11:00:00 | 314.55 | 324.04 | 322.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2026-03-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 12:15:00 | 313.40 | 320.31 | 320.93 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 12:15:00 | 320.30 | 318.41 | 318.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 09:15:00 | 325.30 | 320.48 | 319.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 323.00 | 326.99 | 324.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 323.00 | 326.99 | 324.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 323.00 | 326.99 | 324.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 323.00 | 326.99 | 324.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 321.15 | 325.82 | 323.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:00:00 | 321.15 | 325.82 | 323.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 320.85 | 324.82 | 323.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 12:00:00 | 320.85 | 324.82 | 323.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2026-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 13:15:00 | 317.70 | 322.11 | 322.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 12:15:00 | 313.55 | 317.84 | 319.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 11:15:00 | 315.85 | 315.58 | 317.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 12:00:00 | 315.85 | 315.58 | 317.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 318.00 | 316.14 | 317.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 318.00 | 316.14 | 317.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 316.30 | 316.17 | 317.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:45:00 | 318.30 | 316.17 | 317.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 318.35 | 316.39 | 317.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 318.50 | 316.39 | 317.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 322.45 | 317.60 | 317.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 322.45 | 317.60 | 317.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 322.00 | 318.48 | 318.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 324.00 | 319.58 | 318.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 319.10 | 320.96 | 319.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 319.10 | 320.96 | 319.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 319.10 | 320.96 | 319.77 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 316.00 | 319.13 | 319.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 313.90 | 318.08 | 318.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 297.85 | 295.83 | 302.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 297.85 | 295.83 | 302.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 302.00 | 297.93 | 301.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 15:00:00 | 302.00 | 297.93 | 301.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 300.65 | 298.47 | 301.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 307.60 | 298.47 | 301.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 307.90 | 300.36 | 302.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 306.50 | 300.36 | 302.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 306.30 | 301.55 | 302.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 11:15:00 | 305.50 | 301.55 | 302.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 12:45:00 | 305.15 | 302.89 | 303.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 13:15:00 | 304.40 | 303.19 | 303.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 304.40 | 303.19 | 303.16 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 14:15:00 | 299.90 | 302.53 | 302.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 09:15:00 | 292.85 | 300.19 | 301.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 288.55 | 280.49 | 286.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 288.55 | 280.49 | 286.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 288.55 | 280.49 | 286.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 288.55 | 280.49 | 286.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 280.90 | 280.57 | 285.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:30:00 | 280.55 | 280.66 | 285.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 280.65 | 280.67 | 284.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:00:00 | 280.00 | 280.67 | 284.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 15:15:00 | 289.90 | 284.40 | 283.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 289.90 | 284.40 | 283.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 10:15:00 | 300.00 | 287.54 | 285.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 14:15:00 | 302.00 | 302.12 | 294.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-06 15:00:00 | 302.00 | 302.12 | 294.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 297.85 | 300.00 | 295.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 11:30:00 | 296.05 | 300.00 | 295.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 297.90 | 298.91 | 296.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 14:45:00 | 297.20 | 298.91 | 296.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 297.80 | 298.69 | 296.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 304.00 | 298.69 | 296.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-20 15:15:00 | 334.40 | 327.58 | 323.06 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 327.60 | 332.80 | 333.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 326.75 | 331.59 | 332.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 329.80 | 328.54 | 330.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 329.80 | 328.54 | 330.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 329.80 | 328.54 | 330.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 330.95 | 328.54 | 330.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 329.55 | 328.74 | 330.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 328.50 | 328.74 | 330.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:45:00 | 328.50 | 328.99 | 330.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 09:30:00 | 328.45 | 328.91 | 329.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 10:15:00 | 328.60 | 328.91 | 329.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 327.55 | 326.93 | 328.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:30:00 | 325.85 | 327.10 | 327.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:45:00 | 325.85 | 324.41 | 324.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 324.65 | 324.41 | 324.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 13:15:00 | 325.25 | 324.58 | 324.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2026-05-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 13:15:00 | 325.25 | 324.58 | 324.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 327.55 | 325.49 | 325.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 10:15:00 | 324.50 | 325.49 | 325.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 10:15:00 | 324.50 | 325.49 | 325.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 324.50 | 325.49 | 325.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:45:00 | 323.75 | 325.49 | 325.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 324.95 | 325.38 | 325.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:45:00 | 324.35 | 325.38 | 325.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 326.00 | 325.51 | 325.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 14:15:00 | 327.20 | 325.61 | 325.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:30:00 | 328.20 | 326.86 | 326.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 13:00:00 | 327.55 | 327.25 | 326.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 14:15:00 | 366.00 | 2025-05-21 13:15:00 | 371.40 | STOP_HIT | 1.00 | 1.48% |
| BUY | retest2 | 2025-05-16 09:15:00 | 366.60 | 2025-05-21 13:15:00 | 371.40 | STOP_HIT | 1.00 | 1.31% |
| BUY | retest2 | 2025-05-16 10:45:00 | 365.50 | 2025-05-21 13:15:00 | 371.40 | STOP_HIT | 1.00 | 1.61% |
| BUY | retest2 | 2025-06-04 10:00:00 | 377.00 | 2025-06-05 14:15:00 | 374.45 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-06-04 10:30:00 | 376.95 | 2025-06-05 14:15:00 | 374.45 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-06-04 13:45:00 | 377.00 | 2025-06-05 14:15:00 | 374.45 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-06-05 09:15:00 | 377.20 | 2025-06-05 14:15:00 | 374.45 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-06-09 09:15:00 | 375.45 | 2025-06-11 13:15:00 | 373.15 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-06-09 10:45:00 | 377.00 | 2025-06-11 13:15:00 | 373.15 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-06-17 12:30:00 | 358.50 | 2025-06-20 09:15:00 | 340.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 12:30:00 | 358.50 | 2025-06-20 13:15:00 | 346.90 | STOP_HIT | 0.50 | 3.24% |
| BUY | retest2 | 2025-06-30 09:15:00 | 364.15 | 2025-07-02 13:15:00 | 363.70 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-07-08 09:15:00 | 370.55 | 2025-07-21 09:15:00 | 379.25 | STOP_HIT | 1.00 | 2.35% |
| BUY | retest2 | 2025-07-08 09:45:00 | 370.85 | 2025-07-21 09:15:00 | 379.25 | STOP_HIT | 1.00 | 2.27% |
| BUY | retest2 | 2025-07-25 11:15:00 | 381.45 | 2025-07-25 11:15:00 | 377.40 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-07-29 10:45:00 | 376.20 | 2025-07-30 09:15:00 | 383.50 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-08-05 09:45:00 | 374.90 | 2025-08-07 10:15:00 | 356.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 09:45:00 | 374.90 | 2025-08-07 15:15:00 | 361.00 | STOP_HIT | 0.50 | 3.71% |
| SELL | retest2 | 2025-09-01 14:15:00 | 396.55 | 2025-09-02 11:15:00 | 403.45 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-09-08 12:30:00 | 415.35 | 2025-09-09 13:15:00 | 407.60 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-09-15 14:45:00 | 399.00 | 2025-09-15 15:15:00 | 403.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-16 12:15:00 | 399.45 | 2025-09-16 12:15:00 | 402.55 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-10-09 11:15:00 | 390.65 | 2025-10-10 14:15:00 | 384.95 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-10-09 11:45:00 | 390.90 | 2025-10-10 14:15:00 | 384.95 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-10-09 13:00:00 | 390.75 | 2025-10-10 14:15:00 | 384.95 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-10-09 14:30:00 | 390.65 | 2025-10-10 14:15:00 | 384.95 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-10-15 12:30:00 | 380.45 | 2025-10-15 14:15:00 | 386.45 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-10-21 13:45:00 | 390.70 | 2025-10-24 15:15:00 | 384.70 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-10-24 09:30:00 | 387.35 | 2025-10-24 15:15:00 | 384.70 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-10-24 10:00:00 | 387.10 | 2025-10-24 15:15:00 | 384.70 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-10-24 11:45:00 | 387.90 | 2025-10-24 15:15:00 | 384.70 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-10-30 11:00:00 | 392.50 | 2025-11-03 10:15:00 | 389.10 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-11-10 09:15:00 | 388.80 | 2025-11-18 10:15:00 | 384.85 | STOP_HIT | 1.00 | 1.02% |
| SELL | retest2 | 2025-11-10 10:45:00 | 388.50 | 2025-11-18 10:15:00 | 384.85 | STOP_HIT | 1.00 | 0.94% |
| SELL | retest2 | 2025-11-11 12:30:00 | 388.20 | 2025-11-18 10:15:00 | 384.85 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2025-12-02 09:15:00 | 380.80 | 2025-12-04 13:15:00 | 376.75 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-12-02 10:00:00 | 378.80 | 2025-12-04 13:15:00 | 376.75 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-12-02 12:15:00 | 377.65 | 2025-12-04 13:15:00 | 376.75 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-12-04 09:30:00 | 378.05 | 2025-12-04 13:15:00 | 376.75 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2026-01-14 13:30:00 | 344.40 | 2026-01-20 15:15:00 | 327.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:15:00 | 343.85 | 2026-01-21 09:15:00 | 326.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 13:30:00 | 344.40 | 2026-01-21 12:15:00 | 333.60 | STOP_HIT | 0.50 | 3.14% |
| SELL | retest2 | 2026-01-16 09:15:00 | 343.85 | 2026-01-21 12:15:00 | 333.60 | STOP_HIT | 0.50 | 2.98% |
| BUY | retest2 | 2026-02-05 12:15:00 | 350.60 | 2026-02-11 10:15:00 | 338.45 | STOP_HIT | 1.00 | -3.47% |
| BUY | retest2 | 2026-02-06 09:15:00 | 352.25 | 2026-02-11 10:15:00 | 338.45 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest2 | 2026-02-06 13:30:00 | 350.35 | 2026-02-11 10:15:00 | 338.45 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2026-02-25 11:00:00 | 317.60 | 2026-03-02 09:15:00 | 301.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:45:00 | 317.35 | 2026-03-02 09:15:00 | 301.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 09:15:00 | 315.95 | 2026-03-02 09:15:00 | 300.91 | PARTIAL | 0.50 | 4.76% |
| SELL | retest2 | 2026-02-25 11:00:00 | 317.60 | 2026-03-02 14:15:00 | 312.40 | STOP_HIT | 0.50 | 1.64% |
| SELL | retest2 | 2026-02-25 11:45:00 | 317.35 | 2026-03-02 14:15:00 | 312.40 | STOP_HIT | 0.50 | 1.56% |
| SELL | retest2 | 2026-02-26 09:15:00 | 315.95 | 2026-03-02 14:15:00 | 312.40 | STOP_HIT | 0.50 | 1.12% |
| SELL | retest2 | 2026-02-27 09:15:00 | 316.75 | 2026-03-05 11:15:00 | 317.75 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2026-02-27 13:15:00 | 315.00 | 2026-03-05 11:15:00 | 317.75 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-02-27 15:00:00 | 311.60 | 2026-03-05 11:15:00 | 317.75 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-03-25 11:15:00 | 305.50 | 2026-03-25 13:15:00 | 304.40 | STOP_HIT | 1.00 | 0.36% |
| SELL | retest2 | 2026-03-25 12:45:00 | 305.15 | 2026-03-25 13:15:00 | 304.40 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2026-04-01 11:30:00 | 280.55 | 2026-04-02 15:15:00 | 289.90 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2026-04-01 13:30:00 | 280.65 | 2026-04-02 15:15:00 | 289.90 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2026-04-01 14:00:00 | 280.00 | 2026-04-02 15:15:00 | 289.90 | STOP_HIT | 1.00 | -3.54% |
| BUY | retest2 | 2026-04-08 09:15:00 | 304.00 | 2026-04-20 15:15:00 | 334.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-27 11:15:00 | 328.50 | 2026-05-04 13:15:00 | 325.25 | STOP_HIT | 1.00 | 0.99% |
| SELL | retest2 | 2026-04-27 14:45:00 | 328.50 | 2026-05-04 13:15:00 | 325.25 | STOP_HIT | 1.00 | 0.99% |
| SELL | retest2 | 2026-04-28 09:30:00 | 328.45 | 2026-05-04 13:15:00 | 325.25 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2026-04-28 10:15:00 | 328.60 | 2026-05-04 13:15:00 | 325.25 | STOP_HIT | 1.00 | 1.02% |
| SELL | retest2 | 2026-04-29 13:30:00 | 325.85 | 2026-05-04 13:15:00 | 325.25 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2026-05-04 12:45:00 | 325.85 | 2026-05-04 13:15:00 | 325.25 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2026-05-04 13:15:00 | 324.65 | 2026-05-04 13:15:00 | 325.25 | STOP_HIT | 1.00 | -0.18% |
