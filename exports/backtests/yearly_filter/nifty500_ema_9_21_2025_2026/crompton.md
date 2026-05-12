# Crompton Greaves Consumer Electricals Ltd. (CROMPTON)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 293.55
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 67 |
| ALERT1 | 45 |
| ALERT2 | 44 |
| ALERT2_SKIP | 24 |
| ALERT3 | 124 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 47 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 53 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 32
- **Target hits / Stop hits / Partials:** 2 / 48 / 3
- **Avg / median % per leg:** 0.44% / -0.48%
- **Sum % (uncompounded):** 23.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 8 | 50.0% | 2 | 14 | 0 | 1.07% | 17.0% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.70% | -3.4% |
| BUY @ 3rd Alert (retest2) | 14 | 8 | 57.1% | 2 | 12 | 0 | 1.46% | 20.4% |
| SELL (all) | 37 | 13 | 35.1% | 0 | 34 | 3 | 0.17% | 6.3% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.41% | 8.8% |
| SELL @ 3rd Alert (retest2) | 35 | 11 | 31.4% | 0 | 33 | 2 | -0.07% | -2.5% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | 1.35% | 5.4% |
| retest2 (combined) | 49 | 19 | 38.8% | 2 | 45 | 2 | 0.37% | 17.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 332.60 | 327.72 | 327.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 333.65 | 328.91 | 328.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 329.85 | 332.63 | 331.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 12:15:00 | 329.85 | 332.63 | 331.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 329.85 | 332.63 | 331.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:00:00 | 329.85 | 332.63 | 331.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 330.30 | 332.17 | 330.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:30:00 | 330.25 | 332.17 | 330.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 331.35 | 331.78 | 330.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 333.45 | 331.78 | 330.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-14 09:15:00 | 329.45 | 331.32 | 330.84 | SL hit (close<static) qty=1.00 sl=330.50 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 13:15:00 | 329.65 | 330.45 | 330.54 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 15:15:00 | 331.80 | 330.84 | 330.71 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 09:15:00 | 327.00 | 330.07 | 330.37 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 09:15:00 | 347.35 | 331.95 | 330.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 11:15:00 | 350.50 | 337.96 | 333.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 348.10 | 349.87 | 345.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 09:45:00 | 347.65 | 349.87 | 345.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 345.75 | 348.50 | 346.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 345.75 | 348.50 | 346.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 346.50 | 348.10 | 346.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 346.10 | 348.10 | 346.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 345.40 | 347.56 | 346.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 345.40 | 347.56 | 346.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 348.10 | 347.67 | 346.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:00:00 | 349.25 | 348.01 | 346.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:45:00 | 349.25 | 348.21 | 347.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 13:15:00 | 345.60 | 346.60 | 346.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 13:15:00 | 345.60 | 346.60 | 346.74 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 348.55 | 346.81 | 346.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 10:15:00 | 349.95 | 347.44 | 347.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 354.55 | 355.06 | 352.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 10:00:00 | 354.55 | 355.06 | 352.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 353.75 | 354.80 | 352.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:00:00 | 353.75 | 354.80 | 352.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 357.80 | 355.40 | 353.18 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 352.60 | 353.63 | 353.65 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 13:15:00 | 358.80 | 354.56 | 353.99 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 13:15:00 | 352.00 | 354.34 | 354.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 351.15 | 353.12 | 353.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 13:15:00 | 352.85 | 352.63 | 353.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 13:15:00 | 352.85 | 352.63 | 353.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 352.85 | 352.63 | 353.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:00:00 | 352.85 | 352.63 | 353.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 354.30 | 352.96 | 353.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 354.30 | 352.96 | 353.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 353.20 | 353.01 | 353.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 354.80 | 353.01 | 353.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 351.60 | 352.73 | 353.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 11:00:00 | 350.10 | 352.20 | 352.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 13:45:00 | 350.35 | 349.05 | 349.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 09:15:00 | 350.95 | 349.84 | 349.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 09:15:00 | 350.95 | 349.84 | 349.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 13:15:00 | 351.95 | 350.66 | 350.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 11:15:00 | 353.80 | 354.08 | 352.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 12:00:00 | 353.80 | 354.08 | 352.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 353.70 | 354.41 | 353.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:00:00 | 353.70 | 354.41 | 353.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 352.25 | 353.98 | 353.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:00:00 | 352.25 | 353.98 | 353.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 350.60 | 353.30 | 353.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:00:00 | 350.60 | 353.30 | 353.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 352.80 | 353.20 | 353.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 348.05 | 352.11 | 352.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 344.05 | 344.04 | 346.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 344.05 | 344.04 | 346.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 344.55 | 344.13 | 346.22 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 15:15:00 | 349.55 | 347.13 | 346.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 351.20 | 347.94 | 347.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 12:15:00 | 348.25 | 348.58 | 347.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 12:15:00 | 348.25 | 348.58 | 347.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 348.25 | 348.58 | 347.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:45:00 | 347.80 | 348.58 | 347.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 350.30 | 348.93 | 348.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 14:15:00 | 350.75 | 348.93 | 348.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 10:15:00 | 347.50 | 348.79 | 348.31 | SL hit (close<static) qty=1.00 sl=347.55 alert=retest2 |

### Cycle 14 — SELL (started 2025-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 13:15:00 | 346.05 | 347.69 | 347.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 345.30 | 347.00 | 347.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 343.70 | 342.27 | 344.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 10:15:00 | 343.70 | 342.27 | 344.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 343.70 | 342.27 | 344.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 343.70 | 342.27 | 344.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 343.15 | 342.44 | 344.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:30:00 | 343.55 | 342.44 | 344.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 344.40 | 342.64 | 343.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 344.40 | 342.64 | 343.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 344.60 | 343.03 | 343.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 341.25 | 343.03 | 343.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 12:15:00 | 345.90 | 343.27 | 343.64 | SL hit (close>static) qty=1.00 sl=345.40 alert=retest2 |

### Cycle 15 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 349.30 | 344.48 | 344.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 349.65 | 346.75 | 345.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 10:15:00 | 351.55 | 352.36 | 350.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 11:00:00 | 351.55 | 352.36 | 350.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 356.00 | 357.11 | 355.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:45:00 | 356.15 | 357.11 | 355.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 356.00 | 356.89 | 355.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:45:00 | 355.80 | 356.89 | 355.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 355.55 | 356.56 | 355.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:30:00 | 355.35 | 356.56 | 355.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 355.10 | 356.26 | 355.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:45:00 | 355.35 | 356.26 | 355.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 355.00 | 356.01 | 355.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 354.20 | 356.01 | 355.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 354.05 | 355.62 | 355.34 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 350.60 | 354.38 | 354.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 347.50 | 351.44 | 353.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 348.45 | 347.88 | 349.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 348.45 | 347.88 | 349.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 348.45 | 347.88 | 349.96 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 10:15:00 | 355.30 | 351.33 | 350.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 15:15:00 | 356.00 | 353.73 | 352.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 11:15:00 | 351.85 | 353.73 | 352.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 11:15:00 | 351.85 | 353.73 | 352.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 351.85 | 353.73 | 352.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:45:00 | 352.00 | 353.73 | 352.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 352.10 | 353.41 | 352.67 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 346.90 | 351.30 | 351.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 13:15:00 | 343.60 | 348.05 | 349.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 10:15:00 | 344.20 | 343.85 | 345.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 10:45:00 | 344.00 | 343.85 | 345.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 339.65 | 338.43 | 339.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 342.00 | 338.43 | 339.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 342.20 | 339.19 | 340.10 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 344.00 | 341.10 | 340.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 13:15:00 | 347.80 | 342.44 | 341.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 12:15:00 | 349.50 | 349.53 | 347.57 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-18 09:30:00 | 351.60 | 349.92 | 348.39 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-18 10:30:00 | 350.25 | 350.20 | 348.66 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 348.70 | 349.90 | 348.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:00:00 | 348.70 | 349.90 | 348.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 344.95 | 348.91 | 348.33 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-18 12:15:00 | 344.95 | 348.91 | 348.33 | SL hit (close<ema400) qty=1.00 sl=348.33 alert=retest1 |

### Cycle 20 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 343.70 | 347.87 | 347.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 340.75 | 345.25 | 346.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 11:15:00 | 338.25 | 337.12 | 339.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 12:00:00 | 338.25 | 337.12 | 339.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 340.00 | 337.70 | 339.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 12:45:00 | 340.25 | 337.70 | 339.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 340.15 | 338.19 | 339.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 337.10 | 338.91 | 339.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 09:15:00 | 320.25 | 322.44 | 323.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 12:15:00 | 324.05 | 322.56 | 323.55 | SL hit (close>ema200) qty=0.50 sl=322.56 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 326.60 | 324.37 | 324.21 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 322.75 | 324.04 | 324.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 10:15:00 | 321.75 | 323.59 | 323.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 14:15:00 | 324.00 | 323.03 | 323.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 14:15:00 | 324.00 | 323.03 | 323.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 324.00 | 323.03 | 323.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 15:00:00 | 324.00 | 323.03 | 323.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 323.00 | 323.02 | 323.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 322.00 | 323.02 | 323.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 323.30 | 319.31 | 318.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 09:15:00 | 323.30 | 319.31 | 318.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 327.20 | 321.63 | 320.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 327.30 | 327.61 | 325.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 325.05 | 327.26 | 326.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 325.05 | 327.26 | 326.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:00:00 | 325.05 | 327.26 | 326.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 324.40 | 326.69 | 326.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 324.40 | 326.69 | 326.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 12:15:00 | 323.55 | 325.54 | 325.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 14:15:00 | 321.10 | 324.28 | 325.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 321.35 | 319.22 | 321.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 321.35 | 319.22 | 321.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 321.35 | 319.22 | 321.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:00:00 | 321.35 | 319.22 | 321.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 321.90 | 319.76 | 321.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:15:00 | 322.50 | 319.76 | 321.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 322.35 | 320.27 | 321.39 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 09:15:00 | 325.70 | 322.55 | 322.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-26 13:15:00 | 327.10 | 324.72 | 323.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-28 09:15:00 | 321.65 | 324.19 | 323.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 09:15:00 | 321.65 | 324.19 | 323.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 321.65 | 324.19 | 323.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:30:00 | 321.95 | 324.19 | 323.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 322.00 | 323.75 | 323.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 11:30:00 | 323.30 | 323.83 | 323.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 11:15:00 | 329.60 | 331.99 | 332.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 329.60 | 331.99 | 332.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 326.65 | 330.52 | 331.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 328.45 | 328.37 | 329.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 13:45:00 | 328.45 | 328.37 | 329.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 328.30 | 328.11 | 329.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 323.75 | 327.17 | 328.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 12:15:00 | 317.55 | 315.69 | 315.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 12:15:00 | 317.55 | 315.69 | 315.48 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 12:15:00 | 314.30 | 315.63 | 315.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 09:15:00 | 311.90 | 314.33 | 315.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 14:15:00 | 299.70 | 299.56 | 302.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 14:30:00 | 300.15 | 299.56 | 302.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 290.40 | 291.14 | 293.05 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 10:15:00 | 294.85 | 293.18 | 293.15 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 10:15:00 | 291.10 | 293.18 | 293.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 11:15:00 | 289.05 | 292.36 | 292.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 289.80 | 285.79 | 286.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 289.80 | 285.79 | 286.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 289.80 | 285.79 | 286.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 289.80 | 285.79 | 286.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 288.55 | 286.34 | 286.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 11:15:00 | 288.00 | 286.34 | 286.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 11:15:00 | 289.80 | 287.03 | 286.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 289.80 | 287.03 | 286.97 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 284.85 | 286.78 | 286.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 12:15:00 | 284.00 | 285.94 | 286.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 285.50 | 282.38 | 283.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 285.50 | 282.38 | 283.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 285.50 | 282.38 | 283.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:45:00 | 286.00 | 282.38 | 283.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 285.60 | 283.02 | 283.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 286.80 | 283.02 | 283.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 288.00 | 284.74 | 284.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 289.00 | 286.84 | 285.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 11:15:00 | 286.80 | 286.99 | 285.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 11:15:00 | 286.80 | 286.99 | 285.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 286.80 | 286.99 | 285.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:45:00 | 286.50 | 286.99 | 285.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 288.00 | 287.96 | 287.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:30:00 | 287.20 | 287.96 | 287.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 286.55 | 287.68 | 287.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 286.55 | 287.68 | 287.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 287.35 | 287.61 | 287.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 286.50 | 287.61 | 287.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 287.60 | 287.61 | 287.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:30:00 | 287.55 | 287.61 | 287.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 287.50 | 287.59 | 287.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:30:00 | 286.60 | 287.34 | 287.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 286.60 | 287.19 | 287.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:30:00 | 285.90 | 287.19 | 287.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 286.65 | 287.08 | 287.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 15:15:00 | 288.00 | 287.20 | 287.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 288.55 | 291.00 | 291.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 288.55 | 291.00 | 291.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 09:15:00 | 286.30 | 289.77 | 290.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 13:15:00 | 283.55 | 282.95 | 284.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 14:00:00 | 283.55 | 282.95 | 284.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 283.95 | 283.15 | 284.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:30:00 | 284.65 | 283.15 | 284.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 284.00 | 283.50 | 284.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:15:00 | 282.85 | 283.43 | 284.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 12:15:00 | 283.00 | 283.39 | 284.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 13:15:00 | 282.75 | 283.36 | 284.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:15:00 | 282.00 | 283.25 | 283.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 281.85 | 282.97 | 283.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:15:00 | 281.10 | 282.97 | 283.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 11:30:00 | 281.05 | 281.97 | 283.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 285.70 | 279.61 | 279.75 | SL hit (close>static) qty=1.00 sl=284.35 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 282.45 | 280.17 | 279.99 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 275.10 | 279.49 | 279.82 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 12:15:00 | 280.05 | 279.38 | 279.29 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 277.25 | 279.54 | 279.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 275.10 | 277.80 | 278.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 11:15:00 | 275.75 | 274.47 | 275.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 11:15:00 | 275.75 | 274.47 | 275.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 275.75 | 274.47 | 275.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 11:45:00 | 275.90 | 274.47 | 275.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 273.90 | 274.36 | 275.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 13:30:00 | 273.10 | 274.23 | 275.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 14:45:00 | 273.30 | 274.07 | 275.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 11:00:00 | 273.20 | 273.86 | 274.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 13:15:00 | 273.25 | 273.80 | 274.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 274.65 | 274.01 | 274.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 273.30 | 274.01 | 274.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 272.50 | 273.71 | 274.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 12:30:00 | 271.90 | 272.96 | 273.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 13:15:00 | 269.00 | 267.47 | 267.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 269.00 | 267.47 | 267.40 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 267.05 | 267.47 | 267.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 13:15:00 | 267.00 | 267.37 | 267.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 10:15:00 | 264.40 | 263.61 | 264.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 10:15:00 | 264.40 | 263.61 | 264.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 264.40 | 263.61 | 264.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:45:00 | 264.65 | 263.61 | 264.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 260.20 | 258.25 | 260.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:45:00 | 261.10 | 258.25 | 260.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 260.20 | 258.64 | 260.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 260.05 | 258.64 | 260.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 260.45 | 259.00 | 260.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:30:00 | 260.15 | 259.00 | 260.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 260.60 | 259.32 | 260.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 13:00:00 | 260.60 | 259.32 | 260.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 259.55 | 259.37 | 260.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 14:30:00 | 258.70 | 259.29 | 260.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 262.35 | 259.67 | 260.02 | SL hit (close>static) qty=1.00 sl=260.95 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 260.60 | 260.28 | 260.25 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 15:15:00 | 260.00 | 260.20 | 260.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 256.70 | 259.50 | 259.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 13:15:00 | 254.15 | 253.91 | 255.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 13:45:00 | 254.35 | 253.91 | 255.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 254.80 | 253.89 | 255.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:30:00 | 255.55 | 253.89 | 255.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 253.40 | 253.79 | 255.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 11:45:00 | 252.55 | 253.61 | 254.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 12:30:00 | 252.45 | 253.46 | 254.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:30:00 | 252.50 | 253.23 | 254.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 09:45:00 | 252.50 | 251.55 | 252.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 250.90 | 251.42 | 252.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 12:30:00 | 250.50 | 251.34 | 252.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 13:15:00 | 253.05 | 251.68 | 252.16 | SL hit (close>static) qty=1.00 sl=252.30 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 15:15:00 | 255.10 | 252.71 | 252.56 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 09:15:00 | 251.95 | 252.85 | 252.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 10:15:00 | 251.25 | 252.53 | 252.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 09:15:00 | 258.25 | 252.02 | 252.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 09:15:00 | 258.25 | 252.02 | 252.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 258.25 | 252.02 | 252.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:00:00 | 258.25 | 252.02 | 252.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 10:15:00 | 263.05 | 254.23 | 253.14 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 12:15:00 | 251.60 | 253.50 | 253.76 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 256.50 | 254.25 | 254.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 258.50 | 255.10 | 254.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 259.10 | 259.26 | 258.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:15:00 | 260.30 | 259.26 | 258.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 257.85 | 258.79 | 258.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:30:00 | 257.45 | 258.79 | 258.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 257.00 | 258.44 | 257.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:45:00 | 257.10 | 258.44 | 257.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 256.85 | 258.12 | 257.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:00:00 | 256.85 | 258.12 | 257.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 257.25 | 257.92 | 257.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:45:00 | 257.65 | 257.92 | 257.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 257.55 | 257.84 | 257.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 258.10 | 257.84 | 257.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 257.95 | 257.76 | 257.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:45:00 | 258.05 | 257.76 | 257.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 258.50 | 257.91 | 257.81 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 257.05 | 257.68 | 257.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 255.45 | 257.20 | 257.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 11:15:00 | 252.65 | 252.19 | 253.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 12:00:00 | 252.65 | 252.19 | 253.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 253.20 | 252.54 | 253.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:00:00 | 253.20 | 252.54 | 253.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 251.15 | 252.18 | 253.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 11:15:00 | 250.70 | 251.95 | 252.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:30:00 | 250.55 | 251.37 | 252.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 254.50 | 251.27 | 251.90 | SL hit (close>static) qty=1.00 sl=254.15 alert=retest2 |

### Cycle 49 — BUY (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 10:15:00 | 256.50 | 252.84 | 252.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 12:15:00 | 258.95 | 254.42 | 253.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 260.80 | 261.87 | 259.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 10:00:00 | 260.80 | 261.87 | 259.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 259.75 | 261.44 | 259.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 259.75 | 261.44 | 259.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 259.45 | 261.04 | 259.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 259.45 | 261.04 | 259.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 258.80 | 260.59 | 259.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:00:00 | 258.80 | 260.59 | 259.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 257.90 | 260.06 | 259.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:30:00 | 257.25 | 260.06 | 259.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 257.50 | 259.14 | 259.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 255.95 | 258.50 | 258.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 251.35 | 251.11 | 253.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 252.65 | 251.11 | 253.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 249.50 | 250.73 | 252.78 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 11:15:00 | 254.45 | 252.73 | 252.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 255.85 | 253.35 | 252.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 12:15:00 | 251.50 | 253.81 | 253.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 12:15:00 | 251.50 | 253.81 | 253.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 251.50 | 253.81 | 253.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:00:00 | 251.50 | 253.81 | 253.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 13:15:00 | 250.40 | 253.13 | 253.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 249.60 | 251.91 | 252.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 233.55 | 233.24 | 237.21 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 10:45:00 | 232.25 | 232.93 | 236.71 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 12:15:00 | 220.64 | 224.33 | 227.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-27 15:15:00 | 223.40 | 223.21 | 226.42 | SL hit (close>ema200) qty=0.50 sl=223.21 alert=retest1 |

### Cycle 53 — BUY (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 09:15:00 | 230.29 | 223.28 | 223.22 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 220.14 | 223.61 | 223.90 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 232.56 | 225.40 | 224.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 237.80 | 229.43 | 226.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 243.30 | 244.60 | 240.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 254.78 | 246.85 | 243.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 254.78 | 246.85 | 243.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 10:15:00 | 256.38 | 246.85 | 243.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 14:15:00 | 265.58 | 268.66 | 269.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 265.58 | 268.66 | 269.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 265.40 | 268.01 | 268.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 09:15:00 | 269.85 | 268.38 | 268.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 09:15:00 | 269.85 | 268.38 | 268.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 269.85 | 268.38 | 268.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:45:00 | 270.88 | 268.38 | 268.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 268.64 | 268.43 | 268.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 11:30:00 | 268.25 | 268.41 | 268.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 254.84 | 257.51 | 259.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 15:15:00 | 256.40 | 253.89 | 256.52 | SL hit (close>ema200) qty=0.50 sl=253.89 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 247.80 | 245.71 | 245.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 11:15:00 | 248.20 | 246.21 | 245.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 246.80 | 246.87 | 246.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 15:00:00 | 246.80 | 246.87 | 246.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 245.65 | 246.78 | 246.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 244.90 | 246.78 | 246.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 251.10 | 247.65 | 246.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:00:00 | 254.20 | 248.96 | 247.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 240.50 | 246.72 | 246.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 240.50 | 246.72 | 246.97 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 09:15:00 | 251.40 | 246.17 | 245.83 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 243.10 | 247.58 | 247.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 241.90 | 244.88 | 246.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 10:15:00 | 244.55 | 244.49 | 245.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 10:45:00 | 245.00 | 244.49 | 245.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 245.70 | 244.73 | 245.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:45:00 | 246.20 | 244.73 | 245.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 245.45 | 244.87 | 245.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:45:00 | 245.40 | 244.87 | 245.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 244.55 | 244.81 | 245.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:30:00 | 245.40 | 244.81 | 245.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 239.00 | 235.70 | 238.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 238.70 | 235.70 | 238.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 238.60 | 236.28 | 238.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 239.00 | 236.28 | 238.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 237.55 | 236.53 | 238.12 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 243.75 | 238.98 | 238.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 245.10 | 240.93 | 239.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 235.65 | 240.73 | 240.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 235.65 | 240.73 | 240.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 235.65 | 240.73 | 240.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 235.65 | 240.73 | 240.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 234.00 | 239.38 | 239.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 233.35 | 236.63 | 238.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 231.84 | 228.08 | 231.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 231.84 | 228.08 | 231.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 231.84 | 228.08 | 231.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 230.26 | 228.08 | 231.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 229.10 | 228.29 | 231.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 226.24 | 230.75 | 231.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 12:00:00 | 228.43 | 228.93 | 230.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 232.50 | 229.95 | 230.49 | SL hit (close>static) qty=1.00 sl=232.20 alert=retest2 |

### Cycle 63 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 231.18 | 230.77 | 230.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 233.29 | 231.27 | 230.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 231.97 | 232.70 | 231.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 231.97 | 232.70 | 231.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 231.97 | 232.70 | 231.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:30:00 | 235.10 | 232.67 | 231.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 12:30:00 | 234.56 | 233.19 | 232.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 15:00:00 | 234.95 | 233.66 | 232.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 240.06 | 233.73 | 232.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 12:15:00 | 238.80 | 241.84 | 239.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:00:00 | 238.80 | 241.84 | 239.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 237.75 | 241.02 | 239.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:45:00 | 237.62 | 241.02 | 239.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 237.97 | 240.70 | 240.27 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 11:15:00 | 238.70 | 240.01 | 240.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 11:15:00 | 238.70 | 240.01 | 240.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 14:15:00 | 237.62 | 239.34 | 239.69 | Break + close below crossover candle low |

### Cycle 65 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 244.15 | 240.11 | 239.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 10:15:00 | 246.43 | 241.37 | 240.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 13:15:00 | 261.02 | 261.89 | 258.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 14:00:00 | 261.02 | 261.89 | 258.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 260.52 | 261.63 | 259.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 11:45:00 | 260.25 | 261.63 | 259.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 259.30 | 261.17 | 259.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 13:00:00 | 259.30 | 261.17 | 259.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 258.72 | 260.68 | 259.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:00:00 | 258.72 | 260.68 | 259.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 258.88 | 260.32 | 259.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:30:00 | 258.34 | 260.32 | 259.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 258.61 | 259.48 | 259.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:30:00 | 257.94 | 259.48 | 259.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 260.62 | 259.67 | 259.39 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 253.30 | 258.59 | 258.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 10:15:00 | 252.30 | 257.33 | 258.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 252.82 | 251.31 | 253.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 252.82 | 251.31 | 253.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 252.82 | 251.31 | 253.35 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 257.25 | 254.94 | 254.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 258.70 | 256.48 | 255.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 270.45 | 273.73 | 269.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 270.45 | 273.73 | 269.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 270.45 | 273.73 | 269.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:15:00 | 267.36 | 273.73 | 269.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 266.73 | 272.33 | 268.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:00:00 | 266.73 | 272.33 | 268.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 268.53 | 271.57 | 268.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 12:15:00 | 269.88 | 271.57 | 268.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 13:00:00 | 269.24 | 271.10 | 268.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-08 13:15:00 | 296.87 | 291.96 | 287.61 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 09:15:00 | 333.45 | 2025-05-14 09:15:00 | 329.45 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-05-21 13:00:00 | 349.25 | 2025-05-22 13:15:00 | 345.60 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-05-21 14:45:00 | 349.25 | 2025-05-22 13:15:00 | 345.60 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-06-03 11:00:00 | 350.10 | 2025-06-06 09:15:00 | 350.95 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-06-05 13:45:00 | 350.35 | 2025-06-06 09:15:00 | 350.95 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-06-17 14:15:00 | 350.75 | 2025-06-18 10:15:00 | 347.50 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-06-23 09:15:00 | 341.25 | 2025-06-23 12:15:00 | 345.90 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest1 | 2025-07-18 09:30:00 | 351.60 | 2025-07-18 12:15:00 | 344.95 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest1 | 2025-07-18 10:30:00 | 350.25 | 2025-07-18 12:15:00 | 344.95 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-07-24 09:15:00 | 337.10 | 2025-08-04 09:15:00 | 320.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 09:15:00 | 337.10 | 2025-08-04 12:15:00 | 324.05 | STOP_HIT | 0.50 | 3.87% |
| SELL | retest2 | 2025-08-06 09:15:00 | 322.00 | 2025-08-14 09:15:00 | 323.30 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-08-28 11:30:00 | 323.30 | 2025-09-04 11:15:00 | 329.60 | STOP_HIT | 1.00 | 1.95% |
| SELL | retest2 | 2025-09-09 09:15:00 | 323.75 | 2025-09-17 12:15:00 | 317.55 | STOP_HIT | 1.00 | 1.92% |
| SELL | retest2 | 2025-10-10 11:15:00 | 288.00 | 2025-10-10 11:15:00 | 289.80 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-10-20 15:15:00 | 288.00 | 2025-10-28 10:15:00 | 288.55 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2025-11-04 11:15:00 | 282.85 | 2025-11-10 10:15:00 | 285.70 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-11-04 12:15:00 | 283.00 | 2025-11-10 10:15:00 | 285.70 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-11-04 13:15:00 | 282.75 | 2025-11-10 11:15:00 | 282.45 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-11-06 09:15:00 | 282.00 | 2025-11-10 11:15:00 | 282.45 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-11-06 10:15:00 | 281.10 | 2025-11-10 11:15:00 | 282.45 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-11-06 11:30:00 | 281.05 | 2025-11-10 11:15:00 | 282.45 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-11-18 13:30:00 | 273.10 | 2025-11-26 13:15:00 | 269.00 | STOP_HIT | 1.00 | 1.50% |
| SELL | retest2 | 2025-11-18 14:45:00 | 273.30 | 2025-11-26 13:15:00 | 269.00 | STOP_HIT | 1.00 | 1.57% |
| SELL | retest2 | 2025-11-19 11:00:00 | 273.20 | 2025-11-26 13:15:00 | 269.00 | STOP_HIT | 1.00 | 1.54% |
| SELL | retest2 | 2025-11-19 13:15:00 | 273.25 | 2025-11-26 13:15:00 | 269.00 | STOP_HIT | 1.00 | 1.56% |
| SELL | retest2 | 2025-11-20 12:30:00 | 271.90 | 2025-11-26 13:15:00 | 269.00 | STOP_HIT | 1.00 | 1.07% |
| SELL | retest2 | 2025-12-04 14:30:00 | 258.70 | 2025-12-05 10:15:00 | 262.35 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-12-10 11:45:00 | 252.55 | 2025-12-12 13:15:00 | 253.05 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-12-10 12:30:00 | 252.45 | 2025-12-12 15:15:00 | 255.10 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-12-10 13:30:00 | 252.50 | 2025-12-12 15:15:00 | 255.10 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-12-12 09:45:00 | 252.50 | 2025-12-12 15:15:00 | 255.10 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-12-12 12:30:00 | 250.50 | 2025-12-12 15:15:00 | 255.10 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2026-01-01 11:15:00 | 250.70 | 2026-01-02 10:15:00 | 254.50 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2026-01-01 12:30:00 | 250.55 | 2026-01-02 10:15:00 | 254.50 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2026-01-05 09:45:00 | 250.70 | 2026-01-05 10:15:00 | 256.50 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest1 | 2026-01-22 10:45:00 | 232.25 | 2026-01-27 12:15:00 | 220.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-22 10:45:00 | 232.25 | 2026-01-27 15:15:00 | 223.40 | STOP_HIT | 0.50 | 3.81% |
| SELL | retest2 | 2026-01-28 13:15:00 | 223.90 | 2026-02-01 09:15:00 | 230.29 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2026-01-29 09:15:00 | 222.75 | 2026-02-01 09:15:00 | 230.29 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2026-02-09 10:15:00 | 256.38 | 2026-02-19 14:15:00 | 265.58 | STOP_HIT | 1.00 | 3.59% |
| SELL | retest2 | 2026-02-20 11:30:00 | 268.25 | 2026-03-02 09:15:00 | 254.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-20 11:30:00 | 268.25 | 2026-03-02 15:15:00 | 256.40 | STOP_HIT | 0.50 | 4.42% |
| BUY | retest2 | 2026-03-12 12:00:00 | 254.20 | 2026-03-13 09:15:00 | 240.50 | STOP_HIT | 1.00 | -5.39% |
| SELL | retest2 | 2026-04-02 09:15:00 | 226.24 | 2026-04-02 14:15:00 | 232.50 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2026-04-02 12:00:00 | 228.43 | 2026-04-02 14:15:00 | 232.50 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-04-06 09:15:00 | 228.00 | 2026-04-06 12:15:00 | 231.18 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2026-04-07 10:30:00 | 235.10 | 2026-04-13 11:15:00 | 238.70 | STOP_HIT | 1.00 | 1.53% |
| BUY | retest2 | 2026-04-07 12:30:00 | 234.56 | 2026-04-13 11:15:00 | 238.70 | STOP_HIT | 1.00 | 1.77% |
| BUY | retest2 | 2026-04-07 15:00:00 | 234.95 | 2026-04-13 11:15:00 | 238.70 | STOP_HIT | 1.00 | 1.60% |
| BUY | retest2 | 2026-04-08 09:15:00 | 240.06 | 2026-04-13 11:15:00 | 238.70 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2026-04-30 12:15:00 | 269.88 | 2026-05-08 13:15:00 | 296.87 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-30 13:00:00 | 269.24 | 2026-05-08 13:15:00 | 296.16 | TARGET_HIT | 1.00 | 10.00% |
