# Indus Towers Ltd. (INDUSTOWER)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 402.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 146 |
| ALERT1 | 99 |
| ALERT2 | 99 |
| ALERT2_SKIP | 48 |
| ALERT3 | 271 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 130 |
| PARTIAL | 14 |
| TARGET_HIT | 8 |
| STOP_HIT | 128 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 150 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 53 / 97
- **Target hits / Stop hits / Partials:** 8 / 128 / 14
- **Avg / median % per leg:** 0.25% / -0.96%
- **Sum % (uncompounded):** 37.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 63 | 22 | 34.9% | 5 | 58 | 0 | -0.17% | -10.5% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.16% | -2.3% |
| BUY @ 3rd Alert (retest2) | 61 | 22 | 36.1% | 5 | 56 | 0 | -0.13% | -8.2% |
| SELL (all) | 87 | 31 | 35.6% | 3 | 70 | 14 | 0.55% | 47.6% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.12% | -4.5% |
| SELL @ 3rd Alert (retest2) | 83 | 31 | 37.3% | 3 | 66 | 14 | 0.63% | 52.0% |
| retest1 (combined) | 6 | 0 | 0.0% | 0 | 6 | 0 | -1.13% | -6.8% |
| retest2 (combined) | 144 | 53 | 36.8% | 8 | 122 | 14 | 0.30% | 43.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 337.85 | 331.81 | 331.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 12:15:00 | 339.80 | 333.41 | 332.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 09:15:00 | 340.75 | 340.85 | 338.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 10:00:00 | 340.75 | 340.85 | 338.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 337.80 | 340.27 | 338.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 12:00:00 | 337.80 | 340.27 | 338.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 340.40 | 340.29 | 338.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 347.80 | 340.40 | 339.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 12:30:00 | 342.00 | 342.98 | 342.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-21 14:15:00 | 338.70 | 341.78 | 341.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 14:15:00 | 338.70 | 341.78 | 341.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 12:15:00 | 338.00 | 339.79 | 340.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 14:15:00 | 339.55 | 339.37 | 340.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-22 15:00:00 | 339.55 | 339.37 | 340.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 345.95 | 340.74 | 340.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:30:00 | 344.80 | 340.74 | 340.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 10:15:00 | 343.90 | 341.38 | 341.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 09:15:00 | 354.00 | 345.15 | 343.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 13:15:00 | 346.70 | 347.48 | 345.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-24 14:00:00 | 346.70 | 347.48 | 345.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 346.50 | 347.29 | 345.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 14:30:00 | 348.20 | 347.29 | 345.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 344.65 | 346.62 | 345.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:00:00 | 344.65 | 346.62 | 345.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 345.35 | 346.36 | 345.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 11:15:00 | 345.40 | 346.36 | 345.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 11:15:00 | 345.40 | 346.17 | 345.35 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2024-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 15:15:00 | 343.00 | 344.61 | 344.80 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 09:15:00 | 346.90 | 345.07 | 344.99 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 13:15:00 | 344.00 | 345.07 | 345.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 14:15:00 | 341.55 | 344.37 | 344.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 10:15:00 | 346.15 | 342.60 | 343.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 10:15:00 | 346.15 | 342.60 | 343.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 346.15 | 342.60 | 343.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 11:00:00 | 346.15 | 342.60 | 343.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 11:15:00 | 343.70 | 342.82 | 343.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 15:00:00 | 343.05 | 343.20 | 343.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 15:15:00 | 345.65 | 343.69 | 343.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 15:15:00 | 345.65 | 343.69 | 343.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 11:15:00 | 346.65 | 344.65 | 344.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-31 15:15:00 | 344.90 | 345.86 | 344.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 15:15:00 | 344.90 | 345.86 | 344.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 344.90 | 345.86 | 344.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 12:45:00 | 352.50 | 347.55 | 345.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 327.60 | 348.94 | 348.22 | SL hit (close<static) qty=1.00 sl=343.55 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 297.95 | 338.74 | 343.65 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 344.00 | 333.91 | 333.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 12:15:00 | 346.15 | 340.61 | 337.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 347.55 | 347.62 | 344.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 09:15:00 | 347.50 | 347.62 | 344.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 343.00 | 346.70 | 344.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 10:00:00 | 343.00 | 346.70 | 344.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 10:15:00 | 345.00 | 346.36 | 344.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 12:00:00 | 346.15 | 346.32 | 344.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 12:45:00 | 348.00 | 346.90 | 344.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 10:30:00 | 346.05 | 346.95 | 345.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 13:15:00 | 342.00 | 344.66 | 344.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 13:15:00 | 342.00 | 344.66 | 344.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-13 10:15:00 | 340.05 | 342.82 | 343.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-14 09:15:00 | 343.75 | 341.26 | 342.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-14 09:15:00 | 343.75 | 341.26 | 342.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 343.75 | 341.26 | 342.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 10:00:00 | 343.75 | 341.26 | 342.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 10:15:00 | 341.50 | 341.31 | 342.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-14 11:15:00 | 340.95 | 341.31 | 342.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-14 13:15:00 | 345.10 | 342.02 | 342.38 | SL hit (close>static) qty=1.00 sl=344.85 alert=retest2 |

### Cycle 11 — BUY (started 2024-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 14:15:00 | 340.05 | 336.49 | 336.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 09:15:00 | 343.15 | 338.27 | 336.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 09:15:00 | 341.65 | 342.28 | 340.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-26 10:00:00 | 341.65 | 342.28 | 340.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 14:15:00 | 383.30 | 387.10 | 381.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 15:00:00 | 383.30 | 387.10 | 381.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 13:15:00 | 395.65 | 398.36 | 395.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 13:45:00 | 396.35 | 398.36 | 395.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 395.50 | 397.78 | 395.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 09:15:00 | 396.90 | 397.33 | 395.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 10:15:00 | 391.05 | 395.52 | 394.95 | SL hit (close<static) qty=1.00 sl=392.30 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 389.30 | 394.27 | 394.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 13:15:00 | 386.60 | 392.18 | 393.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 14:15:00 | 383.25 | 381.26 | 384.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-10 15:00:00 | 383.25 | 381.26 | 384.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 382.75 | 381.56 | 384.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 389.10 | 381.56 | 384.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 388.70 | 382.99 | 384.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:45:00 | 390.15 | 382.99 | 384.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 389.75 | 384.34 | 385.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:45:00 | 389.70 | 384.34 | 385.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2024-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 12:15:00 | 391.35 | 386.51 | 386.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 14:15:00 | 394.35 | 388.83 | 387.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 14:15:00 | 391.55 | 392.28 | 390.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-12 15:00:00 | 391.55 | 392.28 | 390.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 394.65 | 392.62 | 390.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 12:45:00 | 397.30 | 393.24 | 391.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 14:30:00 | 395.90 | 393.91 | 392.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-23 13:15:00 | 437.03 | 424.96 | 419.36 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 12:15:00 | 432.70 | 439.72 | 440.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 09:15:00 | 430.80 | 435.27 | 437.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 423.55 | 417.55 | 421.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 423.55 | 417.55 | 421.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 423.55 | 417.55 | 421.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:45:00 | 426.35 | 417.55 | 421.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 424.10 | 418.86 | 421.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 11:15:00 | 422.90 | 418.86 | 421.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 11:15:00 | 424.70 | 420.03 | 422.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 12:00:00 | 424.70 | 420.03 | 422.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 12:15:00 | 420.90 | 420.20 | 421.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:30:00 | 419.20 | 419.46 | 421.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:00:00 | 416.50 | 419.46 | 421.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 10:15:00 | 423.25 | 420.29 | 420.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 10:15:00 | 423.25 | 420.29 | 420.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 11:15:00 | 424.60 | 421.15 | 420.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 13:15:00 | 419.65 | 420.99 | 420.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 13:15:00 | 419.65 | 420.99 | 420.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 13:15:00 | 419.65 | 420.99 | 420.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 13:45:00 | 418.35 | 420.99 | 420.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 14:15:00 | 416.95 | 420.18 | 420.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 14:15:00 | 415.00 | 417.94 | 418.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 13:15:00 | 416.80 | 416.78 | 417.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 13:15:00 | 416.80 | 416.78 | 417.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 416.80 | 416.78 | 417.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 13:45:00 | 417.05 | 416.78 | 417.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 415.65 | 416.56 | 417.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:45:00 | 414.90 | 416.56 | 417.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 416.00 | 416.40 | 417.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 10:15:00 | 413.90 | 416.40 | 417.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 15:15:00 | 412.00 | 409.92 | 409.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 15:15:00 | 412.00 | 409.92 | 409.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 414.85 | 410.91 | 410.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 12:15:00 | 424.25 | 425.65 | 422.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 13:00:00 | 424.25 | 425.65 | 422.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 432.35 | 434.07 | 431.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:00:00 | 432.35 | 434.07 | 431.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 429.90 | 433.24 | 431.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:45:00 | 429.80 | 433.24 | 431.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 11:15:00 | 429.55 | 432.50 | 431.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 12:00:00 | 429.55 | 432.50 | 431.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 433.25 | 432.23 | 431.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 09:30:00 | 435.20 | 432.84 | 431.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-02 12:15:00 | 435.70 | 444.43 | 445.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 12:15:00 | 435.70 | 444.43 | 445.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 12:15:00 | 432.95 | 437.72 | 440.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 438.50 | 436.54 | 438.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 09:15:00 | 438.50 | 436.54 | 438.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 438.50 | 436.54 | 438.75 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2024-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 14:15:00 | 443.40 | 439.71 | 439.61 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 424.80 | 437.31 | 438.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 419.00 | 425.94 | 431.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 427.20 | 423.22 | 426.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 427.20 | 423.22 | 426.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 427.20 | 423.22 | 426.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 10:00:00 | 427.20 | 423.22 | 426.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 426.90 | 423.96 | 426.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 10:30:00 | 427.95 | 423.96 | 426.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 11:15:00 | 426.90 | 424.54 | 426.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 11:30:00 | 427.45 | 424.54 | 426.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 12:15:00 | 428.05 | 425.25 | 426.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 13:00:00 | 428.05 | 425.25 | 426.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 427.85 | 425.77 | 426.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 13:30:00 | 428.90 | 425.77 | 426.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 430.05 | 426.37 | 426.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 10:30:00 | 428.95 | 426.92 | 427.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 11:15:00 | 428.85 | 426.92 | 427.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 11:15:00 | 429.55 | 427.44 | 427.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 11:15:00 | 429.55 | 427.44 | 427.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 12:15:00 | 433.80 | 428.72 | 427.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 14:15:00 | 427.60 | 428.96 | 428.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 14:15:00 | 427.60 | 428.96 | 428.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 427.60 | 428.96 | 428.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:45:00 | 425.20 | 428.96 | 428.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 427.00 | 428.57 | 428.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:15:00 | 435.50 | 428.57 | 428.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 10:45:00 | 430.80 | 429.45 | 428.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-16 09:15:00 | 422.75 | 429.25 | 430.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 09:15:00 | 422.75 | 429.25 | 430.05 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 12:15:00 | 431.95 | 429.63 | 429.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 13:15:00 | 432.40 | 430.19 | 429.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 15:15:00 | 430.00 | 430.24 | 429.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 15:15:00 | 430.00 | 430.24 | 429.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 430.00 | 430.24 | 429.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:15:00 | 427.75 | 430.24 | 429.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 427.15 | 429.62 | 429.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:30:00 | 428.05 | 429.62 | 429.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 10:15:00 | 427.15 | 429.13 | 429.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 425.00 | 427.92 | 428.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 14:15:00 | 428.50 | 427.27 | 428.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 14:15:00 | 428.50 | 427.27 | 428.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 428.50 | 427.27 | 428.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 15:00:00 | 428.50 | 427.27 | 428.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 15:15:00 | 426.00 | 427.01 | 428.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:15:00 | 427.20 | 427.01 | 428.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 415.45 | 424.70 | 426.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 10:15:00 | 414.95 | 424.70 | 426.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-19 11:15:00 | 373.45 | 415.17 | 422.03 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 373.65 | 370.26 | 370.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 13:15:00 | 376.90 | 372.59 | 371.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 13:15:00 | 376.80 | 377.16 | 374.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 14:00:00 | 376.80 | 377.16 | 374.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 379.35 | 378.57 | 376.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:15:00 | 379.90 | 378.46 | 377.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 09:15:00 | 382.60 | 384.95 | 385.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 09:15:00 | 382.60 | 384.95 | 385.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 10:15:00 | 377.25 | 382.41 | 383.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 09:15:00 | 343.20 | 341.15 | 348.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-28 10:00:00 | 343.20 | 341.15 | 348.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 346.60 | 342.93 | 348.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:45:00 | 347.15 | 342.93 | 348.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 13:15:00 | 345.65 | 344.14 | 347.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 13:30:00 | 347.20 | 344.14 | 347.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 346.70 | 344.65 | 347.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 14:45:00 | 346.10 | 344.65 | 347.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 349.00 | 345.14 | 346.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 15:00:00 | 349.00 | 345.14 | 346.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 347.70 | 345.65 | 346.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 12:00:00 | 344.50 | 346.04 | 346.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 14:15:00 | 341.95 | 340.94 | 340.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 14:15:00 | 341.95 | 340.94 | 340.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 09:15:00 | 343.20 | 341.55 | 341.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 11:15:00 | 340.15 | 341.42 | 341.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 11:15:00 | 340.15 | 341.42 | 341.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 340.15 | 341.42 | 341.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:00:00 | 340.15 | 341.42 | 341.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 340.35 | 341.21 | 341.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:00:00 | 340.35 | 341.21 | 341.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 340.60 | 341.08 | 341.03 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 14:15:00 | 339.50 | 340.77 | 340.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 333.35 | 339.10 | 340.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 329.85 | 326.96 | 329.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 329.85 | 326.96 | 329.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 329.85 | 326.96 | 329.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:30:00 | 329.40 | 326.96 | 329.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 328.10 | 327.18 | 329.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:30:00 | 328.55 | 327.18 | 329.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 318.85 | 318.43 | 320.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 09:30:00 | 319.05 | 318.43 | 320.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 11:15:00 | 322.30 | 319.17 | 320.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:00:00 | 322.30 | 319.17 | 320.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 324.80 | 320.30 | 320.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:45:00 | 324.65 | 320.30 | 320.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2024-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 13:15:00 | 326.15 | 321.47 | 321.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 330.75 | 323.65 | 322.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 327.45 | 328.98 | 325.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 15:00:00 | 327.45 | 328.98 | 325.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 326.65 | 328.52 | 326.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 09:15:00 | 331.90 | 328.52 | 326.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 10:00:00 | 330.35 | 328.88 | 326.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 11:45:00 | 329.10 | 329.37 | 328.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-05 09:15:00 | 365.09 | 356.11 | 353.36 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 09:15:00 | 355.50 | 360.95 | 361.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 09:15:00 | 354.40 | 358.89 | 359.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 13:15:00 | 345.30 | 344.80 | 349.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 13:45:00 | 345.50 | 344.80 | 349.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 350.10 | 345.86 | 349.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 350.10 | 345.86 | 349.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 350.00 | 346.69 | 349.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 352.50 | 346.69 | 349.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 352.90 | 347.93 | 349.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 354.65 | 347.93 | 349.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 352.10 | 349.54 | 350.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:00:00 | 352.10 | 349.54 | 350.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 352.45 | 350.12 | 350.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 14:00:00 | 352.45 | 350.12 | 350.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 14:15:00 | 352.95 | 350.68 | 350.62 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 11:15:00 | 347.60 | 350.51 | 350.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 346.35 | 349.37 | 350.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 11:15:00 | 344.85 | 342.04 | 344.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 11:15:00 | 344.85 | 342.04 | 344.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 344.85 | 342.04 | 344.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 12:00:00 | 344.85 | 342.04 | 344.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 343.85 | 342.40 | 344.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 13:15:00 | 343.30 | 342.40 | 344.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-19 13:15:00 | 345.15 | 342.95 | 344.44 | SL hit (close>static) qty=1.00 sl=344.95 alert=retest2 |

### Cycle 33 — BUY (started 2024-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 11:15:00 | 346.70 | 345.41 | 345.24 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 12:15:00 | 341.95 | 344.72 | 344.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 13:15:00 | 340.45 | 343.86 | 344.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 10:15:00 | 336.10 | 335.51 | 338.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 11:00:00 | 336.10 | 335.51 | 338.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 11:15:00 | 333.35 | 333.71 | 335.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 11:30:00 | 333.75 | 333.71 | 335.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 335.80 | 333.76 | 334.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:15:00 | 336.40 | 333.76 | 334.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 334.70 | 333.95 | 334.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:30:00 | 336.40 | 333.95 | 334.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 333.90 | 333.94 | 334.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 12:45:00 | 333.35 | 333.78 | 334.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 11:00:00 | 333.40 | 332.64 | 333.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 11:15:00 | 336.00 | 333.31 | 333.81 | SL hit (close>static) qty=1.00 sl=334.95 alert=retest2 |

### Cycle 35 — BUY (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 13:15:00 | 335.90 | 334.27 | 334.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 14:15:00 | 339.10 | 335.24 | 334.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 09:15:00 | 339.00 | 340.01 | 338.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-01 09:45:00 | 339.45 | 340.01 | 338.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 343.35 | 344.86 | 343.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 11:00:00 | 343.35 | 344.86 | 343.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 344.00 | 344.69 | 343.16 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 332.70 | 340.85 | 341.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 332.00 | 339.08 | 341.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 09:15:00 | 330.45 | 330.33 | 333.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 10:00:00 | 330.45 | 330.33 | 333.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 13:15:00 | 332.10 | 329.38 | 331.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 14:00:00 | 332.10 | 329.38 | 331.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 330.80 | 329.67 | 331.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 329.15 | 329.74 | 331.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 09:15:00 | 337.55 | 331.30 | 332.18 | SL hit (close>static) qty=1.00 sl=334.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 10:15:00 | 337.05 | 327.32 | 327.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-14 13:15:00 | 338.45 | 331.50 | 329.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 367.50 | 371.13 | 365.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 367.50 | 371.13 | 365.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 363.35 | 369.57 | 365.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 363.35 | 369.57 | 365.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 368.00 | 369.26 | 365.50 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 354.35 | 363.60 | 364.02 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 364.85 | 361.77 | 361.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-24 09:15:00 | 369.35 | 364.80 | 363.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 13:15:00 | 366.55 | 367.50 | 365.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 14:00:00 | 366.55 | 367.50 | 365.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 367.85 | 367.57 | 365.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:30:00 | 366.40 | 367.57 | 365.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 353.30 | 364.70 | 364.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 09:45:00 | 353.90 | 364.70 | 364.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 10:15:00 | 354.00 | 362.56 | 363.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 15:15:00 | 348.55 | 354.27 | 358.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 352.15 | 351.24 | 355.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 352.15 | 351.24 | 355.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 353.75 | 352.59 | 355.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:30:00 | 356.40 | 352.59 | 355.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 11:15:00 | 347.85 | 345.91 | 347.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 12:00:00 | 347.85 | 345.91 | 347.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 12:15:00 | 344.30 | 345.59 | 347.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 11:45:00 | 341.75 | 345.70 | 346.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 13:15:00 | 350.50 | 344.95 | 346.11 | SL hit (close>static) qty=1.00 sl=348.90 alert=retest2 |

### Cycle 41 — BUY (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 15:15:00 | 347.80 | 346.95 | 346.91 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 343.45 | 346.25 | 346.59 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 10:15:00 | 349.25 | 346.85 | 346.83 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 345.85 | 346.65 | 346.75 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 14:15:00 | 350.00 | 347.41 | 347.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 09:15:00 | 354.50 | 349.21 | 347.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 14:15:00 | 357.90 | 358.13 | 355.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-05 15:00:00 | 357.90 | 358.13 | 355.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 355.70 | 357.40 | 356.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 356.40 | 357.40 | 356.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 355.70 | 357.06 | 356.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:00:00 | 355.70 | 357.06 | 356.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 354.35 | 356.52 | 355.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 364.70 | 356.52 | 355.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 351.80 | 360.24 | 359.23 | SL hit (close<static) qty=1.00 sl=354.05 alert=retest2 |

### Cycle 46 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 11:15:00 | 350.00 | 356.94 | 357.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 347.35 | 352.33 | 354.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 348.75 | 346.62 | 349.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:45:00 | 349.25 | 346.62 | 349.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 350.00 | 347.30 | 349.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 350.00 | 347.30 | 349.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 346.80 | 347.20 | 349.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 09:15:00 | 343.85 | 347.39 | 349.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-13 09:15:00 | 353.30 | 348.57 | 349.60 | SL hit (close>static) qty=1.00 sl=350.75 alert=retest2 |

### Cycle 47 — BUY (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 11:15:00 | 354.20 | 350.79 | 350.49 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 15:15:00 | 348.35 | 350.18 | 350.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 341.50 | 348.45 | 349.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 12:15:00 | 335.55 | 335.47 | 339.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 12:45:00 | 335.15 | 335.47 | 339.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 339.55 | 336.57 | 339.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 339.55 | 336.57 | 339.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 340.15 | 337.29 | 339.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 09:15:00 | 334.40 | 337.29 | 339.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 341.10 | 336.22 | 337.23 | SL hit (close>static) qty=1.00 sl=340.40 alert=retest2 |

### Cycle 49 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 341.00 | 337.89 | 337.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 343.00 | 340.34 | 339.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 12:15:00 | 339.95 | 340.98 | 339.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 12:15:00 | 339.95 | 340.98 | 339.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 12:15:00 | 339.95 | 340.98 | 339.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 13:00:00 | 339.95 | 340.98 | 339.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 13:15:00 | 342.40 | 341.26 | 340.09 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 09:15:00 | 332.50 | 339.50 | 339.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 329.50 | 335.69 | 337.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 335.60 | 331.77 | 333.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 335.60 | 331.77 | 333.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 335.60 | 331.77 | 333.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 10:00:00 | 335.60 | 331.77 | 333.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 330.70 | 331.56 | 333.68 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 12:15:00 | 336.10 | 333.95 | 333.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-27 14:15:00 | 338.40 | 335.00 | 334.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 09:15:00 | 329.35 | 334.27 | 334.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 09:15:00 | 329.35 | 334.27 | 334.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 329.35 | 334.27 | 334.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 10:00:00 | 329.35 | 334.27 | 334.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 10:15:00 | 329.55 | 333.32 | 333.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 11:15:00 | 326.50 | 331.96 | 333.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 327.10 | 323.06 | 326.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 12:15:00 | 327.10 | 323.06 | 326.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 327.10 | 323.06 | 326.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:00:00 | 327.10 | 323.06 | 326.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 327.10 | 323.87 | 326.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 323.60 | 324.96 | 326.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 12:00:00 | 324.65 | 325.28 | 326.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 13:00:00 | 324.35 | 325.09 | 326.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 13:30:00 | 324.10 | 325.60 | 326.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 327.00 | 325.88 | 326.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 14:45:00 | 329.50 | 325.88 | 326.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 332.20 | 327.26 | 327.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 332.20 | 327.26 | 327.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 10:15:00 | 335.80 | 328.97 | 327.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 11:15:00 | 331.25 | 334.65 | 332.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-06 11:15:00 | 331.25 | 334.65 | 332.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 11:15:00 | 331.25 | 334.65 | 332.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 12:00:00 | 331.25 | 334.65 | 332.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 12:15:00 | 331.60 | 334.04 | 332.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 12:30:00 | 333.20 | 334.04 | 332.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 13:15:00 | 331.40 | 333.51 | 332.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 13:30:00 | 331.00 | 333.51 | 332.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 14:15:00 | 330.60 | 332.93 | 331.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 15:00:00 | 330.60 | 332.93 | 331.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-03-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 10:15:00 | 329.05 | 331.36 | 331.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 11:15:00 | 327.20 | 330.53 | 331.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-10 09:15:00 | 334.90 | 329.96 | 330.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 334.90 | 329.96 | 330.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 334.90 | 329.96 | 330.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 334.90 | 329.96 | 330.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-10 10:15:00 | 333.90 | 330.75 | 330.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 13:15:00 | 338.40 | 334.06 | 332.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 09:15:00 | 330.05 | 335.38 | 333.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 09:15:00 | 330.05 | 335.38 | 333.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 330.05 | 335.38 | 333.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:00:00 | 330.05 | 335.38 | 333.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 10:15:00 | 318.45 | 331.99 | 332.37 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 15:15:00 | 331.15 | 329.25 | 329.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 334.50 | 330.30 | 329.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 337.80 | 338.71 | 336.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 09:15:00 | 337.80 | 338.71 | 336.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 337.80 | 338.71 | 336.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:00:00 | 337.80 | 338.71 | 336.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 335.35 | 338.04 | 336.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 11:00:00 | 335.35 | 338.04 | 336.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 11:15:00 | 336.65 | 337.76 | 336.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 11:30:00 | 336.25 | 337.76 | 336.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 12:15:00 | 336.80 | 337.57 | 336.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 13:30:00 | 337.85 | 337.71 | 336.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 13:15:00 | 341.55 | 344.01 | 344.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 13:15:00 | 341.55 | 344.01 | 344.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 11:15:00 | 336.45 | 340.78 | 342.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 09:15:00 | 340.05 | 339.62 | 341.08 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-27 13:15:00 | 335.75 | 338.88 | 340.38 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 340.60 | 338.99 | 340.16 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-27 14:15:00 | 340.60 | 338.99 | 340.16 | SL hit (close>ema400) qty=1.00 sl=340.16 alert=retest1 |

### Cycle 59 — BUY (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 09:15:00 | 359.30 | 340.58 | 339.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 12:15:00 | 361.90 | 355.65 | 350.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 09:15:00 | 355.60 | 357.62 | 353.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-03 09:45:00 | 354.55 | 357.62 | 353.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 353.20 | 356.74 | 353.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 11:00:00 | 353.20 | 356.74 | 353.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 11:15:00 | 355.65 | 356.52 | 353.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 13:15:00 | 355.80 | 356.12 | 353.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 14:15:00 | 356.75 | 355.96 | 353.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 342.35 | 356.94 | 356.56 | SL hit (close<static) qty=1.00 sl=353.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 10:15:00 | 347.45 | 355.05 | 355.73 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 09:15:00 | 361.10 | 355.79 | 355.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 14:15:00 | 370.25 | 363.62 | 359.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 12:15:00 | 406.30 | 407.32 | 400.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 13:00:00 | 406.30 | 407.32 | 400.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 401.00 | 405.42 | 400.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 14:45:00 | 400.00 | 405.42 | 400.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 401.35 | 404.60 | 400.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 09:15:00 | 400.40 | 404.60 | 400.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 399.30 | 403.54 | 400.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 14:00:00 | 405.25 | 401.85 | 400.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 394.75 | 401.89 | 401.18 | SL hit (close<static) qty=1.00 sl=397.20 alert=retest2 |

### Cycle 62 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 393.80 | 400.27 | 400.51 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 408.00 | 400.77 | 400.49 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 09:15:00 | 391.75 | 402.77 | 403.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 15:15:00 | 378.60 | 386.83 | 394.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 382.95 | 381.07 | 384.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:00:00 | 382.95 | 381.07 | 384.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 383.35 | 381.53 | 384.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:30:00 | 384.45 | 381.53 | 384.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 381.65 | 381.55 | 383.98 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 394.95 | 384.82 | 384.76 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-05-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 12:15:00 | 383.75 | 386.05 | 386.14 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-05-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 14:15:00 | 388.20 | 386.48 | 386.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-09 15:15:00 | 389.10 | 387.00 | 386.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 394.25 | 395.07 | 391.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 394.25 | 395.07 | 391.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 394.25 | 395.07 | 391.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 09:30:00 | 394.35 | 395.07 | 391.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 394.80 | 395.62 | 393.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:30:00 | 393.05 | 395.62 | 393.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 398.00 | 395.83 | 393.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:15:00 | 399.20 | 396.16 | 395.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 10:00:00 | 399.90 | 400.72 | 398.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 13:15:00 | 389.15 | 396.93 | 397.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 13:15:00 | 389.15 | 396.93 | 397.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 13:15:00 | 381.90 | 383.54 | 385.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 14:15:00 | 383.70 | 383.57 | 385.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 15:00:00 | 383.70 | 383.57 | 385.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 379.40 | 382.64 | 384.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 11:15:00 | 378.30 | 382.04 | 384.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 09:45:00 | 377.55 | 379.77 | 381.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 15:15:00 | 384.50 | 382.48 | 382.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 15:15:00 | 384.50 | 382.48 | 382.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 392.35 | 384.46 | 383.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 10:15:00 | 390.35 | 391.58 | 388.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 11:00:00 | 390.35 | 391.58 | 388.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 389.70 | 390.89 | 388.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:45:00 | 389.85 | 390.89 | 388.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 388.80 | 390.48 | 388.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:30:00 | 388.00 | 390.48 | 388.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 392.45 | 390.87 | 389.28 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 11:15:00 | 385.60 | 388.53 | 388.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 384.25 | 387.48 | 388.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 14:15:00 | 380.95 | 380.88 | 382.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-03 14:45:00 | 382.50 | 380.88 | 382.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 379.95 | 380.56 | 382.37 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 386.40 | 383.12 | 382.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 13:15:00 | 389.30 | 386.31 | 384.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 393.00 | 393.51 | 390.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 11:00:00 | 393.00 | 393.51 | 390.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 390.80 | 393.37 | 392.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:00:00 | 390.80 | 393.37 | 392.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 389.90 | 392.67 | 392.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:00:00 | 389.90 | 392.67 | 392.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 387.25 | 390.92 | 391.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 382.95 | 387.32 | 389.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 381.85 | 380.98 | 383.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 10:00:00 | 381.85 | 380.98 | 383.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 385.25 | 381.83 | 383.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 385.25 | 381.83 | 383.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 386.45 | 382.76 | 383.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 386.45 | 382.76 | 383.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 391.55 | 385.70 | 385.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 392.55 | 388.51 | 386.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 10:15:00 | 390.85 | 391.46 | 389.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 10:45:00 | 391.50 | 391.46 | 389.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 391.25 | 391.41 | 389.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:30:00 | 389.70 | 391.41 | 389.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 391.80 | 391.49 | 389.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:45:00 | 391.30 | 391.49 | 389.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 392.05 | 391.58 | 390.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 11:00:00 | 393.60 | 391.67 | 390.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 12:15:00 | 393.05 | 391.86 | 390.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 12:15:00 | 389.70 | 391.43 | 390.69 | SL hit (close<static) qty=1.00 sl=390.05 alert=retest2 |

### Cycle 74 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 409.50 | 421.65 | 422.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 14:15:00 | 408.30 | 413.69 | 417.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 15:15:00 | 407.90 | 407.61 | 411.76 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-09 09:15:00 | 404.45 | 407.61 | 411.76 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-09 13:30:00 | 405.90 | 407.18 | 410.02 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-09 14:00:00 | 405.85 | 407.18 | 410.02 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 409.50 | 407.00 | 409.18 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 409.50 | 407.00 | 409.18 | SL hit (close>ema400) qty=1.00 sl=409.18 alert=retest1 |

### Cycle 75 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 406.50 | 404.61 | 404.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 408.00 | 405.47 | 405.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 11:15:00 | 405.45 | 406.15 | 405.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 11:15:00 | 405.45 | 406.15 | 405.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 405.45 | 406.15 | 405.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:45:00 | 405.65 | 406.15 | 405.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 405.20 | 405.96 | 405.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:00:00 | 405.20 | 405.96 | 405.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 405.95 | 405.96 | 405.56 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 09:15:00 | 404.70 | 405.32 | 405.34 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 10:15:00 | 405.90 | 405.44 | 405.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 11:15:00 | 407.70 | 405.89 | 405.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 12:15:00 | 404.40 | 405.59 | 405.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 12:15:00 | 404.40 | 405.59 | 405.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 404.40 | 405.59 | 405.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 404.40 | 405.59 | 405.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 407.25 | 405.92 | 405.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 14:30:00 | 407.40 | 405.98 | 405.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 402.00 | 405.27 | 405.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 402.00 | 405.27 | 405.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 10:15:00 | 401.00 | 402.92 | 403.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 11:15:00 | 404.20 | 403.18 | 403.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 11:15:00 | 404.20 | 403.18 | 403.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 404.20 | 403.18 | 403.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:00:00 | 404.20 | 403.18 | 403.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 404.85 | 403.51 | 404.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:00:00 | 404.85 | 403.51 | 404.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 402.30 | 403.27 | 403.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:15:00 | 401.00 | 403.30 | 403.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 15:15:00 | 406.50 | 401.86 | 401.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 15:15:00 | 406.50 | 401.86 | 401.37 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 399.15 | 401.27 | 401.28 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 14:15:00 | 401.50 | 401.31 | 401.29 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 15:15:00 | 400.60 | 401.17 | 401.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 394.90 | 399.91 | 400.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 395.90 | 394.97 | 397.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 10:00:00 | 395.90 | 394.97 | 397.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 390.45 | 394.07 | 396.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:45:00 | 388.95 | 392.85 | 395.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 387.85 | 388.61 | 390.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 09:15:00 | 369.50 | 382.07 | 386.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 09:15:00 | 368.46 | 382.07 | 386.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-01 12:15:00 | 350.06 | 359.74 | 369.61 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 83 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 10:15:00 | 338.25 | 336.54 | 336.48 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 332.95 | 337.56 | 337.91 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 13:15:00 | 338.50 | 336.41 | 336.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 13:15:00 | 339.20 | 337.68 | 337.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 11:15:00 | 348.75 | 348.86 | 345.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 12:00:00 | 348.75 | 348.86 | 345.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 343.20 | 352.37 | 351.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 342.75 | 352.37 | 351.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 341.60 | 350.22 | 350.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 340.65 | 344.32 | 346.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 345.60 | 340.41 | 342.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 345.60 | 340.41 | 342.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 345.60 | 340.41 | 342.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 345.60 | 340.41 | 342.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 343.70 | 341.07 | 342.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:45:00 | 345.35 | 341.07 | 342.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 341.85 | 341.41 | 342.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 14:30:00 | 340.50 | 340.84 | 342.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:45:00 | 340.45 | 340.70 | 341.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:45:00 | 340.85 | 340.94 | 341.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 12:15:00 | 340.80 | 340.94 | 341.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 340.55 | 340.86 | 341.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 15:15:00 | 339.00 | 340.50 | 341.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 09:15:00 | 323.47 | 330.01 | 334.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 09:15:00 | 323.43 | 330.01 | 334.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 09:15:00 | 323.81 | 330.01 | 334.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 09:15:00 | 323.76 | 330.01 | 334.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 09:15:00 | 322.05 | 330.01 | 334.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 325.85 | 324.84 | 328.99 | SL hit (close>ema200) qty=0.50 sl=324.84 alert=retest2 |

### Cycle 87 — BUY (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 12:15:00 | 333.70 | 328.12 | 328.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 13:15:00 | 341.00 | 330.69 | 329.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 13:15:00 | 359.85 | 362.05 | 358.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 13:15:00 | 359.85 | 362.05 | 358.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 359.85 | 362.05 | 358.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:00:00 | 359.85 | 362.05 | 358.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 359.80 | 361.08 | 358.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 357.95 | 361.08 | 358.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 355.15 | 359.89 | 358.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:00:00 | 355.15 | 359.89 | 358.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 357.70 | 359.45 | 358.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 12:00:00 | 358.40 | 359.24 | 358.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 12:30:00 | 358.00 | 358.91 | 358.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 13:45:00 | 357.90 | 358.68 | 358.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 15:15:00 | 358.05 | 358.49 | 358.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 358.05 | 358.40 | 358.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 356.55 | 358.40 | 358.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 354.70 | 357.66 | 357.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 09:15:00 | 354.70 | 357.66 | 357.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 10:15:00 | 354.00 | 356.93 | 357.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 09:15:00 | 356.40 | 355.23 | 356.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 356.40 | 355.23 | 356.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 356.40 | 355.23 | 356.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:45:00 | 357.60 | 355.23 | 356.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 355.55 | 355.29 | 356.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:00:00 | 355.55 | 355.29 | 356.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 355.80 | 355.39 | 356.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:30:00 | 354.80 | 355.39 | 356.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 353.35 | 354.98 | 355.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:45:00 | 351.65 | 353.94 | 354.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 11:15:00 | 361.55 | 355.46 | 355.57 | SL hit (close>static) qty=1.00 sl=356.25 alert=retest2 |

### Cycle 89 — BUY (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 13:15:00 | 356.35 | 355.66 | 355.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 15:15:00 | 356.80 | 356.02 | 355.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 13:15:00 | 357.80 | 358.05 | 356.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 14:00:00 | 357.80 | 358.05 | 356.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 356.05 | 357.65 | 356.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:45:00 | 356.00 | 357.65 | 356.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 356.65 | 357.45 | 356.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:15:00 | 356.45 | 357.45 | 356.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 355.25 | 357.01 | 356.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:45:00 | 354.20 | 357.01 | 356.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 353.80 | 356.37 | 356.46 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 11:15:00 | 357.30 | 356.56 | 356.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 15:15:00 | 359.80 | 357.84 | 357.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 14:15:00 | 357.70 | 359.21 | 358.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 14:15:00 | 357.70 | 359.21 | 358.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 357.70 | 359.21 | 358.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 15:00:00 | 357.70 | 359.21 | 358.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 360.00 | 359.37 | 358.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 358.40 | 359.37 | 358.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 359.40 | 359.37 | 358.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:45:00 | 358.25 | 359.37 | 358.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 358.70 | 359.25 | 358.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:45:00 | 359.05 | 359.25 | 358.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 12:15:00 | 357.80 | 358.96 | 358.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 13:00:00 | 357.80 | 358.96 | 358.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 358.90 | 358.95 | 358.61 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 355.85 | 357.91 | 358.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 351.80 | 356.69 | 357.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 348.25 | 348.09 | 350.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:15:00 | 349.70 | 348.09 | 350.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 347.05 | 347.89 | 350.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:30:00 | 348.75 | 347.89 | 350.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 345.90 | 344.70 | 346.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 346.30 | 344.70 | 346.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 345.60 | 344.88 | 346.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:30:00 | 346.50 | 344.88 | 346.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 350.15 | 345.93 | 347.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 350.15 | 345.93 | 347.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 349.70 | 346.69 | 347.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:30:00 | 350.15 | 346.69 | 347.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 352.00 | 348.58 | 348.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 15:15:00 | 352.65 | 349.40 | 348.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 11:15:00 | 348.85 | 349.67 | 348.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 11:15:00 | 348.85 | 349.67 | 348.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 348.85 | 349.67 | 348.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 11:45:00 | 348.75 | 349.67 | 348.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 350.80 | 349.90 | 349.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 14:15:00 | 353.45 | 350.24 | 349.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 10:15:00 | 352.00 | 351.34 | 350.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 14:45:00 | 352.00 | 351.91 | 350.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 12:00:00 | 353.20 | 352.92 | 351.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 354.45 | 355.51 | 353.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 354.45 | 355.51 | 353.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 355.75 | 355.56 | 354.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:30:00 | 353.90 | 355.56 | 354.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 354.75 | 356.07 | 354.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 354.75 | 356.07 | 354.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 355.00 | 355.85 | 354.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 356.10 | 355.85 | 354.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 354.75 | 355.63 | 354.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 11:45:00 | 357.10 | 355.27 | 354.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 10:15:00 | 353.65 | 354.67 | 354.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 353.65 | 354.67 | 354.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 11:15:00 | 349.70 | 353.67 | 354.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 344.20 | 343.20 | 346.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 344.20 | 343.20 | 346.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 344.85 | 344.39 | 345.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 13:15:00 | 343.80 | 344.78 | 345.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 15:15:00 | 343.70 | 344.72 | 345.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 347.65 | 344.80 | 345.22 | SL hit (close>static) qty=1.00 sl=346.55 alert=retest2 |

### Cycle 95 — BUY (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 10:15:00 | 350.80 | 345.55 | 345.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 11:15:00 | 352.65 | 346.97 | 345.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 358.10 | 358.16 | 354.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 15:00:00 | 358.10 | 358.16 | 354.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 376.20 | 363.53 | 359.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 384.00 | 368.81 | 363.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 10:15:00 | 382.00 | 371.04 | 365.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 11:15:00 | 383.20 | 372.93 | 366.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 12:30:00 | 384.05 | 376.14 | 369.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 365.30 | 378.42 | 376.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 365.30 | 378.42 | 376.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 370.20 | 376.77 | 375.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 11:15:00 | 371.30 | 376.77 | 375.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 11:15:00 | 368.25 | 375.07 | 375.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 368.25 | 375.07 | 375.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 366.40 | 370.17 | 372.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 367.40 | 367.15 | 369.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 367.40 | 367.15 | 369.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 367.40 | 367.15 | 369.62 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 382.90 | 370.45 | 370.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 09:15:00 | 390.90 | 376.51 | 373.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 15:15:00 | 400.00 | 400.48 | 395.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 09:15:00 | 399.50 | 400.48 | 395.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 396.20 | 398.86 | 397.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 396.20 | 398.86 | 397.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 399.20 | 398.93 | 397.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 12:30:00 | 400.05 | 399.05 | 397.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 13:15:00 | 400.90 | 399.05 | 397.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 10:15:00 | 404.35 | 408.53 | 408.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 404.35 | 408.53 | 408.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 401.55 | 405.69 | 407.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 12:15:00 | 405.05 | 404.69 | 406.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 12:30:00 | 404.85 | 404.69 | 406.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 400.75 | 403.51 | 405.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 398.50 | 402.18 | 403.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 13:00:00 | 399.45 | 400.23 | 402.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 15:15:00 | 399.20 | 401.07 | 401.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 13:15:00 | 404.95 | 401.71 | 401.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 13:15:00 | 404.95 | 401.71 | 401.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 406.25 | 403.39 | 402.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 15:15:00 | 405.05 | 405.13 | 403.88 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 09:15:00 | 406.90 | 405.13 | 403.88 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 10:00:00 | 406.55 | 405.41 | 404.12 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 404.40 | 405.32 | 404.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:45:00 | 404.00 | 405.32 | 404.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 402.00 | 404.66 | 404.10 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-27 12:15:00 | 402.00 | 404.66 | 404.10 | SL hit (close<ema400) qty=1.00 sl=404.10 alert=retest1 |

### Cycle 100 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 401.55 | 403.71 | 403.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 401.00 | 402.93 | 403.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 403.85 | 402.76 | 403.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 403.85 | 402.76 | 403.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 403.85 | 402.76 | 403.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 10:45:00 | 400.10 | 402.11 | 402.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:45:00 | 400.15 | 398.82 | 400.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 13:00:00 | 400.10 | 400.16 | 400.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 409.00 | 402.51 | 401.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 09:15:00 | 409.00 | 402.51 | 401.76 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 13:15:00 | 399.95 | 403.21 | 403.36 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 411.15 | 404.74 | 403.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 11:15:00 | 411.25 | 406.04 | 404.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 409.15 | 411.16 | 408.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 10:00:00 | 409.15 | 411.16 | 408.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 405.05 | 409.68 | 408.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 405.05 | 409.68 | 408.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 401.80 | 408.11 | 407.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 401.80 | 408.11 | 407.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 400.85 | 406.65 | 406.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 398.60 | 403.72 | 405.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 404.00 | 403.15 | 404.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 14:15:00 | 404.00 | 403.15 | 404.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 404.00 | 403.15 | 404.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 404.00 | 403.15 | 404.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 403.30 | 403.18 | 404.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 406.15 | 403.18 | 404.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 405.05 | 403.55 | 404.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:30:00 | 406.35 | 403.55 | 404.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 405.40 | 403.92 | 404.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:15:00 | 405.40 | 403.92 | 404.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 405.15 | 404.17 | 404.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:30:00 | 406.65 | 404.17 | 404.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 403.45 | 404.41 | 404.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 405.20 | 404.41 | 404.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 410.00 | 405.53 | 405.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 412.90 | 408.97 | 407.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 10:15:00 | 411.00 | 412.64 | 410.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 10:15:00 | 411.00 | 412.64 | 410.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 411.00 | 412.64 | 410.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:45:00 | 411.40 | 412.64 | 410.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 410.40 | 412.19 | 410.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:00:00 | 410.40 | 412.19 | 410.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 410.70 | 411.90 | 410.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 13:00:00 | 410.70 | 411.90 | 410.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 13:15:00 | 407.25 | 410.97 | 410.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:00:00 | 407.25 | 410.97 | 410.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 409.95 | 410.76 | 410.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 10:15:00 | 411.20 | 409.92 | 409.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 10:15:00 | 408.80 | 409.69 | 409.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 408.80 | 409.69 | 409.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 407.55 | 408.94 | 409.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 11:15:00 | 408.05 | 407.88 | 408.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 12:00:00 | 408.05 | 407.88 | 408.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 406.45 | 407.59 | 408.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 13:30:00 | 405.90 | 407.36 | 408.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 14:15:00 | 406.05 | 407.36 | 408.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 11:15:00 | 410.35 | 407.93 | 408.14 | SL hit (close>static) qty=1.00 sl=409.20 alert=retest2 |

### Cycle 107 — BUY (started 2025-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 12:15:00 | 410.95 | 408.53 | 408.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 11:15:00 | 412.50 | 410.13 | 409.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 09:15:00 | 410.85 | 411.18 | 410.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 410.85 | 411.18 | 410.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 410.85 | 411.18 | 410.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:30:00 | 409.25 | 411.18 | 410.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 411.15 | 411.17 | 410.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 12:30:00 | 412.00 | 411.25 | 410.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 15:15:00 | 411.95 | 411.22 | 410.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:45:00 | 412.90 | 411.59 | 410.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 10:45:00 | 413.05 | 412.27 | 411.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 414.50 | 414.13 | 412.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 414.45 | 414.13 | 412.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 417.65 | 414.68 | 413.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:15:00 | 420.05 | 414.68 | 413.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:30:00 | 421.00 | 422.24 | 420.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 14:15:00 | 420.85 | 424.42 | 423.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 15:15:00 | 421.00 | 423.02 | 422.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 15:15:00 | 421.00 | 422.62 | 422.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 15:15:00 | 421.00 | 422.62 | 422.73 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 09:15:00 | 434.45 | 424.99 | 423.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 442.40 | 434.53 | 430.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 438.55 | 440.12 | 435.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 438.55 | 440.12 | 435.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 438.55 | 440.12 | 435.82 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 432.00 | 434.94 | 434.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 11:15:00 | 431.05 | 434.16 | 434.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 431.45 | 430.16 | 431.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 431.45 | 430.16 | 431.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 431.45 | 430.16 | 431.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 431.45 | 430.16 | 431.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 429.65 | 430.06 | 431.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:15:00 | 427.70 | 430.06 | 431.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 434.30 | 430.99 | 431.51 | SL hit (close>static) qty=1.00 sl=432.10 alert=retest2 |

### Cycle 111 — BUY (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 09:15:00 | 443.10 | 433.80 | 432.67 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 15:15:00 | 431.95 | 433.87 | 433.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 10:15:00 | 429.40 | 432.45 | 433.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 434.95 | 429.69 | 430.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 434.95 | 429.69 | 430.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 434.95 | 429.69 | 430.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 433.75 | 429.69 | 430.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 441.60 | 432.07 | 431.94 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 430.55 | 433.74 | 433.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 12:15:00 | 425.65 | 430.38 | 432.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 415.85 | 415.41 | 420.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 12:45:00 | 417.05 | 415.41 | 420.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 418.50 | 415.14 | 418.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:45:00 | 414.50 | 414.98 | 418.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:45:00 | 414.50 | 414.79 | 417.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 09:15:00 | 422.10 | 417.84 | 418.33 | SL hit (close>static) qty=1.00 sl=420.35 alert=retest2 |

### Cycle 115 — BUY (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 14:15:00 | 422.30 | 417.11 | 416.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 15:15:00 | 424.00 | 418.49 | 417.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-28 13:15:00 | 422.50 | 422.59 | 420.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-28 14:00:00 | 422.50 | 422.59 | 420.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 438.65 | 442.04 | 437.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:45:00 | 436.50 | 442.04 | 437.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 439.20 | 441.48 | 437.62 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 425.05 | 435.32 | 435.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 421.50 | 432.56 | 434.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 431.65 | 428.61 | 431.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 431.65 | 428.61 | 431.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 431.65 | 428.61 | 431.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 431.65 | 428.61 | 431.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 431.70 | 429.23 | 431.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 431.05 | 429.23 | 431.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 428.15 | 429.01 | 430.84 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 444.20 | 433.09 | 432.44 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 13:15:00 | 437.90 | 439.68 | 439.88 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 443.30 | 440.41 | 440.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 450.90 | 442.54 | 441.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 454.30 | 455.42 | 450.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 13:15:00 | 454.30 | 455.42 | 450.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 454.30 | 455.42 | 450.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:00:00 | 454.30 | 455.42 | 450.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 469.25 | 470.55 | 466.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 11:45:00 | 472.80 | 469.75 | 468.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 13:30:00 | 472.45 | 470.55 | 468.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 10:15:00 | 471.55 | 471.44 | 469.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 13:30:00 | 472.65 | 471.37 | 470.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 476.65 | 472.69 | 471.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 10:30:00 | 479.60 | 473.41 | 471.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 09:15:00 | 472.45 | 473.18 | 473.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 472.45 | 473.18 | 473.21 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 473.75 | 473.25 | 473.24 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 12:15:00 | 471.75 | 472.95 | 473.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 13:15:00 | 470.35 | 472.43 | 472.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 14:15:00 | 474.50 | 472.85 | 473.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 14:15:00 | 474.50 | 472.85 | 473.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 474.50 | 472.85 | 473.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 15:00:00 | 474.50 | 472.85 | 473.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 472.00 | 472.68 | 472.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 474.30 | 472.68 | 472.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 473.90 | 472.92 | 473.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:30:00 | 469.80 | 472.71 | 472.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:00:00 | 470.60 | 471.77 | 472.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 15:00:00 | 471.45 | 471.46 | 472.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 469.85 | 471.77 | 472.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 469.95 | 471.41 | 472.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 11:45:00 | 465.70 | 469.74 | 471.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 13:15:00 | 465.80 | 469.13 | 470.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 10:30:00 | 466.05 | 468.25 | 469.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 446.31 | 456.15 | 459.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 447.07 | 456.15 | 459.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 447.88 | 456.15 | 459.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 446.36 | 456.15 | 459.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 442.41 | 456.15 | 459.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 442.51 | 456.15 | 459.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 442.75 | 456.15 | 459.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 13:15:00 | 443.45 | 443.10 | 448.68 | SL hit (close>ema200) qty=0.50 sl=443.10 alert=retest2 |

### Cycle 123 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 453.90 | 449.97 | 449.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 12:15:00 | 456.60 | 452.68 | 451.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 14:15:00 | 452.35 | 453.02 | 451.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 14:15:00 | 452.35 | 453.02 | 451.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 452.35 | 453.02 | 451.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 452.35 | 453.02 | 451.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 450.00 | 452.41 | 451.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 435.40 | 452.41 | 451.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 439.25 | 449.78 | 450.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 430.80 | 438.26 | 440.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 426.95 | 425.82 | 430.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 428.70 | 425.82 | 430.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 433.80 | 427.68 | 430.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:30:00 | 434.95 | 427.68 | 430.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 433.45 | 428.84 | 430.65 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 436.70 | 432.41 | 431.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 15:15:00 | 437.95 | 433.52 | 432.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 431.00 | 437.09 | 435.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 431.00 | 437.09 | 435.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 431.00 | 437.09 | 435.58 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 428.65 | 433.52 | 434.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 423.05 | 431.43 | 433.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 436.90 | 431.49 | 432.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 436.90 | 431.49 | 432.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 436.90 | 431.49 | 432.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 436.10 | 431.49 | 432.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 437.95 | 432.78 | 433.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 438.50 | 432.78 | 433.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 11:15:00 | 438.35 | 433.89 | 433.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 12:15:00 | 441.45 | 435.40 | 434.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 14:15:00 | 434.00 | 435.60 | 434.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 14:15:00 | 434.00 | 435.60 | 434.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 434.00 | 435.60 | 434.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 434.00 | 435.60 | 434.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 434.15 | 435.31 | 434.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 421.25 | 435.31 | 434.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 417.50 | 431.75 | 433.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 415.55 | 425.19 | 429.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 420.10 | 419.79 | 424.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:00:00 | 420.10 | 419.79 | 424.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 423.15 | 420.46 | 424.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 11:45:00 | 424.50 | 420.46 | 424.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 429.65 | 422.30 | 424.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 428.30 | 422.30 | 424.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 429.75 | 423.79 | 425.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 429.75 | 423.79 | 425.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 431.00 | 427.02 | 426.65 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 12:15:00 | 425.20 | 426.36 | 426.43 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 428.60 | 426.81 | 426.63 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 424.10 | 426.50 | 426.56 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 12:15:00 | 428.05 | 426.84 | 426.68 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 418.75 | 425.13 | 425.93 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 429.75 | 424.96 | 424.70 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 412.35 | 422.53 | 423.78 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 09:15:00 | 428.40 | 423.20 | 423.02 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-04-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 14:15:00 | 422.65 | 423.66 | 423.70 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 433.35 | 425.57 | 424.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 437.60 | 427.98 | 425.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 435.35 | 435.49 | 431.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 435.35 | 435.49 | 431.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 438.10 | 438.29 | 436.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:30:00 | 435.70 | 438.29 | 436.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 432.55 | 437.21 | 435.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 434.20 | 437.21 | 435.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 435.00 | 436.74 | 435.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:30:00 | 436.40 | 436.04 | 435.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 425.00 | 433.83 | 434.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 10:15:00 | 425.00 | 433.83 | 434.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-15 13:15:00 | 422.90 | 429.15 | 432.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 415.60 | 409.11 | 412.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 415.60 | 409.11 | 412.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 415.60 | 409.11 | 412.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:00:00 | 415.60 | 409.11 | 412.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 417.75 | 410.84 | 412.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 11:00:00 | 417.75 | 410.84 | 412.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2026-04-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 13:15:00 | 416.10 | 413.89 | 413.85 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 410.20 | 413.34 | 413.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 11:15:00 | 408.65 | 412.40 | 413.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 13:15:00 | 401.60 | 401.53 | 404.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 14:00:00 | 401.60 | 401.53 | 404.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 402.20 | 401.67 | 404.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:30:00 | 403.55 | 401.67 | 404.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 403.35 | 402.04 | 404.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 10:30:00 | 402.30 | 402.21 | 404.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 401.75 | 402.21 | 404.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 12:45:00 | 401.90 | 401.96 | 403.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:30:00 | 402.50 | 402.15 | 403.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 402.15 | 402.15 | 403.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:30:00 | 403.00 | 402.15 | 403.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-28 09:15:00 | 415.75 | 404.84 | 404.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 415.75 | 404.84 | 404.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 420.20 | 413.48 | 409.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 414.00 | 415.07 | 411.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 13:30:00 | 413.90 | 415.07 | 411.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 414.10 | 414.87 | 412.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:30:00 | 415.45 | 414.87 | 412.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 413.00 | 414.50 | 412.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 409.25 | 414.50 | 412.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 408.45 | 413.29 | 411.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 408.45 | 413.29 | 411.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 410.35 | 412.70 | 411.71 | EMA400 retest candle locked (from upside) |

### Cycle 144 — SELL (started 2026-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 14:15:00 | 410.00 | 411.10 | 411.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 09:15:00 | 395.85 | 407.94 | 409.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 13:15:00 | 403.80 | 403.62 | 406.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 13:45:00 | 403.60 | 403.62 | 406.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 403.10 | 402.58 | 405.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:45:00 | 405.85 | 402.58 | 405.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 402.65 | 402.61 | 404.50 | EMA400 retest candle locked (from downside) |

### Cycle 145 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 410.75 | 405.46 | 405.28 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 12:15:00 | 403.05 | 405.37 | 405.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 09:15:00 | 398.80 | 403.17 | 404.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 13:15:00 | 403.65 | 401.68 | 403.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 13:15:00 | 403.65 | 401.68 | 403.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 403.65 | 401.68 | 403.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:00:00 | 403.65 | 401.68 | 403.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 405.40 | 402.42 | 403.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:30:00 | 405.00 | 402.42 | 403.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-17 09:15:00 | 347.80 | 2024-05-21 14:15:00 | 338.70 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2024-05-21 12:30:00 | 342.00 | 2024-05-21 14:15:00 | 338.70 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-05-30 15:00:00 | 343.05 | 2024-05-30 15:15:00 | 345.65 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-06-03 12:45:00 | 352.50 | 2024-06-04 10:15:00 | 327.60 | STOP_HIT | 1.00 | -7.06% |
| BUY | retest2 | 2024-06-11 12:00:00 | 346.15 | 2024-06-12 13:15:00 | 342.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-06-11 12:45:00 | 348.00 | 2024-06-12 13:15:00 | 342.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-06-12 10:30:00 | 346.05 | 2024-06-12 13:15:00 | 342.00 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-06-14 11:15:00 | 340.95 | 2024-06-14 13:15:00 | 345.10 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-06-14 13:30:00 | 341.05 | 2024-06-18 14:15:00 | 345.15 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-06-14 15:00:00 | 339.60 | 2024-06-18 14:15:00 | 345.15 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-06-19 09:15:00 | 324.40 | 2024-06-24 14:15:00 | 340.05 | STOP_HIT | 1.00 | -4.82% |
| BUY | retest2 | 2024-07-08 09:15:00 | 396.90 | 2024-07-08 10:15:00 | 391.05 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-07-15 12:45:00 | 397.30 | 2024-07-23 13:15:00 | 437.03 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-15 14:30:00 | 395.90 | 2024-07-23 13:15:00 | 435.49 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-06 13:30:00 | 419.20 | 2024-08-08 10:15:00 | 423.25 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-08-06 14:00:00 | 416.50 | 2024-08-08 10:15:00 | 423.25 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-08-13 10:15:00 | 413.90 | 2024-08-16 15:15:00 | 412.00 | STOP_HIT | 1.00 | 0.46% |
| BUY | retest2 | 2024-08-27 09:30:00 | 435.20 | 2024-09-02 12:15:00 | 435.70 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2024-09-11 10:30:00 | 428.95 | 2024-09-11 11:15:00 | 429.55 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2024-09-11 11:15:00 | 428.85 | 2024-09-11 11:15:00 | 429.55 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2024-09-12 09:15:00 | 435.50 | 2024-09-16 09:15:00 | 422.75 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2024-09-12 10:45:00 | 430.80 | 2024-09-16 09:15:00 | 422.75 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2024-09-19 10:15:00 | 414.95 | 2024-09-19 11:15:00 | 373.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-14 09:15:00 | 379.90 | 2024-10-18 09:15:00 | 382.60 | STOP_HIT | 1.00 | 0.71% |
| SELL | retest2 | 2024-10-30 12:00:00 | 344.50 | 2024-11-06 14:15:00 | 341.95 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest2 | 2024-11-21 09:15:00 | 331.90 | 2024-12-05 09:15:00 | 365.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-21 10:00:00 | 330.35 | 2024-12-05 09:15:00 | 363.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-22 11:45:00 | 329.10 | 2024-12-05 09:15:00 | 362.01 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-19 13:15:00 | 343.30 | 2024-12-19 13:15:00 | 345.15 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-12-27 12:45:00 | 333.35 | 2024-12-30 11:15:00 | 336.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-12-30 11:00:00 | 333.40 | 2024-12-30 11:15:00 | 336.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-01-09 09:15:00 | 329.15 | 2025-01-09 09:15:00 | 337.55 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2025-01-09 14:45:00 | 329.75 | 2025-01-14 10:15:00 | 337.05 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-01-10 09:15:00 | 323.75 | 2025-01-14 10:15:00 | 337.05 | STOP_HIT | 1.00 | -4.11% |
| SELL | retest2 | 2025-01-13 11:00:00 | 328.55 | 2025-01-14 10:15:00 | 337.05 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-02-01 11:45:00 | 341.75 | 2025-02-01 13:15:00 | 350.50 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2025-02-07 09:15:00 | 364.70 | 2025-02-10 09:15:00 | 351.80 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2025-02-13 09:15:00 | 343.85 | 2025-02-13 09:15:00 | 353.30 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2025-02-18 09:15:00 | 334.40 | 2025-02-19 09:15:00 | 341.10 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-03-04 09:15:00 | 323.60 | 2025-03-05 09:15:00 | 332.20 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-03-04 12:00:00 | 324.65 | 2025-03-05 09:15:00 | 332.20 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-03-04 13:00:00 | 324.35 | 2025-03-05 09:15:00 | 332.20 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-03-04 13:30:00 | 324.10 | 2025-03-05 09:15:00 | 332.20 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-03-20 13:30:00 | 337.85 | 2025-03-25 13:15:00 | 341.55 | STOP_HIT | 1.00 | 1.10% |
| SELL | retest1 | 2025-03-27 13:15:00 | 335.75 | 2025-03-27 14:15:00 | 340.60 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-03-28 12:15:00 | 337.30 | 2025-04-01 09:15:00 | 359.30 | STOP_HIT | 1.00 | -6.52% |
| SELL | retest2 | 2025-03-28 14:30:00 | 336.30 | 2025-04-01 09:15:00 | 359.30 | STOP_HIT | 1.00 | -6.84% |
| BUY | retest2 | 2025-04-03 13:15:00 | 355.80 | 2025-04-07 09:15:00 | 342.35 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest2 | 2025-04-03 14:15:00 | 356.75 | 2025-04-07 09:15:00 | 342.35 | STOP_HIT | 1.00 | -4.04% |
| BUY | retest2 | 2025-04-24 14:00:00 | 405.25 | 2025-04-25 09:15:00 | 394.75 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2025-05-16 10:15:00 | 399.20 | 2025-05-19 13:15:00 | 389.15 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-05-19 10:00:00 | 399.90 | 2025-05-19 13:15:00 | 389.15 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-05-26 11:15:00 | 378.30 | 2025-05-27 15:15:00 | 384.50 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-05-27 09:45:00 | 377.55 | 2025-05-27 15:15:00 | 384.50 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-06-19 11:00:00 | 393.60 | 2025-06-19 12:15:00 | 389.70 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-06-19 12:15:00 | 393.05 | 2025-06-19 12:15:00 | 389.70 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-06-20 09:15:00 | 397.20 | 2025-07-07 09:15:00 | 409.50 | STOP_HIT | 1.00 | 3.10% |
| SELL | retest1 | 2025-07-09 09:15:00 | 404.45 | 2025-07-10 09:15:00 | 409.50 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest1 | 2025-07-09 13:30:00 | 405.90 | 2025-07-10 09:15:00 | 409.50 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest1 | 2025-07-09 14:00:00 | 405.85 | 2025-07-10 09:15:00 | 409.50 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-07-14 13:30:00 | 402.85 | 2025-07-15 10:15:00 | 406.50 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-07-14 15:15:00 | 403.00 | 2025-07-15 10:15:00 | 406.50 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-07-17 14:30:00 | 407.40 | 2025-07-18 09:15:00 | 402.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-07-22 09:15:00 | 401.00 | 2025-07-23 15:15:00 | 406.50 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-07-28 11:45:00 | 388.95 | 2025-07-31 09:15:00 | 369.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 10:15:00 | 387.85 | 2025-07-31 09:15:00 | 368.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-28 11:45:00 | 388.95 | 2025-08-01 12:15:00 | 350.06 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-30 10:15:00 | 387.85 | 2025-08-01 13:15:00 | 349.07 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-29 14:30:00 | 340.50 | 2025-09-03 09:15:00 | 323.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-01 10:45:00 | 340.45 | 2025-09-03 09:15:00 | 323.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-01 11:45:00 | 340.85 | 2025-09-03 09:15:00 | 323.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-01 12:15:00 | 340.80 | 2025-09-03 09:15:00 | 323.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-01 15:15:00 | 339.00 | 2025-09-03 09:15:00 | 322.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-29 14:30:00 | 340.50 | 2025-09-04 09:15:00 | 325.85 | STOP_HIT | 0.50 | 4.30% |
| SELL | retest2 | 2025-09-01 10:45:00 | 340.45 | 2025-09-04 09:15:00 | 325.85 | STOP_HIT | 0.50 | 4.29% |
| SELL | retest2 | 2025-09-01 11:45:00 | 340.85 | 2025-09-04 09:15:00 | 325.85 | STOP_HIT | 0.50 | 4.40% |
| SELL | retest2 | 2025-09-01 12:15:00 | 340.80 | 2025-09-04 09:15:00 | 325.85 | STOP_HIT | 0.50 | 4.39% |
| SELL | retest2 | 2025-09-01 15:15:00 | 339.00 | 2025-09-04 09:15:00 | 325.85 | STOP_HIT | 0.50 | 3.88% |
| BUY | retest2 | 2025-09-16 12:00:00 | 358.40 | 2025-09-17 09:15:00 | 354.70 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-09-16 12:30:00 | 358.00 | 2025-09-17 09:15:00 | 354.70 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-09-16 13:45:00 | 357.90 | 2025-09-17 09:15:00 | 354.70 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-09-16 15:15:00 | 358.05 | 2025-09-17 09:15:00 | 354.70 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-09-19 10:45:00 | 351.65 | 2025-09-19 11:15:00 | 361.55 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2025-10-03 14:15:00 | 353.45 | 2025-10-13 10:15:00 | 353.65 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-10-06 10:15:00 | 352.00 | 2025-10-13 10:15:00 | 353.65 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest2 | 2025-10-06 14:45:00 | 352.00 | 2025-10-13 10:15:00 | 353.65 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest2 | 2025-10-07 12:00:00 | 353.20 | 2025-10-13 10:15:00 | 353.65 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2025-10-10 11:45:00 | 357.10 | 2025-10-13 10:15:00 | 353.65 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-10-16 13:15:00 | 343.80 | 2025-10-17 11:15:00 | 347.65 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-10-16 15:15:00 | 343.70 | 2025-10-17 11:15:00 | 347.65 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-10-17 14:15:00 | 343.65 | 2025-10-20 10:15:00 | 350.80 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-10-28 09:15:00 | 384.00 | 2025-10-30 11:15:00 | 368.25 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2025-10-28 10:15:00 | 382.00 | 2025-10-30 11:15:00 | 368.25 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2025-10-28 11:15:00 | 383.20 | 2025-10-30 11:15:00 | 368.25 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest2 | 2025-10-28 12:30:00 | 384.05 | 2025-10-30 11:15:00 | 368.25 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest2 | 2025-10-30 11:15:00 | 371.30 | 2025-10-30 11:15:00 | 368.25 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-11-11 12:30:00 | 400.05 | 2025-11-18 10:15:00 | 404.35 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2025-11-11 13:15:00 | 400.90 | 2025-11-18 10:15:00 | 404.35 | STOP_HIT | 1.00 | 0.86% |
| SELL | retest2 | 2025-11-21 09:15:00 | 398.50 | 2025-11-25 13:15:00 | 404.95 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-11-21 13:00:00 | 399.45 | 2025-11-25 13:15:00 | 404.95 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-11-24 15:15:00 | 399.20 | 2025-11-25 13:15:00 | 404.95 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest1 | 2025-11-27 09:15:00 | 406.90 | 2025-11-27 12:15:00 | 402.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest1 | 2025-11-27 10:00:00 | 406.55 | 2025-11-27 12:15:00 | 402.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-11-27 14:30:00 | 406.25 | 2025-11-28 09:15:00 | 401.55 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-12-01 10:45:00 | 400.10 | 2025-12-03 09:15:00 | 409.00 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-12-02 09:45:00 | 400.15 | 2025-12-03 09:15:00 | 409.00 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-12-02 13:00:00 | 400.10 | 2025-12-03 09:15:00 | 409.00 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-12-16 10:15:00 | 411.20 | 2025-12-16 10:15:00 | 408.80 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-12-17 13:30:00 | 405.90 | 2025-12-18 11:15:00 | 410.35 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-12-17 14:15:00 | 406.05 | 2025-12-18 11:15:00 | 410.35 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-12-22 12:30:00 | 412.00 | 2025-12-31 15:15:00 | 421.00 | STOP_HIT | 1.00 | 2.18% |
| BUY | retest2 | 2025-12-22 15:15:00 | 411.95 | 2025-12-31 15:15:00 | 421.00 | STOP_HIT | 1.00 | 2.20% |
| BUY | retest2 | 2025-12-23 09:45:00 | 412.90 | 2025-12-31 15:15:00 | 421.00 | STOP_HIT | 1.00 | 1.96% |
| BUY | retest2 | 2025-12-23 10:45:00 | 413.05 | 2025-12-31 15:15:00 | 421.00 | STOP_HIT | 1.00 | 1.92% |
| BUY | retest2 | 2025-12-24 10:15:00 | 420.05 | 2025-12-31 15:15:00 | 421.00 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2025-12-29 09:30:00 | 421.00 | 2025-12-31 15:15:00 | 421.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-12-31 14:15:00 | 420.85 | 2025-12-31 15:15:00 | 421.00 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2025-12-31 15:15:00 | 421.00 | 2025-12-31 15:15:00 | 421.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2026-01-08 11:15:00 | 427.70 | 2026-01-08 12:15:00 | 434.30 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2026-01-22 10:45:00 | 414.50 | 2026-01-23 09:15:00 | 422.10 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-01-22 12:45:00 | 414.50 | 2026-01-23 09:15:00 | 422.10 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-01-23 11:45:00 | 414.30 | 2026-01-27 14:15:00 | 422.30 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2026-01-27 10:15:00 | 414.75 | 2026-01-27 14:15:00 | 422.30 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2026-01-27 13:45:00 | 415.95 | 2026-01-27 14:15:00 | 422.30 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2026-02-16 11:45:00 | 472.80 | 2026-02-20 09:15:00 | 472.45 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2026-02-16 13:30:00 | 472.45 | 2026-02-20 09:15:00 | 472.45 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2026-02-17 10:15:00 | 471.55 | 2026-02-20 09:15:00 | 472.45 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2026-02-17 13:30:00 | 472.65 | 2026-02-20 09:15:00 | 472.45 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2026-02-18 10:30:00 | 479.60 | 2026-02-20 09:15:00 | 472.45 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2026-02-23 10:30:00 | 469.80 | 2026-03-02 09:15:00 | 446.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 13:00:00 | 470.60 | 2026-03-02 09:15:00 | 447.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 15:00:00 | 471.45 | 2026-03-02 09:15:00 | 447.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 469.85 | 2026-03-02 09:15:00 | 446.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 11:45:00 | 465.70 | 2026-03-02 09:15:00 | 442.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 13:15:00 | 465.80 | 2026-03-02 09:15:00 | 442.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 10:30:00 | 466.05 | 2026-03-02 09:15:00 | 442.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 10:30:00 | 469.80 | 2026-03-04 13:15:00 | 443.45 | STOP_HIT | 0.50 | 5.61% |
| SELL | retest2 | 2026-02-23 13:00:00 | 470.60 | 2026-03-04 13:15:00 | 443.45 | STOP_HIT | 0.50 | 5.77% |
| SELL | retest2 | 2026-02-23 15:00:00 | 471.45 | 2026-03-04 13:15:00 | 443.45 | STOP_HIT | 0.50 | 5.94% |
| SELL | retest2 | 2026-02-24 09:15:00 | 469.85 | 2026-03-04 13:15:00 | 443.45 | STOP_HIT | 0.50 | 5.62% |
| SELL | retest2 | 2026-02-24 11:45:00 | 465.70 | 2026-03-04 13:15:00 | 443.45 | STOP_HIT | 0.50 | 4.78% |
| SELL | retest2 | 2026-02-24 13:15:00 | 465.80 | 2026-03-04 13:15:00 | 443.45 | STOP_HIT | 0.50 | 4.80% |
| SELL | retest2 | 2026-02-25 10:30:00 | 466.05 | 2026-03-04 13:15:00 | 443.45 | STOP_HIT | 0.50 | 4.85% |
| BUY | retest2 | 2026-04-13 10:15:00 | 434.20 | 2026-04-15 10:15:00 | 425.00 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2026-04-13 10:45:00 | 435.00 | 2026-04-15 10:15:00 | 425.00 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2026-04-15 09:30:00 | 436.40 | 2026-04-15 10:15:00 | 425.00 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2026-04-27 10:30:00 | 402.30 | 2026-04-28 09:15:00 | 415.75 | STOP_HIT | 1.00 | -3.34% |
| SELL | retest2 | 2026-04-27 11:15:00 | 401.75 | 2026-04-28 09:15:00 | 415.75 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2026-04-27 12:45:00 | 401.90 | 2026-04-28 09:15:00 | 415.75 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2026-04-27 13:30:00 | 402.50 | 2026-04-28 09:15:00 | 415.75 | STOP_HIT | 1.00 | -3.29% |
