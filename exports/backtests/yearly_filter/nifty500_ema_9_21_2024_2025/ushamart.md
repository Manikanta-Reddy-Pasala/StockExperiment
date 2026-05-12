# Usha Martin Ltd. (USHAMART)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3709 bars)
- **Last close:** 472.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 167 |
| ALERT1 | 110 |
| ALERT2 | 108 |
| ALERT2_SKIP | 63 |
| ALERT3 | 297 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 100 |
| PARTIAL | 15 |
| TARGET_HIT | 2 |
| STOP_HIT | 103 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 120 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 35 / 85
- **Target hits / Stop hits / Partials:** 2 / 103 / 15
- **Avg / median % per leg:** -0.01% / -0.88%
- **Sum % (uncompounded):** -1.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 50 | 4 | 8.0% | 2 | 48 | 0 | -0.92% | -46.1% |
| BUY @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.69% | -3.4% |
| BUY @ 3rd Alert (retest2) | 45 | 4 | 8.9% | 2 | 43 | 0 | -0.95% | -42.6% |
| SELL (all) | 70 | 31 | 44.3% | 0 | 55 | 15 | 0.64% | 44.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 70 | 31 | 44.3% | 0 | 55 | 15 | 0.64% | 44.7% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.69% | -3.4% |
| retest2 (combined) | 115 | 35 | 30.4% | 2 | 98 | 15 | 0.02% | 2.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 13:15:00 | 345.65 | 342.00 | 341.91 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 11:15:00 | 339.30 | 341.70 | 341.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 13:15:00 | 338.75 | 340.68 | 341.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-17 09:15:00 | 335.95 | 335.71 | 337.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-17 10:15:00 | 336.80 | 335.71 | 337.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 10:15:00 | 336.65 | 335.89 | 337.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 10:45:00 | 337.00 | 335.89 | 337.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 11:15:00 | 340.65 | 336.85 | 337.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 11:45:00 | 339.75 | 336.85 | 337.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-05-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 12:15:00 | 345.50 | 338.58 | 338.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 13:15:00 | 352.90 | 341.44 | 339.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 10:15:00 | 350.95 | 351.82 | 347.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 10:45:00 | 351.60 | 351.82 | 347.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 12:15:00 | 347.65 | 350.48 | 347.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 12:30:00 | 347.00 | 350.48 | 347.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 345.90 | 349.56 | 347.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 13:45:00 | 345.20 | 349.56 | 347.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 344.00 | 348.45 | 346.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 15:00:00 | 344.00 | 348.45 | 346.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 341.00 | 346.96 | 346.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:15:00 | 338.55 | 346.96 | 346.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 351.65 | 347.24 | 346.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 12:30:00 | 354.60 | 349.39 | 347.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 15:00:00 | 352.80 | 350.99 | 348.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 10:15:00 | 350.05 | 354.02 | 354.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 10:15:00 | 350.05 | 354.02 | 354.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 11:15:00 | 348.90 | 353.00 | 353.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 15:15:00 | 353.65 | 352.07 | 352.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 15:15:00 | 353.65 | 352.07 | 352.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 15:15:00 | 353.65 | 352.07 | 352.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 09:15:00 | 355.95 | 352.07 | 352.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 09:15:00 | 358.95 | 353.45 | 353.41 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 09:15:00 | 348.95 | 361.22 | 361.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 314.90 | 351.96 | 357.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 342.45 | 338.00 | 344.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 12:15:00 | 342.45 | 338.00 | 344.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 342.45 | 338.00 | 344.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:00:00 | 342.45 | 338.00 | 344.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 349.15 | 340.98 | 343.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:45:00 | 349.45 | 340.98 | 343.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 351.85 | 343.15 | 344.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:45:00 | 351.60 | 343.15 | 344.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 12:15:00 | 353.15 | 346.71 | 346.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 13:15:00 | 357.00 | 348.77 | 347.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 12:15:00 | 370.10 | 370.12 | 365.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 13:00:00 | 370.10 | 370.12 | 365.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 393.85 | 400.47 | 394.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:00:00 | 393.85 | 400.47 | 394.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 394.90 | 399.35 | 394.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 11:15:00 | 393.60 | 399.35 | 394.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 11:15:00 | 393.75 | 398.23 | 394.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 11:30:00 | 393.05 | 398.23 | 394.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 12:15:00 | 393.95 | 397.38 | 394.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 13:00:00 | 393.95 | 397.38 | 394.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 13:15:00 | 395.10 | 396.92 | 394.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 13:45:00 | 399.90 | 396.92 | 394.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 14:15:00 | 396.30 | 396.80 | 394.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 14:45:00 | 394.30 | 396.80 | 394.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 388.60 | 394.98 | 393.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 392.00 | 394.98 | 393.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 388.45 | 393.68 | 393.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:30:00 | 388.50 | 393.68 | 393.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 11:15:00 | 389.75 | 392.89 | 393.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 12:15:00 | 388.00 | 391.91 | 392.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 15:15:00 | 398.90 | 389.98 | 390.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 15:15:00 | 398.90 | 389.98 | 390.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 15:15:00 | 398.90 | 389.98 | 390.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:15:00 | 401.05 | 389.98 | 390.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 401.30 | 392.24 | 391.33 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 10:15:00 | 390.20 | 395.80 | 396.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 12:15:00 | 386.90 | 390.37 | 392.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 09:15:00 | 389.90 | 389.18 | 391.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 09:15:00 | 389.90 | 389.18 | 391.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 389.90 | 389.18 | 391.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 11:45:00 | 386.95 | 388.46 | 390.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 10:15:00 | 385.00 | 384.51 | 387.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 15:15:00 | 385.00 | 385.69 | 386.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 12:15:00 | 395.85 | 388.68 | 387.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 12:15:00 | 395.85 | 388.68 | 387.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 13:15:00 | 396.25 | 390.20 | 388.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 15:15:00 | 402.80 | 403.38 | 400.05 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:15:00 | 408.40 | 403.38 | 400.05 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 12:15:00 | 408.30 | 404.85 | 401.70 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 12:15:00 | 406.65 | 408.66 | 405.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 13:00:00 | 406.65 | 408.66 | 405.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 13:15:00 | 410.60 | 409.04 | 406.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 13:45:00 | 408.80 | 409.04 | 406.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 408.00 | 409.56 | 407.25 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-08 10:15:00 | 403.40 | 408.33 | 406.90 | SL hit (close<ema400) qty=1.00 sl=406.90 alert=retest1 |

### Cycle 12 — SELL (started 2024-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 09:15:00 | 396.35 | 404.49 | 405.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 11:15:00 | 390.65 | 400.05 | 403.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 14:15:00 | 385.15 | 384.04 | 388.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-11 15:00:00 | 385.15 | 384.04 | 388.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 391.00 | 385.43 | 388.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:15:00 | 386.50 | 385.43 | 388.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 386.10 | 385.56 | 388.22 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 392.40 | 387.48 | 387.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 10:15:00 | 393.00 | 388.58 | 387.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 13:15:00 | 388.70 | 388.90 | 388.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 14:00:00 | 388.70 | 388.90 | 388.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 391.70 | 389.46 | 388.43 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 383.90 | 387.99 | 388.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 371.40 | 382.46 | 385.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 11:15:00 | 368.45 | 368.05 | 371.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 11:15:00 | 368.45 | 368.05 | 371.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 368.45 | 368.05 | 371.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 356.65 | 368.05 | 371.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:45:00 | 360.10 | 366.25 | 370.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:30:00 | 361.40 | 365.23 | 369.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 09:15:00 | 361.65 | 364.66 | 366.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 366.45 | 364.55 | 366.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 12:45:00 | 365.70 | 364.55 | 366.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 366.75 | 364.99 | 366.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 13:45:00 | 367.95 | 364.99 | 366.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 362.65 | 364.52 | 365.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 09:45:00 | 361.60 | 363.48 | 365.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 10:15:00 | 370.95 | 364.98 | 365.64 | SL hit (close>static) qty=1.00 sl=366.75 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 13:15:00 | 369.70 | 366.57 | 366.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 14:15:00 | 371.60 | 367.58 | 366.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 11:15:00 | 366.70 | 369.28 | 368.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 11:15:00 | 366.70 | 369.28 | 368.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 11:15:00 | 366.70 | 369.28 | 368.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 11:45:00 | 366.45 | 369.28 | 368.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 12:15:00 | 366.45 | 368.72 | 367.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 12:45:00 | 366.75 | 368.72 | 367.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 14:15:00 | 368.35 | 368.27 | 367.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 15:15:00 | 371.15 | 368.27 | 367.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 10:00:00 | 371.20 | 369.32 | 368.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 11:15:00 | 369.00 | 376.56 | 377.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 11:15:00 | 369.00 | 376.56 | 377.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 357.15 | 369.63 | 373.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 351.35 | 349.15 | 354.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 11:00:00 | 351.35 | 349.15 | 354.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 353.40 | 350.53 | 354.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:45:00 | 354.75 | 350.53 | 354.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 355.65 | 351.56 | 354.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:00:00 | 355.65 | 351.56 | 354.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 355.00 | 352.25 | 354.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 355.00 | 352.25 | 354.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 354.25 | 352.94 | 354.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:00:00 | 346.50 | 351.30 | 353.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 15:15:00 | 345.45 | 351.01 | 351.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 15:15:00 | 329.18 | 337.80 | 341.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 15:15:00 | 328.18 | 337.80 | 341.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-14 11:15:00 | 338.00 | 336.49 | 340.03 | SL hit (close>ema200) qty=0.50 sl=336.49 alert=retest2 |

### Cycle 17 — BUY (started 2024-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 11:15:00 | 338.60 | 331.30 | 331.04 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 11:15:00 | 331.90 | 334.60 | 334.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 12:15:00 | 330.60 | 333.80 | 334.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 15:15:00 | 333.70 | 333.32 | 333.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 15:15:00 | 333.70 | 333.32 | 333.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 333.70 | 333.32 | 333.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 333.40 | 333.32 | 333.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 335.25 | 333.71 | 334.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:00:00 | 335.25 | 333.71 | 334.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 336.10 | 334.19 | 334.21 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2024-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 11:15:00 | 336.20 | 334.59 | 334.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 11:15:00 | 337.00 | 336.12 | 335.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 14:15:00 | 340.20 | 341.98 | 339.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 14:15:00 | 340.20 | 341.98 | 339.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 340.20 | 341.98 | 339.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 15:00:00 | 340.20 | 341.98 | 339.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 341.70 | 341.92 | 340.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:15:00 | 340.55 | 341.92 | 340.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 338.70 | 341.28 | 339.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:00:00 | 338.70 | 341.28 | 339.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 335.40 | 340.10 | 339.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:45:00 | 335.95 | 340.10 | 339.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 11:15:00 | 333.70 | 338.82 | 338.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 13:15:00 | 333.50 | 337.26 | 338.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 14:15:00 | 337.35 | 337.28 | 338.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-29 15:00:00 | 337.35 | 337.28 | 338.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 333.00 | 336.42 | 337.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 15:00:00 | 331.70 | 334.90 | 336.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 10:15:00 | 339.95 | 336.38 | 336.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 10:15:00 | 339.95 | 336.38 | 336.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 09:15:00 | 342.60 | 337.63 | 336.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 09:15:00 | 338.30 | 339.64 | 338.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 09:15:00 | 338.30 | 339.64 | 338.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 338.30 | 339.64 | 338.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 10:00:00 | 338.30 | 339.64 | 338.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 339.20 | 339.55 | 338.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 11:00:00 | 339.20 | 339.55 | 338.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 11:15:00 | 340.20 | 339.68 | 338.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 09:15:00 | 340.65 | 339.90 | 339.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 10:15:00 | 340.65 | 339.92 | 339.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 10:15:00 | 336.00 | 339.14 | 338.94 | SL hit (close<static) qty=1.00 sl=336.25 alert=retest2 |

### Cycle 22 — SELL (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 12:15:00 | 353.60 | 358.49 | 359.06 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-09-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 12:15:00 | 360.90 | 359.36 | 359.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 15:15:00 | 362.00 | 360.07 | 359.55 | Break + close above crossover candle high |

### Cycle 24 — SELL (started 2024-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 09:15:00 | 354.15 | 358.89 | 359.06 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 11:15:00 | 359.65 | 359.21 | 359.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 12:15:00 | 362.70 | 359.91 | 359.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 13:15:00 | 358.75 | 359.68 | 359.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 13:15:00 | 358.75 | 359.68 | 359.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 13:15:00 | 358.75 | 359.68 | 359.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 13:45:00 | 360.55 | 359.68 | 359.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2024-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 14:15:00 | 355.90 | 358.92 | 359.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 09:15:00 | 352.85 | 356.21 | 357.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 350.30 | 343.24 | 347.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 350.30 | 343.24 | 347.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 350.30 | 343.24 | 347.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:30:00 | 354.40 | 343.24 | 347.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 353.05 | 345.20 | 347.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:30:00 | 354.30 | 345.20 | 347.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 346.45 | 347.75 | 348.32 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2024-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 13:15:00 | 353.90 | 349.15 | 348.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 14:15:00 | 356.90 | 354.12 | 352.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 12:15:00 | 354.65 | 355.70 | 353.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 12:15:00 | 354.65 | 355.70 | 353.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 12:15:00 | 354.65 | 355.70 | 353.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 12:30:00 | 354.50 | 355.70 | 353.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 13:15:00 | 353.30 | 355.22 | 353.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 14:00:00 | 353.30 | 355.22 | 353.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 351.20 | 354.41 | 353.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 351.20 | 354.41 | 353.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 350.25 | 353.58 | 353.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 09:15:00 | 353.60 | 353.58 | 353.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 10:15:00 | 350.80 | 352.92 | 353.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 10:15:00 | 350.80 | 352.92 | 353.10 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-09-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 13:15:00 | 356.10 | 353.41 | 353.26 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 11:15:00 | 349.50 | 352.69 | 353.06 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 15:15:00 | 360.50 | 354.60 | 353.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 11:15:00 | 365.65 | 358.39 | 355.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 359.45 | 362.80 | 359.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 09:15:00 | 359.45 | 362.80 | 359.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 359.45 | 362.80 | 359.44 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 10:15:00 | 357.35 | 358.09 | 358.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 12:15:00 | 354.30 | 357.10 | 357.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 12:15:00 | 345.05 | 342.12 | 346.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 13:00:00 | 345.05 | 342.12 | 346.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 345.25 | 342.75 | 346.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:00:00 | 345.25 | 342.75 | 346.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 345.15 | 343.23 | 346.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 345.15 | 343.23 | 346.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 345.30 | 343.64 | 346.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 346.45 | 343.64 | 346.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 346.60 | 344.23 | 346.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 14:15:00 | 343.70 | 346.35 | 346.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 09:15:00 | 353.85 | 346.90 | 346.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 09:15:00 | 353.85 | 346.90 | 346.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 10:15:00 | 355.15 | 348.55 | 347.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 09:15:00 | 435.20 | 436.51 | 423.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-16 10:00:00 | 435.20 | 436.51 | 423.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 424.80 | 433.10 | 427.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:45:00 | 423.25 | 433.10 | 427.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 423.30 | 431.14 | 427.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:15:00 | 422.65 | 431.14 | 427.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 15:15:00 | 420.50 | 425.06 | 425.57 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 14:15:00 | 428.45 | 425.25 | 425.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 15:15:00 | 431.50 | 426.50 | 425.71 | Break + close above crossover candle high |

### Cycle 36 — SELL (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 09:15:00 | 417.90 | 424.78 | 425.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 10:15:00 | 408.95 | 421.62 | 423.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 12:15:00 | 420.50 | 420.19 | 422.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-21 13:00:00 | 420.50 | 420.19 | 422.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 13:15:00 | 422.05 | 420.56 | 422.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 13:45:00 | 427.70 | 420.56 | 422.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 420.40 | 420.53 | 422.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 15:15:00 | 414.90 | 420.53 | 422.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 394.15 | 406.29 | 413.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-24 09:15:00 | 402.90 | 398.97 | 404.45 | SL hit (close>ema200) qty=0.50 sl=398.97 alert=retest2 |

### Cycle 37 — BUY (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 11:15:00 | 422.75 | 397.89 | 396.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 14:15:00 | 425.75 | 418.81 | 414.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 418.00 | 422.04 | 417.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 418.00 | 422.04 | 417.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 418.00 | 422.04 | 417.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 418.00 | 422.04 | 417.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 417.95 | 421.22 | 417.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 417.95 | 421.22 | 417.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 419.00 | 420.78 | 418.06 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 09:15:00 | 416.35 | 416.95 | 417.00 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 15:15:00 | 420.40 | 417.12 | 416.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 11:15:00 | 429.90 | 420.35 | 418.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 14:15:00 | 415.55 | 420.99 | 419.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 14:15:00 | 415.55 | 420.99 | 419.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 415.55 | 420.99 | 419.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 415.55 | 420.99 | 419.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 417.95 | 420.38 | 419.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 415.80 | 420.38 | 419.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 09:15:00 | 412.15 | 418.73 | 418.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 10:15:00 | 407.10 | 416.41 | 417.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 381.10 | 375.25 | 383.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 381.10 | 375.25 | 383.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 378.40 | 375.88 | 382.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 385.05 | 375.88 | 382.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 384.75 | 377.65 | 382.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 12:00:00 | 384.75 | 377.65 | 382.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 385.10 | 379.14 | 383.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:00:00 | 385.10 | 379.14 | 383.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 381.35 | 379.58 | 382.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 14:15:00 | 379.50 | 379.58 | 382.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 15:15:00 | 378.20 | 379.79 | 382.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 10:15:00 | 377.35 | 379.72 | 382.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 398.00 | 385.23 | 383.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 398.00 | 385.23 | 383.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 400.50 | 394.34 | 392.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 14:15:00 | 383.50 | 393.52 | 393.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 14:15:00 | 383.50 | 393.52 | 393.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 383.50 | 393.52 | 393.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 15:00:00 | 383.50 | 393.52 | 393.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2024-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 15:15:00 | 383.50 | 391.51 | 392.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 13:15:00 | 378.45 | 386.55 | 389.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 14:15:00 | 390.55 | 384.65 | 386.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 14:15:00 | 390.55 | 384.65 | 386.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 390.55 | 384.65 | 386.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 15:00:00 | 390.55 | 384.65 | 386.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 15:15:00 | 388.60 | 385.44 | 386.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 10:45:00 | 386.65 | 386.25 | 386.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 11:15:00 | 384.35 | 382.48 | 382.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2024-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 11:15:00 | 384.35 | 382.48 | 382.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 13:15:00 | 388.00 | 383.99 | 383.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 09:15:00 | 403.50 | 403.73 | 399.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-06 10:00:00 | 403.50 | 403.73 | 399.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 406.20 | 409.08 | 406.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:00:00 | 406.20 | 409.08 | 406.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 411.30 | 409.52 | 407.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 11:15:00 | 421.05 | 409.52 | 407.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 14:45:00 | 413.35 | 410.46 | 408.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 09:15:00 | 402.95 | 407.56 | 407.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 402.95 | 407.56 | 407.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 11:15:00 | 398.85 | 405.07 | 406.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-17 12:15:00 | 388.35 | 388.11 | 392.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 12:15:00 | 388.35 | 388.11 | 392.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 388.35 | 388.11 | 392.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:45:00 | 391.50 | 388.11 | 392.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 380.65 | 376.98 | 379.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 11:30:00 | 379.70 | 376.98 | 379.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 380.15 | 377.62 | 379.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 12:30:00 | 381.95 | 377.62 | 379.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 380.50 | 378.19 | 379.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 14:30:00 | 377.50 | 376.67 | 378.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 09:15:00 | 387.00 | 371.81 | 372.93 | SL hit (close>static) qty=1.00 sl=381.50 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 10:15:00 | 385.75 | 374.60 | 374.10 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 11:15:00 | 374.85 | 377.47 | 377.82 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 10:15:00 | 380.00 | 377.62 | 377.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 11:15:00 | 381.05 | 378.31 | 377.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 12:15:00 | 381.10 | 382.17 | 380.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 13:00:00 | 381.10 | 382.17 | 380.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 13:15:00 | 381.00 | 381.94 | 380.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 13:45:00 | 380.25 | 381.94 | 380.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 14:15:00 | 379.20 | 381.39 | 380.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 15:00:00 | 379.20 | 381.39 | 380.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 15:15:00 | 378.10 | 380.73 | 380.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 09:15:00 | 377.85 | 380.73 | 380.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 09:15:00 | 376.85 | 379.96 | 380.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 10:15:00 | 374.05 | 378.77 | 379.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-06 13:15:00 | 366.10 | 366.00 | 370.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-06 13:30:00 | 367.30 | 366.00 | 370.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 369.15 | 366.63 | 370.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 15:00:00 | 369.15 | 366.63 | 370.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 15:15:00 | 367.90 | 366.88 | 370.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:15:00 | 370.95 | 366.88 | 370.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 377.25 | 368.96 | 370.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:00:00 | 377.25 | 368.96 | 370.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 380.65 | 371.30 | 371.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 380.65 | 371.30 | 371.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 12:15:00 | 376.05 | 372.90 | 372.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 14:15:00 | 380.30 | 374.93 | 373.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 09:15:00 | 375.50 | 375.86 | 374.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 09:15:00 | 375.50 | 375.86 | 374.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 375.50 | 375.86 | 374.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:30:00 | 373.65 | 375.86 | 374.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 375.65 | 375.82 | 374.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 11:45:00 | 377.20 | 375.79 | 374.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 12:15:00 | 373.35 | 375.30 | 374.38 | SL hit (close<static) qty=1.00 sl=373.75 alert=retest2 |

### Cycle 50 — SELL (started 2025-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 14:15:00 | 369.45 | 373.69 | 373.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 11:15:00 | 368.60 | 371.24 | 372.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 345.50 | 343.83 | 350.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 11:15:00 | 349.80 | 345.66 | 349.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 349.80 | 345.66 | 349.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:30:00 | 350.70 | 345.66 | 349.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 12:15:00 | 353.10 | 347.15 | 350.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 13:00:00 | 353.10 | 347.15 | 350.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 356.05 | 348.93 | 350.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 13:30:00 | 358.00 | 348.93 | 350.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 349.05 | 349.78 | 350.81 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 357.15 | 351.67 | 351.32 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 14:15:00 | 350.50 | 353.62 | 353.83 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 12:15:00 | 358.35 | 353.97 | 353.77 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 349.30 | 354.35 | 354.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 10:15:00 | 346.20 | 352.72 | 353.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 347.75 | 347.10 | 349.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:00:00 | 347.75 | 347.10 | 349.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 347.95 | 347.27 | 349.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 347.20 | 347.27 | 349.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 346.50 | 347.12 | 349.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 13:30:00 | 343.80 | 346.05 | 348.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 326.61 | 334.22 | 339.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 11:15:00 | 327.00 | 323.20 | 329.37 | SL hit (close>ema200) qty=0.50 sl=323.20 alert=retest2 |

### Cycle 55 — BUY (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 09:15:00 | 345.15 | 331.39 | 331.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 14:15:00 | 350.10 | 344.21 | 340.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 09:15:00 | 344.80 | 345.72 | 341.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 09:30:00 | 344.75 | 345.72 | 341.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 344.10 | 345.39 | 342.03 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-01-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 13:15:00 | 332.60 | 339.39 | 339.85 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-01-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 15:15:00 | 346.00 | 340.73 | 340.38 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 09:15:00 | 336.90 | 339.97 | 340.07 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 14:15:00 | 342.35 | 339.87 | 339.77 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 15:15:00 | 338.80 | 339.66 | 339.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 336.00 | 338.92 | 339.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 321.85 | 321.69 | 326.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 12:15:00 | 325.70 | 322.74 | 325.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 12:15:00 | 325.70 | 322.74 | 325.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 13:00:00 | 325.70 | 322.74 | 325.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 13:15:00 | 330.85 | 324.36 | 326.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 14:00:00 | 330.85 | 324.36 | 326.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 14:15:00 | 331.45 | 325.78 | 326.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 15:00:00 | 331.45 | 325.78 | 326.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 10:15:00 | 328.45 | 327.41 | 327.32 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 12:15:00 | 325.50 | 327.03 | 327.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 321.60 | 325.24 | 326.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 11:15:00 | 325.75 | 325.10 | 325.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 11:15:00 | 325.75 | 325.10 | 325.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 325.75 | 325.10 | 325.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:45:00 | 326.25 | 325.10 | 325.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 12:15:00 | 326.40 | 325.36 | 326.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 12:30:00 | 326.15 | 325.36 | 326.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 328.75 | 326.04 | 326.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 14:00:00 | 328.75 | 326.04 | 326.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2025-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 14:15:00 | 328.30 | 326.49 | 326.44 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 319.85 | 325.46 | 326.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 318.05 | 323.11 | 324.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 301.50 | 300.72 | 307.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 13:00:00 | 301.50 | 300.72 | 307.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 305.25 | 300.86 | 305.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 306.00 | 300.86 | 305.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 305.85 | 301.86 | 305.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 305.85 | 301.86 | 305.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 303.85 | 302.26 | 305.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 12:30:00 | 303.50 | 302.54 | 304.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:00:00 | 303.65 | 302.54 | 304.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 15:00:00 | 302.35 | 299.06 | 301.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-17 15:15:00 | 305.00 | 300.67 | 300.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 15:15:00 | 305.00 | 300.67 | 300.53 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 09:15:00 | 292.80 | 299.09 | 299.83 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 14:15:00 | 301.00 | 295.01 | 294.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-21 09:15:00 | 302.55 | 297.48 | 296.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 303.50 | 303.94 | 300.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 09:15:00 | 303.50 | 303.94 | 300.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 303.50 | 303.94 | 300.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 10:30:00 | 305.85 | 304.70 | 301.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 09:30:00 | 304.95 | 306.23 | 304.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 13:15:00 | 297.85 | 302.47 | 302.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 13:15:00 | 297.85 | 302.47 | 302.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 14:15:00 | 291.45 | 300.27 | 301.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 294.00 | 293.83 | 297.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-27 15:00:00 | 294.00 | 293.83 | 297.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 295.15 | 292.95 | 294.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 14:45:00 | 297.40 | 292.95 | 294.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 296.50 | 293.66 | 295.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:15:00 | 296.50 | 293.66 | 295.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 289.35 | 292.80 | 294.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 11:00:00 | 287.55 | 291.75 | 293.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-03 15:15:00 | 303.00 | 295.97 | 295.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-03-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 15:15:00 | 303.00 | 295.97 | 295.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 14:15:00 | 313.55 | 302.94 | 299.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 329.45 | 329.56 | 322.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 329.45 | 329.56 | 322.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 327.15 | 329.44 | 326.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:00:00 | 327.15 | 329.44 | 326.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 322.20 | 327.99 | 326.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 15:00:00 | 322.20 | 327.99 | 326.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 325.30 | 327.45 | 326.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 316.35 | 327.45 | 326.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 322.90 | 326.54 | 326.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:30:00 | 317.90 | 326.54 | 326.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 11:15:00 | 320.40 | 325.03 | 325.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 09:15:00 | 318.00 | 321.01 | 323.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 312.50 | 308.99 | 311.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 312.50 | 308.99 | 311.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 312.50 | 308.99 | 311.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 312.50 | 308.99 | 311.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 314.95 | 310.18 | 311.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:45:00 | 315.85 | 310.18 | 311.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 11:15:00 | 323.00 | 312.74 | 312.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 12:00:00 | 323.00 | 312.74 | 312.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 12:15:00 | 319.55 | 314.11 | 313.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 14:15:00 | 325.00 | 319.13 | 316.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 337.00 | 338.33 | 334.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 09:15:00 | 340.75 | 338.81 | 336.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 340.75 | 338.81 | 336.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 10:15:00 | 342.50 | 338.81 | 336.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 10:15:00 | 343.15 | 343.81 | 340.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 14:45:00 | 343.15 | 341.75 | 340.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 12:15:00 | 338.00 | 340.30 | 340.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 12:15:00 | 338.00 | 340.30 | 340.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 15:15:00 | 335.50 | 338.62 | 339.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 09:15:00 | 335.20 | 335.16 | 336.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 335.20 | 335.16 | 336.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 335.20 | 335.16 | 336.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 09:15:00 | 324.05 | 332.58 | 334.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 307.85 | 317.98 | 324.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-08 09:15:00 | 304.35 | 304.31 | 312.66 | SL hit (close>ema200) qty=0.50 sl=304.31 alert=retest2 |

### Cycle 73 — BUY (started 2025-04-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 15:15:00 | 306.60 | 304.35 | 304.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 311.15 | 305.71 | 304.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 315.30 | 316.89 | 313.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 315.30 | 316.89 | 313.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 13:15:00 | 313.85 | 315.78 | 313.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 13:45:00 | 313.70 | 315.78 | 313.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 14:15:00 | 311.90 | 315.00 | 313.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 15:00:00 | 311.90 | 315.00 | 313.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 312.50 | 314.50 | 313.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:15:00 | 311.00 | 314.50 | 313.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 10:15:00 | 313.40 | 314.07 | 313.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 13:00:00 | 314.65 | 314.10 | 313.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 09:15:00 | 317.40 | 313.79 | 313.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 12:15:00 | 314.30 | 314.73 | 314.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-22 14:15:00 | 312.00 | 313.65 | 313.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-22 14:15:00 | 312.00 | 313.65 | 313.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-23 09:15:00 | 306.65 | 312.05 | 312.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 09:15:00 | 309.50 | 308.34 | 310.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-24 10:15:00 | 310.30 | 308.34 | 310.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 309.50 | 308.57 | 310.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:45:00 | 310.00 | 308.57 | 310.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 11:15:00 | 309.00 | 308.66 | 309.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 11:30:00 | 309.95 | 308.66 | 309.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 293.00 | 304.38 | 307.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 09:15:00 | 287.05 | 294.88 | 297.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 10:00:00 | 289.10 | 293.72 | 296.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 09:15:00 | 288.05 | 290.17 | 293.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 14:15:00 | 303.50 | 294.63 | 294.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-05-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 14:15:00 | 303.50 | 294.63 | 294.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 11:15:00 | 307.75 | 301.08 | 299.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 15:15:00 | 301.65 | 305.35 | 303.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 15:15:00 | 301.65 | 305.35 | 303.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 301.65 | 305.35 | 303.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 297.55 | 305.35 | 303.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 297.65 | 303.81 | 303.10 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 10:15:00 | 297.70 | 302.59 | 302.61 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 308.00 | 302.25 | 301.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 310.70 | 306.52 | 304.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 11:15:00 | 305.95 | 307.43 | 305.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 11:15:00 | 305.95 | 307.43 | 305.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 11:15:00 | 305.95 | 307.43 | 305.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:00:00 | 305.95 | 307.43 | 305.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 304.60 | 306.86 | 305.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:45:00 | 304.35 | 306.86 | 305.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 307.05 | 306.90 | 305.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:00:00 | 307.05 | 306.90 | 305.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 310.40 | 312.27 | 310.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:45:00 | 309.75 | 312.27 | 310.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 312.20 | 312.26 | 310.66 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 12:15:00 | 307.80 | 310.11 | 310.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 13:15:00 | 306.95 | 309.48 | 309.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 14:15:00 | 312.30 | 310.04 | 310.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 14:15:00 | 312.30 | 310.04 | 310.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 312.30 | 310.04 | 310.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 15:00:00 | 312.30 | 310.04 | 310.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 15:15:00 | 311.10 | 310.26 | 310.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 313.15 | 310.83 | 310.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-21 09:15:00 | 329.45 | 332.78 | 327.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 329.45 | 332.78 | 327.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 329.45 | 332.78 | 327.78 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 319.45 | 324.86 | 325.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 10:15:00 | 318.90 | 323.66 | 324.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 13:15:00 | 325.00 | 323.03 | 324.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 13:15:00 | 325.00 | 323.03 | 324.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 325.00 | 323.03 | 324.27 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 329.90 | 325.28 | 324.97 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 10:15:00 | 321.95 | 324.97 | 325.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 11:15:00 | 319.55 | 323.89 | 324.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 312.80 | 311.09 | 312.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 312.80 | 311.09 | 312.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 312.80 | 311.09 | 312.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 312.80 | 311.09 | 312.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 312.75 | 311.42 | 312.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:30:00 | 313.00 | 311.42 | 312.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 312.45 | 311.62 | 312.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:30:00 | 312.85 | 311.62 | 312.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 310.65 | 311.43 | 312.37 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 13:15:00 | 313.75 | 312.09 | 312.06 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 09:15:00 | 310.85 | 312.01 | 312.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 12:15:00 | 308.00 | 310.67 | 311.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 12:15:00 | 308.85 | 308.68 | 309.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 12:15:00 | 308.85 | 308.68 | 309.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 308.85 | 308.68 | 309.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:30:00 | 309.25 | 308.68 | 309.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 310.05 | 308.95 | 309.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:00:00 | 310.05 | 308.95 | 309.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 306.60 | 308.48 | 309.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 13:45:00 | 306.10 | 307.50 | 308.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 15:00:00 | 306.30 | 307.26 | 308.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 09:15:00 | 313.20 | 308.32 | 308.61 | SL hit (close>static) qty=1.00 sl=310.05 alert=retest2 |

### Cycle 85 — BUY (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 11:15:00 | 310.15 | 309.07 | 308.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 12:15:00 | 311.55 | 309.56 | 309.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 11:15:00 | 311.20 | 312.44 | 311.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 11:15:00 | 311.20 | 312.44 | 311.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 311.20 | 312.44 | 311.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:45:00 | 311.95 | 312.44 | 311.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 310.55 | 312.06 | 311.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:00:00 | 310.55 | 312.06 | 311.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 310.45 | 311.74 | 311.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 14:15:00 | 309.85 | 311.74 | 311.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 311.50 | 311.07 | 310.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 12:15:00 | 315.95 | 311.07 | 310.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 12:15:00 | 309.00 | 311.11 | 311.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 309.00 | 311.11 | 311.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 308.80 | 310.29 | 310.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 14:15:00 | 306.30 | 306.25 | 307.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 15:00:00 | 306.30 | 306.25 | 307.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 306.30 | 306.26 | 307.60 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 12:15:00 | 312.30 | 308.96 | 308.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 09:15:00 | 313.20 | 311.12 | 310.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 12:15:00 | 309.60 | 311.19 | 310.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 12:15:00 | 309.60 | 311.19 | 310.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 309.60 | 311.19 | 310.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:00:00 | 309.60 | 311.19 | 310.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 309.20 | 310.80 | 310.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:00:00 | 309.20 | 310.80 | 310.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 315.85 | 311.81 | 311.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 315.85 | 311.81 | 311.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 310.55 | 312.39 | 311.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 310.55 | 312.39 | 311.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 309.55 | 311.82 | 311.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:45:00 | 309.45 | 311.82 | 311.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 310.60 | 311.37 | 311.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:45:00 | 310.55 | 311.37 | 311.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 13:15:00 | 310.00 | 311.10 | 311.16 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 332.45 | 315.09 | 312.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 337.60 | 322.30 | 316.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 332.05 | 334.52 | 329.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:00:00 | 332.05 | 334.52 | 329.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 366.50 | 369.99 | 367.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:45:00 | 366.50 | 369.99 | 367.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 366.90 | 369.38 | 367.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 14:45:00 | 370.20 | 369.87 | 367.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 09:15:00 | 364.00 | 368.34 | 367.29 | SL hit (close<static) qty=1.00 sl=365.20 alert=retest2 |

### Cycle 90 — SELL (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 13:15:00 | 368.70 | 371.17 | 371.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 11:15:00 | 367.75 | 369.78 | 370.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 12:15:00 | 368.90 | 368.14 | 368.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 12:15:00 | 368.90 | 368.14 | 368.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 368.90 | 368.14 | 368.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:00:00 | 368.90 | 368.14 | 368.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 371.00 | 368.71 | 369.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:00:00 | 371.00 | 368.71 | 369.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 370.45 | 369.06 | 369.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 368.65 | 369.27 | 369.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 376.40 | 370.69 | 370.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 09:15:00 | 376.40 | 370.69 | 370.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 10:15:00 | 385.90 | 373.73 | 371.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 14:15:00 | 390.00 | 390.65 | 384.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 14:45:00 | 388.90 | 390.65 | 384.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 388.25 | 389.68 | 385.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 12:15:00 | 389.65 | 386.37 | 385.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 381.00 | 385.39 | 385.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 381.00 | 385.39 | 385.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 10:15:00 | 379.70 | 384.25 | 384.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 384.50 | 382.95 | 383.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 384.50 | 382.95 | 383.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 384.50 | 382.95 | 383.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:45:00 | 384.95 | 382.95 | 383.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 385.65 | 383.49 | 383.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:45:00 | 384.40 | 383.49 | 383.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 385.20 | 383.83 | 384.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 12:45:00 | 383.65 | 382.87 | 383.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:30:00 | 383.45 | 382.61 | 383.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 09:15:00 | 364.47 | 369.93 | 373.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 09:15:00 | 364.28 | 369.93 | 373.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 14:15:00 | 366.00 | 365.33 | 369.59 | SL hit (close>ema200) qty=0.50 sl=365.33 alert=retest2 |

### Cycle 93 — BUY (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 11:15:00 | 377.75 | 371.46 | 371.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 383.80 | 377.34 | 374.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 14:15:00 | 378.65 | 380.14 | 377.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 14:15:00 | 378.65 | 380.14 | 377.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 378.65 | 380.14 | 377.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 14:45:00 | 378.55 | 380.14 | 377.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 377.40 | 379.59 | 377.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 373.75 | 379.59 | 377.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 380.70 | 379.81 | 377.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:15:00 | 382.00 | 380.00 | 377.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:45:00 | 382.50 | 380.50 | 378.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 10:15:00 | 372.20 | 376.97 | 377.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 10:15:00 | 372.20 | 376.97 | 377.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 369.50 | 375.48 | 376.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 369.85 | 369.36 | 372.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 12:45:00 | 369.55 | 369.36 | 372.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 373.45 | 369.65 | 371.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:30:00 | 375.05 | 369.65 | 371.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 370.15 | 369.75 | 371.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 11:15:00 | 369.60 | 369.75 | 371.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 09:15:00 | 351.12 | 359.37 | 363.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 12:15:00 | 342.65 | 341.29 | 346.75 | SL hit (close>ema200) qty=0.50 sl=341.29 alert=retest2 |

### Cycle 95 — BUY (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 15:15:00 | 349.00 | 346.57 | 346.26 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 12:15:00 | 339.50 | 345.87 | 346.20 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 09:15:00 | 351.80 | 347.13 | 346.65 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 343.60 | 346.47 | 346.57 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 14:15:00 | 355.15 | 348.21 | 347.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 12:15:00 | 361.50 | 352.88 | 350.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 13:15:00 | 367.70 | 369.82 | 365.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 14:15:00 | 364.40 | 368.74 | 365.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 364.40 | 368.74 | 365.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:45:00 | 365.45 | 368.74 | 365.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 364.70 | 367.93 | 365.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 10:00:00 | 368.15 | 367.98 | 365.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 12:15:00 | 383.50 | 385.48 | 385.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 12:15:00 | 383.50 | 385.48 | 385.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 13:15:00 | 380.05 | 384.39 | 385.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 385.35 | 383.99 | 384.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 09:15:00 | 385.35 | 383.99 | 384.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 385.35 | 383.99 | 384.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:00:00 | 385.35 | 383.99 | 384.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 381.60 | 383.51 | 384.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 12:45:00 | 380.80 | 382.94 | 384.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 13:15:00 | 381.10 | 379.74 | 381.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 14:15:00 | 381.00 | 380.05 | 381.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 14:45:00 | 381.10 | 380.29 | 381.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 381.00 | 380.43 | 381.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 381.60 | 380.43 | 381.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 385.10 | 381.36 | 381.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:45:00 | 387.00 | 381.36 | 381.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 381.45 | 381.38 | 381.64 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-05 15:15:00 | 383.05 | 381.93 | 381.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 15:15:00 | 383.05 | 381.93 | 381.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 384.30 | 382.41 | 382.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 11:15:00 | 381.85 | 382.48 | 382.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 11:15:00 | 381.85 | 382.48 | 382.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 381.85 | 382.48 | 382.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:00:00 | 381.85 | 382.48 | 382.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 380.50 | 382.08 | 382.01 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 380.00 | 381.67 | 381.83 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 15:15:00 | 382.65 | 382.01 | 381.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 11:15:00 | 385.90 | 382.86 | 382.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 10:15:00 | 387.10 | 387.48 | 385.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 11:00:00 | 387.10 | 387.48 | 385.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 380.60 | 386.10 | 384.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 380.60 | 386.10 | 384.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 383.50 | 385.58 | 384.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 386.15 | 385.07 | 384.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 10:15:00 | 383.35 | 384.43 | 384.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 10:15:00 | 383.35 | 384.43 | 384.44 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 11:15:00 | 387.15 | 384.98 | 384.69 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 15:15:00 | 382.90 | 384.61 | 384.64 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 10:15:00 | 389.40 | 385.30 | 384.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 11:15:00 | 390.00 | 386.24 | 385.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 15:15:00 | 386.00 | 387.68 | 386.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 15:15:00 | 386.00 | 387.68 | 386.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 386.00 | 387.68 | 386.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 384.00 | 387.68 | 386.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 393.20 | 388.78 | 387.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 11:45:00 | 396.10 | 390.08 | 388.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-22 09:15:00 | 435.71 | 423.92 | 416.30 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 12:15:00 | 453.35 | 458.60 | 459.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-03 09:15:00 | 450.75 | 454.92 | 457.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 11:15:00 | 456.35 | 454.89 | 456.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 11:15:00 | 456.35 | 454.89 | 456.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 456.35 | 454.89 | 456.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 12:00:00 | 456.35 | 454.89 | 456.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 462.30 | 456.37 | 457.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 12:45:00 | 462.25 | 456.37 | 457.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 458.30 | 456.76 | 457.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:30:00 | 462.60 | 456.76 | 457.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 463.15 | 458.04 | 457.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 09:15:00 | 464.60 | 460.14 | 458.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 12:15:00 | 457.70 | 460.38 | 459.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 12:15:00 | 457.70 | 460.38 | 459.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 457.70 | 460.38 | 459.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:45:00 | 458.25 | 460.38 | 459.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 457.15 | 459.74 | 459.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:45:00 | 455.60 | 459.74 | 459.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 14:15:00 | 450.90 | 457.97 | 458.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 448.95 | 451.61 | 453.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 15:15:00 | 448.50 | 448.48 | 451.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 09:15:00 | 475.25 | 448.48 | 451.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 460.75 | 450.93 | 452.06 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 461.25 | 453.00 | 452.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 11:15:00 | 472.20 | 456.84 | 454.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 13:15:00 | 470.85 | 474.81 | 468.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-10 14:00:00 | 470.85 | 474.81 | 468.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 469.00 | 473.65 | 468.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:00:00 | 469.00 | 473.65 | 468.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 465.00 | 471.15 | 467.93 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 459.50 | 465.62 | 465.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 13:15:00 | 458.70 | 464.24 | 465.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 12:15:00 | 462.35 | 460.86 | 462.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-14 12:45:00 | 461.40 | 460.86 | 462.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 470.90 | 462.87 | 463.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:00:00 | 470.90 | 462.87 | 463.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 14:15:00 | 480.45 | 466.39 | 464.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 15:15:00 | 485.00 | 470.11 | 466.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 11:15:00 | 470.50 | 471.21 | 468.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-15 12:00:00 | 470.50 | 471.21 | 468.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 469.00 | 470.87 | 468.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:45:00 | 469.00 | 470.87 | 468.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 468.05 | 470.30 | 468.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 472.25 | 470.30 | 468.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 473.05 | 470.85 | 469.13 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 10:15:00 | 465.70 | 468.60 | 468.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 11:15:00 | 464.40 | 467.76 | 468.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 462.50 | 456.92 | 459.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 462.50 | 456.92 | 459.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 462.50 | 456.92 | 459.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:00:00 | 462.50 | 456.92 | 459.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 455.00 | 456.54 | 459.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 11:00:00 | 454.45 | 455.92 | 458.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 10:15:00 | 453.75 | 455.44 | 457.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 11:00:00 | 454.70 | 455.30 | 456.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 461.50 | 454.04 | 453.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 09:15:00 | 461.50 | 454.04 | 453.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 10:15:00 | 467.80 | 456.79 | 454.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 10:15:00 | 461.05 | 462.37 | 459.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 11:00:00 | 461.05 | 462.37 | 459.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 460.20 | 461.93 | 459.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:45:00 | 460.25 | 461.93 | 459.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 462.90 | 462.94 | 460.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:45:00 | 462.40 | 462.94 | 460.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 459.80 | 462.13 | 460.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 459.80 | 462.13 | 460.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 461.20 | 461.94 | 460.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 13:00:00 | 463.50 | 462.22 | 461.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 463.00 | 461.54 | 460.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 12:00:00 | 462.50 | 461.86 | 461.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 464.55 | 463.59 | 462.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 460.15 | 462.90 | 462.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 460.15 | 462.90 | 462.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 465.90 | 463.50 | 462.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 11:30:00 | 468.00 | 464.84 | 463.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 458.60 | 467.75 | 468.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 458.60 | 467.75 | 468.52 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 13:15:00 | 476.25 | 467.36 | 466.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 489.95 | 472.86 | 469.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 13:15:00 | 478.75 | 478.97 | 474.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 13:45:00 | 479.95 | 478.97 | 474.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 469.95 | 478.16 | 475.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:45:00 | 467.15 | 478.16 | 475.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 462.80 | 475.08 | 473.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 462.80 | 475.08 | 473.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 11:15:00 | 462.60 | 472.59 | 472.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 14:15:00 | 448.95 | 461.34 | 465.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 09:15:00 | 459.90 | 459.40 | 463.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-14 09:45:00 | 460.70 | 459.40 | 463.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 462.85 | 460.60 | 463.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:30:00 | 464.75 | 460.60 | 463.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 463.40 | 461.16 | 463.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:00:00 | 463.40 | 461.16 | 463.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 461.10 | 461.15 | 463.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:45:00 | 465.00 | 461.15 | 463.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 461.00 | 460.65 | 462.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:30:00 | 464.80 | 460.65 | 462.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 460.60 | 460.64 | 462.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:45:00 | 462.25 | 460.64 | 462.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 459.00 | 455.87 | 458.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:00:00 | 459.00 | 455.87 | 458.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 454.90 | 455.68 | 457.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:30:00 | 458.55 | 455.68 | 457.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 456.95 | 454.94 | 456.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:30:00 | 456.90 | 454.94 | 456.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 457.10 | 455.37 | 456.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:45:00 | 457.20 | 455.37 | 456.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 451.40 | 454.58 | 456.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 13:15:00 | 449.25 | 453.96 | 455.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 09:30:00 | 448.60 | 451.28 | 453.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 10:45:00 | 449.25 | 450.83 | 453.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 09:15:00 | 426.79 | 432.92 | 439.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 09:15:00 | 426.17 | 432.92 | 439.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 09:15:00 | 426.79 | 432.92 | 439.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 421.45 | 421.02 | 428.84 | SL hit (close>ema200) qty=0.50 sl=421.02 alert=retest2 |

### Cycle 119 — BUY (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 10:15:00 | 425.65 | 422.98 | 422.84 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 422.75 | 423.60 | 423.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 12:15:00 | 419.55 | 422.13 | 422.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 14:15:00 | 426.85 | 422.79 | 423.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 14:15:00 | 426.85 | 422.79 | 423.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 426.85 | 422.79 | 423.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 426.85 | 422.79 | 423.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 423.30 | 422.89 | 423.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 422.30 | 422.89 | 423.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 15:15:00 | 423.95 | 422.52 | 422.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 15:15:00 | 423.95 | 422.52 | 422.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 09:15:00 | 442.90 | 426.59 | 424.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 10:15:00 | 431.55 | 433.08 | 430.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-05 10:30:00 | 432.75 | 433.08 | 430.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 432.10 | 433.58 | 431.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 432.10 | 433.58 | 431.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 432.00 | 433.26 | 431.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 430.95 | 433.26 | 431.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 424.60 | 431.53 | 430.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 424.60 | 431.53 | 430.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 423.05 | 429.83 | 430.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 421.10 | 428.09 | 429.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 425.65 | 421.41 | 424.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 11:15:00 | 425.65 | 421.41 | 424.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 425.65 | 421.41 | 424.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 425.65 | 421.41 | 424.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 427.30 | 422.59 | 424.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 427.30 | 422.59 | 424.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 429.05 | 423.88 | 424.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:00:00 | 429.05 | 423.88 | 424.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 435.40 | 426.18 | 425.91 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 11:15:00 | 424.95 | 427.04 | 427.12 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 428.35 | 427.30 | 427.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 13:15:00 | 429.00 | 427.64 | 427.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 453.30 | 454.81 | 446.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:45:00 | 452.50 | 454.81 | 446.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 447.45 | 453.60 | 452.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:00:00 | 447.45 | 453.60 | 452.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 448.55 | 452.59 | 451.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 11:45:00 | 451.00 | 452.71 | 452.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 14:15:00 | 447.75 | 451.44 | 451.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 14:15:00 | 447.75 | 451.44 | 451.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 09:15:00 | 444.30 | 449.37 | 450.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 452.15 | 446.58 | 448.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 14:15:00 | 452.15 | 446.58 | 448.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 452.15 | 446.58 | 448.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 452.15 | 446.58 | 448.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 447.35 | 446.74 | 448.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 455.40 | 446.74 | 448.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 454.55 | 448.30 | 448.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 11:15:00 | 450.00 | 448.88 | 448.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 13:15:00 | 450.50 | 449.33 | 449.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 13:15:00 | 450.50 | 449.33 | 449.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 451.40 | 449.74 | 449.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 10:15:00 | 456.90 | 458.48 | 455.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 11:00:00 | 456.90 | 458.48 | 455.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 456.60 | 458.10 | 455.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:00:00 | 456.60 | 458.10 | 455.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 454.10 | 456.97 | 455.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 454.10 | 456.97 | 455.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 454.95 | 456.57 | 455.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:30:00 | 454.25 | 456.57 | 455.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 452.05 | 455.67 | 455.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 452.45 | 455.67 | 455.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 448.20 | 454.17 | 454.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 443.25 | 447.96 | 450.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 10:15:00 | 450.00 | 447.02 | 449.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 10:15:00 | 450.00 | 447.02 | 449.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 450.00 | 447.02 | 449.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:00:00 | 450.00 | 447.02 | 449.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 449.40 | 447.50 | 449.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:45:00 | 452.30 | 447.50 | 449.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 449.70 | 447.94 | 449.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 12:45:00 | 449.40 | 447.94 | 449.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 450.85 | 448.52 | 449.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 450.85 | 448.52 | 449.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 451.70 | 449.16 | 449.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 451.70 | 449.16 | 449.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 450.00 | 449.32 | 449.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 443.15 | 449.32 | 449.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 458.50 | 450.54 | 449.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 458.50 | 450.54 | 449.71 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 11:15:00 | 445.20 | 450.35 | 450.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 12:15:00 | 444.15 | 449.11 | 450.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 446.60 | 446.51 | 448.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 09:15:00 | 446.60 | 446.51 | 448.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 446.60 | 446.51 | 448.36 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 14:15:00 | 451.35 | 449.35 | 449.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 454.35 | 450.66 | 449.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 14:15:00 | 456.90 | 457.94 | 455.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 14:15:00 | 456.90 | 457.94 | 455.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 456.90 | 457.94 | 455.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:30:00 | 457.00 | 457.94 | 455.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 455.90 | 457.53 | 455.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 452.30 | 457.53 | 455.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 448.85 | 455.79 | 455.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:30:00 | 446.85 | 455.79 | 455.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 449.90 | 454.62 | 454.56 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 11:15:00 | 450.70 | 453.83 | 454.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 439.65 | 448.41 | 450.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 438.85 | 438.67 | 441.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 10:15:00 | 438.70 | 438.67 | 441.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 438.70 | 438.67 | 441.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:30:00 | 439.50 | 438.67 | 441.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 437.55 | 434.62 | 437.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:45:00 | 438.30 | 434.62 | 437.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 437.00 | 435.09 | 437.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:45:00 | 438.25 | 435.09 | 437.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 437.40 | 435.55 | 437.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:30:00 | 435.65 | 436.18 | 437.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:30:00 | 435.75 | 435.82 | 437.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 413.87 | 420.27 | 424.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 413.96 | 420.27 | 424.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 15:15:00 | 417.20 | 416.37 | 420.66 | SL hit (close>ema200) qty=0.50 sl=416.37 alert=retest2 |

### Cycle 133 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 415.00 | 413.77 | 413.66 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 411.80 | 413.38 | 413.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 10:15:00 | 410.00 | 412.70 | 413.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 410.85 | 410.19 | 411.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 410.85 | 410.19 | 411.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 410.85 | 410.19 | 411.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:30:00 | 410.00 | 410.19 | 411.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 408.75 | 409.91 | 411.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:30:00 | 412.80 | 409.91 | 411.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 409.95 | 409.43 | 410.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:30:00 | 410.85 | 409.43 | 410.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 409.00 | 409.27 | 410.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 404.75 | 409.27 | 410.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 11:45:00 | 407.45 | 408.84 | 409.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 13:15:00 | 413.60 | 409.85 | 410.13 | SL hit (close>static) qty=1.00 sl=411.00 alert=retest2 |

### Cycle 135 — BUY (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 15:15:00 | 423.00 | 412.18 | 411.12 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 397.85 | 409.02 | 409.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 15:15:00 | 394.65 | 400.72 | 404.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 406.00 | 401.78 | 405.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 406.00 | 401.78 | 405.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 406.00 | 401.78 | 405.04 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2026-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 13:15:00 | 410.65 | 407.00 | 406.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 417.55 | 409.77 | 408.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 415.80 | 417.05 | 413.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:45:00 | 414.80 | 417.05 | 413.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 413.40 | 416.32 | 413.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 413.40 | 416.32 | 413.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 416.10 | 416.27 | 413.63 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 13:15:00 | 412.55 | 413.31 | 413.40 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 417.50 | 414.15 | 413.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 419.75 | 415.42 | 414.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 431.55 | 432.82 | 427.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 14:00:00 | 431.55 | 432.82 | 427.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 427.70 | 431.36 | 427.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 423.60 | 429.83 | 427.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 424.00 | 428.66 | 427.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:15:00 | 422.45 | 428.66 | 427.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2026-02-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 13:15:00 | 422.10 | 425.56 | 425.97 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 15:15:00 | 426.50 | 425.47 | 425.42 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 413.60 | 423.10 | 424.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 408.45 | 416.41 | 420.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 10:15:00 | 422.40 | 416.31 | 419.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 10:15:00 | 422.40 | 416.31 | 419.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 422.40 | 416.31 | 419.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:00:00 | 422.40 | 416.31 | 419.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 422.70 | 417.58 | 419.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:45:00 | 424.45 | 417.58 | 419.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 418.00 | 418.01 | 419.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:30:00 | 419.70 | 418.01 | 419.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 426.30 | 418.87 | 419.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 426.30 | 418.87 | 419.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 423.70 | 419.83 | 419.73 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 15:15:00 | 415.20 | 419.40 | 419.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 10:15:00 | 415.10 | 417.84 | 418.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 11:15:00 | 406.60 | 405.74 | 408.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 12:00:00 | 406.60 | 405.74 | 408.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 406.60 | 405.91 | 408.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:45:00 | 407.90 | 405.91 | 408.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 405.60 | 403.51 | 405.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:00:00 | 405.60 | 403.51 | 405.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 408.15 | 404.44 | 405.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 411.45 | 404.44 | 405.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 435.75 | 410.70 | 408.47 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 15:15:00 | 417.50 | 420.03 | 420.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 415.70 | 419.16 | 419.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 10:15:00 | 421.65 | 419.66 | 419.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 10:15:00 | 421.65 | 419.66 | 419.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 10:15:00 | 421.65 | 419.66 | 419.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 10:30:00 | 420.45 | 419.66 | 419.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 11:15:00 | 419.20 | 419.57 | 419.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 12:30:00 | 418.05 | 419.13 | 419.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 15:15:00 | 421.60 | 419.85 | 419.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2026-03-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 15:15:00 | 421.60 | 419.85 | 419.81 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 416.25 | 419.13 | 419.49 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 13:15:00 | 420.25 | 418.34 | 418.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 14:15:00 | 422.25 | 419.12 | 418.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 10:15:00 | 419.10 | 420.31 | 419.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 10:15:00 | 419.10 | 420.31 | 419.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 419.10 | 420.31 | 419.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:45:00 | 420.55 | 420.31 | 419.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 419.35 | 420.12 | 419.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:15:00 | 420.30 | 420.12 | 419.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 420.40 | 420.17 | 419.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 13:15:00 | 421.10 | 420.17 | 419.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 15:15:00 | 418.00 | 419.51 | 419.27 | SL hit (close<static) qty=1.00 sl=419.00 alert=retest2 |

### Cycle 150 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 405.00 | 416.61 | 417.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 12:15:00 | 401.55 | 410.30 | 414.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 416.50 | 409.78 | 412.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 416.50 | 409.78 | 412.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 416.50 | 409.78 | 412.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:00:00 | 416.50 | 409.78 | 412.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 416.00 | 411.02 | 412.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 11:15:00 | 417.00 | 411.02 | 412.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 419.70 | 414.58 | 414.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 420.25 | 415.71 | 414.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 15:15:00 | 422.20 | 423.34 | 420.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-12 09:15:00 | 419.95 | 423.34 | 420.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 418.40 | 422.35 | 420.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:30:00 | 425.20 | 423.79 | 421.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 413.45 | 420.46 | 420.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 413.45 | 420.46 | 420.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 412.00 | 418.77 | 419.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 13:15:00 | 399.00 | 398.52 | 403.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 14:00:00 | 399.00 | 398.52 | 403.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 405.45 | 400.11 | 402.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 405.60 | 400.11 | 402.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 408.30 | 401.75 | 403.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 408.30 | 401.75 | 403.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 409.90 | 404.55 | 404.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 412.40 | 406.12 | 405.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 401.15 | 406.86 | 405.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 401.15 | 406.86 | 405.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 401.15 | 406.86 | 405.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 401.15 | 406.86 | 405.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 404.00 | 406.29 | 405.71 | EMA400 retest candle locked (from upside) |

### Cycle 154 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 402.55 | 405.19 | 405.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 400.35 | 404.22 | 404.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 405.25 | 402.53 | 403.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 405.25 | 402.53 | 403.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 405.25 | 402.53 | 403.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 404.10 | 402.53 | 403.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 405.15 | 403.05 | 403.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 404.55 | 403.05 | 403.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 404.40 | 403.32 | 403.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 402.20 | 403.32 | 403.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:00:00 | 402.85 | 403.23 | 403.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 392.85 | 402.56 | 403.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 382.09 | 396.45 | 400.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 382.71 | 396.45 | 400.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 393.30 | 391.78 | 395.94 | SL hit (close>ema200) qty=0.50 sl=391.78 alert=retest2 |

### Cycle 155 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 406.15 | 397.10 | 396.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 411.80 | 401.39 | 398.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 408.35 | 408.85 | 404.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:15:00 | 406.15 | 408.85 | 404.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 404.15 | 407.91 | 404.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 404.15 | 407.91 | 404.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 402.65 | 406.86 | 404.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 402.65 | 406.86 | 404.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 405.00 | 406.49 | 404.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 15:00:00 | 407.85 | 406.52 | 404.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 401.05 | 404.98 | 404.20 | SL hit (close<static) qty=1.00 sl=402.20 alert=retest2 |

### Cycle 156 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 394.00 | 402.79 | 403.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 392.30 | 398.65 | 401.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 406.70 | 399.08 | 400.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 406.70 | 399.08 | 400.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 406.70 | 399.08 | 400.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 403.55 | 399.08 | 400.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 405.00 | 400.26 | 401.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:30:00 | 407.60 | 400.26 | 401.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 407.50 | 402.79 | 402.18 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 391.45 | 402.12 | 402.25 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 405.40 | 402.30 | 402.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 10:15:00 | 409.75 | 404.28 | 403.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 404.95 | 407.77 | 405.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 10:15:00 | 404.95 | 407.77 | 405.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 404.95 | 407.77 | 405.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 11:00:00 | 404.95 | 407.77 | 405.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 405.00 | 407.21 | 405.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 12:00:00 | 405.00 | 407.21 | 405.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 407.45 | 407.26 | 405.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 413.20 | 406.72 | 406.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-20 12:15:00 | 454.52 | 448.71 | 446.00 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 15:15:00 | 442.10 | 445.28 | 445.63 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 10:15:00 | 448.45 | 446.30 | 446.06 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 15:15:00 | 445.80 | 446.00 | 446.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 10:15:00 | 441.00 | 444.65 | 445.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 09:15:00 | 443.00 | 442.30 | 443.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 09:15:00 | 443.00 | 442.30 | 443.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 443.00 | 442.30 | 443.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 10:00:00 | 443.00 | 442.30 | 443.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 449.05 | 443.65 | 444.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:00:00 | 449.05 | 443.65 | 444.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — BUY (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 11:15:00 | 453.40 | 445.60 | 444.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 463.45 | 448.94 | 446.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 11:15:00 | 459.30 | 459.61 | 455.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 12:00:00 | 459.30 | 459.61 | 455.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 453.90 | 458.35 | 455.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:00:00 | 453.90 | 458.35 | 455.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 458.25 | 458.33 | 455.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:30:00 | 463.40 | 458.86 | 456.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 10:00:00 | 462.85 | 458.86 | 456.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 12:15:00 | 464.05 | 460.14 | 457.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 15:15:00 | 463.10 | 461.08 | 458.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 462.90 | 461.77 | 459.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 13:30:00 | 470.70 | 461.84 | 460.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 14:15:00 | 453.70 | 460.21 | 459.60 | SL hit (close<static) qty=1.00 sl=459.15 alert=retest2 |

### Cycle 164 — SELL (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 15:15:00 | 448.00 | 457.77 | 458.55 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 465.65 | 459.72 | 459.26 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 453.15 | 458.54 | 459.09 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 463.50 | 458.66 | 458.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 11:15:00 | 469.00 | 460.73 | 459.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 15:15:00 | 465.00 | 465.43 | 462.49 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:15:00 | 473.50 | 465.43 | 462.49 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 13:15:00 | 472.80 | 467.88 | 464.67 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 14:00:00 | 474.50 | 469.20 | 465.57 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 473.40 | 476.94 | 472.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 473.40 | 476.94 | 472.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 472.00 | 475.95 | 472.76 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-05-08 15:15:00 | 472.00 | 475.95 | 472.76 | SL hit (close<ema400) qty=1.00 sl=472.76 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-22 12:30:00 | 354.60 | 2024-05-27 10:15:00 | 350.05 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-05-22 15:00:00 | 352.80 | 2024-05-27 10:15:00 | 350.05 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-06-27 11:45:00 | 386.95 | 2024-07-01 12:15:00 | 395.85 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2024-06-28 10:15:00 | 385.00 | 2024-07-01 12:15:00 | 395.85 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2024-06-28 15:15:00 | 385.00 | 2024-07-01 12:15:00 | 395.85 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest1 | 2024-07-04 09:15:00 | 408.40 | 2024-07-08 10:15:00 | 403.40 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest1 | 2024-07-04 12:15:00 | 408.30 | 2024-07-08 10:15:00 | 403.40 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-07-23 12:15:00 | 356.65 | 2024-07-26 10:15:00 | 370.95 | STOP_HIT | 1.00 | -4.01% |
| SELL | retest2 | 2024-07-23 12:45:00 | 360.10 | 2024-07-26 13:15:00 | 369.70 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2024-07-23 13:30:00 | 361.40 | 2024-07-26 13:15:00 | 369.70 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2024-07-25 09:15:00 | 361.65 | 2024-07-26 13:15:00 | 369.70 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2024-07-26 09:45:00 | 361.60 | 2024-07-26 13:15:00 | 369.70 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-07-29 15:15:00 | 371.15 | 2024-08-02 11:15:00 | 369.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-07-30 10:00:00 | 371.20 | 2024-08-02 11:15:00 | 369.00 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2024-08-08 13:00:00 | 346.50 | 2024-08-13 15:15:00 | 329.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-09 15:15:00 | 345.45 | 2024-08-13 15:15:00 | 328.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-08 13:00:00 | 346.50 | 2024-08-14 11:15:00 | 338.00 | STOP_HIT | 0.50 | 2.45% |
| SELL | retest2 | 2024-08-09 15:15:00 | 345.45 | 2024-08-14 11:15:00 | 338.00 | STOP_HIT | 0.50 | 2.16% |
| SELL | retest2 | 2024-08-30 15:00:00 | 331.70 | 2024-09-03 10:15:00 | 339.95 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2024-09-06 09:15:00 | 340.65 | 2024-09-06 10:15:00 | 336.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-09-06 10:15:00 | 340.65 | 2024-09-06 10:15:00 | 336.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-09-06 12:15:00 | 344.65 | 2024-09-12 12:15:00 | 353.60 | STOP_HIT | 1.00 | 2.60% |
| BUY | retest2 | 2024-09-27 09:15:00 | 353.60 | 2024-09-27 10:15:00 | 350.80 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-10-09 14:15:00 | 343.70 | 2024-10-10 09:15:00 | 353.85 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2024-10-21 15:15:00 | 414.90 | 2024-10-22 14:15:00 | 394.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 15:15:00 | 414.90 | 2024-10-24 09:15:00 | 402.90 | STOP_HIT | 0.50 | 2.89% |
| SELL | retest2 | 2024-11-14 14:15:00 | 379.50 | 2024-11-19 09:15:00 | 398.00 | STOP_HIT | 1.00 | -4.87% |
| SELL | retest2 | 2024-11-14 15:15:00 | 378.20 | 2024-11-19 09:15:00 | 398.00 | STOP_HIT | 1.00 | -5.24% |
| SELL | retest2 | 2024-11-18 10:15:00 | 377.35 | 2024-11-19 09:15:00 | 398.00 | STOP_HIT | 1.00 | -5.47% |
| SELL | retest2 | 2024-11-28 10:45:00 | 386.65 | 2024-12-03 11:15:00 | 384.35 | STOP_HIT | 1.00 | 0.59% |
| BUY | retest2 | 2024-12-10 11:15:00 | 421.05 | 2024-12-12 09:15:00 | 402.95 | STOP_HIT | 1.00 | -4.30% |
| BUY | retest2 | 2024-12-10 14:45:00 | 413.35 | 2024-12-12 09:15:00 | 402.95 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2024-12-20 14:30:00 | 377.50 | 2024-12-27 09:15:00 | 387.00 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-01-08 11:45:00 | 377.20 | 2025-01-08 12:15:00 | 373.35 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-01-23 13:30:00 | 343.80 | 2025-01-27 09:15:00 | 326.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 13:30:00 | 343.80 | 2025-01-28 11:15:00 | 327.00 | STOP_HIT | 0.50 | 4.89% |
| SELL | retest2 | 2025-02-13 12:30:00 | 303.50 | 2025-02-17 15:15:00 | 305.00 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-02-13 13:00:00 | 303.65 | 2025-02-17 15:15:00 | 305.00 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-02-14 15:00:00 | 302.35 | 2025-02-17 15:15:00 | 305.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-02-24 10:30:00 | 305.85 | 2025-02-25 13:15:00 | 297.85 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-02-25 09:30:00 | 304.95 | 2025-02-25 13:15:00 | 297.85 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-03-03 11:00:00 | 287.55 | 2025-03-03 15:15:00 | 303.00 | STOP_HIT | 1.00 | -5.37% |
| BUY | retest2 | 2025-03-26 10:15:00 | 342.50 | 2025-03-28 12:15:00 | 338.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-03-27 10:15:00 | 343.15 | 2025-03-28 12:15:00 | 338.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-03-27 14:45:00 | 343.15 | 2025-03-28 12:15:00 | 338.00 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-04-04 09:15:00 | 324.05 | 2025-04-07 09:15:00 | 307.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 09:15:00 | 324.05 | 2025-04-08 09:15:00 | 304.35 | STOP_HIT | 0.50 | 6.08% |
| BUY | retest2 | 2025-04-21 13:00:00 | 314.65 | 2025-04-22 14:15:00 | 312.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-04-22 09:15:00 | 317.40 | 2025-04-22 14:15:00 | 312.00 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-04-22 12:15:00 | 314.30 | 2025-04-22 14:15:00 | 312.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-04-30 09:15:00 | 287.05 | 2025-05-02 14:15:00 | 303.50 | STOP_HIT | 1.00 | -5.73% |
| SELL | retest2 | 2025-04-30 10:00:00 | 289.10 | 2025-05-02 14:15:00 | 303.50 | STOP_HIT | 1.00 | -4.98% |
| SELL | retest2 | 2025-05-02 09:15:00 | 288.05 | 2025-05-02 14:15:00 | 303.50 | STOP_HIT | 1.00 | -5.36% |
| SELL | retest2 | 2025-06-06 13:45:00 | 306.10 | 2025-06-09 09:15:00 | 313.20 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-06-06 15:00:00 | 306.30 | 2025-06-09 09:15:00 | 313.20 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-06-11 12:15:00 | 315.95 | 2025-06-12 12:15:00 | 309.00 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-07-04 14:45:00 | 370.20 | 2025-07-07 09:15:00 | 364.00 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-07-07 12:15:00 | 372.65 | 2025-07-09 13:15:00 | 368.70 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-07-07 13:00:00 | 369.55 | 2025-07-09 13:15:00 | 368.70 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-07-09 13:15:00 | 368.85 | 2025-07-09 13:15:00 | 368.70 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-07-14 09:15:00 | 368.65 | 2025-07-14 09:15:00 | 376.40 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-07-17 12:15:00 | 389.65 | 2025-07-18 09:15:00 | 381.00 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-07-21 12:45:00 | 383.65 | 2025-07-28 09:15:00 | 364.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 12:30:00 | 383.45 | 2025-07-28 09:15:00 | 364.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 12:45:00 | 383.65 | 2025-07-28 14:15:00 | 366.00 | STOP_HIT | 0.50 | 4.60% |
| SELL | retest2 | 2025-07-22 12:30:00 | 383.45 | 2025-07-28 14:15:00 | 366.00 | STOP_HIT | 0.50 | 4.55% |
| BUY | retest2 | 2025-07-31 11:15:00 | 382.00 | 2025-08-01 10:15:00 | 372.20 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-07-31 11:45:00 | 382.50 | 2025-08-01 10:15:00 | 372.20 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-08-05 11:15:00 | 369.60 | 2025-08-07 09:15:00 | 351.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 11:15:00 | 369.60 | 2025-08-11 12:15:00 | 342.65 | STOP_HIT | 0.50 | 7.29% |
| BUY | retest2 | 2025-08-21 10:00:00 | 368.15 | 2025-09-02 12:15:00 | 383.50 | STOP_HIT | 1.00 | 4.17% |
| SELL | retest2 | 2025-09-03 12:45:00 | 380.80 | 2025-09-05 15:15:00 | 383.05 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-09-04 13:15:00 | 381.10 | 2025-09-05 15:15:00 | 383.05 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-09-04 14:15:00 | 381.00 | 2025-09-05 15:15:00 | 383.05 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-09-04 14:45:00 | 381.10 | 2025-09-05 15:15:00 | 383.05 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-09-11 09:15:00 | 386.15 | 2025-09-11 10:15:00 | 383.35 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-09-16 11:45:00 | 396.10 | 2025-09-22 09:15:00 | 435.71 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-23 11:00:00 | 454.45 | 2025-10-28 09:15:00 | 461.50 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-10-24 10:15:00 | 453.75 | 2025-10-28 09:15:00 | 461.50 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-10-24 11:00:00 | 454.70 | 2025-10-28 09:15:00 | 461.50 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-10-30 13:00:00 | 463.50 | 2025-11-06 11:15:00 | 458.60 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-10-31 09:15:00 | 463.00 | 2025-11-06 11:15:00 | 458.60 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-10-31 12:00:00 | 462.50 | 2025-11-06 11:15:00 | 458.60 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-11-03 09:15:00 | 464.55 | 2025-11-06 11:15:00 | 458.60 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-11-03 11:30:00 | 468.00 | 2025-11-06 11:15:00 | 458.60 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-11-19 13:15:00 | 449.25 | 2025-11-24 09:15:00 | 426.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 09:30:00 | 448.60 | 2025-11-24 09:15:00 | 426.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 10:45:00 | 449.25 | 2025-11-24 09:15:00 | 426.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-19 13:15:00 | 449.25 | 2025-11-25 09:15:00 | 421.45 | STOP_HIT | 0.50 | 6.19% |
| SELL | retest2 | 2025-11-20 09:30:00 | 448.60 | 2025-11-25 09:15:00 | 421.45 | STOP_HIT | 0.50 | 6.05% |
| SELL | retest2 | 2025-11-20 10:45:00 | 449.25 | 2025-11-25 09:15:00 | 421.45 | STOP_HIT | 0.50 | 6.19% |
| SELL | retest2 | 2025-12-03 09:15:00 | 422.30 | 2025-12-03 15:15:00 | 423.95 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-12-18 11:45:00 | 451.00 | 2025-12-18 14:15:00 | 447.75 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-12-22 11:15:00 | 450.00 | 2025-12-22 13:15:00 | 450.50 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-12-30 09:15:00 | 443.15 | 2025-12-31 09:15:00 | 458.50 | STOP_HIT | 1.00 | -3.46% |
| SELL | retest2 | 2026-01-14 13:30:00 | 435.65 | 2026-01-21 10:15:00 | 413.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 14:30:00 | 435.75 | 2026-01-21 10:15:00 | 413.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 13:30:00 | 435.65 | 2026-01-21 15:15:00 | 417.20 | STOP_HIT | 0.50 | 4.24% |
| SELL | retest2 | 2026-01-14 14:30:00 | 435.75 | 2026-01-21 15:15:00 | 417.20 | STOP_HIT | 0.50 | 4.26% |
| SELL | retest2 | 2026-02-01 09:15:00 | 404.75 | 2026-02-01 13:15:00 | 413.60 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2026-02-01 11:45:00 | 407.45 | 2026-02-01 13:15:00 | 413.60 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2026-02-01 15:00:00 | 408.00 | 2026-02-01 15:15:00 | 423.00 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2026-03-02 12:30:00 | 418.05 | 2026-03-02 15:15:00 | 421.60 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2026-03-06 13:15:00 | 421.10 | 2026-03-06 15:15:00 | 418.00 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2026-03-12 11:30:00 | 425.20 | 2026-03-13 09:15:00 | 413.45 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2026-03-20 12:15:00 | 402.20 | 2026-03-23 10:15:00 | 382.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 13:00:00 | 402.85 | 2026-03-23 10:15:00 | 382.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 402.20 | 2026-03-24 09:15:00 | 393.30 | STOP_HIT | 0.50 | 2.21% |
| SELL | retest2 | 2026-03-20 13:00:00 | 402.85 | 2026-03-24 09:15:00 | 393.30 | STOP_HIT | 0.50 | 2.37% |
| SELL | retest2 | 2026-03-23 09:15:00 | 392.85 | 2026-03-25 09:15:00 | 406.15 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2026-03-27 15:00:00 | 407.85 | 2026-03-30 09:15:00 | 401.05 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2026-04-08 09:15:00 | 413.20 | 2026-04-20 12:15:00 | 454.52 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-29 09:30:00 | 463.40 | 2026-04-30 14:15:00 | 453.70 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2026-04-29 10:00:00 | 462.85 | 2026-04-30 15:15:00 | 448.00 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2026-04-29 12:15:00 | 464.05 | 2026-04-30 15:15:00 | 448.00 | STOP_HIT | 1.00 | -3.46% |
| BUY | retest2 | 2026-04-29 15:15:00 | 463.10 | 2026-04-30 15:15:00 | 448.00 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2026-04-30 13:30:00 | 470.70 | 2026-04-30 15:15:00 | 448.00 | STOP_HIT | 1.00 | -4.82% |
| BUY | retest1 | 2026-05-07 09:15:00 | 473.50 | 2026-05-08 15:15:00 | 472.00 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-05-07 13:15:00 | 472.80 | 2026-05-08 15:15:00 | 472.00 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2026-05-07 14:00:00 | 474.50 | 2026-05-08 15:15:00 | 472.00 | STOP_HIT | 1.00 | -0.53% |
