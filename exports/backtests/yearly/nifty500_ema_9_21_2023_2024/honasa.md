# Honasa Consumer Ltd. (HONASA)

## Backtest Summary

- **Window:** 2023-11-07 09:15:00 → 2026-05-08 15:15:00 (4316 bars)
- **Last close:** 358.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 191 |
| ALERT1 | 115 |
| ALERT2 | 114 |
| ALERT2_SKIP | 63 |
| ALERT3 | 287 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 131 |
| PARTIAL | 24 |
| TARGET_HIT | 10 |
| STOP_HIT | 123 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 157 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 69 / 88
- **Target hits / Stop hits / Partials:** 10 / 123 / 24
- **Avg / median % per leg:** 1.00% / -0.60%
- **Sum % (uncompounded):** 156.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 53 | 19 | 35.8% | 9 | 43 | 1 | 1.00% | 53.1% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 1 | 1 | 1.93% | 3.9% |
| BUY @ 3rd Alert (retest2) | 51 | 18 | 35.3% | 9 | 42 | 0 | 0.96% | 49.2% |
| SELL (all) | 104 | 50 | 48.1% | 1 | 80 | 23 | 0.99% | 103.3% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.95% | -0.9% |
| SELL @ 3rd Alert (retest2) | 103 | 50 | 48.5% | 1 | 79 | 23 | 1.01% | 104.2% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.97% | 2.9% |
| retest2 (combined) | 154 | 68 | 44.2% | 10 | 121 | 23 | 1.00% | 153.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 09:15:00 | 321.80 | 316.58 | 315.95 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-11-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-15 10:15:00 | 310.80 | 315.42 | 315.49 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-11-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-16 09:15:00 | 319.65 | 315.27 | 315.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 10:15:00 | 323.35 | 316.88 | 315.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 14:15:00 | 352.55 | 353.02 | 341.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-20 15:00:00 | 352.55 | 353.02 | 341.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 15:15:00 | 347.05 | 351.83 | 341.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 09:30:00 | 353.20 | 352.15 | 342.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 10:00:00 | 353.45 | 352.15 | 342.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-22 11:00:00 | 356.50 | 361.27 | 353.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-22 14:15:00 | 354.60 | 357.56 | 353.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 14:15:00 | 352.10 | 356.47 | 353.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 14:45:00 | 353.55 | 356.47 | 353.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 15:15:00 | 349.00 | 354.98 | 353.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-23 09:15:00 | 375.75 | 354.98 | 353.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-11-23 09:15:00 | 388.52 | 368.73 | 359.44 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2023-11-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-29 14:15:00 | 427.00 | 432.62 | 433.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-29 15:15:00 | 423.00 | 430.69 | 432.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-30 13:15:00 | 422.35 | 421.99 | 426.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-30 13:45:00 | 424.95 | 421.99 | 426.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 09:15:00 | 377.80 | 387.63 | 398.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-05 13:30:00 | 367.15 | 379.64 | 390.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-07 14:15:00 | 390.00 | 385.20 | 384.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2023-12-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 14:15:00 | 390.00 | 385.20 | 384.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 09:15:00 | 407.30 | 390.39 | 387.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 12:15:00 | 388.20 | 392.06 | 389.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-08 12:15:00 | 388.20 | 392.06 | 389.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 12:15:00 | 388.20 | 392.06 | 389.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 13:00:00 | 388.20 | 392.06 | 389.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 13:15:00 | 388.85 | 391.42 | 389.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 13:45:00 | 387.50 | 391.42 | 389.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 14:15:00 | 400.45 | 393.23 | 390.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-11 09:30:00 | 403.75 | 396.41 | 392.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-11 15:00:00 | 404.95 | 400.93 | 396.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-13 14:15:00 | 389.85 | 403.30 | 404.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2023-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 14:15:00 | 389.85 | 403.30 | 404.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-14 09:15:00 | 387.30 | 397.79 | 401.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-15 09:15:00 | 401.75 | 391.26 | 395.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-15 09:15:00 | 401.75 | 391.26 | 395.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 09:15:00 | 401.75 | 391.26 | 395.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-15 09:30:00 | 401.90 | 391.26 | 395.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 10:15:00 | 401.00 | 393.21 | 395.67 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 12:15:00 | 404.95 | 397.46 | 397.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 14:15:00 | 407.00 | 400.74 | 398.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 09:15:00 | 399.00 | 401.09 | 399.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-18 09:15:00 | 399.00 | 401.09 | 399.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 09:15:00 | 399.00 | 401.09 | 399.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 10:00:00 | 399.00 | 401.09 | 399.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 10:15:00 | 401.35 | 401.14 | 399.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 10:30:00 | 400.05 | 401.14 | 399.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 11:15:00 | 405.15 | 401.94 | 400.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 11:30:00 | 404.55 | 401.94 | 400.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 10:15:00 | 410.40 | 414.38 | 410.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 11:00:00 | 410.40 | 414.38 | 410.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 11:15:00 | 408.10 | 413.12 | 410.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 11:45:00 | 406.75 | 413.12 | 410.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 12:15:00 | 405.05 | 411.51 | 409.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 13:15:00 | 405.40 | 411.51 | 409.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 14:15:00 | 406.75 | 408.66 | 408.53 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 15:15:00 | 404.10 | 407.75 | 408.12 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-12-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-21 12:15:00 | 410.95 | 408.40 | 408.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 09:15:00 | 436.30 | 415.01 | 411.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 10:15:00 | 451.20 | 454.13 | 444.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-28 11:00:00 | 451.20 | 454.13 | 444.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 451.10 | 454.65 | 449.15 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-29 14:15:00 | 440.00 | 446.56 | 446.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-01 10:15:00 | 437.00 | 442.83 | 444.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-02 12:15:00 | 429.65 | 428.62 | 434.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-02 12:15:00 | 429.65 | 428.62 | 434.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 12:15:00 | 429.65 | 428.62 | 434.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-02 12:45:00 | 430.05 | 428.62 | 434.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 14:15:00 | 431.70 | 429.45 | 433.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-02 15:00:00 | 431.70 | 429.45 | 433.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 15:15:00 | 432.90 | 430.14 | 433.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-03 09:15:00 | 431.40 | 430.14 | 433.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 09:15:00 | 422.00 | 428.51 | 432.68 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2024-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 14:15:00 | 442.05 | 434.84 | 434.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 09:15:00 | 461.30 | 441.28 | 437.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 13:15:00 | 447.30 | 452.80 | 448.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-05 13:15:00 | 447.30 | 452.80 | 448.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 13:15:00 | 447.30 | 452.80 | 448.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 14:00:00 | 447.30 | 452.80 | 448.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 14:15:00 | 446.20 | 451.48 | 448.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 15:15:00 | 444.50 | 451.48 | 448.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 15:15:00 | 444.50 | 450.08 | 448.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-08 09:15:00 | 448.70 | 450.08 | 448.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-08 13:15:00 | 461.20 | 447.62 | 447.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-09 09:15:00 | 493.57 | 467.21 | 458.02 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 12:15:00 | 462.45 | 470.74 | 471.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 11:15:00 | 457.45 | 463.79 | 467.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 10:15:00 | 461.90 | 460.59 | 463.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 10:15:00 | 461.90 | 460.59 | 463.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 10:15:00 | 461.90 | 460.59 | 463.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 10:30:00 | 462.50 | 460.59 | 463.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 11:15:00 | 459.40 | 460.35 | 463.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 11:30:00 | 461.70 | 460.35 | 463.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 13:15:00 | 462.85 | 460.88 | 463.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 14:00:00 | 462.85 | 460.88 | 463.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 14:15:00 | 469.00 | 462.50 | 463.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 15:00:00 | 469.00 | 462.50 | 463.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 15:15:00 | 464.00 | 462.80 | 463.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 09:15:00 | 485.55 | 462.80 | 463.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2024-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 09:15:00 | 490.35 | 468.31 | 466.06 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-01-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 11:15:00 | 470.50 | 478.37 | 478.98 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-01-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 09:15:00 | 486.55 | 479.55 | 479.05 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 15:15:00 | 475.00 | 482.47 | 482.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-31 12:15:00 | 470.00 | 476.55 | 479.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-05 09:15:00 | 442.20 | 440.54 | 449.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-05 09:30:00 | 441.45 | 440.54 | 449.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 09:15:00 | 430.25 | 434.35 | 439.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-08 09:30:00 | 426.45 | 430.31 | 434.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-12 09:15:00 | 442.60 | 427.36 | 426.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-12 09:15:00 | 442.60 | 427.36 | 426.65 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-13 11:15:00 | 419.30 | 428.22 | 429.17 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-02-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 12:15:00 | 440.60 | 429.62 | 428.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 13:15:00 | 444.00 | 432.49 | 429.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 13:15:00 | 444.05 | 444.32 | 440.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-16 14:00:00 | 444.05 | 444.32 | 440.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 15:15:00 | 437.00 | 443.15 | 440.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 10:00:00 | 436.05 | 441.73 | 440.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 10:15:00 | 437.20 | 440.82 | 439.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 13:00:00 | 439.60 | 440.27 | 439.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 13:30:00 | 439.90 | 439.94 | 439.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 14:15:00 | 439.20 | 439.94 | 439.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-19 14:15:00 | 437.50 | 439.45 | 439.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 14:15:00 | 437.50 | 439.45 | 439.46 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 15:15:00 | 440.00 | 439.56 | 439.51 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 09:15:00 | 428.45 | 437.34 | 438.50 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 09:15:00 | 451.60 | 436.92 | 435.84 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 12:15:00 | 436.50 | 439.05 | 439.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 15:15:00 | 435.15 | 437.58 | 438.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 09:15:00 | 438.05 | 437.67 | 438.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 09:15:00 | 438.05 | 437.67 | 438.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 438.05 | 437.67 | 438.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-27 13:45:00 | 432.80 | 436.20 | 437.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-27 15:00:00 | 429.05 | 434.77 | 436.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 09:30:00 | 429.55 | 433.49 | 435.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-29 09:15:00 | 411.16 | 424.52 | 429.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-29 09:15:00 | 407.60 | 424.52 | 429.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-29 09:15:00 | 408.07 | 424.52 | 429.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-01 13:15:00 | 416.00 | 415.85 | 420.67 | SL hit (close>ema200) qty=0.50 sl=415.85 alert=retest2 |

### Cycle 25 — BUY (started 2024-03-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 13:15:00 | 371.50 | 371.09 | 371.06 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-03-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 14:15:00 | 370.30 | 370.93 | 370.99 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-03-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 15:15:00 | 372.00 | 371.15 | 371.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-26 10:15:00 | 373.15 | 371.68 | 371.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-02 10:15:00 | 401.80 | 404.53 | 399.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-02 10:45:00 | 402.65 | 404.53 | 399.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 14:15:00 | 412.85 | 412.81 | 409.97 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-04-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 15:15:00 | 404.10 | 408.26 | 408.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-08 14:15:00 | 395.60 | 404.99 | 406.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-09 14:15:00 | 400.00 | 398.43 | 401.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 14:15:00 | 400.00 | 398.43 | 401.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 14:15:00 | 400.00 | 398.43 | 401.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-09 15:00:00 | 400.00 | 398.43 | 401.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 395.20 | 394.98 | 397.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-15 09:15:00 | 391.35 | 396.64 | 397.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 09:15:00 | 386.25 | 395.22 | 396.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 13:15:00 | 391.35 | 393.50 | 394.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 09:15:00 | 400.10 | 388.80 | 387.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 09:15:00 | 400.10 | 388.80 | 387.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 15:15:00 | 406.00 | 399.17 | 394.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 09:15:00 | 426.80 | 428.04 | 415.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-24 10:00:00 | 426.80 | 428.04 | 415.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 15:15:00 | 422.00 | 425.35 | 422.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 09:15:00 | 427.30 | 425.35 | 422.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 14:30:00 | 429.95 | 424.17 | 423.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 10:30:00 | 425.60 | 428.61 | 428.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-02 11:15:00 | 423.05 | 427.50 | 427.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 11:15:00 | 423.05 | 427.50 | 427.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 09:15:00 | 420.30 | 423.83 | 425.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-03 14:15:00 | 420.50 | 419.33 | 422.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-03 14:15:00 | 420.50 | 419.33 | 422.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 14:15:00 | 420.50 | 419.33 | 422.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-03 15:00:00 | 420.50 | 419.33 | 422.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 418.20 | 419.47 | 421.82 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2024-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 14:15:00 | 428.60 | 423.22 | 422.92 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 12:15:00 | 418.60 | 422.30 | 422.69 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-07 14:15:00 | 427.65 | 423.36 | 423.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 14:15:00 | 430.40 | 427.40 | 425.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 09:15:00 | 427.15 | 428.08 | 426.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-09 10:00:00 | 427.15 | 428.08 | 426.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 10:15:00 | 425.00 | 427.47 | 426.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 10:45:00 | 424.45 | 427.47 | 426.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 11:15:00 | 429.35 | 427.84 | 426.48 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 11:15:00 | 425.30 | 427.21 | 427.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-14 09:15:00 | 420.35 | 424.94 | 426.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 15:15:00 | 413.00 | 412.17 | 415.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-17 09:15:00 | 408.90 | 412.17 | 415.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 408.05 | 411.35 | 414.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 10:45:00 | 405.50 | 410.09 | 413.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-21 09:15:00 | 426.60 | 409.61 | 410.22 | SL hit (close>static) qty=1.00 sl=417.10 alert=retest2 |

### Cycle 35 — BUY (started 2024-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 10:15:00 | 425.15 | 412.72 | 411.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 09:15:00 | 430.00 | 422.50 | 417.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 09:15:00 | 423.10 | 425.58 | 422.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-23 09:45:00 | 423.00 | 425.58 | 422.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 419.95 | 424.45 | 421.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:45:00 | 420.30 | 424.45 | 421.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 420.80 | 423.72 | 421.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 12:00:00 | 420.80 | 423.72 | 421.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2024-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 15:15:00 | 418.90 | 420.69 | 420.80 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 09:15:00 | 430.00 | 422.55 | 421.64 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 15:15:00 | 416.80 | 423.37 | 424.19 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 14:15:00 | 427.65 | 424.87 | 424.61 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 418.80 | 424.00 | 424.28 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 10:15:00 | 430.70 | 424.18 | 423.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 14:15:00 | 441.00 | 430.21 | 426.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 444.10 | 450.30 | 442.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 444.10 | 450.30 | 442.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 444.10 | 450.30 | 442.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 441.20 | 450.30 | 442.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 430.00 | 446.24 | 441.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 430.00 | 446.24 | 441.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 421.70 | 441.33 | 439.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 421.70 | 441.33 | 439.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 430.05 | 439.08 | 438.50 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2024-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 13:15:00 | 426.05 | 436.47 | 437.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 14:15:00 | 417.35 | 432.65 | 435.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 439.40 | 432.74 | 435.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 09:15:00 | 439.40 | 432.74 | 435.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 439.40 | 432.74 | 435.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 439.40 | 432.74 | 435.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 444.75 | 435.14 | 435.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:45:00 | 451.10 | 435.14 | 435.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 11:15:00 | 450.95 | 438.30 | 437.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 12:15:00 | 457.85 | 442.21 | 439.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 09:15:00 | 449.65 | 449.94 | 444.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 10:15:00 | 446.30 | 449.21 | 444.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 446.30 | 449.21 | 444.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:45:00 | 445.30 | 449.21 | 444.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 11:15:00 | 444.30 | 448.23 | 444.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:30:00 | 444.60 | 448.23 | 444.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 12:15:00 | 444.35 | 447.45 | 444.49 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2024-06-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-07 10:15:00 | 439.95 | 442.68 | 442.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-07 14:15:00 | 438.75 | 440.85 | 441.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-10 09:15:00 | 444.65 | 441.28 | 441.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-10 09:15:00 | 444.65 | 441.28 | 441.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 09:15:00 | 444.65 | 441.28 | 441.91 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2024-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-10 10:15:00 | 453.70 | 443.76 | 442.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 11:15:00 | 456.75 | 446.36 | 444.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 09:15:00 | 441.05 | 450.18 | 447.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-11 09:15:00 | 441.05 | 450.18 | 447.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 441.05 | 450.18 | 447.44 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2024-06-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-11 11:15:00 | 433.35 | 443.54 | 444.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 12:15:00 | 423.05 | 429.45 | 431.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 12:15:00 | 424.00 | 423.90 | 427.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-20 13:00:00 | 424.00 | 423.90 | 427.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 427.35 | 424.77 | 427.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 15:00:00 | 427.35 | 424.77 | 427.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 15:15:00 | 427.30 | 425.27 | 427.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:15:00 | 428.60 | 425.27 | 427.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 437.35 | 427.69 | 428.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:00:00 | 437.35 | 427.69 | 428.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 10:15:00 | 445.00 | 431.15 | 429.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 13:15:00 | 448.80 | 438.68 | 433.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 11:15:00 | 450.25 | 451.00 | 445.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-25 12:00:00 | 450.25 | 451.00 | 445.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 447.40 | 449.56 | 446.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 15:00:00 | 447.40 | 449.56 | 446.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 444.20 | 449.20 | 446.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:00:00 | 444.20 | 449.20 | 446.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 446.85 | 448.73 | 446.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 11:30:00 | 449.80 | 448.45 | 446.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 09:15:00 | 435.10 | 444.31 | 445.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 09:15:00 | 435.10 | 444.31 | 445.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 10:15:00 | 434.15 | 442.28 | 444.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 14:15:00 | 438.75 | 437.93 | 441.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-27 15:00:00 | 438.75 | 437.93 | 441.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 437.05 | 435.51 | 438.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 12:30:00 | 439.30 | 435.51 | 438.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 434.30 | 435.51 | 438.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 15:15:00 | 428.00 | 435.51 | 438.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 09:15:00 | 447.50 | 436.70 | 438.04 | SL hit (close>static) qty=1.00 sl=446.95 alert=retest2 |

### Cycle 49 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 441.25 | 438.93 | 438.90 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 433.35 | 438.26 | 438.78 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 09:15:00 | 454.45 | 439.76 | 438.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 10:15:00 | 468.90 | 445.59 | 441.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 09:15:00 | 468.25 | 475.27 | 466.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 09:15:00 | 468.25 | 475.27 | 466.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 468.25 | 475.27 | 466.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:45:00 | 466.40 | 475.27 | 466.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 461.85 | 472.59 | 466.13 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2024-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 10:15:00 | 461.75 | 464.32 | 464.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 15:15:00 | 455.50 | 459.91 | 461.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 09:15:00 | 468.75 | 461.68 | 462.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 09:15:00 | 468.75 | 461.68 | 462.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 468.75 | 461.68 | 462.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 09:45:00 | 471.00 | 461.68 | 462.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 465.70 | 462.48 | 462.86 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2024-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 12:15:00 | 467.75 | 463.96 | 463.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 13:15:00 | 470.30 | 465.22 | 464.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 11:15:00 | 468.25 | 471.45 | 469.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 11:15:00 | 468.25 | 471.45 | 469.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 468.25 | 471.45 | 469.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 11:30:00 | 469.10 | 471.45 | 469.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 470.05 | 471.17 | 469.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 13:15:00 | 471.05 | 471.17 | 469.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 09:15:00 | 469.05 | 478.81 | 478.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 469.05 | 478.81 | 478.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 14:15:00 | 465.15 | 471.31 | 474.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 15:15:00 | 464.00 | 463.64 | 468.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-23 09:15:00 | 460.15 | 463.64 | 468.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 462.40 | 462.88 | 466.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 455.20 | 462.88 | 466.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 14:15:00 | 468.15 | 464.65 | 466.48 | SL hit (close>static) qty=1.00 sl=466.90 alert=retest2 |

### Cycle 55 — BUY (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 09:15:00 | 473.45 | 465.64 | 464.80 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 12:15:00 | 459.25 | 463.96 | 464.21 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 10:15:00 | 469.30 | 464.96 | 464.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 14:15:00 | 475.00 | 468.81 | 466.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 09:15:00 | 469.35 | 469.92 | 467.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 10:00:00 | 469.35 | 469.92 | 467.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 467.20 | 469.38 | 467.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:00:00 | 467.20 | 469.38 | 467.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 11:15:00 | 468.15 | 469.13 | 467.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:30:00 | 468.45 | 469.13 | 467.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 12:15:00 | 467.70 | 468.85 | 467.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 12:45:00 | 467.25 | 468.85 | 467.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 468.25 | 468.73 | 467.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 14:30:00 | 469.30 | 468.18 | 467.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 10:15:00 | 465.90 | 467.82 | 467.52 | SL hit (close<static) qty=1.00 sl=466.40 alert=retest2 |

### Cycle 58 — SELL (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 11:15:00 | 465.20 | 467.30 | 467.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 12:15:00 | 461.60 | 466.16 | 466.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-05 14:15:00 | 455.40 | 451.75 | 455.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 14:15:00 | 455.40 | 451.75 | 455.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 14:15:00 | 455.40 | 451.75 | 455.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-05 15:00:00 | 455.40 | 451.75 | 455.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 15:15:00 | 456.80 | 452.76 | 455.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:15:00 | 458.70 | 452.76 | 455.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 456.40 | 453.49 | 455.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 10:45:00 | 455.00 | 453.65 | 455.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:45:00 | 455.00 | 454.91 | 455.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 11:00:00 | 453.90 | 452.52 | 454.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 14:15:00 | 455.25 | 453.35 | 454.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 10:15:00 | 457.80 | 455.05 | 454.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 10:15:00 | 457.80 | 455.05 | 454.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 11:15:00 | 461.60 | 456.36 | 455.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 09:15:00 | 476.15 | 476.70 | 467.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 09:45:00 | 479.30 | 476.70 | 467.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 472.00 | 475.02 | 470.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:15:00 | 462.60 | 475.02 | 470.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 448.95 | 469.81 | 468.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:45:00 | 447.65 | 469.81 | 468.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 10:15:00 | 453.30 | 466.51 | 467.12 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 09:15:00 | 470.00 | 464.11 | 463.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-14 10:15:00 | 471.50 | 465.59 | 464.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-16 09:15:00 | 461.05 | 467.31 | 466.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 461.05 | 467.31 | 466.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 461.05 | 467.31 | 466.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:45:00 | 461.60 | 467.31 | 466.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 461.90 | 466.23 | 465.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:30:00 | 460.35 | 466.23 | 465.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2024-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 12:15:00 | 462.75 | 465.10 | 465.25 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 10:15:00 | 469.30 | 465.54 | 465.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 13:15:00 | 471.55 | 468.13 | 466.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 09:15:00 | 465.70 | 468.39 | 467.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 09:15:00 | 465.70 | 468.39 | 467.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 465.70 | 468.39 | 467.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 10:00:00 | 465.70 | 468.39 | 467.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 466.25 | 467.96 | 467.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 11:30:00 | 467.50 | 467.95 | 467.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-20 15:15:00 | 465.05 | 466.58 | 466.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2024-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 15:15:00 | 465.05 | 466.58 | 466.72 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 10:15:00 | 468.90 | 466.84 | 466.80 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 09:15:00 | 463.35 | 466.69 | 466.84 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 11:15:00 | 473.75 | 467.87 | 467.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 09:15:00 | 519.25 | 479.07 | 473.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-26 15:15:00 | 506.00 | 506.73 | 492.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 15:15:00 | 506.00 | 506.73 | 492.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 15:15:00 | 506.00 | 506.73 | 492.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 15:15:00 | 515.20 | 505.23 | 497.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 09:45:00 | 517.70 | 509.08 | 500.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 09:15:00 | 524.15 | 512.93 | 506.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 12:15:00 | 516.85 | 514.78 | 509.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 511.30 | 514.16 | 510.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 15:00:00 | 511.30 | 514.16 | 510.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 515.00 | 514.33 | 510.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:30:00 | 505.60 | 514.40 | 511.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 513.60 | 513.90 | 511.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:30:00 | 511.80 | 513.90 | 511.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 12:15:00 | 510.45 | 513.21 | 511.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 12:45:00 | 512.50 | 513.21 | 511.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 13:15:00 | 506.15 | 511.80 | 510.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 14:00:00 | 506.15 | 511.80 | 510.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-30 14:15:00 | 504.25 | 510.29 | 510.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 14:15:00 | 504.25 | 510.29 | 510.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 09:15:00 | 503.00 | 507.98 | 509.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 14:15:00 | 505.25 | 504.50 | 506.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-02 15:00:00 | 505.25 | 504.50 | 506.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 514.95 | 506.53 | 507.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:30:00 | 520.30 | 506.53 | 507.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 10:15:00 | 518.05 | 508.83 | 508.18 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 11:15:00 | 504.00 | 508.31 | 508.78 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 11:15:00 | 513.30 | 508.67 | 508.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 12:15:00 | 514.40 | 509.82 | 508.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-09 09:15:00 | 524.20 | 526.68 | 521.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 09:15:00 | 524.20 | 526.68 | 521.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 524.20 | 526.68 | 521.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 09:15:00 | 531.00 | 521.59 | 520.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 13:15:00 | 521.60 | 528.03 | 528.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2024-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 13:15:00 | 521.60 | 528.03 | 528.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 15:15:00 | 520.00 | 525.61 | 526.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 09:15:00 | 501.45 | 501.14 | 510.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-13 10:15:00 | 503.20 | 501.14 | 510.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 480.15 | 474.32 | 481.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 14:45:00 | 479.85 | 474.32 | 481.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 471.25 | 474.46 | 480.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 10:15:00 | 470.70 | 474.46 | 480.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 10:45:00 | 470.35 | 473.12 | 479.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 13:00:00 | 470.55 | 465.48 | 471.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 14:00:00 | 467.40 | 465.86 | 470.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 470.50 | 466.79 | 470.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 15:00:00 | 470.50 | 466.79 | 470.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 470.00 | 467.43 | 470.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 463.55 | 467.43 | 470.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 463.35 | 466.62 | 469.95 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-20 14:15:00 | 479.75 | 471.08 | 470.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2024-09-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 14:15:00 | 479.75 | 471.08 | 470.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 15:15:00 | 487.30 | 474.32 | 472.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 13:15:00 | 480.40 | 485.49 | 481.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 13:15:00 | 480.40 | 485.49 | 481.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 13:15:00 | 480.40 | 485.49 | 481.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 14:00:00 | 480.40 | 485.49 | 481.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 474.30 | 483.25 | 481.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 14:45:00 | 475.55 | 483.25 | 481.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 475.00 | 481.60 | 480.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 09:15:00 | 477.10 | 481.60 | 480.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 09:15:00 | 471.50 | 479.58 | 479.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 09:15:00 | 471.50 | 479.58 | 479.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 12:15:00 | 468.20 | 473.24 | 474.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 14:15:00 | 459.15 | 456.41 | 462.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 14:15:00 | 459.15 | 456.41 | 462.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 459.15 | 456.41 | 462.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 15:00:00 | 459.15 | 456.41 | 462.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 454.80 | 456.08 | 462.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 09:15:00 | 447.90 | 456.08 | 462.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 14:00:00 | 451.25 | 450.99 | 456.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 451.90 | 452.24 | 456.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 10:00:00 | 449.30 | 451.66 | 455.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 440.15 | 447.00 | 451.13 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 429.30 | 447.00 | 451.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 13:15:00 | 425.50 | 441.09 | 446.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 13:15:00 | 428.69 | 441.09 | 446.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 13:15:00 | 426.83 | 441.09 | 446.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 14:00:00 | 427.45 | 441.09 | 446.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 15:00:00 | 428.00 | 438.47 | 445.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 09:45:00 | 427.85 | 434.21 | 441.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 11:45:00 | 429.10 | 433.06 | 440.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 11:15:00 | 431.70 | 429.50 | 434.48 | SL hit (close>ema200) qty=0.50 sl=429.50 alert=retest2 |

### Cycle 75 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 438.35 | 436.39 | 436.34 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 13:15:00 | 433.10 | 435.92 | 436.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-09 14:15:00 | 432.90 | 435.31 | 435.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 09:15:00 | 428.85 | 427.16 | 430.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 09:15:00 | 428.85 | 427.16 | 430.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 428.85 | 427.16 | 430.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 10:30:00 | 422.80 | 427.64 | 429.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 11:15:00 | 423.95 | 427.64 | 429.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 11:15:00 | 423.60 | 425.95 | 427.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 09:45:00 | 421.00 | 425.91 | 426.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 425.60 | 417.29 | 420.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 13:00:00 | 425.60 | 417.29 | 420.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 422.45 | 418.32 | 420.36 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-18 10:15:00 | 426.95 | 422.04 | 421.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 10:15:00 | 426.95 | 422.04 | 421.66 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-10-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 15:15:00 | 416.80 | 421.54 | 422.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 409.20 | 419.07 | 420.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 09:15:00 | 404.90 | 402.63 | 407.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 09:15:00 | 404.90 | 402.63 | 407.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 404.90 | 402.63 | 407.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:45:00 | 399.65 | 405.57 | 406.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 09:15:00 | 399.35 | 400.65 | 403.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 11:15:00 | 408.20 | 402.23 | 403.29 | SL hit (close>static) qty=1.00 sl=407.85 alert=retest2 |

### Cycle 79 — BUY (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 13:15:00 | 407.50 | 404.38 | 404.15 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 398.85 | 403.33 | 403.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 12:15:00 | 395.00 | 400.93 | 402.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 14:15:00 | 401.75 | 400.03 | 401.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-29 14:15:00 | 401.75 | 400.03 | 401.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 401.75 | 400.03 | 401.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 15:00:00 | 401.75 | 400.03 | 401.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 399.70 | 399.96 | 401.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:15:00 | 405.40 | 399.96 | 401.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 397.50 | 399.47 | 401.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:30:00 | 403.85 | 399.47 | 401.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 11:15:00 | 401.30 | 399.84 | 401.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 11:30:00 | 399.80 | 399.84 | 401.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 12:15:00 | 402.45 | 400.36 | 401.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 13:00:00 | 402.45 | 400.36 | 401.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 13:15:00 | 402.60 | 400.81 | 401.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 13:45:00 | 404.80 | 400.81 | 401.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 15:15:00 | 404.50 | 402.19 | 401.95 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 11:15:00 | 394.60 | 400.42 | 401.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 13:15:00 | 391.80 | 395.76 | 397.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 15:15:00 | 389.15 | 388.71 | 391.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-06 09:15:00 | 390.75 | 388.71 | 391.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 392.05 | 389.38 | 391.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:45:00 | 393.55 | 389.38 | 391.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 391.80 | 389.86 | 391.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 13:15:00 | 389.90 | 390.11 | 391.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 14:00:00 | 390.00 | 390.09 | 391.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 14:45:00 | 388.20 | 389.89 | 391.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 09:45:00 | 389.10 | 389.82 | 390.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 388.95 | 389.64 | 390.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 11:15:00 | 386.90 | 389.64 | 390.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 15:15:00 | 370.40 | 383.70 | 387.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 15:15:00 | 370.50 | 383.70 | 387.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 15:15:00 | 368.79 | 383.70 | 387.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 15:15:00 | 369.64 | 383.70 | 387.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 15:15:00 | 367.55 | 383.70 | 387.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-08 15:15:00 | 381.90 | 381.14 | 383.87 | SL hit (close>ema200) qty=0.50 sl=381.14 alert=retest2 |

### Cycle 83 — BUY (started 2024-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 09:15:00 | 252.65 | 233.80 | 233.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 09:15:00 | 268.70 | 251.91 | 244.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 09:15:00 | 258.00 | 260.15 | 253.30 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:15:00 | 273.50 | 261.62 | 259.24 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 09:15:00 | 287.18 | 266.46 | 261.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 270.40 | 275.47 | 270.11 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-05 09:15:00 | 270.40 | 275.47 | 270.11 | SL hit (close<ema200) qty=0.50 sl=275.47 alert=retest1 |

### Cycle 84 — SELL (started 2024-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 12:15:00 | 267.70 | 270.18 | 270.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 15:15:00 | 265.30 | 268.69 | 269.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 10:15:00 | 259.75 | 257.86 | 261.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 10:15:00 | 259.75 | 257.86 | 261.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 259.75 | 257.86 | 261.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 10:45:00 | 259.25 | 257.86 | 261.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 11:15:00 | 263.20 | 258.93 | 261.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 11:45:00 | 264.40 | 258.93 | 261.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 12:15:00 | 263.65 | 259.87 | 261.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 13:00:00 | 263.65 | 259.87 | 261.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 261.10 | 260.22 | 261.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 15:00:00 | 261.10 | 260.22 | 261.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 15:15:00 | 260.90 | 260.36 | 261.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:15:00 | 257.30 | 260.36 | 261.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 256.15 | 259.51 | 260.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 11:30:00 | 253.95 | 257.46 | 259.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 13:45:00 | 253.75 | 256.25 | 258.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 15:00:00 | 250.00 | 255.00 | 257.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 11:00:00 | 253.50 | 247.84 | 248.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 11:15:00 | 251.35 | 248.54 | 249.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 11:30:00 | 254.40 | 248.54 | 249.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-18 13:15:00 | 251.60 | 249.64 | 249.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2024-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 13:15:00 | 251.60 | 249.64 | 249.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 10:15:00 | 253.40 | 250.97 | 250.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 12:15:00 | 254.00 | 256.09 | 253.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 12:15:00 | 254.00 | 256.09 | 253.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 254.00 | 256.09 | 253.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 13:00:00 | 254.00 | 256.09 | 253.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 253.50 | 255.57 | 253.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 15:00:00 | 259.45 | 256.35 | 254.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 09:15:00 | 253.00 | 254.64 | 254.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2024-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 09:15:00 | 253.00 | 254.64 | 254.65 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2024-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 10:15:00 | 255.60 | 254.55 | 254.45 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 12:15:00 | 253.85 | 254.37 | 254.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 13:15:00 | 252.75 | 254.05 | 254.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 14:15:00 | 252.40 | 248.71 | 249.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 14:15:00 | 252.40 | 248.71 | 249.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 252.40 | 248.71 | 249.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 252.40 | 248.71 | 249.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 251.90 | 249.35 | 250.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 09:15:00 | 248.55 | 249.35 | 250.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 10:00:00 | 249.80 | 249.44 | 249.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 11:15:00 | 250.15 | 249.70 | 250.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 12:15:00 | 250.00 | 250.09 | 250.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 250.00 | 250.07 | 250.18 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-31 13:15:00 | 253.25 | 250.71 | 250.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2024-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 13:15:00 | 253.25 | 250.71 | 250.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 14:15:00 | 255.05 | 251.58 | 250.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 09:15:00 | 252.00 | 252.21 | 251.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-01 10:00:00 | 252.00 | 252.21 | 251.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 251.05 | 251.98 | 251.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:45:00 | 251.35 | 251.98 | 251.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 11:15:00 | 250.20 | 251.62 | 251.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 11:30:00 | 250.00 | 251.62 | 251.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 12:15:00 | 249.75 | 251.25 | 251.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 13:00:00 | 249.75 | 251.25 | 251.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2025-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 14:15:00 | 249.55 | 250.70 | 250.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 09:15:00 | 249.25 | 250.29 | 250.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 14:15:00 | 250.00 | 249.61 | 250.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 14:15:00 | 250.00 | 249.61 | 250.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 14:15:00 | 250.00 | 249.61 | 250.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 15:00:00 | 250.00 | 249.61 | 250.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 15:15:00 | 250.10 | 249.71 | 250.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 09:15:00 | 251.40 | 249.71 | 250.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 251.35 | 250.04 | 250.20 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 10:15:00 | 253.05 | 250.64 | 250.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 11:15:00 | 254.40 | 251.39 | 250.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 15:15:00 | 252.50 | 252.52 | 251.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 15:15:00 | 252.50 | 252.52 | 251.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 252.50 | 252.52 | 251.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 246.60 | 252.52 | 251.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 248.00 | 251.62 | 251.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 248.90 | 251.62 | 251.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 245.95 | 250.48 | 250.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 244.45 | 249.28 | 250.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 247.50 | 246.84 | 248.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 09:45:00 | 246.35 | 246.84 | 248.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 245.50 | 246.25 | 247.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:15:00 | 253.00 | 246.25 | 247.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 252.65 | 247.53 | 247.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 11:15:00 | 247.60 | 247.81 | 247.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 10:15:00 | 249.35 | 247.70 | 247.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 10:15:00 | 249.35 | 247.70 | 247.64 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 12:15:00 | 246.05 | 247.38 | 247.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 243.15 | 246.03 | 246.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-13 15:15:00 | 242.00 | 241.34 | 242.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 09:15:00 | 242.45 | 241.34 | 242.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 244.00 | 241.87 | 243.03 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 09:15:00 | 245.75 | 243.66 | 243.51 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-01-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 15:15:00 | 242.80 | 243.64 | 243.67 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 244.70 | 243.85 | 243.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 10:15:00 | 247.95 | 246.33 | 245.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 249.80 | 250.02 | 248.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 249.80 | 250.02 | 248.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 248.40 | 249.69 | 248.40 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 246.15 | 248.18 | 248.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 13:15:00 | 246.05 | 247.44 | 247.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 247.95 | 247.39 | 247.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 247.95 | 247.39 | 247.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 247.95 | 247.39 | 247.73 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2025-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 11:15:00 | 248.90 | 248.07 | 248.00 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 14:15:00 | 244.25 | 247.86 | 248.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 229.90 | 243.94 | 246.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 09:15:00 | 225.45 | 224.48 | 227.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-30 09:45:00 | 225.35 | 224.48 | 227.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 11:15:00 | 219.15 | 218.51 | 222.15 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 13:15:00 | 230.00 | 222.79 | 222.53 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 15:15:00 | 224.00 | 225.93 | 226.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 09:15:00 | 221.82 | 225.11 | 225.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 206.39 | 204.03 | 207.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:00:00 | 206.39 | 204.03 | 207.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 207.15 | 204.66 | 207.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:30:00 | 207.93 | 204.66 | 207.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 207.99 | 205.32 | 207.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:45:00 | 207.46 | 205.32 | 207.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 204.64 | 205.19 | 207.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 14:15:00 | 204.00 | 205.19 | 207.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 14:45:00 | 203.66 | 205.21 | 207.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-13 09:15:00 | 216.81 | 207.62 | 207.96 | SL hit (close>static) qty=1.00 sl=208.20 alert=retest2 |

### Cycle 103 — BUY (started 2025-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 10:15:00 | 222.17 | 210.53 | 209.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 12:15:00 | 225.06 | 215.27 | 211.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 12:15:00 | 219.10 | 222.12 | 218.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-14 12:15:00 | 219.10 | 222.12 | 218.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 12:15:00 | 219.10 | 222.12 | 218.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 12:45:00 | 219.91 | 222.12 | 218.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 13:15:00 | 219.34 | 221.56 | 218.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 13:45:00 | 219.00 | 221.56 | 218.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 09:15:00 | 220.44 | 221.20 | 218.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 10:30:00 | 230.33 | 224.85 | 222.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 09:15:00 | 221.32 | 224.19 | 224.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 221.32 | 224.19 | 224.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 217.67 | 221.40 | 222.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 15:15:00 | 211.40 | 211.00 | 214.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 09:15:00 | 213.12 | 211.43 | 214.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 213.12 | 211.43 | 214.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:45:00 | 213.03 | 211.43 | 214.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 13:15:00 | 214.93 | 212.11 | 213.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 13:45:00 | 214.07 | 212.11 | 213.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 217.10 | 213.11 | 214.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 15:00:00 | 217.10 | 213.11 | 214.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 217.60 | 214.68 | 214.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 220.39 | 216.48 | 215.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 15:15:00 | 222.70 | 223.26 | 221.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 09:15:00 | 219.26 | 223.26 | 221.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 218.53 | 222.31 | 220.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 218.53 | 222.31 | 220.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 222.01 | 222.25 | 220.90 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 216.39 | 219.78 | 220.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 14:15:00 | 209.49 | 212.04 | 214.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 208.99 | 208.32 | 210.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 208.99 | 208.32 | 210.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 208.99 | 208.32 | 210.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:30:00 | 209.71 | 208.32 | 210.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 11:15:00 | 211.20 | 208.87 | 210.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 12:00:00 | 211.20 | 208.87 | 210.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 12:15:00 | 211.33 | 209.36 | 210.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 13:15:00 | 212.24 | 209.36 | 210.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 13:15:00 | 211.22 | 209.74 | 210.53 | EMA400 retest candle locked (from downside) |

### Cycle 107 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 217.58 | 212.02 | 211.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 12:15:00 | 220.41 | 215.77 | 213.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 11:15:00 | 221.45 | 221.99 | 219.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 12:00:00 | 221.45 | 221.99 | 219.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 221.23 | 221.84 | 219.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 13:45:00 | 220.33 | 221.84 | 219.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 220.73 | 221.39 | 220.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:15:00 | 223.76 | 221.39 | 220.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 13:15:00 | 229.61 | 232.22 | 232.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 13:15:00 | 229.61 | 232.22 | 232.28 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 14:15:00 | 235.10 | 232.80 | 232.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 15:15:00 | 237.00 | 233.64 | 232.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 09:15:00 | 232.53 | 233.42 | 232.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 09:15:00 | 232.53 | 233.42 | 232.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 232.53 | 233.42 | 232.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:30:00 | 233.80 | 233.42 | 232.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 232.23 | 233.18 | 232.84 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 12:15:00 | 229.67 | 232.22 | 232.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 13:15:00 | 228.84 | 231.54 | 232.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 14:15:00 | 233.00 | 231.83 | 232.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 14:15:00 | 233.00 | 231.83 | 232.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 233.00 | 231.83 | 232.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 15:00:00 | 233.00 | 231.83 | 232.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 233.00 | 232.07 | 232.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 09:15:00 | 232.35 | 232.07 | 232.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 10:15:00 | 233.67 | 232.38 | 232.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 10:15:00 | 233.67 | 232.38 | 232.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 11:15:00 | 235.38 | 232.98 | 232.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 14:15:00 | 233.89 | 233.95 | 233.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-01 15:00:00 | 233.89 | 233.95 | 233.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 236.21 | 234.73 | 233.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 10:30:00 | 240.77 | 236.88 | 235.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 12:15:00 | 234.17 | 235.85 | 235.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 234.17 | 235.85 | 235.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 233.64 | 235.41 | 235.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 11:15:00 | 225.00 | 224.99 | 228.05 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-09 09:15:00 | 221.44 | 224.90 | 227.03 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 223.54 | 220.79 | 223.19 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 223.54 | 220.79 | 223.19 | SL hit (close>ema400) qty=1.00 sl=223.19 alert=retest1 |

### Cycle 113 — BUY (started 2025-04-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 12:15:00 | 230.82 | 224.97 | 224.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 13:15:00 | 232.25 | 226.43 | 225.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 14:15:00 | 229.20 | 229.29 | 227.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-15 14:45:00 | 229.17 | 229.29 | 227.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 11:15:00 | 237.22 | 231.39 | 229.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:30:00 | 240.30 | 233.99 | 231.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 230.49 | 232.39 | 232.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 09:15:00 | 230.49 | 232.39 | 232.45 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 11:15:00 | 234.23 | 232.47 | 232.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 13:15:00 | 235.33 | 233.26 | 232.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 15:15:00 | 236.66 | 237.59 | 236.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-25 09:15:00 | 236.10 | 237.59 | 236.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 231.04 | 236.28 | 235.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 09:45:00 | 230.19 | 236.28 | 235.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 229.18 | 234.86 | 235.02 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 10:15:00 | 240.15 | 234.80 | 234.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 11:15:00 | 244.65 | 236.77 | 235.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 14:15:00 | 246.16 | 246.66 | 242.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-30 15:00:00 | 246.16 | 246.66 | 242.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 248.24 | 246.87 | 243.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 11:15:00 | 249.66 | 247.17 | 244.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 12:30:00 | 249.69 | 248.01 | 245.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 11:15:00 | 248.01 | 250.25 | 250.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2025-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 11:15:00 | 248.01 | 250.25 | 250.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 246.00 | 248.50 | 249.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 249.81 | 246.24 | 247.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 249.81 | 246.24 | 247.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 249.81 | 246.24 | 247.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 10:30:00 | 248.34 | 246.68 | 247.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-13 09:15:00 | 251.60 | 248.03 | 247.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 09:15:00 | 251.60 | 248.03 | 247.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 10:15:00 | 255.46 | 249.51 | 248.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 15:15:00 | 255.10 | 256.45 | 254.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 09:15:00 | 258.95 | 256.45 | 254.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 258.75 | 256.91 | 254.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:15:00 | 261.00 | 256.16 | 255.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-23 09:15:00 | 287.10 | 280.35 | 274.13 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 312.35 | 313.72 | 313.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 11:15:00 | 311.75 | 313.32 | 313.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 14:15:00 | 312.10 | 311.99 | 312.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 14:15:00 | 312.10 | 311.99 | 312.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 312.10 | 311.99 | 312.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:00:00 | 312.10 | 311.99 | 312.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 312.00 | 311.99 | 312.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:15:00 | 316.20 | 311.99 | 312.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 316.80 | 312.95 | 313.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:15:00 | 323.80 | 312.95 | 313.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 318.10 | 313.98 | 313.61 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 10:15:00 | 313.75 | 316.26 | 316.27 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 11:15:00 | 318.95 | 316.80 | 316.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 12:15:00 | 320.05 | 317.45 | 316.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 320.80 | 321.50 | 320.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 10:15:00 | 320.80 | 321.50 | 320.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 320.80 | 321.50 | 320.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:00:00 | 320.80 | 321.50 | 320.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 320.00 | 321.20 | 320.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:00:00 | 320.00 | 321.20 | 320.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 316.15 | 320.19 | 319.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:45:00 | 316.15 | 320.19 | 319.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 318.00 | 319.75 | 319.58 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2025-06-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 15:15:00 | 318.20 | 319.29 | 319.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 12:15:00 | 316.40 | 318.61 | 319.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 308.35 | 308.00 | 310.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:30:00 | 308.20 | 308.00 | 310.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 310.25 | 308.45 | 310.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:30:00 | 310.20 | 308.45 | 310.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 310.00 | 308.76 | 310.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 307.25 | 309.45 | 310.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:15:00 | 306.70 | 309.12 | 310.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 11:45:00 | 308.70 | 306.82 | 306.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 12:15:00 | 308.20 | 307.10 | 306.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 12:15:00 | 308.20 | 307.10 | 306.98 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 13:15:00 | 306.00 | 306.88 | 306.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 14:15:00 | 297.45 | 304.99 | 306.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 14:15:00 | 304.45 | 303.05 | 304.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 14:15:00 | 304.45 | 303.05 | 304.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 304.45 | 303.05 | 304.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:45:00 | 304.65 | 303.05 | 304.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 303.15 | 303.07 | 304.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 306.25 | 303.07 | 304.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 304.70 | 303.40 | 304.15 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 311.60 | 305.81 | 305.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 15:15:00 | 313.00 | 308.22 | 306.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 10:15:00 | 308.50 | 308.76 | 307.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 11:00:00 | 308.50 | 308.76 | 307.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 307.75 | 310.12 | 308.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 307.75 | 310.12 | 308.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 305.90 | 309.27 | 308.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:45:00 | 305.35 | 309.27 | 308.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 308.90 | 309.13 | 308.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:45:00 | 308.45 | 309.13 | 308.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 308.65 | 312.63 | 311.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 308.65 | 312.63 | 311.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 310.30 | 312.16 | 311.33 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2025-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 10:15:00 | 308.25 | 310.81 | 310.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 09:15:00 | 305.85 | 309.40 | 310.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 13:15:00 | 298.40 | 296.00 | 298.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 13:15:00 | 298.40 | 296.00 | 298.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 298.40 | 296.00 | 298.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:45:00 | 299.30 | 296.00 | 298.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 299.05 | 296.61 | 298.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 15:15:00 | 298.05 | 296.61 | 298.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:45:00 | 297.45 | 295.78 | 297.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 300.40 | 296.97 | 297.01 | SL hit (close>static) qty=1.00 sl=300.15 alert=retest2 |

### Cycle 129 — BUY (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 10:15:00 | 298.30 | 297.24 | 297.13 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 12:15:00 | 295.85 | 296.90 | 296.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 13:15:00 | 295.35 | 296.59 | 296.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 297.15 | 296.70 | 296.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 14:15:00 | 297.15 | 296.70 | 296.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 297.15 | 296.70 | 296.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 297.15 | 296.70 | 296.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 297.95 | 296.95 | 296.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 299.05 | 296.95 | 296.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 299.40 | 297.44 | 297.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 10:15:00 | 300.45 | 298.04 | 297.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 300.35 | 300.69 | 299.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 11:00:00 | 300.35 | 300.69 | 299.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 299.80 | 300.44 | 299.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:30:00 | 300.55 | 300.44 | 299.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 300.85 | 300.52 | 299.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:45:00 | 299.00 | 300.52 | 299.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 296.30 | 300.00 | 299.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 296.30 | 300.00 | 299.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 296.85 | 299.37 | 299.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 09:15:00 | 291.10 | 295.41 | 297.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 292.35 | 291.05 | 293.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 292.35 | 291.05 | 293.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 292.35 | 291.05 | 293.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 292.10 | 291.05 | 293.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 292.90 | 291.42 | 293.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:30:00 | 293.55 | 291.42 | 293.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 295.65 | 292.27 | 293.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:30:00 | 295.60 | 292.27 | 293.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 298.90 | 293.59 | 294.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 298.90 | 293.59 | 294.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 297.15 | 294.30 | 294.35 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 299.45 | 295.33 | 294.82 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-07-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 15:15:00 | 293.95 | 295.96 | 296.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 291.75 | 295.12 | 295.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 13:15:00 | 283.95 | 282.99 | 285.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 14:00:00 | 283.95 | 282.99 | 285.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 284.20 | 283.23 | 285.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:30:00 | 284.70 | 283.23 | 285.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 280.90 | 282.95 | 284.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 10:30:00 | 279.35 | 281.80 | 284.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:15:00 | 265.38 | 273.16 | 276.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 13:15:00 | 266.25 | 264.80 | 268.65 | SL hit (close>ema200) qty=0.50 sl=264.80 alert=retest2 |

### Cycle 135 — BUY (started 2025-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 15:15:00 | 268.45 | 266.65 | 266.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 09:15:00 | 271.85 | 267.69 | 267.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 14:15:00 | 271.45 | 273.25 | 271.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 14:15:00 | 271.45 | 273.25 | 271.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 271.45 | 273.25 | 271.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 271.45 | 273.25 | 271.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 272.25 | 273.05 | 271.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:15:00 | 268.85 | 273.05 | 271.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 269.35 | 272.31 | 271.29 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 11:15:00 | 265.15 | 270.16 | 270.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 10:15:00 | 264.10 | 266.61 | 268.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 13:15:00 | 262.00 | 261.63 | 263.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-06 14:00:00 | 262.00 | 261.63 | 263.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 262.80 | 261.86 | 263.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:45:00 | 263.30 | 261.86 | 263.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 266.10 | 262.87 | 263.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:30:00 | 268.45 | 262.87 | 263.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 264.85 | 263.26 | 263.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 11:30:00 | 262.30 | 262.80 | 263.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:30:00 | 263.25 | 262.45 | 263.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 14:15:00 | 266.35 | 262.39 | 262.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 266.35 | 262.39 | 262.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 10:15:00 | 269.15 | 264.59 | 263.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 15:15:00 | 284.40 | 284.67 | 277.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 09:15:00 | 282.65 | 284.67 | 277.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 284.45 | 283.84 | 278.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 14:00:00 | 286.75 | 281.47 | 279.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 09:45:00 | 286.75 | 284.19 | 282.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 12:15:00 | 295.90 | 298.06 | 298.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 12:15:00 | 295.90 | 298.06 | 298.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 15:15:00 | 295.05 | 297.05 | 297.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 15:15:00 | 295.45 | 294.87 | 295.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 15:15:00 | 295.45 | 294.87 | 295.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 295.45 | 294.87 | 295.89 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 12:15:00 | 296.00 | 295.70 | 295.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 299.10 | 296.46 | 296.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 297.45 | 298.47 | 297.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 297.45 | 298.47 | 297.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 297.45 | 298.47 | 297.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 297.45 | 298.47 | 297.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 297.70 | 298.32 | 297.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:00:00 | 297.70 | 298.32 | 297.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 296.65 | 297.98 | 297.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 304.30 | 297.98 | 297.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 14:15:00 | 299.55 | 300.40 | 300.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 14:15:00 | 299.55 | 300.40 | 300.43 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 15:15:00 | 301.80 | 300.68 | 300.55 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 11:15:00 | 300.10 | 300.47 | 300.49 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 09:15:00 | 305.50 | 301.38 | 300.86 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 11:15:00 | 300.45 | 301.68 | 301.71 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 12:15:00 | 305.80 | 301.31 | 300.76 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 15:15:00 | 301.35 | 303.32 | 303.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 299.95 | 301.55 | 302.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 13:15:00 | 300.75 | 300.64 | 301.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 14:00:00 | 300.75 | 300.64 | 301.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 300.25 | 300.65 | 301.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:45:00 | 301.35 | 300.85 | 301.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 299.40 | 300.56 | 301.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:30:00 | 296.85 | 299.76 | 300.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 14:15:00 | 297.50 | 298.68 | 299.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 12:15:00 | 282.01 | 289.54 | 294.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 12:15:00 | 282.62 | 289.54 | 294.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 280.75 | 279.62 | 284.35 | SL hit (close>ema200) qty=0.50 sl=279.62 alert=retest2 |

### Cycle 147 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 286.50 | 283.01 | 282.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 292.15 | 285.73 | 284.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 290.05 | 293.68 | 292.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 290.05 | 293.68 | 292.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 290.05 | 293.68 | 292.03 | EMA400 retest candle locked (from upside) |

### Cycle 148 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 285.00 | 290.78 | 290.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 15:15:00 | 284.10 | 287.36 | 289.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 287.70 | 285.69 | 287.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 14:15:00 | 287.70 | 285.69 | 287.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 287.70 | 285.69 | 287.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 287.70 | 285.69 | 287.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 287.10 | 285.97 | 287.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 289.20 | 286.56 | 287.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 289.15 | 287.08 | 287.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:30:00 | 289.70 | 287.08 | 287.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 291.35 | 287.93 | 287.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 292.00 | 288.74 | 288.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 288.35 | 289.68 | 288.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 288.35 | 289.68 | 288.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 288.35 | 289.68 | 288.98 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 287.45 | 288.63 | 288.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 14:15:00 | 282.95 | 287.49 | 288.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 285.05 | 284.09 | 285.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-14 14:45:00 | 285.15 | 284.09 | 285.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 285.15 | 284.30 | 285.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 282.65 | 284.30 | 285.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 14:45:00 | 284.50 | 284.76 | 285.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 284.80 | 284.87 | 285.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 09:45:00 | 284.50 | 284.70 | 285.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 284.80 | 284.72 | 285.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:30:00 | 284.50 | 284.72 | 285.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 284.35 | 284.65 | 285.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:30:00 | 284.80 | 284.65 | 285.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 282.80 | 284.28 | 284.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:30:00 | 282.95 | 284.28 | 284.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 279.80 | 277.23 | 278.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 280.05 | 277.23 | 278.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 278.80 | 277.54 | 278.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 278.80 | 277.54 | 278.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 276.55 | 274.77 | 276.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:00:00 | 276.55 | 274.77 | 276.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 276.35 | 275.09 | 276.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:45:00 | 277.35 | 275.09 | 276.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 275.85 | 275.24 | 276.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 273.30 | 275.37 | 276.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 11:15:00 | 274.30 | 274.69 | 275.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 14:15:00 | 270.27 | 273.00 | 274.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 14:15:00 | 270.56 | 273.00 | 274.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 14:15:00 | 270.27 | 273.00 | 274.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 12:15:00 | 268.52 | 271.27 | 272.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 12:15:00 | 271.30 | 271.27 | 272.89 | SL hit (close>static) qty=0.50 sl=271.27 alert=retest2 |

### Cycle 151 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 278.20 | 273.46 | 273.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 09:15:00 | 283.75 | 278.34 | 276.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 15:15:00 | 283.75 | 283.89 | 280.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-03 09:15:00 | 286.25 | 283.89 | 280.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 280.80 | 283.05 | 281.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:00:00 | 280.80 | 283.05 | 281.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 280.60 | 282.56 | 281.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:30:00 | 281.50 | 282.56 | 281.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 279.65 | 281.98 | 281.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 279.65 | 281.98 | 281.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 280.00 | 281.58 | 280.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 279.15 | 281.58 | 280.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 279.30 | 280.63 | 280.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:15:00 | 278.85 | 280.63 | 280.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 278.50 | 280.21 | 280.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 277.35 | 279.61 | 280.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 281.20 | 279.29 | 279.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 12:15:00 | 281.20 | 279.29 | 279.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 281.20 | 279.29 | 279.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:45:00 | 281.10 | 279.29 | 279.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 280.90 | 279.61 | 279.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:30:00 | 281.85 | 279.61 | 279.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 277.00 | 273.38 | 275.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:30:00 | 275.55 | 273.38 | 275.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 277.55 | 274.21 | 275.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:30:00 | 277.80 | 274.21 | 275.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 274.20 | 274.62 | 275.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:45:00 | 273.85 | 274.66 | 275.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 11:15:00 | 273.80 | 274.66 | 275.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 15:15:00 | 277.00 | 274.28 | 274.69 | SL hit (close>static) qty=1.00 sl=276.90 alert=retest2 |

### Cycle 153 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 278.10 | 275.04 | 275.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 293.30 | 282.59 | 279.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 288.75 | 289.51 | 284.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 288.75 | 289.51 | 284.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 285.15 | 287.79 | 284.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:30:00 | 284.70 | 287.79 | 284.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 284.95 | 287.22 | 284.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:45:00 | 284.40 | 287.22 | 284.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 291.30 | 288.04 | 285.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 13:15:00 | 292.85 | 288.04 | 285.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 14:15:00 | 291.75 | 288.73 | 286.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:00:00 | 292.05 | 289.39 | 286.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 12:30:00 | 296.50 | 290.94 | 288.37 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 290.30 | 293.27 | 290.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 11:00:00 | 290.30 | 293.27 | 290.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 291.25 | 292.87 | 290.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 12:15:00 | 291.95 | 292.87 | 290.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 14:15:00 | 288.40 | 291.47 | 290.73 | SL hit (close<static) qty=1.00 sl=290.20 alert=retest2 |

### Cycle 154 — SELL (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 09:15:00 | 286.70 | 289.91 | 290.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 10:15:00 | 284.75 | 288.88 | 289.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 12:15:00 | 290.05 | 288.72 | 289.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 12:15:00 | 290.05 | 288.72 | 289.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 290.05 | 288.72 | 289.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:45:00 | 289.90 | 288.72 | 289.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 13:15:00 | 294.40 | 289.85 | 289.85 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 289.45 | 290.31 | 290.42 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 10:15:00 | 291.80 | 290.61 | 290.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 11:15:00 | 293.10 | 291.11 | 290.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 13:15:00 | 289.00 | 290.77 | 290.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 13:15:00 | 289.00 | 290.77 | 290.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 289.00 | 290.77 | 290.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:00:00 | 289.00 | 290.77 | 290.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 287.15 | 290.04 | 290.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 285.80 | 289.19 | 289.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 13:15:00 | 288.60 | 287.32 | 288.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 13:15:00 | 288.60 | 287.32 | 288.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 288.60 | 287.32 | 288.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:00:00 | 288.60 | 287.32 | 288.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 290.05 | 287.86 | 288.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:45:00 | 290.00 | 287.86 | 288.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 290.00 | 288.29 | 288.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 286.75 | 288.29 | 288.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 11:15:00 | 291.00 | 289.23 | 289.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 11:15:00 | 291.00 | 289.23 | 289.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 12:15:00 | 293.00 | 289.98 | 289.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 10:15:00 | 292.40 | 292.56 | 291.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-26 10:30:00 | 292.05 | 292.56 | 291.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 293.20 | 292.69 | 291.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:30:00 | 289.85 | 292.69 | 291.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 294.05 | 293.87 | 292.55 | EMA400 retest candle locked (from upside) |

### Cycle 160 — SELL (started 2025-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 15:15:00 | 289.55 | 291.66 | 291.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 10:15:00 | 287.70 | 290.42 | 291.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 14:15:00 | 290.40 | 289.42 | 290.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 14:15:00 | 290.40 | 289.42 | 290.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 290.40 | 289.42 | 290.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:00:00 | 290.40 | 289.42 | 290.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 290.60 | 289.65 | 290.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 288.65 | 289.65 | 290.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 11:15:00 | 274.22 | 277.40 | 280.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-08 10:15:00 | 259.78 | 270.38 | 275.54 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 161 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 264.00 | 259.94 | 259.51 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 10:15:00 | 256.40 | 259.60 | 259.73 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 13:15:00 | 261.70 | 258.91 | 258.77 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 258.50 | 258.77 | 258.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 252.05 | 256.80 | 257.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 12:15:00 | 257.70 | 256.13 | 257.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 12:15:00 | 257.70 | 256.13 | 257.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 257.70 | 256.13 | 257.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 257.70 | 256.13 | 257.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 256.80 | 256.26 | 257.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:45:00 | 257.55 | 256.26 | 257.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 258.05 | 256.62 | 257.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 258.05 | 256.62 | 257.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 257.90 | 256.88 | 257.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 261.70 | 256.88 | 257.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 264.50 | 258.40 | 257.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 10:15:00 | 269.80 | 260.68 | 259.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 12:15:00 | 267.10 | 268.13 | 265.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 13:00:00 | 267.10 | 268.13 | 265.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 268.05 | 267.60 | 265.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 12:00:00 | 270.70 | 268.16 | 266.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 15:15:00 | 267.30 | 270.69 | 270.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 267.30 | 270.69 | 270.75 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 10:15:00 | 272.50 | 271.11 | 270.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 11:15:00 | 272.60 | 271.41 | 271.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 12:15:00 | 271.20 | 271.37 | 271.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 12:15:00 | 271.20 | 271.37 | 271.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 271.20 | 271.37 | 271.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 12:30:00 | 271.15 | 271.37 | 271.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 272.05 | 271.50 | 271.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 272.05 | 271.50 | 271.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 283.90 | 275.28 | 273.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:15:00 | 286.85 | 275.28 | 273.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 12:00:00 | 286.55 | 286.00 | 281.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 14:15:00 | 286.45 | 286.06 | 282.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 15:00:00 | 286.25 | 286.10 | 282.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 284.00 | 285.66 | 283.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 12:15:00 | 287.20 | 284.49 | 283.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 10:15:00 | 293.85 | 295.49 | 295.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — SELL (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 10:15:00 | 293.85 | 295.49 | 295.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 292.55 | 294.71 | 295.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 10:15:00 | 288.25 | 288.13 | 290.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 11:00:00 | 288.25 | 288.13 | 290.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 290.60 | 288.62 | 290.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:00:00 | 290.60 | 288.62 | 290.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 287.60 | 288.42 | 290.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 286.10 | 288.49 | 289.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:45:00 | 285.05 | 287.89 | 288.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 14:00:00 | 286.20 | 286.70 | 287.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 271.80 | 276.37 | 280.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 270.80 | 276.37 | 280.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 271.89 | 276.37 | 280.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 274.00 | 273.83 | 277.00 | SL hit (close>ema200) qty=0.50 sl=273.83 alert=retest2 |

### Cycle 169 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 11:15:00 | 274.95 | 268.65 | 268.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 14:15:00 | 276.90 | 271.93 | 269.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 270.75 | 273.49 | 271.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 270.75 | 273.49 | 271.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 270.75 | 273.49 | 271.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:00:00 | 270.75 | 273.49 | 271.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 280.55 | 274.90 | 272.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 14:30:00 | 284.10 | 281.59 | 279.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-13 09:15:00 | 312.51 | 299.92 | 295.80 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 170 — SELL (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 13:15:00 | 296.00 | 300.87 | 301.35 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 302.20 | 301.29 | 301.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 09:15:00 | 304.35 | 301.90 | 301.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 09:15:00 | 300.40 | 303.49 | 302.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 09:15:00 | 300.40 | 303.49 | 302.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 300.40 | 303.49 | 302.83 | EMA400 retest candle locked (from upside) |

### Cycle 172 — SELL (started 2026-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 15:15:00 | 301.20 | 302.50 | 302.57 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 306.40 | 303.28 | 302.92 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 11:15:00 | 303.00 | 304.25 | 304.31 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2026-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 13:15:00 | 305.70 | 304.54 | 304.43 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 15:15:00 | 304.00 | 304.33 | 304.35 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 308.60 | 305.18 | 304.73 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 14:15:00 | 301.85 | 305.27 | 305.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 294.05 | 302.43 | 304.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 294.50 | 290.54 | 293.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 294.50 | 290.54 | 293.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 294.50 | 290.54 | 293.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:30:00 | 295.05 | 290.54 | 293.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 289.70 | 290.37 | 293.40 | EMA400 retest candle locked (from downside) |

### Cycle 179 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 299.00 | 294.58 | 294.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 300.20 | 295.70 | 294.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 13:15:00 | 294.55 | 295.90 | 295.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 13:15:00 | 294.55 | 295.90 | 295.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 294.55 | 295.90 | 295.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:30:00 | 294.80 | 295.90 | 295.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 291.40 | 295.00 | 294.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 291.40 | 295.00 | 294.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — SELL (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 15:15:00 | 288.20 | 293.64 | 294.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 280.35 | 290.98 | 293.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 14:15:00 | 287.40 | 286.31 | 289.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-09 15:00:00 | 287.40 | 286.31 | 289.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 290.10 | 287.31 | 289.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 12:30:00 | 287.75 | 287.97 | 289.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 294.35 | 289.82 | 289.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 294.35 | 289.82 | 289.76 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 285.25 | 289.47 | 290.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 15:15:00 | 280.00 | 283.89 | 286.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 15:15:00 | 265.55 | 265.00 | 268.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-18 09:15:00 | 268.95 | 265.00 | 268.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 268.00 | 265.60 | 268.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:00:00 | 268.00 | 265.60 | 268.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 268.15 | 266.11 | 268.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:45:00 | 268.55 | 266.11 | 268.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 272.90 | 267.47 | 268.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 12:00:00 | 272.90 | 267.47 | 268.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 278.60 | 269.70 | 269.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 286.75 | 273.11 | 271.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 275.40 | 275.99 | 273.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 09:45:00 | 274.20 | 275.99 | 273.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 280.40 | 284.12 | 280.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:30:00 | 282.35 | 284.12 | 280.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 281.55 | 283.61 | 280.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:30:00 | 279.50 | 283.61 | 280.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 11:15:00 | 281.30 | 283.15 | 280.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 12:00:00 | 281.30 | 283.15 | 280.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 12:15:00 | 281.00 | 282.72 | 280.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 12:30:00 | 281.85 | 282.72 | 280.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 13:15:00 | 282.10 | 282.59 | 280.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 283.80 | 281.69 | 280.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 09:15:00 | 312.18 | 304.71 | 303.46 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 184 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 344.10 | 346.96 | 347.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 342.15 | 345.85 | 346.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 13:15:00 | 345.90 | 345.71 | 346.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 13:15:00 | 345.90 | 345.71 | 346.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 345.90 | 345.71 | 346.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:00:00 | 345.90 | 345.71 | 346.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 348.55 | 346.28 | 346.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 15:00:00 | 348.55 | 346.28 | 346.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 347.10 | 346.44 | 346.57 | EMA400 retest candle locked (from downside) |

### Cycle 185 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 347.55 | 346.79 | 346.71 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 12:15:00 | 344.75 | 346.33 | 346.51 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 349.85 | 347.00 | 346.75 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 15:15:00 | 346.20 | 346.59 | 346.62 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 347.00 | 346.67 | 346.66 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 14:15:00 | 345.25 | 346.64 | 346.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 343.95 | 345.93 | 346.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 14:15:00 | 341.50 | 340.76 | 342.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 15:00:00 | 341.50 | 340.76 | 342.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 15:15:00 | 339.20 | 340.45 | 342.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 338.45 | 340.45 | 342.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:45:00 | 337.90 | 339.70 | 341.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 15:15:00 | 344.00 | 340.66 | 341.27 | SL hit (close>static) qty=1.00 sl=343.75 alert=retest2 |

### Cycle 191 — BUY (started 2026-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 13:15:00 | 347.25 | 342.46 | 341.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 10:15:00 | 353.50 | 345.76 | 343.66 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-11-21 09:30:00 | 353.20 | 2023-11-23 09:15:00 | 388.52 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-21 10:00:00 | 353.45 | 2023-11-23 09:15:00 | 388.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-22 11:00:00 | 356.50 | 2023-11-23 09:15:00 | 392.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-22 14:15:00 | 354.60 | 2023-11-23 09:15:00 | 390.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-23 09:15:00 | 375.75 | 2023-11-23 09:15:00 | 413.33 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-12-05 13:30:00 | 367.15 | 2023-12-07 14:15:00 | 390.00 | STOP_HIT | 1.00 | -6.22% |
| BUY | retest2 | 2023-12-11 09:30:00 | 403.75 | 2023-12-13 14:15:00 | 389.85 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest2 | 2023-12-11 15:00:00 | 404.95 | 2023-12-13 14:15:00 | 389.85 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2024-01-08 09:15:00 | 448.70 | 2024-01-09 09:15:00 | 493.57 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-08 13:15:00 | 461.20 | 2024-01-16 12:15:00 | 462.45 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2024-02-08 09:30:00 | 426.45 | 2024-02-12 09:15:00 | 442.60 | STOP_HIT | 1.00 | -3.79% |
| BUY | retest2 | 2024-02-19 13:00:00 | 439.60 | 2024-02-19 14:15:00 | 437.50 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-02-19 13:30:00 | 439.90 | 2024-02-19 14:15:00 | 437.50 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-02-19 14:15:00 | 439.20 | 2024-02-19 14:15:00 | 437.50 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2024-02-27 13:45:00 | 432.80 | 2024-02-29 09:15:00 | 411.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-27 15:00:00 | 429.05 | 2024-02-29 09:15:00 | 407.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-28 09:30:00 | 429.55 | 2024-02-29 09:15:00 | 408.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-27 13:45:00 | 432.80 | 2024-03-01 13:15:00 | 416.00 | STOP_HIT | 0.50 | 3.88% |
| SELL | retest2 | 2024-02-27 15:00:00 | 429.05 | 2024-03-01 13:15:00 | 416.00 | STOP_HIT | 0.50 | 3.04% |
| SELL | retest2 | 2024-02-28 09:30:00 | 429.55 | 2024-03-01 13:15:00 | 416.00 | STOP_HIT | 0.50 | 3.15% |
| SELL | retest2 | 2024-04-15 09:15:00 | 391.35 | 2024-04-22 09:15:00 | 400.10 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2024-04-16 09:15:00 | 386.25 | 2024-04-22 09:15:00 | 400.10 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2024-04-16 13:15:00 | 391.35 | 2024-04-22 09:15:00 | 400.10 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-04-26 09:15:00 | 427.30 | 2024-05-02 11:15:00 | 423.05 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-04-26 14:30:00 | 429.95 | 2024-05-02 11:15:00 | 423.05 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-05-02 10:30:00 | 425.60 | 2024-05-02 11:15:00 | 423.05 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-05-17 10:45:00 | 405.50 | 2024-05-21 09:15:00 | 426.60 | STOP_HIT | 1.00 | -5.20% |
| BUY | retest2 | 2024-06-26 11:30:00 | 449.80 | 2024-06-27 09:15:00 | 435.10 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2024-06-28 15:15:00 | 428.00 | 2024-07-01 09:15:00 | 447.50 | STOP_HIT | 1.00 | -4.56% |
| BUY | retest2 | 2024-07-12 13:15:00 | 471.05 | 2024-07-19 09:15:00 | 469.05 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2024-07-23 12:15:00 | 455.20 | 2024-07-23 14:15:00 | 468.15 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2024-07-26 10:15:00 | 459.65 | 2024-07-26 15:15:00 | 467.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-07-26 10:45:00 | 459.50 | 2024-07-26 15:15:00 | 467.00 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-07-31 14:30:00 | 469.30 | 2024-08-01 10:15:00 | 465.90 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-08-06 10:45:00 | 455.00 | 2024-08-08 10:15:00 | 457.80 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-08-06 13:45:00 | 455.00 | 2024-08-08 10:15:00 | 457.80 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-08-07 11:00:00 | 453.90 | 2024-08-08 10:15:00 | 457.80 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-08-07 14:15:00 | 455.25 | 2024-08-08 10:15:00 | 457.80 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-08-20 11:30:00 | 467.50 | 2024-08-20 15:15:00 | 465.05 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2024-08-27 15:15:00 | 515.20 | 2024-08-30 14:15:00 | 504.25 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-08-28 09:45:00 | 517.70 | 2024-08-30 14:15:00 | 504.25 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2024-08-29 09:15:00 | 524.15 | 2024-08-30 14:15:00 | 504.25 | STOP_HIT | 1.00 | -3.80% |
| BUY | retest2 | 2024-08-29 12:15:00 | 516.85 | 2024-08-30 14:15:00 | 504.25 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2024-09-10 09:15:00 | 531.00 | 2024-09-11 13:15:00 | 521.60 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2024-09-18 10:15:00 | 470.70 | 2024-09-20 14:15:00 | 479.75 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-09-18 10:45:00 | 470.35 | 2024-09-20 14:15:00 | 479.75 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-09-19 13:00:00 | 470.55 | 2024-09-20 14:15:00 | 479.75 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-09-19 14:00:00 | 467.40 | 2024-09-20 14:15:00 | 479.75 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2024-09-25 09:15:00 | 477.10 | 2024-09-25 09:15:00 | 471.50 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-10-01 09:15:00 | 447.90 | 2024-10-04 09:15:00 | 429.30 | PARTIAL | 0.50 | 4.15% |
| SELL | retest2 | 2024-10-01 14:00:00 | 451.25 | 2024-10-04 13:15:00 | 425.50 | PARTIAL | 0.50 | 5.71% |
| SELL | retest2 | 2024-10-03 09:15:00 | 451.90 | 2024-10-04 13:15:00 | 428.69 | PARTIAL | 0.50 | 5.14% |
| SELL | retest2 | 2024-10-03 10:00:00 | 449.30 | 2024-10-04 13:15:00 | 426.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 09:15:00 | 447.90 | 2024-10-08 11:15:00 | 431.70 | STOP_HIT | 0.50 | 3.62% |
| SELL | retest2 | 2024-10-01 14:00:00 | 451.25 | 2024-10-08 11:15:00 | 431.70 | STOP_HIT | 0.50 | 4.33% |
| SELL | retest2 | 2024-10-03 09:15:00 | 451.90 | 2024-10-08 11:15:00 | 431.70 | STOP_HIT | 0.50 | 4.47% |
| SELL | retest2 | 2024-10-03 10:00:00 | 449.30 | 2024-10-08 11:15:00 | 431.70 | STOP_HIT | 0.50 | 3.92% |
| SELL | retest2 | 2024-10-04 14:00:00 | 427.45 | 2024-10-09 09:15:00 | 438.35 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2024-10-04 15:00:00 | 428.00 | 2024-10-09 09:15:00 | 438.35 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2024-10-07 09:45:00 | 427.85 | 2024-10-09 09:15:00 | 438.35 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2024-10-07 11:45:00 | 429.10 | 2024-10-09 09:15:00 | 438.35 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2024-10-14 10:30:00 | 422.80 | 2024-10-18 10:15:00 | 426.95 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-10-14 11:15:00 | 423.95 | 2024-10-18 10:15:00 | 426.95 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-10-15 11:15:00 | 423.60 | 2024-10-18 10:15:00 | 426.95 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-10-16 09:45:00 | 421.00 | 2024-10-18 10:15:00 | 426.95 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-10-25 09:45:00 | 399.65 | 2024-10-28 11:15:00 | 408.20 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-10-28 09:15:00 | 399.35 | 2024-10-28 11:15:00 | 408.20 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2024-11-06 13:15:00 | 389.90 | 2024-11-07 15:15:00 | 370.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-06 14:00:00 | 390.00 | 2024-11-07 15:15:00 | 370.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-06 14:45:00 | 388.20 | 2024-11-07 15:15:00 | 368.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 09:45:00 | 389.10 | 2024-11-07 15:15:00 | 369.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 11:15:00 | 386.90 | 2024-11-07 15:15:00 | 367.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-06 13:15:00 | 389.90 | 2024-11-08 15:15:00 | 381.90 | STOP_HIT | 0.50 | 2.05% |
| SELL | retest2 | 2024-11-06 14:00:00 | 390.00 | 2024-11-08 15:15:00 | 381.90 | STOP_HIT | 0.50 | 2.08% |
| SELL | retest2 | 2024-11-06 14:45:00 | 388.20 | 2024-11-08 15:15:00 | 381.90 | STOP_HIT | 0.50 | 1.62% |
| SELL | retest2 | 2024-11-07 09:45:00 | 389.10 | 2024-11-08 15:15:00 | 381.90 | STOP_HIT | 0.50 | 1.85% |
| SELL | retest2 | 2024-11-07 11:15:00 | 386.90 | 2024-11-08 15:15:00 | 381.90 | STOP_HIT | 0.50 | 1.29% |
| BUY | retest1 | 2024-12-04 09:15:00 | 273.50 | 2024-12-04 09:15:00 | 287.18 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-12-04 09:15:00 | 273.50 | 2024-12-05 09:15:00 | 270.40 | STOP_HIT | 0.50 | -1.13% |
| BUY | retest2 | 2024-12-05 12:15:00 | 273.50 | 2024-12-06 09:15:00 | 267.25 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2024-12-05 15:00:00 | 273.10 | 2024-12-06 09:15:00 | 267.25 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-12-12 11:30:00 | 253.95 | 2024-12-18 13:15:00 | 251.60 | STOP_HIT | 1.00 | 0.93% |
| SELL | retest2 | 2024-12-12 13:45:00 | 253.75 | 2024-12-18 13:15:00 | 251.60 | STOP_HIT | 1.00 | 0.85% |
| SELL | retest2 | 2024-12-12 15:00:00 | 250.00 | 2024-12-18 13:15:00 | 251.60 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-12-18 11:00:00 | 253.50 | 2024-12-18 13:15:00 | 251.60 | STOP_HIT | 1.00 | 0.75% |
| BUY | retest2 | 2024-12-20 15:00:00 | 259.45 | 2024-12-24 09:15:00 | 253.00 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2024-12-31 09:15:00 | 248.55 | 2024-12-31 13:15:00 | 253.25 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-12-31 10:00:00 | 249.80 | 2024-12-31 13:15:00 | 253.25 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-12-31 11:15:00 | 250.15 | 2024-12-31 13:15:00 | 253.25 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-12-31 12:15:00 | 250.00 | 2024-12-31 13:15:00 | 253.25 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-01-08 11:15:00 | 247.60 | 2025-01-09 10:15:00 | 249.35 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-02-12 14:15:00 | 204.00 | 2025-02-13 09:15:00 | 216.81 | STOP_HIT | 1.00 | -6.28% |
| SELL | retest2 | 2025-02-12 14:45:00 | 203.66 | 2025-02-13 09:15:00 | 216.81 | STOP_HIT | 1.00 | -6.46% |
| BUY | retest2 | 2025-02-20 10:30:00 | 230.33 | 2025-02-24 09:15:00 | 221.32 | STOP_HIT | 1.00 | -3.91% |
| BUY | retest2 | 2025-03-24 09:15:00 | 223.76 | 2025-03-27 13:15:00 | 229.61 | STOP_HIT | 1.00 | 2.61% |
| SELL | retest2 | 2025-04-01 09:15:00 | 232.35 | 2025-04-01 10:15:00 | 233.67 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-04-03 10:30:00 | 240.77 | 2025-04-04 12:15:00 | 234.17 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest1 | 2025-04-09 09:15:00 | 221.44 | 2025-04-11 09:15:00 | 223.54 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-04-21 09:30:00 | 240.30 | 2025-04-23 09:15:00 | 230.49 | STOP_HIT | 1.00 | -4.08% |
| BUY | retest2 | 2025-05-02 11:15:00 | 249.66 | 2025-05-07 11:15:00 | 248.01 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-05-02 12:30:00 | 249.69 | 2025-05-07 11:15:00 | 248.01 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-05-12 10:30:00 | 248.34 | 2025-05-13 09:15:00 | 251.60 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-05-16 10:15:00 | 261.00 | 2025-05-23 09:15:00 | 287.10 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-17 09:15:00 | 307.25 | 2025-06-20 12:15:00 | 308.20 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-06-17 11:15:00 | 306.70 | 2025-06-20 12:15:00 | 308.20 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-06-20 11:45:00 | 308.70 | 2025-06-20 12:15:00 | 308.20 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2025-07-03 15:15:00 | 298.05 | 2025-07-08 09:15:00 | 300.40 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-07-07 09:45:00 | 297.45 | 2025-07-08 09:15:00 | 300.40 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-07-23 10:30:00 | 279.35 | 2025-07-25 10:15:00 | 265.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 10:30:00 | 279.35 | 2025-07-28 13:15:00 | 266.25 | STOP_HIT | 0.50 | 4.69% |
| SELL | retest2 | 2025-08-07 11:30:00 | 262.30 | 2025-08-11 14:15:00 | 266.35 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-08-08 09:30:00 | 263.25 | 2025-08-11 14:15:00 | 266.35 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-08-18 14:00:00 | 286.75 | 2025-08-26 12:15:00 | 295.90 | STOP_HIT | 1.00 | 3.19% |
| BUY | retest2 | 2025-08-20 09:45:00 | 286.75 | 2025-08-26 12:15:00 | 295.90 | STOP_HIT | 1.00 | 3.19% |
| BUY | retest2 | 2025-09-03 09:15:00 | 304.30 | 2025-09-05 14:15:00 | 299.55 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-09-25 09:30:00 | 296.85 | 2025-09-26 12:15:00 | 282.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 14:15:00 | 297.50 | 2025-09-26 12:15:00 | 282.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 09:30:00 | 296.85 | 2025-09-30 09:15:00 | 280.75 | STOP_HIT | 0.50 | 5.42% |
| SELL | retest2 | 2025-09-25 14:15:00 | 297.50 | 2025-09-30 09:15:00 | 280.75 | STOP_HIT | 0.50 | 5.63% |
| SELL | retest2 | 2025-10-15 09:15:00 | 282.65 | 2025-10-27 14:15:00 | 270.27 | PARTIAL | 0.50 | 4.38% |
| SELL | retest2 | 2025-10-15 14:45:00 | 284.50 | 2025-10-27 14:15:00 | 270.56 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2025-10-16 09:15:00 | 284.80 | 2025-10-27 14:15:00 | 270.27 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2025-10-16 09:45:00 | 284.50 | 2025-10-28 12:15:00 | 268.52 | PARTIAL | 0.50 | 5.62% |
| SELL | retest2 | 2025-10-15 09:15:00 | 282.65 | 2025-10-28 12:15:00 | 271.30 | STOP_HIT | 0.50 | 4.02% |
| SELL | retest2 | 2025-10-15 14:45:00 | 284.50 | 2025-10-28 12:15:00 | 271.30 | STOP_HIT | 0.50 | 4.64% |
| SELL | retest2 | 2025-10-16 09:15:00 | 284.80 | 2025-10-28 12:15:00 | 271.30 | STOP_HIT | 0.50 | 4.74% |
| SELL | retest2 | 2025-10-16 09:45:00 | 284.50 | 2025-10-28 12:15:00 | 271.30 | STOP_HIT | 0.50 | 4.64% |
| SELL | retest2 | 2025-10-27 09:15:00 | 273.30 | 2025-10-29 12:15:00 | 278.20 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-10-27 11:15:00 | 274.30 | 2025-10-29 12:15:00 | 278.20 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-11-11 09:45:00 | 273.85 | 2025-11-11 15:15:00 | 277.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-11-11 11:15:00 | 273.80 | 2025-11-11 15:15:00 | 277.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-11-14 13:15:00 | 292.85 | 2025-11-18 14:15:00 | 288.40 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-11-14 14:15:00 | 291.75 | 2025-11-19 09:15:00 | 286.70 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-11-14 15:00:00 | 292.05 | 2025-11-19 09:15:00 | 286.70 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-11-17 12:30:00 | 296.50 | 2025-11-19 09:15:00 | 286.70 | STOP_HIT | 1.00 | -3.31% |
| BUY | retest2 | 2025-11-18 12:15:00 | 291.95 | 2025-11-19 09:15:00 | 286.70 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-11-25 09:15:00 | 286.75 | 2025-11-25 11:15:00 | 291.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-12-01 09:15:00 | 288.65 | 2025-12-05 11:15:00 | 274.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 09:15:00 | 288.65 | 2025-12-08 10:15:00 | 259.78 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-12-23 12:00:00 | 270.70 | 2025-12-26 15:15:00 | 267.30 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-12-30 10:15:00 | 286.85 | 2026-01-09 10:15:00 | 293.85 | STOP_HIT | 1.00 | 2.44% |
| BUY | retest2 | 2025-12-31 12:00:00 | 286.55 | 2026-01-09 10:15:00 | 293.85 | STOP_HIT | 1.00 | 2.55% |
| BUY | retest2 | 2025-12-31 14:15:00 | 286.45 | 2026-01-09 10:15:00 | 293.85 | STOP_HIT | 1.00 | 2.58% |
| BUY | retest2 | 2025-12-31 15:00:00 | 286.25 | 2026-01-09 10:15:00 | 293.85 | STOP_HIT | 1.00 | 2.66% |
| BUY | retest2 | 2026-01-02 12:15:00 | 287.20 | 2026-01-09 10:15:00 | 293.85 | STOP_HIT | 1.00 | 2.32% |
| SELL | retest2 | 2026-01-16 09:15:00 | 286.10 | 2026-01-21 10:15:00 | 271.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:45:00 | 285.05 | 2026-01-21 10:15:00 | 270.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 14:00:00 | 286.20 | 2026-01-21 10:15:00 | 271.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:15:00 | 286.10 | 2026-01-22 09:15:00 | 274.00 | STOP_HIT | 0.50 | 4.23% |
| SELL | retest2 | 2026-01-16 09:45:00 | 285.05 | 2026-01-22 09:15:00 | 274.00 | STOP_HIT | 0.50 | 3.88% |
| SELL | retest2 | 2026-01-16 14:00:00 | 286.20 | 2026-01-22 09:15:00 | 274.00 | STOP_HIT | 0.50 | 4.26% |
| BUY | retest2 | 2026-02-06 14:30:00 | 284.10 | 2026-02-13 09:15:00 | 312.51 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-10 12:30:00 | 287.75 | 2026-03-11 09:15:00 | 294.35 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2026-03-24 09:15:00 | 283.80 | 2026-04-08 09:15:00 | 312.18 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-05-05 09:15:00 | 338.45 | 2026-05-05 15:15:00 | 344.00 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-05-05 09:45:00 | 337.90 | 2026-05-05 15:15:00 | 344.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-05-06 10:00:00 | 338.30 | 2026-05-06 12:15:00 | 345.60 | STOP_HIT | 1.00 | -2.16% |
