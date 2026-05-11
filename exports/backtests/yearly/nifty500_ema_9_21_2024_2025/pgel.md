# PG Electroplast Ltd. (PGEL)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 530.45
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 154 |
| ALERT1 | 99 |
| ALERT2 | 97 |
| ALERT2_SKIP | 47 |
| ALERT3 | 234 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 111 |
| PARTIAL | 19 |
| TARGET_HIT | 23 |
| STOP_HIT | 93 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 135 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 58 / 77
- **Target hits / Stop hits / Partials:** 23 / 93 / 19
- **Avg / median % per leg:** 1.15% / -1.08%
- **Sum % (uncompounded):** 154.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 58 | 22 | 37.9% | 10 | 48 | 0 | 0.81% | 47.2% |
| BUY @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.13% | -10.6% |
| BUY @ 3rd Alert (retest2) | 53 | 22 | 41.5% | 10 | 43 | 0 | 1.09% | 57.9% |
| SELL (all) | 77 | 36 | 46.8% | 13 | 45 | 19 | 1.40% | 107.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 77 | 36 | 46.8% | 13 | 45 | 19 | 1.40% | 107.8% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.13% | -10.6% |
| retest2 (combined) | 130 | 58 | 44.6% | 23 | 88 | 19 | 1.27% | 165.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 11:15:00 | 211.14 | 204.87 | 204.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 09:15:00 | 212.83 | 210.13 | 208.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 14:15:00 | 211.38 | 211.66 | 209.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 14:45:00 | 211.01 | 211.66 | 209.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 14:15:00 | 212.24 | 213.24 | 212.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 15:00:00 | 212.24 | 213.24 | 212.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 15:15:00 | 211.50 | 212.89 | 212.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-18 09:15:00 | 214.12 | 212.89 | 212.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-21 09:15:00 | 235.53 | 218.82 | 215.42 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 14:15:00 | 253.07 | 255.13 | 255.39 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 12:15:00 | 256.70 | 255.40 | 255.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 13:15:00 | 260.00 | 256.32 | 255.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 263.39 | 268.82 | 264.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 263.39 | 268.82 | 264.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 263.39 | 268.82 | 264.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 262.19 | 268.82 | 264.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 251.36 | 265.33 | 263.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 251.36 | 265.33 | 263.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 246.50 | 261.57 | 261.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 234.89 | 250.83 | 256.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 256.01 | 250.41 | 254.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 12:15:00 | 256.01 | 250.41 | 254.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 256.01 | 250.41 | 254.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:00:00 | 256.01 | 250.41 | 254.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 256.15 | 251.56 | 254.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:45:00 | 256.19 | 251.56 | 254.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 258.90 | 253.03 | 254.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 15:00:00 | 258.90 | 253.03 | 254.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 12:15:00 | 257.81 | 256.21 | 256.05 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-06 14:15:00 | 254.98 | 255.88 | 255.92 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 15:15:00 | 256.40 | 255.98 | 255.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 280.00 | 260.78 | 258.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 13:15:00 | 294.30 | 294.65 | 289.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 14:00:00 | 294.30 | 294.65 | 289.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 15:15:00 | 291.30 | 293.49 | 290.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 09:15:00 | 293.45 | 293.49 | 290.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-13 10:15:00 | 289.57 | 292.15 | 290.00 | SL hit (close<static) qty=1.00 sl=290.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 14:15:00 | 369.51 | 374.43 | 374.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 09:15:00 | 363.49 | 371.85 | 373.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 12:15:00 | 371.93 | 369.69 | 371.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 12:15:00 | 371.93 | 369.69 | 371.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 12:15:00 | 371.93 | 369.69 | 371.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 13:00:00 | 371.93 | 369.69 | 371.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 13:15:00 | 370.53 | 369.86 | 371.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 13:45:00 | 370.21 | 369.86 | 371.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 371.42 | 370.05 | 371.40 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2024-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 09:15:00 | 380.55 | 372.90 | 372.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 10:15:00 | 384.15 | 375.15 | 373.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 11:15:00 | 390.10 | 391.21 | 384.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-10 12:00:00 | 390.10 | 391.21 | 384.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 376.50 | 391.05 | 387.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:45:00 | 380.00 | 391.05 | 387.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 380.05 | 388.85 | 386.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:45:00 | 377.80 | 388.85 | 386.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 390.25 | 391.95 | 389.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:30:00 | 388.05 | 391.95 | 389.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 378.55 | 389.27 | 388.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 11:00:00 | 378.55 | 389.27 | 388.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 11:15:00 | 376.45 | 386.71 | 387.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 09:15:00 | 364.55 | 377.39 | 382.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 09:15:00 | 370.40 | 369.97 | 375.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 370.40 | 369.97 | 375.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 370.40 | 369.97 | 375.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 09:15:00 | 360.10 | 366.24 | 370.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 12:45:00 | 359.55 | 361.45 | 366.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-22 12:15:00 | 373.10 | 362.75 | 361.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 12:15:00 | 373.10 | 362.75 | 361.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 13:15:00 | 377.00 | 369.79 | 366.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 14:15:00 | 441.00 | 441.68 | 426.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 15:00:00 | 441.00 | 441.68 | 426.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 11:15:00 | 444.05 | 449.21 | 441.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 11:30:00 | 441.10 | 449.21 | 441.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 443.95 | 447.84 | 442.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 15:00:00 | 443.95 | 447.84 | 442.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 443.40 | 446.95 | 443.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:30:00 | 439.00 | 444.22 | 442.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 433.05 | 441.99 | 441.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 10:45:00 | 431.80 | 441.99 | 441.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 12:15:00 | 439.45 | 440.68 | 440.79 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 14:15:00 | 444.00 | 441.06 | 440.93 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 14:15:00 | 436.00 | 440.65 | 441.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 428.15 | 437.40 | 439.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 14:15:00 | 434.75 | 432.72 | 435.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-02 15:00:00 | 434.75 | 432.72 | 435.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 428.20 | 419.72 | 425.42 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2024-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 12:15:00 | 431.85 | 426.49 | 425.96 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 15:15:00 | 424.00 | 426.62 | 426.75 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-08-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 12:15:00 | 427.60 | 426.75 | 426.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 09:15:00 | 437.95 | 430.37 | 428.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 12:15:00 | 443.70 | 444.02 | 439.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 13:00:00 | 443.70 | 444.02 | 439.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 440.05 | 443.23 | 439.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 440.05 | 443.23 | 439.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 430.60 | 440.70 | 438.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:00:00 | 430.60 | 440.70 | 438.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 432.80 | 439.12 | 437.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:15:00 | 418.30 | 439.12 | 437.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 10:15:00 | 431.85 | 436.61 | 436.93 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 12:15:00 | 439.05 | 437.15 | 437.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 09:15:00 | 458.40 | 441.87 | 439.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 14:15:00 | 525.90 | 529.73 | 517.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 15:00:00 | 525.90 | 529.73 | 517.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 539.65 | 552.96 | 544.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:45:00 | 536.25 | 552.96 | 544.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 523.55 | 547.08 | 542.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:00:00 | 523.55 | 547.08 | 542.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 13:15:00 | 528.10 | 539.04 | 539.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 11:15:00 | 515.75 | 526.90 | 532.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 503.45 | 497.29 | 505.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-30 10:00:00 | 503.45 | 497.29 | 505.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 507.90 | 499.42 | 505.28 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 14:15:00 | 526.00 | 508.74 | 508.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 15:15:00 | 530.90 | 513.18 | 510.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 11:15:00 | 516.10 | 517.06 | 512.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 12:00:00 | 516.10 | 517.06 | 512.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 514.10 | 516.47 | 513.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 14:00:00 | 518.30 | 516.83 | 513.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 15:15:00 | 518.00 | 516.27 | 513.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-04 14:15:00 | 570.13 | 553.00 | 540.18 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 10:15:00 | 610.65 | 614.57 | 614.97 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 13:15:00 | 616.25 | 615.31 | 615.22 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 14:15:00 | 612.30 | 614.70 | 614.96 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 09:15:00 | 630.00 | 617.65 | 616.25 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 09:15:00 | 601.45 | 613.88 | 615.30 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 15:15:00 | 624.00 | 615.55 | 615.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 09:15:00 | 639.05 | 620.25 | 617.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 09:15:00 | 622.35 | 631.47 | 626.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 622.35 | 631.47 | 626.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 622.35 | 631.47 | 626.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:00:00 | 622.35 | 631.47 | 626.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 614.80 | 628.13 | 625.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:45:00 | 617.40 | 628.13 | 625.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2024-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 12:15:00 | 611.95 | 623.13 | 623.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 13:15:00 | 605.95 | 619.69 | 621.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 625.35 | 620.19 | 621.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 625.35 | 620.19 | 621.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 625.35 | 620.19 | 621.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:30:00 | 625.60 | 620.19 | 621.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 617.70 | 619.69 | 621.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 11:30:00 | 616.45 | 618.64 | 620.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 09:15:00 | 615.35 | 617.05 | 618.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 11:30:00 | 617.00 | 616.13 | 617.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 13:45:00 | 616.95 | 615.28 | 617.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 618.65 | 614.95 | 616.48 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-24 11:15:00 | 623.65 | 617.50 | 617.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 11:15:00 | 623.65 | 617.50 | 617.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 12:15:00 | 625.50 | 619.10 | 618.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 09:15:00 | 661.10 | 663.04 | 647.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-26 10:00:00 | 661.10 | 663.04 | 647.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 664.80 | 667.67 | 661.38 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 11:15:00 | 646.30 | 657.21 | 658.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 10:15:00 | 639.90 | 646.98 | 652.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 13:15:00 | 615.80 | 614.15 | 621.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 13:45:00 | 612.80 | 614.15 | 621.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 608.55 | 613.03 | 620.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:30:00 | 617.75 | 613.03 | 620.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 558.50 | 602.31 | 614.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:30:00 | 548.05 | 593.40 | 609.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 11:45:00 | 550.75 | 586.47 | 604.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 12:15:00 | 555.95 | 586.47 | 604.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 15:15:00 | 608.90 | 592.17 | 591.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 608.90 | 592.17 | 591.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 616.00 | 596.93 | 593.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 613.50 | 614.98 | 608.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 13:00:00 | 613.50 | 614.98 | 608.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 616.60 | 615.00 | 609.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 14:45:00 | 614.60 | 615.00 | 609.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 614.85 | 614.81 | 610.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 13:30:00 | 620.80 | 614.13 | 611.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:45:00 | 623.00 | 615.55 | 612.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 09:15:00 | 603.30 | 615.41 | 614.39 | SL hit (close<static) qty=1.00 sl=608.90 alert=retest2 |

### Cycle 32 — SELL (started 2024-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 11:15:00 | 607.50 | 612.54 | 613.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-15 12:15:00 | 605.05 | 611.04 | 612.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 13:15:00 | 615.65 | 611.96 | 612.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 13:15:00 | 615.65 | 611.96 | 612.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 615.65 | 611.96 | 612.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:00:00 | 615.65 | 611.96 | 612.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 614.05 | 612.38 | 612.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:45:00 | 616.35 | 612.38 | 612.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 15:15:00 | 611.75 | 612.26 | 612.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 09:15:00 | 607.30 | 612.26 | 612.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 09:30:00 | 610.35 | 600.16 | 604.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 10:45:00 | 606.40 | 601.13 | 604.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 576.93 | 600.00 | 602.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-18 09:15:00 | 611.05 | 600.00 | 602.01 | SL hit (close>static) qty=0.50 sl=600.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 10:15:00 | 617.90 | 603.58 | 603.46 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 11:15:00 | 589.45 | 603.46 | 605.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 15:15:00 | 584.80 | 594.46 | 599.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 588.10 | 582.24 | 589.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 09:15:00 | 588.10 | 582.24 | 589.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 588.10 | 582.24 | 589.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:00:00 | 588.10 | 582.24 | 589.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 582.70 | 582.33 | 588.75 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2024-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 11:15:00 | 597.20 | 589.80 | 589.56 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 571.65 | 587.54 | 588.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 10:15:00 | 554.70 | 580.97 | 585.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-25 15:15:00 | 581.00 | 571.47 | 578.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 15:15:00 | 581.00 | 571.47 | 578.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 15:15:00 | 581.00 | 571.47 | 578.21 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 11:15:00 | 588.00 | 580.81 | 579.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 11:15:00 | 605.40 | 590.87 | 585.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 611.75 | 624.73 | 612.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 611.75 | 624.73 | 612.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 611.75 | 624.73 | 612.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 09:15:00 | 642.50 | 631.26 | 625.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-11 09:15:00 | 706.75 | 684.58 | 672.55 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 14:15:00 | 647.65 | 670.64 | 670.86 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 687.05 | 639.69 | 639.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 13:15:00 | 722.50 | 709.27 | 702.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 10:15:00 | 738.50 | 742.31 | 730.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-03 11:00:00 | 738.50 | 742.31 | 730.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 754.50 | 751.88 | 740.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:00:00 | 754.50 | 751.88 | 740.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 764.30 | 754.36 | 742.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 14:00:00 | 774.90 | 760.14 | 748.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 09:15:00 | 779.45 | 763.90 | 752.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-09 11:15:00 | 852.39 | 822.13 | 802.84 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 09:15:00 | 900.70 | 942.85 | 947.15 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 15:15:00 | 950.10 | 947.89 | 947.74 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 09:15:00 | 941.05 | 946.53 | 947.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 12:15:00 | 924.15 | 939.89 | 943.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 11:15:00 | 928.65 | 922.97 | 931.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-23 11:15:00 | 928.65 | 922.97 | 931.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 928.65 | 922.97 | 931.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:45:00 | 926.20 | 922.97 | 931.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 12:15:00 | 940.80 | 926.54 | 932.65 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2024-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 15:15:00 | 953.70 | 938.33 | 937.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 09:15:00 | 974.85 | 945.63 | 940.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-24 14:15:00 | 965.45 | 966.99 | 954.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-24 15:00:00 | 965.45 | 966.99 | 954.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 968.45 | 967.09 | 956.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 09:45:00 | 982.15 | 967.09 | 956.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 959.75 | 966.12 | 960.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 959.75 | 966.12 | 960.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 964.10 | 965.72 | 960.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 09:15:00 | 979.75 | 965.72 | 960.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 11:15:00 | 977.30 | 983.60 | 977.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 14:15:00 | 984.95 | 977.94 | 975.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 15:15:00 | 994.00 | 1007.01 | 1007.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 15:15:00 | 994.00 | 1007.01 | 1007.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 10:15:00 | 977.40 | 998.85 | 1003.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 14:15:00 | 853.65 | 847.30 | 867.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 14:15:00 | 853.65 | 847.30 | 867.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 853.65 | 847.30 | 867.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 15:00:00 | 853.65 | 847.30 | 867.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 827.70 | 843.80 | 862.83 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 900.15 | 869.42 | 866.45 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-01-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 13:15:00 | 861.55 | 873.15 | 873.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 10:15:00 | 855.00 | 864.74 | 869.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 782.50 | 769.30 | 795.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 782.50 | 769.30 | 795.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 782.50 | 769.30 | 795.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 784.10 | 769.30 | 795.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 792.25 | 773.89 | 795.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:00:00 | 792.25 | 773.89 | 795.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 795.85 | 778.28 | 795.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:30:00 | 791.05 | 778.28 | 795.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 782.00 | 779.03 | 794.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 15:00:00 | 767.10 | 776.56 | 790.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:45:00 | 762.70 | 775.69 | 787.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 13:15:00 | 768.10 | 775.63 | 784.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 728.75 | 750.45 | 769.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 724.57 | 750.45 | 769.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 729.69 | 750.45 | 769.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-28 09:15:00 | 690.39 | 714.45 | 739.70 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 738.65 | 710.72 | 707.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 767.00 | 729.59 | 719.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 09:15:00 | 788.25 | 789.76 | 771.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-03 09:30:00 | 785.50 | 789.76 | 771.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 13:15:00 | 780.35 | 787.73 | 776.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 14:00:00 | 780.35 | 787.73 | 776.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 15:15:00 | 782.80 | 786.65 | 777.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:15:00 | 796.45 | 786.65 | 777.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-07 09:15:00 | 876.10 | 854.11 | 837.32 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 11:15:00 | 802.80 | 839.07 | 841.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-12 09:15:00 | 772.95 | 805.05 | 817.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 808.00 | 805.64 | 816.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 10:45:00 | 803.60 | 805.64 | 816.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 819.00 | 808.31 | 817.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:45:00 | 822.70 | 808.31 | 817.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 833.40 | 813.33 | 818.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 833.40 | 813.33 | 818.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 819.50 | 814.56 | 818.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 14:15:00 | 811.05 | 814.56 | 818.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-13 09:15:00 | 834.85 | 822.33 | 821.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 09:15:00 | 834.85 | 822.33 | 821.51 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 14:15:00 | 813.75 | 821.73 | 821.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 792.85 | 814.52 | 818.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 13:15:00 | 770.15 | 766.31 | 781.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 13:30:00 | 777.90 | 766.31 | 781.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 779.25 | 768.90 | 781.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 779.25 | 768.90 | 781.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 779.05 | 770.93 | 781.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 09:15:00 | 772.30 | 770.93 | 781.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-18 14:15:00 | 733.68 | 755.94 | 769.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-19 10:15:00 | 762.25 | 753.73 | 764.42 | SL hit (close>ema200) qty=0.50 sl=753.73 alert=retest2 |

### Cycle 51 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 801.80 | 774.30 | 772.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 809.10 | 781.26 | 775.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 812.35 | 812.58 | 798.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 812.35 | 812.58 | 798.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 803.80 | 819.05 | 810.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 09:30:00 | 816.25 | 811.04 | 808.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 10:15:00 | 788.70 | 808.21 | 810.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 10:15:00 | 788.70 | 808.21 | 810.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 15:15:00 | 785.05 | 796.38 | 802.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 14:15:00 | 797.75 | 773.76 | 785.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 14:15:00 | 797.75 | 773.76 | 785.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 797.75 | 773.76 | 785.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 797.75 | 773.76 | 785.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 779.75 | 774.96 | 784.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:15:00 | 806.25 | 774.96 | 784.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 795.95 | 779.16 | 785.73 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2025-03-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 12:15:00 | 822.15 | 792.53 | 790.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 09:15:00 | 824.90 | 804.68 | 797.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 14:15:00 | 846.50 | 847.90 | 832.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-05 15:00:00 | 846.50 | 847.90 | 832.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 874.30 | 891.42 | 881.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:30:00 | 864.80 | 891.42 | 881.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 885.15 | 890.16 | 881.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 10:30:00 | 875.35 | 890.16 | 881.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 882.00 | 888.53 | 881.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 13:15:00 | 890.20 | 886.94 | 881.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 09:15:00 | 874.50 | 887.24 | 884.04 | SL hit (close<static) qty=1.00 sl=877.10 alert=retest2 |

### Cycle 54 — SELL (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 11:15:00 | 859.25 | 879.61 | 881.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 09:15:00 | 845.90 | 862.80 | 871.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 870.00 | 843.44 | 848.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 870.00 | 843.44 | 848.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 870.00 | 843.44 | 848.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:45:00 | 870.00 | 843.44 | 848.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 867.50 | 848.25 | 850.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 11:15:00 | 871.60 | 848.25 | 850.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 874.35 | 853.47 | 852.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 12:15:00 | 883.70 | 859.52 | 855.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 15:15:00 | 924.95 | 930.79 | 914.63 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 09:15:00 | 951.90 | 930.79 | 914.63 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 896.45 | 933.82 | 927.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-25 09:15:00 | 896.45 | 933.82 | 927.28 | SL hit (close<ema400) qty=1.00 sl=927.28 alert=retest1 |

### Cycle 56 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 11:15:00 | 898.40 | 921.32 | 922.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 15:15:00 | 878.05 | 901.30 | 911.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 09:15:00 | 909.80 | 903.00 | 911.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 09:15:00 | 909.80 | 903.00 | 911.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 909.80 | 903.00 | 911.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:00:00 | 909.80 | 903.00 | 911.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 914.95 | 905.39 | 911.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:30:00 | 919.75 | 905.39 | 911.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 907.00 | 905.71 | 911.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 12:30:00 | 899.05 | 903.95 | 910.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 14:15:00 | 925.00 | 894.08 | 897.24 | SL hit (close>static) qty=1.00 sl=916.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 929.00 | 901.07 | 900.12 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 13:15:00 | 899.20 | 912.20 | 913.20 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 912.00 | 910.42 | 910.33 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 11:15:00 | 906.40 | 909.63 | 909.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-03 12:15:00 | 902.60 | 908.22 | 909.32 | Break + close below crossover candle low |

### Cycle 61 — BUY (started 2025-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 13:15:00 | 927.25 | 912.03 | 910.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 941.25 | 917.87 | 913.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 909.95 | 919.99 | 915.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 909.95 | 919.99 | 915.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 909.95 | 919.99 | 915.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:45:00 | 908.00 | 919.99 | 915.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 910.00 | 917.99 | 915.11 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 891.40 | 910.25 | 911.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 869.05 | 902.01 | 908.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 859.50 | 833.25 | 856.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 859.50 | 833.25 | 856.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 859.50 | 833.25 | 856.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:30:00 | 860.50 | 833.25 | 856.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 873.10 | 841.22 | 858.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 847.30 | 857.29 | 861.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 12:00:00 | 851.40 | 852.84 | 857.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 12:45:00 | 852.00 | 852.72 | 857.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 13:30:00 | 850.00 | 851.41 | 856.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 903.65 | 862.25 | 860.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 903.65 | 862.25 | 860.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 954.50 | 911.42 | 889.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 13:15:00 | 958.50 | 959.57 | 946.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 13:45:00 | 958.70 | 959.57 | 946.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 14:15:00 | 948.50 | 957.36 | 947.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 15:00:00 | 948.50 | 957.36 | 947.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 950.00 | 955.89 | 947.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:15:00 | 954.65 | 955.89 | 947.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-21 12:15:00 | 937.20 | 949.53 | 946.85 | SL hit (close<static) qty=1.00 sl=943.15 alert=retest2 |

### Cycle 64 — SELL (started 2025-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-21 14:15:00 | 933.25 | 943.51 | 944.40 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 10:15:00 | 983.05 | 951.56 | 947.79 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 09:15:00 | 940.75 | 954.24 | 955.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 12:15:00 | 933.05 | 945.75 | 950.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 911.65 | 893.03 | 911.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 911.65 | 893.03 | 911.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 911.65 | 893.03 | 911.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 911.65 | 893.03 | 911.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 900.55 | 894.54 | 910.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 11:15:00 | 892.80 | 894.54 | 910.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 09:30:00 | 887.70 | 876.98 | 886.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 13:15:00 | 848.16 | 869.74 | 880.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 14:15:00 | 843.32 | 864.96 | 877.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-05-06 13:15:00 | 803.52 | 816.43 | 829.20 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 67 — BUY (started 2025-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 14:15:00 | 846.00 | 827.31 | 826.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 10:15:00 | 855.00 | 838.74 | 832.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 840.20 | 843.18 | 836.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 14:00:00 | 840.20 | 843.18 | 836.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 836.15 | 841.77 | 836.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 836.15 | 841.77 | 836.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 819.00 | 837.22 | 834.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 803.05 | 837.22 | 834.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 799.50 | 829.67 | 831.59 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 857.00 | 828.56 | 827.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 893.75 | 851.99 | 839.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 853.85 | 854.85 | 844.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 11:00:00 | 853.85 | 854.85 | 844.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 11:15:00 | 843.85 | 852.65 | 844.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:00:00 | 843.85 | 852.65 | 844.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 845.10 | 851.14 | 844.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 14:00:00 | 853.30 | 851.57 | 845.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 14:45:00 | 851.15 | 850.86 | 845.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-14 10:15:00 | 834.45 | 846.25 | 844.58 | SL hit (close<static) qty=1.00 sl=840.10 alert=retest2 |

### Cycle 70 — SELL (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 12:15:00 | 827.15 | 840.23 | 842.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 09:15:00 | 819.05 | 832.49 | 837.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-20 09:15:00 | 823.20 | 800.59 | 806.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 823.20 | 800.59 | 806.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 823.20 | 800.59 | 806.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:00:00 | 823.20 | 800.59 | 806.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 822.15 | 804.91 | 807.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:45:00 | 824.85 | 804.91 | 807.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 12:15:00 | 821.85 | 810.34 | 809.87 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 12:15:00 | 802.20 | 810.14 | 810.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 14:15:00 | 792.35 | 804.62 | 808.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 12:15:00 | 765.10 | 763.94 | 777.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 12:45:00 | 765.15 | 763.94 | 777.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 776.95 | 766.54 | 777.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:30:00 | 773.50 | 766.54 | 777.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 774.50 | 768.13 | 777.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 770.10 | 769.51 | 777.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 14:30:00 | 771.25 | 766.94 | 771.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 09:15:00 | 771.50 | 768.34 | 772.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 11:45:00 | 771.00 | 771.29 | 772.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 771.40 | 771.31 | 772.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 14:15:00 | 767.50 | 770.73 | 772.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 784.80 | 772.07 | 772.28 | SL hit (close>static) qty=1.00 sl=777.60 alert=retest2 |

### Cycle 73 — BUY (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 10:15:00 | 798.15 | 777.28 | 774.63 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 12:15:00 | 771.10 | 776.31 | 776.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 11:15:00 | 766.75 | 771.31 | 773.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 14:15:00 | 771.50 | 770.09 | 772.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 14:15:00 | 771.50 | 770.09 | 772.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 771.50 | 770.09 | 772.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 771.50 | 770.09 | 772.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 774.80 | 771.34 | 772.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 774.80 | 771.34 | 772.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 780.30 | 773.13 | 773.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:30:00 | 779.90 | 773.13 | 773.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 11:15:00 | 778.20 | 774.14 | 773.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 14:15:00 | 790.45 | 779.85 | 776.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 09:15:00 | 776.65 | 780.79 | 777.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 09:15:00 | 776.65 | 780.79 | 777.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 776.65 | 780.79 | 777.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 10:00:00 | 776.65 | 780.79 | 777.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 777.90 | 780.21 | 777.80 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 764.90 | 775.53 | 776.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 759.65 | 770.51 | 773.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 09:15:00 | 773.40 | 764.59 | 768.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 09:15:00 | 773.40 | 764.59 | 768.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 773.40 | 764.59 | 768.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:45:00 | 773.40 | 764.59 | 768.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 774.60 | 766.59 | 768.68 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 781.65 | 772.04 | 770.95 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 12:15:00 | 769.00 | 772.14 | 772.15 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 15:15:00 | 772.90 | 772.20 | 772.16 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 09:15:00 | 770.90 | 771.94 | 772.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 12:15:00 | 767.85 | 770.69 | 771.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 14:15:00 | 770.00 | 769.53 | 770.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 14:15:00 | 770.00 | 769.53 | 770.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 770.00 | 769.53 | 770.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 15:00:00 | 770.00 | 769.53 | 770.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 767.00 | 769.02 | 770.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:15:00 | 765.90 | 769.02 | 770.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 767.25 | 768.67 | 770.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 12:00:00 | 761.85 | 766.63 | 768.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 761.05 | 763.38 | 766.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 11:15:00 | 777.05 | 767.98 | 767.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-06-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 11:15:00 | 777.05 | 767.98 | 767.81 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 767.75 | 767.95 | 767.96 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 09:15:00 | 769.20 | 768.20 | 768.07 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 765.15 | 767.59 | 767.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 12:15:00 | 763.35 | 766.21 | 767.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 753.75 | 750.16 | 754.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 13:15:00 | 753.75 | 750.16 | 754.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 753.75 | 750.16 | 754.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 753.75 | 750.16 | 754.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 762.00 | 752.53 | 754.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 762.00 | 752.53 | 754.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 755.05 | 753.03 | 754.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 767.05 | 753.03 | 754.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 771.00 | 756.63 | 756.32 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 752.00 | 762.39 | 762.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 14:15:00 | 748.60 | 757.98 | 760.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 15:15:00 | 737.85 | 737.31 | 743.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-24 09:15:00 | 756.15 | 737.31 | 743.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 758.80 | 741.61 | 744.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:15:00 | 760.80 | 741.61 | 744.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 762.45 | 749.12 | 747.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 12:15:00 | 770.70 | 753.43 | 749.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 14:15:00 | 761.60 | 763.61 | 758.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 15:00:00 | 761.60 | 763.61 | 758.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 760.00 | 762.89 | 758.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:15:00 | 756.80 | 762.89 | 758.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 751.15 | 760.54 | 758.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 751.15 | 760.54 | 758.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 761.40 | 760.71 | 758.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 11:15:00 | 763.10 | 760.71 | 758.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 15:15:00 | 752.00 | 757.84 | 757.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 15:15:00 | 752.00 | 757.84 | 757.88 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 768.35 | 759.94 | 758.83 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 13:15:00 | 756.50 | 758.00 | 758.14 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 14:15:00 | 761.20 | 758.64 | 758.42 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 10:15:00 | 746.00 | 756.22 | 757.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 09:15:00 | 740.20 | 751.33 | 754.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 09:15:00 | 733.15 | 731.75 | 740.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 09:15:00 | 733.15 | 731.75 | 740.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 733.15 | 731.75 | 740.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:30:00 | 744.55 | 731.75 | 740.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 728.00 | 731.00 | 739.31 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 11:15:00 | 748.80 | 741.76 | 741.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 12:15:00 | 760.85 | 745.58 | 743.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 750.65 | 754.47 | 749.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 11:15:00 | 750.65 | 754.47 | 749.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 750.65 | 754.47 | 749.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:45:00 | 746.60 | 754.47 | 749.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 748.75 | 753.33 | 749.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:00:00 | 748.75 | 753.33 | 749.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 748.40 | 752.34 | 749.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:45:00 | 750.00 | 752.34 | 749.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 754.80 | 752.83 | 750.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 10:45:00 | 756.20 | 754.00 | 751.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 744.00 | 755.39 | 753.80 | SL hit (close<static) qty=1.00 sl=748.10 alert=retest2 |

### Cycle 94 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 741.65 | 752.64 | 752.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 15:15:00 | 738.85 | 745.21 | 748.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 768.25 | 749.81 | 750.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 768.25 | 749.81 | 750.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 768.25 | 749.81 | 750.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 768.25 | 749.81 | 750.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 769.30 | 753.71 | 752.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 11:15:00 | 774.35 | 757.84 | 754.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 12:15:00 | 776.25 | 776.83 | 768.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 13:00:00 | 776.25 | 776.83 | 768.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 765.65 | 773.30 | 769.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 765.65 | 773.30 | 769.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 758.95 | 770.43 | 768.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 758.95 | 770.43 | 768.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 765.40 | 768.13 | 767.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:30:00 | 765.50 | 768.13 | 767.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2025-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 15:15:00 | 765.05 | 767.38 | 767.50 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 09:15:00 | 772.40 | 768.38 | 767.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 11:15:00 | 782.35 | 771.83 | 769.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 823.80 | 824.98 | 817.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 10:15:00 | 814.75 | 824.98 | 817.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 820.00 | 823.98 | 817.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:00:00 | 820.00 | 823.98 | 817.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 814.10 | 822.01 | 817.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:00:00 | 814.10 | 822.01 | 817.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 805.70 | 818.74 | 816.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:00:00 | 805.70 | 818.74 | 816.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 805.50 | 813.27 | 814.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 09:15:00 | 803.25 | 808.54 | 811.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 794.05 | 791.66 | 797.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 15:00:00 | 794.05 | 791.66 | 797.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 798.90 | 793.48 | 797.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:15:00 | 804.00 | 793.48 | 797.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 812.25 | 797.23 | 798.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:00:00 | 812.25 | 797.23 | 798.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 11:15:00 | 809.50 | 799.69 | 799.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-25 09:15:00 | 815.70 | 804.91 | 802.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 11:15:00 | 804.00 | 805.04 | 802.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 11:15:00 | 804.00 | 805.04 | 802.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 804.00 | 805.04 | 802.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:45:00 | 802.70 | 805.04 | 802.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 809.80 | 805.99 | 803.55 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 12:15:00 | 789.00 | 800.39 | 801.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 788.45 | 798.01 | 800.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 795.05 | 792.59 | 796.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:00:00 | 795.05 | 792.59 | 796.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 801.50 | 794.37 | 796.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 801.50 | 794.37 | 796.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 805.85 | 796.67 | 797.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 805.85 | 796.67 | 797.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 15:15:00 | 807.00 | 798.73 | 798.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 10:15:00 | 814.30 | 803.17 | 800.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 15:15:00 | 805.50 | 805.54 | 802.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 09:15:00 | 798.50 | 805.54 | 802.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 794.00 | 803.23 | 802.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:00:00 | 805.00 | 803.59 | 802.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:30:00 | 804.40 | 809.85 | 807.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 10:15:00 | 803.80 | 809.85 | 807.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 11:15:00 | 791.35 | 804.75 | 805.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 791.35 | 804.75 | 805.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 777.40 | 795.38 | 800.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 787.85 | 787.81 | 794.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 12:30:00 | 788.10 | 787.81 | 794.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 792.50 | 788.54 | 792.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 792.65 | 788.54 | 792.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 790.00 | 788.83 | 792.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 768.45 | 789.36 | 791.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 10:15:00 | 730.03 | 752.14 | 767.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 15:15:00 | 741.40 | 738.20 | 753.23 | SL hit (close>ema200) qty=0.50 sl=738.20 alert=retest2 |

### Cycle 103 — BUY (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 13:15:00 | 533.55 | 520.88 | 519.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 12:15:00 | 537.70 | 531.08 | 525.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 12:15:00 | 546.75 | 547.18 | 537.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 13:00:00 | 546.75 | 547.18 | 537.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 536.15 | 543.79 | 537.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 536.15 | 543.79 | 537.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 537.40 | 542.51 | 537.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:15:00 | 545.60 | 542.51 | 537.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 13:15:00 | 552.15 | 562.59 | 563.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 552.15 | 562.59 | 563.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 548.30 | 558.86 | 561.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 546.50 | 540.62 | 546.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 546.50 | 540.62 | 546.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 546.50 | 540.62 | 546.66 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 556.50 | 549.81 | 549.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 574.45 | 554.74 | 551.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 561.60 | 561.84 | 556.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 14:00:00 | 561.60 | 561.84 | 556.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 557.40 | 560.63 | 557.26 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 14:15:00 | 554.30 | 555.97 | 555.97 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 571.00 | 558.74 | 557.22 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 554.15 | 557.17 | 557.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 548.85 | 555.51 | 556.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 555.20 | 554.54 | 555.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 555.20 | 554.54 | 555.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 555.20 | 554.54 | 555.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 555.20 | 554.54 | 555.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 553.75 | 554.38 | 555.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:30:00 | 556.90 | 554.38 | 555.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 555.40 | 554.59 | 555.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 560.20 | 554.59 | 555.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 558.05 | 555.28 | 555.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:30:00 | 562.25 | 555.28 | 555.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 554.30 | 555.08 | 555.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 13:30:00 | 552.10 | 554.08 | 555.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:30:00 | 550.90 | 552.94 | 554.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 10:15:00 | 561.95 | 554.74 | 554.98 | SL hit (close>static) qty=1.00 sl=559.15 alert=retest2 |

### Cycle 109 — BUY (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 11:15:00 | 564.20 | 556.63 | 555.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 570.65 | 564.38 | 560.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 13:15:00 | 576.30 | 576.79 | 570.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 14:00:00 | 576.30 | 576.79 | 570.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 570.25 | 575.48 | 570.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 570.25 | 575.48 | 570.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 570.25 | 574.44 | 570.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 573.55 | 574.44 | 570.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 10:00:00 | 574.35 | 574.42 | 570.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 566.40 | 572.19 | 570.46 | SL hit (close<static) qty=1.00 sl=569.50 alert=retest2 |

### Cycle 110 — SELL (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 09:15:00 | 566.70 | 569.06 | 569.28 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 574.20 | 570.33 | 569.81 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 09:15:00 | 568.05 | 570.32 | 570.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 11:15:00 | 563.45 | 568.39 | 569.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 14:15:00 | 568.90 | 567.91 | 569.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-17 15:00:00 | 568.90 | 567.91 | 569.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 569.50 | 568.23 | 569.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 571.15 | 568.23 | 569.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 566.50 | 567.89 | 568.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 13:00:00 | 563.45 | 567.11 | 568.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:15:00 | 563.50 | 566.40 | 567.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 15:15:00 | 572.50 | 568.33 | 568.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 15:15:00 | 572.50 | 568.33 | 568.12 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 566.15 | 567.91 | 568.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 560.90 | 566.13 | 567.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 11:15:00 | 564.15 | 563.22 | 565.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 11:15:00 | 564.15 | 563.22 | 565.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 564.15 | 563.22 | 565.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:45:00 | 564.00 | 563.22 | 565.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 562.50 | 563.08 | 564.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 14:00:00 | 560.70 | 562.60 | 564.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 532.66 | 542.58 | 549.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-09-29 12:15:00 | 504.63 | 515.57 | 527.79 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 115 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 514.00 | 512.00 | 511.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 15:15:00 | 514.55 | 512.51 | 512.22 | Break + close above crossover candle high |

### Cycle 116 — SELL (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 09:15:00 | 508.25 | 511.66 | 511.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 10:15:00 | 504.95 | 510.32 | 511.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-06 11:15:00 | 513.75 | 511.00 | 511.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 11:15:00 | 513.75 | 511.00 | 511.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 513.75 | 511.00 | 511.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:00:00 | 513.75 | 511.00 | 511.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 514.80 | 511.76 | 511.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:00:00 | 514.80 | 511.76 | 511.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 13:15:00 | 514.45 | 512.30 | 512.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 15:15:00 | 520.00 | 514.34 | 513.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 511.60 | 518.92 | 517.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 511.60 | 518.92 | 517.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 511.60 | 518.92 | 517.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 511.60 | 518.92 | 517.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 513.20 | 517.77 | 516.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:30:00 | 514.05 | 517.77 | 516.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 12:15:00 | 513.30 | 516.19 | 516.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 13:15:00 | 509.70 | 514.90 | 515.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 15:15:00 | 514.80 | 514.68 | 515.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 15:15:00 | 514.80 | 514.68 | 515.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 514.80 | 514.68 | 515.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 530.25 | 514.68 | 515.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 09:15:00 | 548.90 | 521.52 | 518.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 15:15:00 | 556.55 | 542.71 | 531.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 575.45 | 581.07 | 571.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 11:00:00 | 575.45 | 581.07 | 571.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 572.35 | 579.33 | 571.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:00:00 | 572.35 | 579.33 | 571.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 572.45 | 577.95 | 571.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:30:00 | 572.30 | 577.95 | 571.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 566.15 | 575.59 | 571.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:00:00 | 566.15 | 575.59 | 571.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 568.25 | 574.12 | 570.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 11:30:00 | 572.95 | 572.11 | 570.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:30:00 | 571.65 | 571.91 | 571.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 11:15:00 | 563.80 | 569.84 | 570.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 11:15:00 | 563.80 | 569.84 | 570.35 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 583.50 | 571.58 | 570.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 13:15:00 | 587.00 | 579.58 | 575.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 14:15:00 | 584.10 | 585.51 | 581.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 15:00:00 | 584.10 | 585.51 | 581.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 585.00 | 585.21 | 582.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 585.00 | 585.21 | 582.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 581.20 | 584.41 | 582.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:45:00 | 580.65 | 584.41 | 582.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 575.85 | 582.70 | 581.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:30:00 | 575.65 | 582.70 | 581.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 12:15:00 | 577.35 | 580.84 | 580.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 15:15:00 | 572.00 | 577.88 | 579.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 09:15:00 | 577.95 | 577.89 | 579.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 09:15:00 | 577.95 | 577.89 | 579.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 577.95 | 577.89 | 579.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:45:00 | 579.90 | 577.89 | 579.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 576.35 | 577.50 | 578.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 573.55 | 576.51 | 577.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 12:15:00 | 574.70 | 570.67 | 571.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 14:15:00 | 574.15 | 572.52 | 572.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 14:15:00 | 574.15 | 572.52 | 572.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 11:15:00 | 577.35 | 574.42 | 573.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 12:15:00 | 573.50 | 574.24 | 573.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 12:15:00 | 573.50 | 574.24 | 573.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 573.50 | 574.24 | 573.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:00:00 | 573.50 | 574.24 | 573.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 574.40 | 574.27 | 573.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:30:00 | 573.25 | 574.27 | 573.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 573.40 | 574.10 | 573.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:30:00 | 573.80 | 574.10 | 573.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 574.00 | 574.08 | 573.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 577.55 | 574.08 | 573.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 574.40 | 574.14 | 573.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:15:00 | 573.50 | 574.14 | 573.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 568.45 | 573.00 | 573.12 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 11:15:00 | 574.25 | 573.25 | 573.22 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 570.65 | 572.81 | 573.03 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 10:15:00 | 578.05 | 573.90 | 573.46 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 12:15:00 | 570.65 | 573.78 | 573.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 13:15:00 | 567.00 | 572.42 | 573.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 11:15:00 | 535.00 | 531.50 | 541.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 11:45:00 | 533.95 | 531.50 | 541.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 533.00 | 530.31 | 534.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 10:00:00 | 521.55 | 528.55 | 533.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 12:00:00 | 525.40 | 528.67 | 532.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 13:45:00 | 525.75 | 527.93 | 531.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 09:15:00 | 543.70 | 531.00 | 532.05 | SL hit (close>static) qty=1.00 sl=536.75 alert=retest2 |

### Cycle 129 — BUY (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 10:15:00 | 551.65 | 535.13 | 533.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 12:15:00 | 566.75 | 544.57 | 538.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 10:15:00 | 562.55 | 568.52 | 559.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-17 11:00:00 | 562.55 | 568.52 | 559.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 578.30 | 581.35 | 576.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:00:00 | 578.30 | 581.35 | 576.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 578.70 | 580.82 | 576.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:00:00 | 578.70 | 580.82 | 576.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 578.15 | 580.29 | 576.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:45:00 | 576.00 | 580.29 | 576.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 576.50 | 579.53 | 576.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 14:00:00 | 576.50 | 579.53 | 576.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 579.70 | 579.56 | 576.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 09:15:00 | 585.30 | 579.65 | 577.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 11:00:00 | 583.95 | 580.93 | 578.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 12:15:00 | 577.60 | 586.86 | 587.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 12:15:00 | 577.60 | 586.86 | 587.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 13:15:00 | 571.95 | 583.88 | 586.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 594.40 | 577.75 | 579.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 594.40 | 577.75 | 579.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 594.40 | 577.75 | 579.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 594.40 | 577.75 | 579.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 591.75 | 580.55 | 580.74 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 591.70 | 582.78 | 581.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 12:15:00 | 595.00 | 585.22 | 582.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 591.85 | 593.55 | 588.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:00:00 | 591.85 | 593.55 | 588.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 588.55 | 592.55 | 588.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:30:00 | 590.75 | 592.55 | 588.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 585.60 | 591.16 | 588.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 585.60 | 591.16 | 588.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 585.60 | 590.05 | 588.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 588.70 | 588.25 | 587.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 10:30:00 | 587.90 | 588.26 | 587.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 11:00:00 | 588.50 | 590.58 | 589.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 13:15:00 | 587.55 | 589.90 | 589.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 13:15:00 | 587.55 | 589.90 | 589.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 575.85 | 586.46 | 588.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 11:15:00 | 576.35 | 575.10 | 579.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 11:30:00 | 575.40 | 575.10 | 579.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 589.25 | 577.93 | 580.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:45:00 | 588.05 | 577.93 | 580.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 581.50 | 578.64 | 580.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 14:15:00 | 580.85 | 578.64 | 580.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 14:15:00 | 551.81 | 568.24 | 573.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-08 13:15:00 | 522.76 | 542.65 | 557.22 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 133 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 555.10 | 550.39 | 550.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 570.00 | 555.19 | 552.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 15:15:00 | 564.35 | 565.21 | 560.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:15:00 | 554.90 | 565.21 | 560.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 552.50 | 562.67 | 559.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 552.50 | 562.67 | 559.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 556.20 | 561.38 | 559.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:45:00 | 553.15 | 561.38 | 559.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 569.75 | 562.93 | 560.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 569.75 | 562.93 | 560.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 562.25 | 565.08 | 562.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 562.25 | 565.08 | 562.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 568.00 | 565.66 | 563.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 12:15:00 | 570.00 | 564.98 | 563.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 13:15:00 | 568.90 | 565.62 | 564.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 13:45:00 | 568.95 | 565.81 | 564.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 09:15:00 | 571.80 | 566.21 | 564.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 577.80 | 579.98 | 576.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:15:00 | 574.80 | 579.98 | 576.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 579.45 | 579.88 | 576.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 582.25 | 576.72 | 576.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:00:00 | 580.95 | 577.57 | 576.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 588.10 | 580.89 | 579.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 10:15:00 | 570.50 | 580.34 | 580.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 570.50 | 580.34 | 580.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 11:15:00 | 569.15 | 578.10 | 579.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 568.90 | 566.94 | 570.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 568.90 | 566.94 | 570.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 568.90 | 566.94 | 570.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 569.60 | 566.94 | 570.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 571.90 | 567.93 | 571.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 571.90 | 567.93 | 571.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 576.50 | 569.64 | 571.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:00:00 | 576.50 | 569.64 | 571.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 581.55 | 572.03 | 572.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:00:00 | 581.55 | 572.03 | 572.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 580.75 | 573.77 | 573.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 595.75 | 580.70 | 577.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 622.80 | 623.04 | 610.90 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 11:30:00 | 626.25 | 623.20 | 612.07 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 12:15:00 | 625.80 | 623.20 | 612.07 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 14:30:00 | 626.10 | 624.34 | 615.43 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 09:15:00 | 626.05 | 624.55 | 616.34 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 618.50 | 622.33 | 619.66 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 618.50 | 622.33 | 619.66 | SL hit (close<ema400) qty=1.00 sl=619.66 alert=retest1 |

### Cycle 136 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 611.95 | 617.84 | 618.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 608.50 | 615.97 | 617.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 597.90 | 596.86 | 603.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 588.80 | 594.76 | 600.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 588.80 | 594.76 | 600.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:45:00 | 584.50 | 593.19 | 598.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 584.60 | 591.47 | 597.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 583.20 | 589.27 | 595.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 14:15:00 | 584.75 | 589.84 | 591.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 555.27 | 567.89 | 576.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 555.37 | 567.89 | 576.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 554.04 | 567.89 | 576.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 555.51 | 567.89 | 576.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-21 11:15:00 | 526.05 | 547.72 | 561.07 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 137 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 538.15 | 526.89 | 526.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 15:15:00 | 543.25 | 530.16 | 527.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 529.00 | 530.56 | 528.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 10:15:00 | 529.00 | 530.56 | 528.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 529.00 | 530.56 | 528.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 526.95 | 530.56 | 528.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 526.85 | 529.82 | 528.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:00:00 | 526.85 | 529.82 | 528.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 524.65 | 528.78 | 528.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:45:00 | 525.35 | 528.78 | 528.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 527.00 | 528.49 | 528.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:45:00 | 527.20 | 528.49 | 528.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 530.00 | 528.79 | 528.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:15:00 | 534.00 | 528.79 | 528.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 561.40 | 535.32 | 531.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 15:00:00 | 562.60 | 550.60 | 545.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 11:30:00 | 563.70 | 560.51 | 552.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 12:15:00 | 562.25 | 560.51 | 552.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 14:45:00 | 562.55 | 558.78 | 553.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 567.95 | 580.66 | 576.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 567.95 | 580.66 | 576.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 569.20 | 578.37 | 575.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 565.25 | 578.37 | 575.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-06 12:15:00 | 564.90 | 573.60 | 573.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 564.90 | 573.60 | 573.97 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 585.90 | 576.23 | 575.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 597.95 | 581.97 | 578.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 12:15:00 | 586.60 | 586.74 | 581.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-09 13:00:00 | 586.60 | 586.74 | 581.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 609.00 | 617.08 | 611.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 607.95 | 617.08 | 611.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 610.55 | 615.78 | 611.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:45:00 | 613.35 | 615.82 | 611.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 12:15:00 | 618.00 | 620.74 | 620.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 618.00 | 620.74 | 620.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 614.75 | 619.54 | 620.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 615.25 | 613.60 | 616.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 11:00:00 | 615.25 | 613.60 | 616.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 615.80 | 614.04 | 616.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:00:00 | 615.80 | 614.04 | 616.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 614.40 | 614.11 | 616.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:30:00 | 610.00 | 614.54 | 615.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:15:00 | 612.65 | 614.54 | 615.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 11:00:00 | 612.10 | 611.74 | 613.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 09:30:00 | 610.80 | 612.54 | 612.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 624.05 | 614.84 | 613.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 624.05 | 614.84 | 613.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 627.75 | 620.68 | 617.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 12:15:00 | 620.90 | 621.30 | 618.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 12:45:00 | 620.60 | 621.30 | 618.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 619.00 | 621.67 | 619.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 619.00 | 621.67 | 619.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 628.00 | 622.94 | 620.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 11:45:00 | 629.10 | 623.94 | 621.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 13:30:00 | 629.85 | 626.14 | 622.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 612.40 | 623.72 | 622.47 | SL hit (close<static) qty=1.00 sl=617.70 alert=retest2 |

### Cycle 142 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 608.30 | 620.63 | 621.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 597.30 | 613.04 | 616.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 596.05 | 594.83 | 603.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 596.05 | 594.83 | 603.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 596.05 | 594.83 | 603.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 10:45:00 | 592.50 | 594.12 | 602.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 612.55 | 599.67 | 602.15 | SL hit (close>static) qty=1.00 sl=606.55 alert=retest2 |

### Cycle 143 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 614.40 | 605.07 | 604.33 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 579.70 | 603.64 | 604.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 10:15:00 | 575.60 | 598.03 | 602.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 14:15:00 | 543.80 | 538.10 | 556.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-10 15:00:00 | 543.80 | 538.10 | 556.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 559.75 | 544.33 | 555.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:45:00 | 559.95 | 544.33 | 555.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 555.10 | 546.49 | 555.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:30:00 | 558.10 | 546.49 | 555.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 552.15 | 548.01 | 554.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 12:30:00 | 560.10 | 548.01 | 554.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 551.15 | 548.64 | 554.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:30:00 | 556.65 | 548.64 | 554.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 549.65 | 548.84 | 554.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:30:00 | 553.55 | 548.84 | 554.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 520.25 | 507.13 | 513.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:45:00 | 523.05 | 507.13 | 513.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 514.45 | 508.60 | 514.02 | EMA400 retest candle locked (from downside) |

### Cycle 145 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 524.30 | 516.97 | 516.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 535.05 | 521.64 | 518.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 520.00 | 530.81 | 526.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 520.00 | 530.81 | 526.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 520.00 | 530.81 | 526.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:15:00 | 517.20 | 530.81 | 526.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 515.20 | 523.33 | 523.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 506.20 | 519.90 | 522.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 515.35 | 515.12 | 519.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 515.35 | 515.12 | 519.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 515.35 | 515.12 | 519.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 517.20 | 515.12 | 519.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 512.80 | 515.04 | 518.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 516.30 | 515.04 | 518.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 512.50 | 514.27 | 517.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:30:00 | 515.50 | 514.27 | 517.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 490.90 | 495.42 | 503.10 | EMA400 retest candle locked (from downside) |

### Cycle 147 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 529.70 | 504.90 | 503.91 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 499.10 | 507.63 | 507.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 12:15:00 | 496.55 | 505.41 | 506.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 497.05 | 484.29 | 491.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 497.05 | 484.29 | 491.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 497.05 | 484.29 | 491.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 497.05 | 484.29 | 491.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 486.60 | 484.75 | 490.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:45:00 | 485.45 | 485.54 | 490.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:15:00 | 486.10 | 486.09 | 490.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 461.18 | 479.03 | 485.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 461.80 | 479.03 | 485.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-04-06 09:15:00 | 436.90 | 455.29 | 467.85 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 149 — BUY (started 2026-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 11:15:00 | 477.00 | 457.12 | 455.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 12:15:00 | 479.00 | 461.50 | 457.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 466.40 | 470.43 | 463.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 466.40 | 470.43 | 463.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 466.40 | 470.43 | 463.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 466.40 | 470.43 | 463.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 477.85 | 484.18 | 478.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:30:00 | 487.80 | 484.24 | 479.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 13:15:00 | 536.58 | 514.66 | 499.05 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 13:15:00 | 552.00 | 559.00 | 559.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 549.45 | 557.09 | 558.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 560.90 | 551.11 | 553.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 560.90 | 551.11 | 553.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 560.90 | 551.11 | 553.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 564.60 | 551.11 | 553.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 561.45 | 553.18 | 554.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 563.00 | 553.18 | 554.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 567.35 | 556.01 | 555.33 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 12:15:00 | 557.40 | 559.81 | 559.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 13:15:00 | 552.00 | 558.24 | 559.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 15:15:00 | 535.50 | 535.43 | 543.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 09:15:00 | 535.35 | 535.43 | 543.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 538.60 | 534.70 | 538.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:30:00 | 541.05 | 534.70 | 538.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 534.00 | 534.56 | 538.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 14:00:00 | 531.55 | 533.88 | 537.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 15:00:00 | 532.05 | 533.51 | 536.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 539.00 | 534.21 | 536.35 | SL hit (close>static) qty=1.00 sl=538.65 alert=retest2 |

### Cycle 153 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 544.00 | 537.41 | 537.15 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2026-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 14:15:00 | 534.85 | 537.05 | 537.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 09:15:00 | 528.00 | 535.06 | 536.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 14:15:00 | 531.00 | 530.94 | 533.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-08 14:45:00 | 531.25 | 530.94 | 533.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 09:15:00 | 203.77 | 2024-05-13 11:15:00 | 211.14 | STOP_HIT | 1.00 | -3.62% |
| BUY | retest2 | 2024-05-18 09:15:00 | 214.12 | 2024-05-21 09:15:00 | 235.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-13 09:15:00 | 293.45 | 2024-06-13 10:15:00 | 289.57 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-06-13 11:45:00 | 293.16 | 2024-06-19 12:15:00 | 322.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-18 09:45:00 | 294.00 | 2024-06-19 13:15:00 | 323.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-18 09:15:00 | 360.10 | 2024-07-22 12:15:00 | 373.10 | STOP_HIT | 1.00 | -3.61% |
| SELL | retest2 | 2024-07-18 12:45:00 | 359.55 | 2024-07-22 12:15:00 | 373.10 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest2 | 2024-09-02 14:00:00 | 518.30 | 2024-09-04 14:15:00 | 570.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-02 15:15:00 | 518.00 | 2024-09-04 14:15:00 | 569.80 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-20 11:30:00 | 616.45 | 2024-09-24 11:15:00 | 623.65 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-09-23 09:15:00 | 615.35 | 2024-09-24 11:15:00 | 623.65 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-09-23 11:30:00 | 617.00 | 2024-09-24 11:15:00 | 623.65 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-09-23 13:45:00 | 616.95 | 2024-09-24 11:15:00 | 623.65 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-10-07 10:30:00 | 548.05 | 2024-10-08 15:15:00 | 608.90 | STOP_HIT | 1.00 | -11.10% |
| SELL | retest2 | 2024-10-07 11:45:00 | 550.75 | 2024-10-08 15:15:00 | 608.90 | STOP_HIT | 1.00 | -10.56% |
| SELL | retest2 | 2024-10-07 12:15:00 | 555.95 | 2024-10-08 15:15:00 | 608.90 | STOP_HIT | 1.00 | -9.52% |
| BUY | retest2 | 2024-10-11 13:30:00 | 620.80 | 2024-10-15 09:15:00 | 603.30 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2024-10-14 09:45:00 | 623.00 | 2024-10-15 09:15:00 | 603.30 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2024-10-16 09:15:00 | 607.30 | 2024-10-18 09:15:00 | 576.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-16 09:15:00 | 607.30 | 2024-10-18 09:15:00 | 611.05 | STOP_HIT | 0.50 | -0.62% |
| SELL | retest2 | 2024-10-17 09:30:00 | 610.35 | 2024-10-18 09:15:00 | 579.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 09:30:00 | 610.35 | 2024-10-18 09:15:00 | 611.05 | STOP_HIT | 0.50 | -0.11% |
| SELL | retest2 | 2024-10-17 10:45:00 | 606.40 | 2024-10-18 09:15:00 | 576.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 10:45:00 | 606.40 | 2024-10-18 09:15:00 | 611.05 | STOP_HIT | 0.50 | -0.77% |
| BUY | retest2 | 2024-11-06 09:15:00 | 642.50 | 2024-11-11 09:15:00 | 706.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-11 14:15:00 | 655.50 | 2024-11-11 14:15:00 | 647.65 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-12-04 14:00:00 | 774.90 | 2024-12-09 11:15:00 | 852.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-05 09:15:00 | 779.45 | 2024-12-09 12:15:00 | 857.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-27 09:15:00 | 979.75 | 2025-01-06 15:15:00 | 994.00 | STOP_HIT | 1.00 | 1.45% |
| BUY | retest2 | 2024-12-30 11:15:00 | 977.30 | 2025-01-06 15:15:00 | 994.00 | STOP_HIT | 1.00 | 1.71% |
| BUY | retest2 | 2024-12-30 14:15:00 | 984.95 | 2025-01-06 15:15:00 | 994.00 | STOP_HIT | 1.00 | 0.92% |
| SELL | retest2 | 2025-01-23 15:00:00 | 767.10 | 2025-01-27 09:15:00 | 728.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:45:00 | 762.70 | 2025-01-27 09:15:00 | 724.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 13:15:00 | 768.10 | 2025-01-27 09:15:00 | 729.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 15:00:00 | 767.10 | 2025-01-28 09:15:00 | 690.39 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 09:45:00 | 762.70 | 2025-01-28 09:15:00 | 686.43 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 13:15:00 | 768.10 | 2025-01-28 09:15:00 | 691.29 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-04 09:15:00 | 796.45 | 2025-02-07 09:15:00 | 876.10 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-12 14:15:00 | 811.05 | 2025-02-13 09:15:00 | 834.85 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2025-02-18 09:15:00 | 772.30 | 2025-02-18 14:15:00 | 733.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-18 09:15:00 | 772.30 | 2025-02-19 10:15:00 | 762.25 | STOP_HIT | 0.50 | 1.30% |
| BUY | retest2 | 2025-02-25 09:30:00 | 816.25 | 2025-02-27 10:15:00 | 788.70 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2025-03-11 13:15:00 | 890.20 | 2025-03-12 09:15:00 | 874.50 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest1 | 2025-03-24 09:15:00 | 951.90 | 2025-03-25 09:15:00 | 896.45 | STOP_HIT | 1.00 | -5.83% |
| SELL | retest2 | 2025-03-26 12:30:00 | 899.05 | 2025-03-27 14:15:00 | 925.00 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2025-04-09 09:15:00 | 847.30 | 2025-04-11 09:15:00 | 903.65 | STOP_HIT | 1.00 | -6.65% |
| SELL | retest2 | 2025-04-09 12:00:00 | 851.40 | 2025-04-11 09:15:00 | 903.65 | STOP_HIT | 1.00 | -6.14% |
| SELL | retest2 | 2025-04-09 12:45:00 | 852.00 | 2025-04-11 09:15:00 | 903.65 | STOP_HIT | 1.00 | -6.06% |
| SELL | retest2 | 2025-04-09 13:30:00 | 850.00 | 2025-04-11 09:15:00 | 903.65 | STOP_HIT | 1.00 | -6.31% |
| BUY | retest2 | 2025-04-21 09:15:00 | 954.65 | 2025-04-21 12:15:00 | 937.20 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-04-28 11:15:00 | 892.80 | 2025-04-30 13:15:00 | 848.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 09:30:00 | 887.70 | 2025-04-30 14:15:00 | 843.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-28 11:15:00 | 892.80 | 2025-05-06 13:15:00 | 803.52 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-30 09:30:00 | 887.70 | 2025-05-06 14:15:00 | 798.93 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-05-13 14:00:00 | 853.30 | 2025-05-14 10:15:00 | 834.45 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-05-13 14:45:00 | 851.15 | 2025-05-14 10:15:00 | 834.45 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-05-26 09:15:00 | 770.10 | 2025-05-28 09:15:00 | 784.80 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-05-26 14:30:00 | 771.25 | 2025-05-28 09:15:00 | 784.80 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-05-27 09:15:00 | 771.50 | 2025-05-28 09:15:00 | 784.80 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-05-27 11:45:00 | 771.00 | 2025-05-28 09:15:00 | 784.80 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-05-27 14:15:00 | 767.50 | 2025-05-28 09:15:00 | 784.80 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-06-10 12:00:00 | 761.85 | 2025-06-11 11:15:00 | 777.05 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-06-11 09:15:00 | 761.05 | 2025-06-11 11:15:00 | 777.05 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-06-26 11:15:00 | 763.10 | 2025-06-26 15:15:00 | 752.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-07-07 10:45:00 | 756.20 | 2025-07-08 09:15:00 | 744.00 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-07-31 11:00:00 | 805.00 | 2025-08-01 11:15:00 | 791.35 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-08-01 09:30:00 | 804.40 | 2025-08-01 11:15:00 | 791.35 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-08-01 10:15:00 | 803.80 | 2025-08-01 11:15:00 | 791.35 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-08-06 09:15:00 | 768.45 | 2025-08-07 10:15:00 | 730.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-06 09:15:00 | 768.45 | 2025-08-07 15:15:00 | 741.40 | STOP_HIT | 0.50 | 3.52% |
| BUY | retest2 | 2025-08-21 09:15:00 | 545.60 | 2025-08-26 13:15:00 | 552.15 | STOP_HIT | 1.00 | 1.20% |
| SELL | retest2 | 2025-09-08 13:30:00 | 552.10 | 2025-09-09 10:15:00 | 561.95 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-09-09 09:30:00 | 550.90 | 2025-09-09 10:15:00 | 561.95 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-09-12 09:15:00 | 573.55 | 2025-09-12 11:15:00 | 566.40 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-09-12 10:00:00 | 574.35 | 2025-09-12 11:15:00 | 566.40 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-09-15 09:15:00 | 573.10 | 2025-09-15 09:15:00 | 566.70 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-09-18 13:00:00 | 563.45 | 2025-09-19 15:15:00 | 572.50 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-09-19 14:15:00 | 563.50 | 2025-09-19 15:15:00 | 572.50 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-09-23 14:00:00 | 560.70 | 2025-09-26 09:15:00 | 532.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 14:00:00 | 560.70 | 2025-09-29 12:15:00 | 504.63 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-10-15 11:30:00 | 572.95 | 2025-10-16 11:15:00 | 563.80 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-10-16 09:30:00 | 571.65 | 2025-10-16 11:15:00 | 563.80 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-10-27 09:15:00 | 573.55 | 2025-10-29 14:15:00 | 574.15 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-10-29 12:15:00 | 574.70 | 2025-10-29 14:15:00 | 574.15 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-11-12 10:00:00 | 521.55 | 2025-11-13 09:15:00 | 543.70 | STOP_HIT | 1.00 | -4.25% |
| SELL | retest2 | 2025-11-12 12:00:00 | 525.40 | 2025-11-13 09:15:00 | 543.70 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2025-11-12 13:45:00 | 525.75 | 2025-11-13 09:15:00 | 543.70 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2025-11-20 09:15:00 | 585.30 | 2025-11-24 12:15:00 | 577.60 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-11-20 11:00:00 | 583.95 | 2025-11-24 12:15:00 | 577.60 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-11-28 09:15:00 | 588.70 | 2025-12-02 13:15:00 | 587.55 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2025-11-28 10:30:00 | 587.90 | 2025-12-02 13:15:00 | 587.55 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2025-12-01 11:00:00 | 588.50 | 2025-12-02 13:15:00 | 587.55 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-12-04 14:15:00 | 580.85 | 2025-12-05 14:15:00 | 551.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 14:15:00 | 580.85 | 2025-12-08 13:15:00 | 522.76 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-12-18 12:15:00 | 570.00 | 2025-12-29 10:15:00 | 570.50 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-12-18 13:15:00 | 568.90 | 2025-12-29 10:15:00 | 570.50 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2025-12-18 13:45:00 | 568.95 | 2025-12-29 10:15:00 | 570.50 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2025-12-19 09:15:00 | 571.80 | 2025-12-29 10:15:00 | 570.50 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-12-24 09:15:00 | 582.25 | 2025-12-29 10:15:00 | 570.50 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-12-24 10:00:00 | 580.95 | 2025-12-29 10:15:00 | 570.50 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-12-26 09:15:00 | 588.10 | 2025-12-29 10:15:00 | 570.50 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest1 | 2026-01-06 11:30:00 | 626.25 | 2026-01-08 09:15:00 | 618.50 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest1 | 2026-01-06 12:15:00 | 625.80 | 2026-01-08 09:15:00 | 618.50 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest1 | 2026-01-06 14:30:00 | 626.10 | 2026-01-08 09:15:00 | 618.50 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest1 | 2026-01-07 09:15:00 | 626.05 | 2026-01-08 09:15:00 | 618.50 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2026-01-08 12:15:00 | 616.65 | 2026-01-08 12:15:00 | 611.95 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-01-13 10:45:00 | 584.50 | 2026-01-20 11:15:00 | 555.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 584.60 | 2026-01-20 11:15:00 | 555.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 14:00:00 | 583.20 | 2026-01-20 11:15:00 | 554.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 14:15:00 | 584.75 | 2026-01-20 11:15:00 | 555.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 10:45:00 | 584.50 | 2026-01-21 11:15:00 | 526.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 584.60 | 2026-01-21 11:15:00 | 526.14 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-13 14:00:00 | 583.20 | 2026-01-21 11:15:00 | 524.88 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-16 14:15:00 | 584.75 | 2026-01-21 11:15:00 | 526.27 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-23 11:15:00 | 535.40 | 2026-01-23 14:15:00 | 508.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 11:15:00 | 535.40 | 2026-01-27 14:15:00 | 517.55 | STOP_HIT | 0.50 | 3.33% |
| BUY | retest2 | 2026-02-02 15:00:00 | 562.60 | 2026-02-06 12:15:00 | 564.90 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest2 | 2026-02-03 11:30:00 | 563.70 | 2026-02-06 12:15:00 | 564.90 | STOP_HIT | 1.00 | 0.21% |
| BUY | retest2 | 2026-02-03 12:15:00 | 562.25 | 2026-02-06 12:15:00 | 564.90 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest2 | 2026-02-03 14:45:00 | 562.55 | 2026-02-06 12:15:00 | 564.90 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2026-02-13 11:45:00 | 613.35 | 2026-02-19 12:15:00 | 618.00 | STOP_HIT | 1.00 | 0.76% |
| SELL | retest2 | 2026-02-23 10:30:00 | 610.00 | 2026-02-25 10:15:00 | 624.05 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2026-02-23 11:15:00 | 612.65 | 2026-02-25 10:15:00 | 624.05 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2026-02-24 11:00:00 | 612.10 | 2026-02-25 10:15:00 | 624.05 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2026-02-25 09:30:00 | 610.80 | 2026-02-25 10:15:00 | 624.05 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2026-02-27 11:45:00 | 629.10 | 2026-03-02 09:15:00 | 612.40 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2026-02-27 13:30:00 | 629.85 | 2026-03-02 09:15:00 | 612.40 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2026-03-05 10:45:00 | 592.50 | 2026-03-05 14:15:00 | 612.55 | STOP_HIT | 1.00 | -3.38% |
| SELL | retest2 | 2026-04-01 11:45:00 | 485.45 | 2026-04-02 09:15:00 | 461.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 13:15:00 | 486.10 | 2026-04-02 09:15:00 | 461.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 11:45:00 | 485.45 | 2026-04-06 09:15:00 | 436.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-01 13:15:00 | 486.10 | 2026-04-06 09:15:00 | 437.49 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-04-13 11:30:00 | 487.80 | 2026-04-15 13:15:00 | 536.58 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-05-05 14:00:00 | 531.55 | 2026-05-06 09:15:00 | 539.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-05-05 15:00:00 | 532.05 | 2026-05-06 09:15:00 | 539.00 | STOP_HIT | 1.00 | -1.31% |
