# Brainbees Solutions Ltd. (FIRSTCRY)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-11 15:15:00 (1983 bars)
- **Last close:** 230.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 78 |
| ALERT1 | 52 |
| ALERT2 | 49 |
| ALERT2_SKIP | 22 |
| ALERT3 | 128 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 59 |
| PARTIAL | 22 |
| TARGET_HIT | 5 |
| STOP_HIT | 57 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 83 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 47 / 36
- **Target hits / Stop hits / Partials:** 5 / 56 / 22
- **Avg / median % per leg:** 2.04% / 2.94%
- **Sum % (uncompounded):** 169.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 1 | 6 | 0 | 0.53% | 3.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 3 | 42.9% | 1 | 6 | 0 | 0.53% | 3.7% |
| SELL (all) | 76 | 44 | 57.9% | 4 | 50 | 22 | 2.18% | 166.0% |
| SELL @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 1.39% | 4.2% |
| SELL @ 3rd Alert (retest2) | 73 | 43 | 58.9% | 4 | 48 | 21 | 2.22% | 161.9% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 2 | 1 | 1.39% | 4.2% |
| retest2 (combined) | 80 | 46 | 57.5% | 5 | 54 | 21 | 2.07% | 165.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 346.80 | 325.89 | 325.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 351.15 | 335.86 | 330.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 345.10 | 345.11 | 338.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 345.10 | 345.11 | 338.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 340.25 | 343.60 | 340.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:00:00 | 340.25 | 343.60 | 340.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 340.20 | 342.92 | 340.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:15:00 | 353.35 | 341.28 | 340.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:30:00 | 346.55 | 342.67 | 341.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 12:15:00 | 347.25 | 351.05 | 351.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 347.25 | 351.05 | 351.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 345.15 | 349.87 | 350.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 349.55 | 348.77 | 349.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 10:00:00 | 349.55 | 348.77 | 349.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 354.25 | 349.65 | 350.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 354.25 | 349.65 | 350.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 12:15:00 | 356.15 | 350.95 | 350.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 13:15:00 | 360.20 | 352.80 | 351.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 358.85 | 370.00 | 366.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 358.85 | 370.00 | 366.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 358.85 | 370.00 | 366.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 358.85 | 370.00 | 366.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 355.30 | 367.06 | 365.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:00:00 | 355.30 | 367.06 | 365.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 11:15:00 | 354.00 | 364.45 | 364.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 12:15:00 | 352.55 | 362.07 | 363.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 11:15:00 | 358.30 | 357.42 | 359.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-28 12:00:00 | 358.30 | 357.42 | 359.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 357.85 | 357.51 | 359.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:30:00 | 359.00 | 357.51 | 359.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 342.30 | 348.29 | 351.51 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 11:15:00 | 375.00 | 354.39 | 351.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 12:15:00 | 388.10 | 361.13 | 354.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 15:15:00 | 362.10 | 366.82 | 359.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 15:15:00 | 362.10 | 366.82 | 359.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 362.10 | 366.82 | 359.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:45:00 | 386.25 | 371.51 | 365.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-06 10:15:00 | 424.88 | 401.83 | 387.00 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 14:15:00 | 396.45 | 401.10 | 401.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 09:15:00 | 390.20 | 398.10 | 399.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 391.60 | 389.62 | 393.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 391.60 | 389.62 | 393.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 391.60 | 389.62 | 393.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 13:45:00 | 383.50 | 386.66 | 390.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 376.50 | 378.93 | 383.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 09:45:00 | 380.05 | 376.92 | 378.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 09:15:00 | 364.32 | 371.86 | 375.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:15:00 | 357.68 | 369.14 | 373.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:15:00 | 361.05 | 369.14 | 373.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-06-19 15:15:00 | 345.15 | 357.30 | 365.64 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 10:15:00 | 362.80 | 343.94 | 342.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 11:15:00 | 374.00 | 349.96 | 345.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 15:15:00 | 383.20 | 385.61 | 374.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 15:15:00 | 383.20 | 385.61 | 374.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 383.20 | 385.61 | 374.10 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 374.00 | 378.15 | 378.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 09:15:00 | 368.35 | 373.48 | 375.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 13:15:00 | 371.60 | 367.21 | 369.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 13:15:00 | 371.60 | 367.21 | 369.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 371.60 | 367.21 | 369.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 14:00:00 | 371.60 | 367.21 | 369.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 371.95 | 368.16 | 369.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 15:15:00 | 370.15 | 368.16 | 369.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 376.35 | 370.12 | 370.20 | SL hit (close>static) qty=1.00 sl=374.45 alert=retest2 |

### Cycle 9 — BUY (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 10:15:00 | 373.25 | 370.74 | 370.48 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 11:15:00 | 368.40 | 370.27 | 370.29 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 372.55 | 370.73 | 370.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 13:15:00 | 373.20 | 371.22 | 370.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 379.10 | 379.66 | 376.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 11:00:00 | 379.10 | 379.66 | 376.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 377.00 | 378.73 | 377.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 378.75 | 378.73 | 377.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 374.50 | 377.89 | 376.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 374.50 | 377.89 | 376.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 372.90 | 376.89 | 376.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:45:00 | 372.00 | 376.89 | 376.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 11:15:00 | 372.45 | 376.00 | 376.23 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 14:15:00 | 380.20 | 377.08 | 376.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 09:15:00 | 382.30 | 378.59 | 377.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 10:15:00 | 383.10 | 383.64 | 381.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 11:00:00 | 383.10 | 383.64 | 381.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 379.50 | 382.81 | 381.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:45:00 | 379.85 | 382.81 | 381.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 383.40 | 382.93 | 381.29 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2025-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 12:15:00 | 380.40 | 380.72 | 380.73 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 382.35 | 381.04 | 380.88 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 377.05 | 380.49 | 380.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 13:15:00 | 376.55 | 378.64 | 379.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 12:15:00 | 356.80 | 356.41 | 359.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-25 13:00:00 | 356.80 | 356.41 | 359.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 357.05 | 356.79 | 358.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:15:00 | 351.55 | 355.33 | 357.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 10:15:00 | 351.55 | 354.39 | 356.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 12:45:00 | 351.50 | 352.92 | 355.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 14:45:00 | 352.10 | 352.46 | 354.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 349.00 | 351.77 | 353.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 347.60 | 351.77 | 353.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 11:30:00 | 348.30 | 350.39 | 352.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 10:15:00 | 357.20 | 347.68 | 348.66 | SL hit (close>static) qty=1.00 sl=356.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 09:15:00 | 355.20 | 350.03 | 349.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 12:15:00 | 364.10 | 354.76 | 351.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 13:15:00 | 365.40 | 367.05 | 361.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-05 14:00:00 | 365.40 | 367.05 | 361.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 361.30 | 365.90 | 361.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 14:45:00 | 357.00 | 365.90 | 361.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 361.10 | 364.94 | 361.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:15:00 | 364.35 | 364.94 | 361.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 358.60 | 363.67 | 360.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 358.60 | 363.67 | 360.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 358.95 | 362.73 | 360.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:15:00 | 358.30 | 362.73 | 360.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 359.40 | 362.06 | 360.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 12:15:00 | 360.30 | 362.06 | 360.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 14:15:00 | 356.50 | 359.89 | 359.87 | SL hit (close<static) qty=1.00 sl=357.60 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 15:15:00 | 356.45 | 359.20 | 359.56 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 367.40 | 360.60 | 359.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 368.75 | 366.14 | 364.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 368.90 | 372.53 | 369.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 368.90 | 372.53 | 369.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 368.90 | 372.53 | 369.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:00:00 | 368.90 | 372.53 | 369.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 375.30 | 373.08 | 370.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:30:00 | 368.30 | 373.08 | 370.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 370.25 | 373.79 | 372.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:45:00 | 368.30 | 373.79 | 372.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 370.40 | 373.11 | 371.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:30:00 | 368.85 | 373.11 | 371.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 373.15 | 374.27 | 372.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 15:00:00 | 373.15 | 374.27 | 372.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 370.00 | 373.42 | 372.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:30:00 | 367.60 | 372.41 | 372.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 10:15:00 | 367.75 | 371.48 | 371.89 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 12:15:00 | 373.70 | 371.83 | 371.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 381.85 | 374.00 | 372.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 10:15:00 | 383.25 | 386.83 | 383.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 10:15:00 | 383.25 | 386.83 | 383.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 383.25 | 386.83 | 383.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 383.25 | 386.83 | 383.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 384.80 | 386.42 | 384.02 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 10:15:00 | 376.75 | 382.23 | 382.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 11:15:00 | 376.10 | 379.58 | 380.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 361.85 | 360.68 | 367.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:45:00 | 362.85 | 360.68 | 367.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 363.00 | 360.89 | 364.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 09:15:00 | 359.30 | 360.89 | 364.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 13:30:00 | 359.55 | 360.61 | 363.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 10:30:00 | 359.05 | 359.70 | 361.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 11:15:00 | 359.50 | 359.70 | 361.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 360.00 | 359.71 | 361.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:30:00 | 362.60 | 359.71 | 361.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 360.15 | 359.80 | 361.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:00:00 | 360.15 | 359.80 | 361.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 359.70 | 359.78 | 361.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 359.70 | 359.78 | 361.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 360.25 | 359.87 | 360.99 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 375.55 | 364.13 | 362.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 375.55 | 364.13 | 362.61 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 352.35 | 361.05 | 361.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 351.70 | 359.18 | 360.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 373.75 | 361.19 | 361.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 373.75 | 361.19 | 361.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 373.75 | 361.19 | 361.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:45:00 | 378.60 | 361.19 | 361.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 10:15:00 | 390.00 | 366.96 | 363.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 12:15:00 | 394.70 | 376.51 | 369.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 15:15:00 | 407.70 | 407.73 | 394.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 09:15:00 | 403.00 | 407.73 | 394.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 395.65 | 406.18 | 401.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 395.65 | 406.18 | 401.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 393.40 | 403.63 | 400.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:30:00 | 394.60 | 403.63 | 400.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 13:15:00 | 386.70 | 396.37 | 397.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 13:15:00 | 383.35 | 389.88 | 393.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 15:15:00 | 395.00 | 390.14 | 392.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 15:15:00 | 395.00 | 390.14 | 392.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 395.00 | 390.14 | 392.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 398.50 | 390.14 | 392.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 395.05 | 391.12 | 393.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 10:45:00 | 392.90 | 391.34 | 392.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 11:45:00 | 392.55 | 391.63 | 392.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 13:30:00 | 390.55 | 391.83 | 392.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 11:15:00 | 396.85 | 393.12 | 393.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 396.85 | 393.12 | 393.03 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 09:15:00 | 390.60 | 393.53 | 393.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 12:15:00 | 388.50 | 391.61 | 392.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 10:15:00 | 390.45 | 390.00 | 391.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 10:15:00 | 390.45 | 390.00 | 391.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 390.45 | 390.00 | 391.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 12:00:00 | 389.15 | 389.83 | 391.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 13:00:00 | 389.10 | 389.68 | 390.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:30:00 | 387.45 | 389.73 | 390.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:15:00 | 388.80 | 389.73 | 390.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 390.25 | 389.83 | 390.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 11:00:00 | 390.25 | 389.83 | 390.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 390.60 | 389.98 | 390.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 11:30:00 | 390.60 | 389.98 | 390.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 390.70 | 390.13 | 390.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:45:00 | 390.55 | 390.13 | 390.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 390.00 | 390.10 | 390.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 389.30 | 390.01 | 390.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 11:00:00 | 388.45 | 389.29 | 389.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 13:30:00 | 389.20 | 384.29 | 385.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 15:15:00 | 386.45 | 385.35 | 386.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 380.80 | 384.62 | 385.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:15:00 | 380.30 | 384.62 | 385.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:30:00 | 379.35 | 382.60 | 383.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 14:00:00 | 380.50 | 382.18 | 383.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:15:00 | 369.69 | 375.70 | 379.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:15:00 | 369.64 | 375.70 | 379.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:15:00 | 368.08 | 375.70 | 379.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:15:00 | 369.36 | 375.70 | 379.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:15:00 | 369.83 | 375.70 | 379.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:15:00 | 369.03 | 375.70 | 379.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:15:00 | 369.74 | 375.70 | 379.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:15:00 | 367.13 | 375.70 | 379.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 15:15:00 | 361.28 | 368.83 | 374.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 15:15:00 | 361.47 | 368.83 | 374.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 09:15:00 | 360.38 | 368.24 | 373.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 366.85 | 364.61 | 368.73 | SL hit (close>ema200) qty=0.50 sl=364.61 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 372.35 | 368.28 | 367.73 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 13:15:00 | 368.00 | 368.94 | 368.97 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 14:15:00 | 370.05 | 369.16 | 369.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 377.40 | 370.94 | 369.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 15:15:00 | 376.95 | 376.95 | 373.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 09:15:00 | 376.80 | 376.95 | 373.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 375.75 | 376.71 | 374.15 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 15:15:00 | 371.80 | 373.02 | 373.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 369.90 | 372.40 | 372.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 12:15:00 | 372.10 | 371.93 | 372.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 12:15:00 | 372.10 | 371.93 | 372.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 372.10 | 371.93 | 372.47 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 15:15:00 | 374.60 | 373.06 | 372.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 10:15:00 | 376.50 | 373.96 | 373.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 13:15:00 | 374.30 | 374.37 | 373.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-10 13:30:00 | 375.30 | 374.37 | 373.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 373.25 | 374.19 | 373.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 368.80 | 374.19 | 373.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 368.35 | 373.02 | 373.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 365.50 | 371.52 | 372.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 362.70 | 362.58 | 365.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:45:00 | 363.15 | 362.58 | 365.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 365.50 | 363.27 | 365.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:00:00 | 365.50 | 363.27 | 365.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 364.55 | 363.53 | 365.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:30:00 | 364.05 | 363.53 | 365.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 364.00 | 363.76 | 365.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 368.45 | 363.76 | 365.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 364.95 | 364.00 | 365.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:15:00 | 364.35 | 364.00 | 365.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:45:00 | 363.25 | 363.92 | 364.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 09:30:00 | 362.50 | 363.85 | 364.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:45:00 | 364.25 | 363.92 | 364.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 364.00 | 363.93 | 364.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 11:15:00 | 362.05 | 363.54 | 363.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 11:15:00 | 346.13 | 351.78 | 355.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 11:15:00 | 345.09 | 351.78 | 355.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 11:15:00 | 344.38 | 351.78 | 355.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 11:15:00 | 346.04 | 351.78 | 355.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 11:15:00 | 343.95 | 351.78 | 355.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 14:15:00 | 351.85 | 351.24 | 354.26 | SL hit (close>ema200) qty=0.50 sl=351.24 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 11:15:00 | 351.00 | 350.21 | 350.18 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 14:15:00 | 346.80 | 349.81 | 350.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 345.25 | 348.77 | 349.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 11:15:00 | 338.40 | 338.28 | 341.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 11:45:00 | 338.15 | 338.28 | 341.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 337.85 | 338.28 | 340.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 13:30:00 | 340.45 | 338.28 | 340.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 338.30 | 336.06 | 337.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:45:00 | 340.20 | 336.06 | 337.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 341.65 | 337.18 | 338.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:00:00 | 341.65 | 337.18 | 338.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 12:15:00 | 340.60 | 338.96 | 338.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 13:15:00 | 344.50 | 342.49 | 340.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 15:15:00 | 342.50 | 342.61 | 341.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 09:15:00 | 338.70 | 342.61 | 341.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 341.35 | 342.36 | 341.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:30:00 | 340.85 | 342.36 | 341.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 331.00 | 340.09 | 340.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 323.45 | 332.08 | 334.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 13:15:00 | 322.65 | 320.25 | 323.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 14:00:00 | 322.65 | 320.25 | 323.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 317.20 | 317.68 | 319.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 11:15:00 | 315.35 | 317.61 | 319.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 14:15:00 | 314.85 | 312.55 | 312.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 14:15:00 | 315.45 | 313.13 | 313.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 315.45 | 313.13 | 313.04 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 310.30 | 313.08 | 313.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 11:15:00 | 309.00 | 312.26 | 312.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 12:15:00 | 300.10 | 299.11 | 301.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 13:00:00 | 300.10 | 299.11 | 301.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 300.55 | 299.40 | 301.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:30:00 | 300.90 | 299.40 | 301.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 300.50 | 299.62 | 301.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 300.50 | 299.62 | 301.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 300.05 | 299.70 | 300.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:15:00 | 301.05 | 299.70 | 300.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 300.20 | 299.80 | 300.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 14:45:00 | 297.60 | 299.64 | 300.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:15:00 | 282.72 | 289.68 | 293.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 290.25 | 284.28 | 288.51 | SL hit (close>ema200) qty=0.50 sl=284.28 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 288.00 | 286.83 | 286.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 290.50 | 288.02 | 287.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 10:15:00 | 288.30 | 288.94 | 288.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 10:15:00 | 288.30 | 288.94 | 288.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 288.30 | 288.94 | 288.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 288.30 | 288.94 | 288.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 287.35 | 288.62 | 288.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:30:00 | 286.60 | 288.62 | 288.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 287.75 | 288.45 | 288.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:45:00 | 286.90 | 288.45 | 288.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 14:15:00 | 285.60 | 287.73 | 287.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 15:15:00 | 285.25 | 287.23 | 287.53 | Break + close below crossover candle low |

### Cycle 43 — BUY (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 09:15:00 | 291.05 | 287.99 | 287.85 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 287.30 | 288.39 | 288.40 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 11:15:00 | 290.30 | 288.77 | 288.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 09:15:00 | 293.00 | 289.88 | 289.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 12:15:00 | 291.60 | 291.76 | 290.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-18 13:00:00 | 291.60 | 291.76 | 290.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 297.15 | 297.34 | 294.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 297.15 | 297.34 | 294.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 294.60 | 296.58 | 294.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:30:00 | 294.15 | 296.58 | 294.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 291.65 | 295.59 | 294.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 11:00:00 | 291.65 | 295.59 | 294.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 11:15:00 | 290.20 | 294.51 | 294.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 12:00:00 | 290.20 | 294.51 | 294.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 12:15:00 | 291.00 | 293.81 | 293.89 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 09:15:00 | 299.05 | 294.65 | 294.19 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 285.60 | 295.21 | 296.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 10:15:00 | 285.00 | 293.17 | 295.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 288.15 | 287.89 | 291.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 09:30:00 | 288.10 | 287.89 | 291.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 286.35 | 284.54 | 286.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 286.35 | 284.54 | 286.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 287.35 | 285.11 | 286.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 292.20 | 285.11 | 286.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 287.45 | 285.57 | 286.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 289.10 | 285.57 | 286.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 286.75 | 285.81 | 286.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:30:00 | 287.00 | 285.81 | 286.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 287.80 | 286.21 | 286.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:45:00 | 287.85 | 286.21 | 286.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 287.85 | 286.54 | 286.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:30:00 | 287.95 | 286.54 | 286.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 15:15:00 | 288.00 | 287.01 | 286.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 288.45 | 287.40 | 287.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 293.80 | 296.12 | 293.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 10:00:00 | 293.80 | 296.12 | 293.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 289.65 | 294.83 | 292.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 289.65 | 294.83 | 292.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 290.10 | 293.88 | 292.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:45:00 | 289.10 | 293.88 | 292.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 290.90 | 293.29 | 292.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:30:00 | 289.50 | 293.29 | 292.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 292.95 | 292.86 | 292.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 15:15:00 | 289.65 | 292.86 | 292.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 289.65 | 292.22 | 292.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 286.00 | 292.22 | 292.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 09:15:00 | 285.50 | 290.88 | 291.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 283.90 | 286.64 | 288.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 10:15:00 | 273.20 | 273.12 | 276.93 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 12:30:00 | 271.20 | 272.81 | 276.12 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 13:45:00 | 270.80 | 272.30 | 275.59 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 10:15:00 | 257.64 | 268.24 | 272.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 272.00 | 268.99 | 272.41 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-14 11:15:00 | 272.00 | 268.99 | 272.41 | SL hit (close>ema200) qty=0.50 sl=268.99 alert=retest1 |

### Cycle 51 — BUY (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 15:15:00 | 273.20 | 271.72 | 271.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 12:15:00 | 274.40 | 273.14 | 272.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 269.85 | 273.28 | 272.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 269.85 | 273.28 | 272.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 269.85 | 273.28 | 272.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:30:00 | 270.15 | 273.28 | 272.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 270.15 | 272.65 | 272.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:00:00 | 270.15 | 272.65 | 272.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 270.00 | 272.12 | 272.36 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 12:15:00 | 274.10 | 272.27 | 272.16 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 271.05 | 272.35 | 272.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 09:15:00 | 260.35 | 269.70 | 271.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 266.45 | 263.45 | 266.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 09:15:00 | 266.45 | 263.45 | 266.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 266.45 | 263.45 | 266.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:00:00 | 266.45 | 263.45 | 266.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 267.15 | 264.19 | 266.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 267.15 | 264.19 | 266.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 267.80 | 264.91 | 266.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:00:00 | 267.80 | 264.91 | 266.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 269.05 | 265.74 | 266.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 13:00:00 | 269.05 | 265.74 | 266.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 272.05 | 267.81 | 267.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 274.05 | 269.74 | 268.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 272.55 | 272.96 | 271.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 272.55 | 272.96 | 271.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 272.55 | 272.96 | 271.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:30:00 | 272.20 | 272.96 | 271.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 271.30 | 273.17 | 271.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 271.30 | 273.17 | 271.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 266.90 | 271.92 | 271.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 268.95 | 271.92 | 271.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 266.40 | 270.81 | 270.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 266.40 | 270.81 | 270.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 264.05 | 269.46 | 270.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 264.00 | 268.37 | 269.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 269.30 | 268.55 | 269.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 269.30 | 268.55 | 269.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 269.30 | 268.55 | 269.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:45:00 | 269.20 | 268.55 | 269.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 269.00 | 268.64 | 269.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 265.35 | 269.44 | 269.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 271.00 | 269.75 | 269.75 | SL hit (close>static) qty=1.00 sl=270.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-02-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 15:15:00 | 270.95 | 269.73 | 269.64 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 10:15:00 | 268.95 | 269.50 | 269.54 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 11:15:00 | 270.65 | 269.73 | 269.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 14:15:00 | 272.05 | 270.70 | 270.15 | Break + close above crossover candle high |

### Cycle 60 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 265.50 | 269.74 | 269.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 261.65 | 265.91 | 267.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 12:15:00 | 269.80 | 265.90 | 267.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 12:15:00 | 269.80 | 265.90 | 267.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 269.80 | 265.90 | 267.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 269.80 | 265.90 | 267.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 268.85 | 266.49 | 267.27 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 270.65 | 268.23 | 267.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 275.50 | 269.68 | 268.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 280.35 | 280.73 | 276.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 10:00:00 | 280.35 | 280.73 | 276.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 276.80 | 279.47 | 276.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:00:00 | 276.80 | 279.47 | 276.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 275.55 | 278.69 | 276.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:00:00 | 275.55 | 278.69 | 276.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 274.70 | 277.89 | 276.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:30:00 | 274.40 | 277.89 | 276.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 272.25 | 275.87 | 275.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 261.30 | 269.23 | 272.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 13:15:00 | 267.00 | 266.96 | 269.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 14:00:00 | 267.00 | 266.96 | 269.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 15:15:00 | 270.05 | 267.93 | 269.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 247.35 | 267.93 | 269.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 09:15:00 | 234.98 | 243.99 | 254.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-18 09:15:00 | 222.62 | 231.64 | 241.99 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 63 — BUY (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 13:15:00 | 217.15 | 215.85 | 215.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 14:15:00 | 220.50 | 216.78 | 216.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 12:15:00 | 217.40 | 217.76 | 217.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 12:15:00 | 217.40 | 217.76 | 217.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 217.40 | 217.76 | 217.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:45:00 | 217.25 | 217.76 | 217.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 218.80 | 217.96 | 217.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:45:00 | 215.60 | 217.96 | 217.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 217.80 | 217.93 | 217.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 215.05 | 217.93 | 217.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 216.75 | 217.69 | 217.25 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 213.10 | 216.54 | 216.79 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 12:15:00 | 217.76 | 216.54 | 216.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-04 13:15:00 | 217.77 | 216.78 | 216.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-05 11:15:00 | 217.21 | 217.23 | 216.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 11:15:00 | 217.21 | 217.23 | 216.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 217.21 | 217.23 | 216.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:00:00 | 217.21 | 217.23 | 216.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 216.18 | 217.02 | 216.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:00:00 | 216.18 | 217.02 | 216.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 216.45 | 216.91 | 216.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 14:15:00 | 216.78 | 216.91 | 216.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 15:15:00 | 215.25 | 216.52 | 216.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 15:15:00 | 215.25 | 216.52 | 216.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 210.07 | 214.81 | 215.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 214.03 | 211.92 | 213.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 214.03 | 211.92 | 213.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 214.03 | 211.92 | 213.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:15:00 | 218.27 | 211.92 | 213.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 218.23 | 213.18 | 213.77 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 218.46 | 215.01 | 214.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 224.78 | 218.18 | 216.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 221.25 | 222.58 | 220.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 221.25 | 222.58 | 220.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 221.25 | 222.58 | 220.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 219.71 | 222.58 | 220.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 228.42 | 224.45 | 222.28 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2026-03-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 10:15:00 | 213.53 | 220.85 | 221.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 15:15:00 | 212.90 | 216.15 | 218.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 14:15:00 | 213.10 | 213.03 | 215.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 14:45:00 | 214.40 | 213.03 | 215.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 214.40 | 213.76 | 214.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 14:45:00 | 214.58 | 213.76 | 214.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 15:15:00 | 209.21 | 210.57 | 212.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:15:00 | 214.38 | 210.57 | 212.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 235.50 | 215.56 | 214.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 10:15:00 | 249.86 | 222.42 | 217.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 232.70 | 239.77 | 230.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-23 10:00:00 | 232.70 | 239.77 | 230.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 229.58 | 237.73 | 230.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 11:00:00 | 229.58 | 237.73 | 230.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 11:15:00 | 227.70 | 235.73 | 230.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 11:45:00 | 226.65 | 235.73 | 230.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 09:15:00 | 224.47 | 226.93 | 227.23 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 10:15:00 | 229.62 | 227.46 | 227.45 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 14:15:00 | 225.17 | 227.39 | 227.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 09:15:00 | 213.98 | 223.00 | 225.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 12:15:00 | 220.59 | 219.97 | 222.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-27 13:00:00 | 220.59 | 219.97 | 222.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 224.69 | 220.71 | 222.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 224.69 | 220.71 | 222.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 221.51 | 220.87 | 222.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 09:15:00 | 216.29 | 220.87 | 222.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 09:30:00 | 219.90 | 215.02 | 217.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 220.01 | 218.77 | 218.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 224.75 | 219.65 | 219.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 224.75 | 219.65 | 219.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 11:15:00 | 225.98 | 223.21 | 221.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 14:15:00 | 244.00 | 244.87 | 236.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 15:00:00 | 244.00 | 244.87 | 236.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 240.34 | 244.21 | 240.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 240.34 | 244.21 | 240.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 243.65 | 244.10 | 241.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:30:00 | 248.30 | 243.48 | 241.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 237.85 | 243.49 | 243.10 | SL hit (close<static) qty=1.00 sl=240.20 alert=retest2 |

### Cycle 74 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 248.11 | 251.75 | 252.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 244.50 | 249.25 | 250.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 247.66 | 246.02 | 248.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 247.66 | 246.02 | 248.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 247.66 | 246.02 | 248.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 248.50 | 246.02 | 248.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 247.80 | 246.38 | 248.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 247.93 | 246.38 | 248.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 248.15 | 246.73 | 248.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:15:00 | 251.91 | 246.73 | 248.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 251.49 | 247.68 | 248.45 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 251.72 | 249.23 | 249.07 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 12:15:00 | 247.01 | 248.83 | 249.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 14:15:00 | 245.39 | 247.82 | 248.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 13:15:00 | 240.70 | 240.66 | 243.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 14:00:00 | 240.70 | 240.66 | 243.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 239.82 | 239.90 | 242.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:45:00 | 238.72 | 239.89 | 242.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 237.82 | 239.48 | 241.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 15:00:00 | 239.35 | 236.97 | 237.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 240.00 | 237.57 | 237.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 240.00 | 237.57 | 237.36 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 11:15:00 | 236.36 | 237.87 | 237.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 14:15:00 | 235.42 | 237.06 | 237.54 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 09:15:00 | 353.35 | 2025-05-20 12:15:00 | 347.25 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-05-15 13:30:00 | 346.55 | 2025-05-20 12:15:00 | 347.25 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-06-05 09:45:00 | 386.25 | 2025-06-06 10:15:00 | 424.88 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-12 13:45:00 | 383.50 | 2025-06-19 09:15:00 | 364.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-16 09:15:00 | 376.50 | 2025-06-19 10:15:00 | 357.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 09:45:00 | 380.05 | 2025-06-19 10:15:00 | 361.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 13:45:00 | 383.50 | 2025-06-19 15:15:00 | 345.15 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-06-16 09:15:00 | 376.50 | 2025-06-20 12:15:00 | 342.05 | TARGET_HIT | 0.50 | 9.15% |
| SELL | retest2 | 2025-06-18 09:45:00 | 380.05 | 2025-06-20 14:15:00 | 338.85 | TARGET_HIT | 0.50 | 10.84% |
| SELL | retest2 | 2025-07-07 15:15:00 | 370.15 | 2025-07-08 09:15:00 | 376.35 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-07-28 12:15:00 | 351.55 | 2025-08-01 10:15:00 | 357.20 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-07-29 10:15:00 | 351.55 | 2025-08-01 10:15:00 | 357.20 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-07-29 12:45:00 | 351.50 | 2025-08-04 09:15:00 | 355.20 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-07-29 14:45:00 | 352.10 | 2025-08-04 09:15:00 | 355.20 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-07-30 10:15:00 | 347.60 | 2025-08-04 09:15:00 | 355.20 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-07-30 11:30:00 | 348.30 | 2025-08-04 09:15:00 | 355.20 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-08-01 14:15:00 | 348.30 | 2025-08-04 09:15:00 | 355.20 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-08-01 15:15:00 | 347.55 | 2025-08-04 09:15:00 | 355.20 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-08-06 12:15:00 | 360.30 | 2025-08-06 14:15:00 | 356.50 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-09-01 09:15:00 | 359.30 | 2025-09-04 09:15:00 | 375.55 | STOP_HIT | 1.00 | -4.52% |
| SELL | retest2 | 2025-09-01 13:30:00 | 359.55 | 2025-09-04 09:15:00 | 375.55 | STOP_HIT | 1.00 | -4.45% |
| SELL | retest2 | 2025-09-02 10:30:00 | 359.05 | 2025-09-04 09:15:00 | 375.55 | STOP_HIT | 1.00 | -4.60% |
| SELL | retest2 | 2025-09-02 11:15:00 | 359.50 | 2025-09-04 09:15:00 | 375.55 | STOP_HIT | 1.00 | -4.46% |
| SELL | retest2 | 2025-09-12 10:45:00 | 392.90 | 2025-09-15 11:15:00 | 396.85 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-09-12 11:45:00 | 392.55 | 2025-09-15 11:15:00 | 396.85 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-09-12 13:30:00 | 390.55 | 2025-09-15 11:15:00 | 396.85 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-09-18 12:00:00 | 389.15 | 2025-09-26 11:15:00 | 369.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 13:00:00 | 389.10 | 2025-09-26 11:15:00 | 369.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 09:30:00 | 387.45 | 2025-09-26 11:15:00 | 368.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 10:15:00 | 388.80 | 2025-09-26 11:15:00 | 369.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 09:15:00 | 389.30 | 2025-09-26 11:15:00 | 369.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 11:00:00 | 388.45 | 2025-09-26 11:15:00 | 369.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 13:30:00 | 389.20 | 2025-09-26 11:15:00 | 369.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 15:15:00 | 386.45 | 2025-09-26 11:15:00 | 367.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 10:15:00 | 380.30 | 2025-09-26 15:15:00 | 361.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 12:30:00 | 379.35 | 2025-09-26 15:15:00 | 361.47 | PARTIAL | 0.50 | 4.71% |
| SELL | retest2 | 2025-09-25 14:00:00 | 380.50 | 2025-09-29 09:15:00 | 360.38 | PARTIAL | 0.50 | 5.29% |
| SELL | retest2 | 2025-09-18 12:00:00 | 389.15 | 2025-09-30 09:15:00 | 366.85 | STOP_HIT | 0.50 | 5.73% |
| SELL | retest2 | 2025-09-18 13:00:00 | 389.10 | 2025-09-30 09:15:00 | 366.85 | STOP_HIT | 0.50 | 5.72% |
| SELL | retest2 | 2025-09-19 09:30:00 | 387.45 | 2025-09-30 09:15:00 | 366.85 | STOP_HIT | 0.50 | 5.32% |
| SELL | retest2 | 2025-09-19 10:15:00 | 388.80 | 2025-09-30 09:15:00 | 366.85 | STOP_HIT | 0.50 | 5.65% |
| SELL | retest2 | 2025-09-22 09:15:00 | 389.30 | 2025-09-30 09:15:00 | 366.85 | STOP_HIT | 0.50 | 5.77% |
| SELL | retest2 | 2025-09-22 11:00:00 | 388.45 | 2025-09-30 09:15:00 | 366.85 | STOP_HIT | 0.50 | 5.56% |
| SELL | retest2 | 2025-09-23 13:30:00 | 389.20 | 2025-09-30 09:15:00 | 366.85 | STOP_HIT | 0.50 | 5.74% |
| SELL | retest2 | 2025-09-23 15:15:00 | 386.45 | 2025-09-30 09:15:00 | 366.85 | STOP_HIT | 0.50 | 5.07% |
| SELL | retest2 | 2025-09-24 10:15:00 | 380.30 | 2025-09-30 09:15:00 | 366.85 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2025-09-25 12:30:00 | 379.35 | 2025-09-30 09:15:00 | 366.85 | STOP_HIT | 0.50 | 3.30% |
| SELL | retest2 | 2025-09-25 14:00:00 | 380.50 | 2025-09-30 09:15:00 | 366.85 | STOP_HIT | 0.50 | 3.59% |
| SELL | retest2 | 2025-10-16 10:15:00 | 364.35 | 2025-10-28 11:15:00 | 346.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-16 10:45:00 | 363.25 | 2025-10-28 11:15:00 | 345.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-17 09:30:00 | 362.50 | 2025-10-28 11:15:00 | 344.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-17 10:45:00 | 364.25 | 2025-10-28 11:15:00 | 346.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-23 11:15:00 | 362.05 | 2025-10-28 11:15:00 | 343.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-16 10:15:00 | 364.35 | 2025-10-28 14:15:00 | 351.85 | STOP_HIT | 0.50 | 3.43% |
| SELL | retest2 | 2025-10-16 10:45:00 | 363.25 | 2025-10-28 14:15:00 | 351.85 | STOP_HIT | 0.50 | 3.14% |
| SELL | retest2 | 2025-10-17 09:30:00 | 362.50 | 2025-10-28 14:15:00 | 351.85 | STOP_HIT | 0.50 | 2.94% |
| SELL | retest2 | 2025-10-17 10:45:00 | 364.25 | 2025-10-28 14:15:00 | 351.85 | STOP_HIT | 0.50 | 3.40% |
| SELL | retest2 | 2025-10-23 11:15:00 | 362.05 | 2025-10-28 14:15:00 | 351.85 | STOP_HIT | 0.50 | 2.82% |
| SELL | retest2 | 2025-11-24 11:15:00 | 315.35 | 2025-11-26 14:15:00 | 315.45 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-11-26 14:15:00 | 314.85 | 2025-11-26 14:15:00 | 315.45 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-12-04 14:45:00 | 297.60 | 2025-12-08 10:15:00 | 282.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 14:45:00 | 297.60 | 2025-12-09 09:15:00 | 290.25 | STOP_HIT | 0.50 | 2.47% |
| SELL | retest2 | 2025-12-09 14:00:00 | 292.05 | 2025-12-12 10:15:00 | 288.00 | STOP_HIT | 1.00 | 1.39% |
| SELL | retest1 | 2026-01-13 12:30:00 | 271.20 | 2026-01-14 10:15:00 | 257.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-13 12:30:00 | 271.20 | 2026-01-14 11:15:00 | 272.00 | STOP_HIT | 0.50 | -0.29% |
| SELL | retest1 | 2026-01-13 13:45:00 | 270.80 | 2026-01-16 12:15:00 | 272.25 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-01-14 13:45:00 | 267.60 | 2026-01-16 15:15:00 | 273.20 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2026-01-16 09:15:00 | 267.55 | 2026-01-16 15:15:00 | 273.20 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2026-01-16 09:45:00 | 268.00 | 2026-01-16 15:15:00 | 273.20 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-02-03 09:15:00 | 265.35 | 2026-02-03 09:15:00 | 271.00 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2026-02-03 11:00:00 | 266.20 | 2026-02-03 11:15:00 | 270.10 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-02-03 13:15:00 | 267.85 | 2026-02-03 14:15:00 | 270.05 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-02-16 09:15:00 | 247.35 | 2026-02-17 09:15:00 | 234.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-16 09:15:00 | 247.35 | 2026-02-18 09:15:00 | 222.62 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-03-05 14:15:00 | 216.78 | 2026-03-05 15:15:00 | 215.25 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-03-30 09:15:00 | 216.29 | 2026-04-02 13:15:00 | 224.75 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2026-04-01 09:30:00 | 219.90 | 2026-04-02 13:15:00 | 224.75 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2026-04-01 13:30:00 | 220.01 | 2026-04-02 13:15:00 | 224.75 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2026-04-10 09:30:00 | 248.30 | 2026-04-13 09:15:00 | 237.85 | STOP_HIT | 1.00 | -4.21% |
| BUY | retest2 | 2026-04-13 15:00:00 | 245.19 | 2026-04-23 14:15:00 | 248.11 | STOP_HIT | 1.00 | 1.19% |
| SELL | retest2 | 2026-05-04 10:45:00 | 238.72 | 2026-05-06 15:15:00 | 240.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-05-04 12:00:00 | 237.82 | 2026-05-06 15:15:00 | 240.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-05-06 15:00:00 | 239.35 | 2026-05-06 15:15:00 | 240.00 | STOP_HIT | 1.00 | -0.27% |
