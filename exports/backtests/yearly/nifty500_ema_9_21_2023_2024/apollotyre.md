# Apollo Tyres Ltd. (APOLLOTYRE)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 408.65
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 216 |
| ALERT1 | 150 |
| ALERT2 | 147 |
| ALERT2_SKIP | 74 |
| ALERT3 | 415 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 228 |
| PARTIAL | 14 |
| TARGET_HIT | 4 |
| STOP_HIT | 231 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 249 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 62 / 187
- **Target hits / Stop hits / Partials:** 4 / 231 / 14
- **Avg / median % per leg:** -0.06% / -0.75%
- **Sum % (uncompounded):** -13.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 119 | 24 | 20.2% | 1 | 118 | 0 | -0.32% | -38.2% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.14% | -3.4% |
| BUY @ 3rd Alert (retest2) | 116 | 24 | 20.7% | 1 | 115 | 0 | -0.30% | -34.8% |
| SELL (all) | 130 | 38 | 29.2% | 3 | 113 | 14 | 0.19% | 24.4% |
| SELL @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 0 | 4 | 2 | 3.15% | 18.9% |
| SELL @ 3rd Alert (retest2) | 124 | 34 | 27.4% | 3 | 109 | 12 | 0.04% | 5.5% |
| retest1 (combined) | 9 | 4 | 44.4% | 0 | 7 | 2 | 1.72% | 15.5% |
| retest2 (combined) | 240 | 58 | 24.2% | 4 | 224 | 12 | -0.12% | -29.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 12:15:00 | 369.75 | 366.92 | 366.76 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 09:15:00 | 361.75 | 367.60 | 367.87 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 12:15:00 | 370.25 | 367.80 | 367.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 13:15:00 | 370.85 | 368.41 | 367.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 15:15:00 | 391.00 | 391.84 | 389.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-30 09:15:00 | 390.10 | 391.84 | 389.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 09:15:00 | 391.75 | 391.82 | 389.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 10:00:00 | 391.75 | 391.82 | 389.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 10:15:00 | 387.10 | 390.88 | 389.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 11:00:00 | 387.10 | 390.88 | 389.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 11:15:00 | 385.50 | 389.80 | 389.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 12:00:00 | 385.50 | 389.80 | 389.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2023-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 13:15:00 | 386.10 | 388.45 | 388.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-30 15:15:00 | 384.00 | 387.11 | 387.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-31 09:15:00 | 389.00 | 387.49 | 388.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 09:15:00 | 389.00 | 387.49 | 388.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 389.00 | 387.49 | 388.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-31 09:45:00 | 389.95 | 387.49 | 388.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 10:15:00 | 389.45 | 387.88 | 388.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-31 10:45:00 | 389.80 | 387.88 | 388.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2023-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 12:15:00 | 390.40 | 388.64 | 388.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-31 14:15:00 | 391.60 | 389.60 | 388.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-01 09:15:00 | 389.55 | 389.81 | 389.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 09:15:00 | 389.55 | 389.81 | 389.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 09:15:00 | 389.55 | 389.81 | 389.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-01 10:00:00 | 389.55 | 389.81 | 389.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 15:15:00 | 389.90 | 390.65 | 389.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 09:15:00 | 391.75 | 390.65 | 389.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-02 10:15:00 | 389.00 | 390.30 | 389.92 | SL hit (close<static) qty=1.00 sl=389.20 alert=retest2 |

### Cycle 6 — SELL (started 2023-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 15:15:00 | 407.95 | 409.25 | 409.41 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 09:15:00 | 420.60 | 410.95 | 409.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-22 09:15:00 | 424.00 | 417.96 | 414.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 11:15:00 | 418.40 | 418.74 | 415.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-22 12:00:00 | 418.40 | 418.74 | 415.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 12:15:00 | 416.20 | 418.23 | 415.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 13:00:00 | 416.20 | 418.23 | 415.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 13:15:00 | 414.30 | 417.45 | 415.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 14:00:00 | 414.30 | 417.45 | 415.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 14:15:00 | 414.60 | 416.88 | 415.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-23 09:45:00 | 419.65 | 417.02 | 415.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-23 12:30:00 | 417.40 | 416.55 | 415.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-23 15:15:00 | 417.20 | 416.31 | 415.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-26 09:15:00 | 412.50 | 415.69 | 415.60 | SL hit (close<static) qty=1.00 sl=413.20 alert=retest2 |

### Cycle 8 — SELL (started 2023-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-26 10:15:00 | 406.60 | 413.87 | 414.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-26 12:15:00 | 403.65 | 410.73 | 413.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-28 14:15:00 | 408.00 | 401.41 | 403.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 14:15:00 | 408.00 | 401.41 | 403.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 14:15:00 | 408.00 | 401.41 | 403.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-28 15:00:00 | 408.00 | 401.41 | 403.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 15:15:00 | 412.50 | 403.63 | 404.35 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2023-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 09:15:00 | 412.30 | 405.36 | 405.07 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-03 09:15:00 | 398.65 | 404.46 | 404.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 12:15:00 | 396.90 | 398.72 | 400.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-05 09:15:00 | 398.90 | 398.18 | 399.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-05 10:00:00 | 398.90 | 398.18 | 399.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 15:15:00 | 399.70 | 397.86 | 398.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-06 09:15:00 | 403.40 | 397.86 | 398.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 09:15:00 | 403.25 | 398.94 | 399.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-06 09:30:00 | 403.75 | 398.94 | 399.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2023-07-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 10:15:00 | 409.25 | 401.00 | 400.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 12:15:00 | 411.65 | 404.07 | 401.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 14:15:00 | 432.55 | 432.93 | 429.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-12 14:45:00 | 432.50 | 432.93 | 429.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 10:15:00 | 429.35 | 431.97 | 429.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 10:45:00 | 428.40 | 431.97 | 429.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 11:15:00 | 429.10 | 431.39 | 429.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 12:00:00 | 429.10 | 431.39 | 429.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 12:15:00 | 430.35 | 431.18 | 429.57 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-07-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 15:15:00 | 427.00 | 428.52 | 428.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-14 09:15:00 | 421.70 | 427.16 | 427.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-17 15:15:00 | 416.20 | 416.15 | 419.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-18 09:15:00 | 421.15 | 416.15 | 419.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 09:15:00 | 424.75 | 417.87 | 419.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-18 10:00:00 | 424.75 | 417.87 | 419.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 10:15:00 | 420.70 | 418.44 | 419.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-18 11:30:00 | 416.50 | 418.06 | 419.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-18 14:15:00 | 418.95 | 418.73 | 419.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-18 15:15:00 | 418.00 | 419.08 | 419.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-19 14:00:00 | 419.35 | 419.37 | 419.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 14:15:00 | 420.30 | 419.55 | 419.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-19 14:45:00 | 420.55 | 419.55 | 419.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-07-19 15:15:00 | 420.00 | 419.64 | 419.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2023-07-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 15:15:00 | 420.00 | 419.64 | 419.61 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-07-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 10:15:00 | 417.75 | 419.38 | 419.51 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-21 12:15:00 | 420.70 | 419.58 | 419.51 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 15:15:00 | 418.70 | 419.39 | 419.44 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 10:15:00 | 421.75 | 419.74 | 419.59 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 14:15:00 | 415.35 | 419.12 | 419.40 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 15:15:00 | 420.50 | 418.79 | 418.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-26 09:15:00 | 427.95 | 420.62 | 419.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-27 09:15:00 | 420.80 | 423.87 | 422.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 09:15:00 | 420.80 | 423.87 | 422.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 420.80 | 423.87 | 422.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 10:00:00 | 420.80 | 423.87 | 422.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 10:15:00 | 421.50 | 423.40 | 422.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 10:30:00 | 421.50 | 423.40 | 422.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 13:15:00 | 421.95 | 422.49 | 422.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 13:30:00 | 421.40 | 422.49 | 422.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 14:15:00 | 419.85 | 421.96 | 421.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 14:45:00 | 418.60 | 421.96 | 421.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2023-07-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 15:15:00 | 420.30 | 421.63 | 421.71 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-28 10:15:00 | 424.20 | 422.17 | 421.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-28 14:15:00 | 424.75 | 423.32 | 422.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 10:15:00 | 428.15 | 428.77 | 426.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-01 10:30:00 | 428.10 | 428.77 | 426.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 11:15:00 | 427.10 | 428.43 | 426.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 12:00:00 | 427.10 | 428.43 | 426.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 12:15:00 | 426.65 | 428.08 | 426.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 12:30:00 | 425.00 | 428.08 | 426.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 13:15:00 | 426.55 | 427.77 | 426.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-01 15:00:00 | 427.40 | 427.70 | 426.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-02 09:30:00 | 428.50 | 428.11 | 427.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-02 11:15:00 | 425.00 | 427.10 | 426.73 | SL hit (close<static) qty=1.00 sl=425.20 alert=retest2 |

### Cycle 22 — SELL (started 2023-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 12:15:00 | 422.45 | 426.17 | 426.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 13:15:00 | 419.70 | 424.88 | 425.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 09:15:00 | 425.45 | 424.71 | 425.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-03 09:15:00 | 425.45 | 424.71 | 425.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 09:15:00 | 425.45 | 424.71 | 425.43 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-08-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 14:15:00 | 430.55 | 425.87 | 425.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 11:15:00 | 435.20 | 429.68 | 427.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-07 09:15:00 | 434.00 | 434.39 | 431.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-07 09:45:00 | 433.70 | 434.39 | 431.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 10:15:00 | 432.45 | 434.12 | 432.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 11:00:00 | 432.45 | 434.12 | 432.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 11:15:00 | 433.60 | 434.02 | 432.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 11:30:00 | 432.40 | 434.02 | 432.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 10:15:00 | 436.35 | 435.87 | 434.52 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-08-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 14:15:00 | 431.50 | 435.01 | 435.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 09:15:00 | 410.45 | 429.44 | 432.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-18 13:15:00 | 393.50 | 390.91 | 394.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 13:15:00 | 393.50 | 390.91 | 394.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 13:15:00 | 393.50 | 390.91 | 394.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-18 13:30:00 | 394.65 | 390.91 | 394.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 394.50 | 391.88 | 394.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 09:45:00 | 395.00 | 391.88 | 394.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 10:15:00 | 396.70 | 392.84 | 394.28 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 13:15:00 | 397.45 | 395.35 | 395.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 09:15:00 | 399.60 | 397.07 | 396.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 11:15:00 | 396.60 | 397.26 | 396.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-22 12:00:00 | 396.60 | 397.26 | 396.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 12:15:00 | 394.45 | 396.70 | 396.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-22 12:45:00 | 394.70 | 396.70 | 396.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 13:15:00 | 394.50 | 396.26 | 396.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-22 14:00:00 | 394.50 | 396.26 | 396.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2023-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-22 14:15:00 | 393.85 | 395.78 | 395.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-23 09:15:00 | 389.30 | 394.36 | 395.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-23 13:15:00 | 394.55 | 392.76 | 393.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 13:15:00 | 394.55 | 392.76 | 393.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 13:15:00 | 394.55 | 392.76 | 393.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 13:45:00 | 394.80 | 392.76 | 393.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 14:15:00 | 393.30 | 392.87 | 393.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 14:45:00 | 394.65 | 392.87 | 393.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 09:15:00 | 396.40 | 393.71 | 394.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-24 10:30:00 | 394.70 | 394.00 | 394.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-24 12:00:00 | 395.00 | 394.20 | 394.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-24 12:15:00 | 399.05 | 395.17 | 394.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2023-08-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-24 12:15:00 | 399.05 | 395.17 | 394.72 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 12:15:00 | 392.50 | 394.60 | 394.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 13:15:00 | 390.60 | 393.80 | 394.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 10:15:00 | 392.00 | 391.78 | 393.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-28 11:00:00 | 392.00 | 391.78 | 393.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 12:15:00 | 392.90 | 391.89 | 392.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 13:00:00 | 392.90 | 391.89 | 392.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 13:15:00 | 390.90 | 391.69 | 392.70 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 12:15:00 | 397.10 | 393.78 | 393.33 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 14:15:00 | 389.50 | 392.93 | 393.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 14:15:00 | 388.60 | 390.58 | 391.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 11:15:00 | 389.25 | 389.25 | 390.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 11:15:00 | 389.25 | 389.25 | 390.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 11:15:00 | 389.25 | 389.25 | 390.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 11:30:00 | 389.80 | 389.25 | 390.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 13:15:00 | 387.90 | 389.14 | 390.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-01 14:30:00 | 387.00 | 388.08 | 389.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-05 09:45:00 | 387.40 | 388.28 | 388.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-05 10:15:00 | 387.35 | 388.28 | 388.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-05 11:00:00 | 386.40 | 387.91 | 388.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 15:15:00 | 382.00 | 381.45 | 383.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-07 09:15:00 | 385.00 | 381.45 | 383.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 09:15:00 | 384.95 | 382.15 | 383.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-07 10:00:00 | 384.95 | 382.15 | 383.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 10:15:00 | 383.70 | 382.46 | 383.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-07 10:30:00 | 384.20 | 382.46 | 383.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 11:15:00 | 386.10 | 383.19 | 383.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-07 12:00:00 | 386.10 | 383.19 | 383.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 12:15:00 | 385.45 | 383.64 | 383.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-07 13:15:00 | 385.95 | 383.64 | 383.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 15:15:00 | 383.85 | 384.06 | 384.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-08 09:15:00 | 384.60 | 384.06 | 384.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 09:15:00 | 384.30 | 384.11 | 384.14 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-09-08 10:15:00 | 384.65 | 384.21 | 384.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2023-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 10:15:00 | 384.65 | 384.21 | 384.19 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-08 12:15:00 | 382.10 | 383.84 | 384.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-08 14:15:00 | 381.50 | 383.24 | 383.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-11 11:15:00 | 384.25 | 383.03 | 383.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 11:15:00 | 384.25 | 383.03 | 383.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 11:15:00 | 384.25 | 383.03 | 383.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-11 11:30:00 | 385.00 | 383.03 | 383.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 12:15:00 | 383.40 | 383.10 | 383.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-12 09:45:00 | 380.60 | 382.81 | 383.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-12 11:30:00 | 381.70 | 382.26 | 382.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-12 14:45:00 | 381.55 | 381.72 | 382.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-13 10:15:00 | 381.90 | 381.89 | 382.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 10:15:00 | 383.05 | 382.12 | 382.45 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-09-13 11:15:00 | 386.05 | 382.91 | 382.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2023-09-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 11:15:00 | 386.05 | 382.91 | 382.78 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-09-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-14 14:15:00 | 382.55 | 383.34 | 383.36 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 09:15:00 | 388.30 | 384.24 | 383.77 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-15 12:15:00 | 378.80 | 382.69 | 383.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 09:15:00 | 375.75 | 378.92 | 381.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-20 10:15:00 | 375.20 | 374.78 | 377.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-20 11:00:00 | 375.20 | 374.78 | 377.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 12:15:00 | 377.15 | 375.41 | 377.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-20 13:00:00 | 377.15 | 375.41 | 377.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 13:15:00 | 377.75 | 375.88 | 377.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-20 13:45:00 | 377.45 | 375.88 | 377.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 14:15:00 | 378.45 | 376.39 | 377.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-21 09:15:00 | 371.10 | 376.91 | 377.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-25 11:15:00 | 375.90 | 372.99 | 372.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2023-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 11:15:00 | 375.90 | 372.99 | 372.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 12:15:00 | 376.90 | 373.77 | 373.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 09:15:00 | 375.40 | 375.45 | 374.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-26 10:00:00 | 375.40 | 375.45 | 374.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 10:15:00 | 375.20 | 375.40 | 374.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 10:45:00 | 372.95 | 375.40 | 374.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 11:15:00 | 374.70 | 375.26 | 374.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 12:00:00 | 374.70 | 375.26 | 374.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 12:15:00 | 372.00 | 374.61 | 374.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 13:00:00 | 372.00 | 374.61 | 374.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 13:15:00 | 371.85 | 374.06 | 374.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 13:45:00 | 371.80 | 374.06 | 374.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2023-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 14:15:00 | 370.80 | 373.41 | 373.75 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-09-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 12:15:00 | 375.55 | 373.87 | 373.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 14:15:00 | 376.15 | 374.62 | 374.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 09:15:00 | 373.10 | 374.85 | 374.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 09:15:00 | 373.10 | 374.85 | 374.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 09:15:00 | 373.10 | 374.85 | 374.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 09:30:00 | 373.85 | 374.85 | 374.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 10:15:00 | 372.10 | 374.30 | 374.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 10:30:00 | 371.85 | 374.30 | 374.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2023-09-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 11:15:00 | 371.00 | 373.64 | 373.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 12:15:00 | 370.05 | 372.92 | 373.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 09:15:00 | 373.75 | 370.95 | 372.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-29 09:15:00 | 373.75 | 370.95 | 372.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 09:15:00 | 373.75 | 370.95 | 372.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 09:45:00 | 373.75 | 370.95 | 372.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 10:15:00 | 372.35 | 371.23 | 372.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-29 14:45:00 | 370.00 | 371.24 | 371.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-04 09:15:00 | 369.20 | 370.66 | 371.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 10:15:00 | 369.65 | 370.21 | 370.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 11:00:00 | 371.30 | 370.43 | 370.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 11:15:00 | 370.60 | 370.46 | 370.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-05 11:30:00 | 372.70 | 370.46 | 370.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-10-05 12:15:00 | 370.80 | 370.53 | 370.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2023-10-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 12:15:00 | 370.80 | 370.53 | 370.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-05 13:15:00 | 371.30 | 370.68 | 370.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 370.05 | 376.50 | 374.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 370.05 | 376.50 | 374.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 370.05 | 376.50 | 374.68 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2023-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 14:15:00 | 370.95 | 373.25 | 373.56 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2023-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 10:15:00 | 375.95 | 373.78 | 373.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 11:15:00 | 378.40 | 375.88 | 374.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-12 15:15:00 | 378.70 | 378.98 | 377.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-13 09:15:00 | 378.50 | 378.98 | 377.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 378.20 | 378.82 | 377.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 12:15:00 | 380.35 | 378.19 | 377.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 11:15:00 | 379.90 | 379.66 | 378.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 12:00:00 | 380.05 | 379.73 | 378.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-19 09:15:00 | 379.80 | 383.41 | 383.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2023-10-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 09:15:00 | 379.80 | 383.41 | 383.72 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-10-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-19 13:15:00 | 385.15 | 383.96 | 383.88 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2023-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 10:15:00 | 382.80 | 383.69 | 383.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 09:15:00 | 379.95 | 382.73 | 383.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 374.80 | 372.61 | 374.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 374.80 | 372.61 | 374.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 374.80 | 372.61 | 374.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:30:00 | 374.50 | 372.61 | 374.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 377.35 | 373.56 | 374.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 11:00:00 | 377.35 | 373.56 | 374.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 11:15:00 | 375.40 | 373.93 | 374.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-27 12:45:00 | 374.75 | 374.21 | 374.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-27 15:15:00 | 374.50 | 374.93 | 375.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-31 09:15:00 | 378.40 | 375.18 | 374.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2023-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 09:15:00 | 378.40 | 375.18 | 374.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 10:15:00 | 380.30 | 376.20 | 375.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 14:15:00 | 380.30 | 381.11 | 379.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-01 15:00:00 | 380.30 | 381.11 | 379.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 12:15:00 | 385.20 | 386.54 | 384.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-03 12:45:00 | 385.10 | 386.54 | 384.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 14:15:00 | 384.55 | 385.86 | 384.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-03 14:30:00 | 384.55 | 385.86 | 384.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 15:15:00 | 384.00 | 385.49 | 384.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 09:15:00 | 386.05 | 385.49 | 384.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-07 11:15:00 | 383.60 | 387.38 | 386.95 | SL hit (close<static) qty=1.00 sl=383.70 alert=retest2 |

### Cycle 48 — SELL (started 2023-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-07 12:15:00 | 383.65 | 386.63 | 386.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-07 13:15:00 | 382.60 | 385.83 | 386.28 | Break + close below crossover candle low |

### Cycle 49 — BUY (started 2023-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 09:15:00 | 404.40 | 389.15 | 387.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-08 10:15:00 | 409.00 | 393.12 | 389.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-10 09:15:00 | 415.75 | 416.06 | 408.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-10 10:15:00 | 415.00 | 416.06 | 408.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 423.65 | 429.09 | 427.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 10:00:00 | 423.65 | 429.09 | 427.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 10:15:00 | 421.75 | 427.62 | 426.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 11:00:00 | 421.75 | 427.62 | 426.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2023-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 13:15:00 | 424.15 | 426.03 | 426.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 09:15:00 | 422.45 | 424.51 | 425.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 09:15:00 | 421.90 | 421.30 | 422.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 09:15:00 | 421.90 | 421.30 | 422.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 421.90 | 421.30 | 422.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 09:30:00 | 421.65 | 421.30 | 422.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 415.70 | 419.80 | 421.30 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2023-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 15:15:00 | 423.35 | 419.57 | 419.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 10:15:00 | 423.40 | 420.76 | 419.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 12:15:00 | 425.40 | 425.42 | 423.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-30 13:00:00 | 425.40 | 425.42 | 423.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 14:15:00 | 455.55 | 462.28 | 458.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-07 15:00:00 | 455.55 | 462.28 | 458.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 15:15:00 | 456.00 | 461.02 | 458.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-08 09:15:00 | 461.80 | 461.02 | 458.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-08 12:30:00 | 457.20 | 459.50 | 458.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-08 14:30:00 | 457.30 | 458.55 | 458.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-11 11:15:00 | 456.70 | 457.97 | 458.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2023-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 11:15:00 | 456.70 | 457.97 | 458.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-11 12:15:00 | 455.00 | 457.38 | 457.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-12 10:15:00 | 456.75 | 455.26 | 456.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 10:15:00 | 456.75 | 455.26 | 456.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 10:15:00 | 456.75 | 455.26 | 456.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-12 10:30:00 | 455.90 | 455.26 | 456.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 11:15:00 | 455.40 | 455.29 | 456.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-12 12:30:00 | 455.10 | 455.04 | 456.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-14 13:15:00 | 461.55 | 453.33 | 452.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2023-12-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 13:15:00 | 461.55 | 453.33 | 452.35 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2023-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 12:15:00 | 448.65 | 454.27 | 454.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 13:15:00 | 438.95 | 451.20 | 453.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 435.50 | 434.50 | 440.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-22 10:00:00 | 435.50 | 434.50 | 440.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 09:15:00 | 439.35 | 431.48 | 432.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-27 10:00:00 | 439.35 | 431.48 | 432.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 10:15:00 | 439.70 | 433.12 | 433.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-27 10:45:00 | 440.15 | 433.12 | 433.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2023-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 11:15:00 | 440.25 | 434.55 | 434.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-29 09:15:00 | 443.70 | 439.92 | 437.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 10:15:00 | 450.50 | 451.94 | 448.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-02 10:15:00 | 450.50 | 451.94 | 448.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 450.50 | 451.94 | 448.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:30:00 | 449.60 | 451.94 | 448.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 10:15:00 | 450.90 | 455.65 | 454.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-04 10:30:00 | 450.55 | 455.65 | 454.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 11:15:00 | 448.60 | 454.24 | 453.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-04 11:45:00 | 448.25 | 454.24 | 453.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2024-01-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-04 12:15:00 | 448.70 | 453.13 | 453.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-04 13:15:00 | 447.85 | 452.07 | 452.73 | Break + close below crossover candle low |

### Cycle 57 — BUY (started 2024-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 09:15:00 | 460.45 | 452.92 | 452.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-05 12:15:00 | 462.25 | 456.76 | 454.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 09:15:00 | 458.15 | 458.46 | 456.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-08 09:15:00 | 458.15 | 458.46 | 456.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 09:15:00 | 458.15 | 458.46 | 456.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 09:45:00 | 457.35 | 458.46 | 456.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 10:15:00 | 455.85 | 457.94 | 456.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 11:00:00 | 455.85 | 457.94 | 456.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 11:15:00 | 454.15 | 457.18 | 456.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 12:00:00 | 454.15 | 457.18 | 456.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 12:15:00 | 455.65 | 456.87 | 456.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 09:15:00 | 469.30 | 456.05 | 455.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-20 12:15:00 | 516.23 | 506.52 | 497.34 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2024-02-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 11:15:00 | 542.75 | 548.45 | 548.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 14:15:00 | 538.30 | 544.46 | 546.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-12 13:15:00 | 517.25 | 514.73 | 522.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-12 14:00:00 | 517.25 | 514.73 | 522.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 13:15:00 | 517.25 | 511.87 | 514.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 14:00:00 | 517.25 | 511.87 | 514.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 14:15:00 | 517.85 | 513.06 | 514.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 15:00:00 | 517.85 | 513.06 | 514.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2024-02-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 10:15:00 | 520.35 | 516.08 | 515.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 12:15:00 | 522.85 | 517.96 | 516.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 14:15:00 | 531.20 | 531.44 | 527.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-19 15:00:00 | 531.20 | 531.44 | 527.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 522.90 | 530.30 | 527.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 10:00:00 | 522.90 | 530.30 | 527.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 10:15:00 | 517.10 | 527.66 | 526.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 11:00:00 | 517.10 | 527.66 | 526.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2024-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 12:15:00 | 520.45 | 524.97 | 525.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 13:15:00 | 517.90 | 523.56 | 524.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-21 09:15:00 | 522.60 | 522.03 | 523.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 09:15:00 | 522.60 | 522.03 | 523.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 09:15:00 | 522.60 | 522.03 | 523.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-21 10:00:00 | 522.60 | 522.03 | 523.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 10:15:00 | 522.85 | 522.20 | 523.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-21 10:30:00 | 522.60 | 522.20 | 523.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 12:15:00 | 523.40 | 522.27 | 523.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-21 12:45:00 | 523.85 | 522.27 | 523.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 13:15:00 | 520.60 | 521.94 | 523.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-21 13:30:00 | 523.70 | 521.94 | 523.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 13:15:00 | 519.90 | 518.57 | 520.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 14:00:00 | 519.90 | 518.57 | 520.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 522.50 | 519.35 | 520.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 14:45:00 | 524.40 | 519.35 | 520.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 523.00 | 520.08 | 520.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:15:00 | 525.15 | 520.08 | 520.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2024-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 10:15:00 | 523.55 | 521.40 | 521.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 11:15:00 | 524.50 | 522.02 | 521.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-23 15:15:00 | 522.20 | 523.38 | 522.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-23 15:15:00 | 522.20 | 523.38 | 522.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 15:15:00 | 522.20 | 523.38 | 522.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 09:15:00 | 524.75 | 523.38 | 522.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 521.95 | 523.09 | 522.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 09:45:00 | 521.45 | 523.09 | 522.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 10:15:00 | 522.90 | 523.05 | 522.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 10:30:00 | 520.30 | 523.05 | 522.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 11:15:00 | 524.35 | 523.31 | 522.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-26 12:15:00 | 526.00 | 523.31 | 522.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-26 12:45:00 | 525.70 | 523.57 | 522.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-26 13:45:00 | 526.95 | 524.38 | 523.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-27 09:15:00 | 526.40 | 524.84 | 523.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 523.80 | 524.63 | 523.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 10:00:00 | 523.80 | 524.63 | 523.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 10:15:00 | 525.55 | 524.82 | 523.88 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-02-27 11:15:00 | 521.75 | 524.20 | 523.69 | SL hit (close<static) qty=1.00 sl=522.20 alert=retest2 |

### Cycle 62 — SELL (started 2024-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 13:15:00 | 517.50 | 522.45 | 522.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 10:15:00 | 516.10 | 519.76 | 521.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 11:15:00 | 512.75 | 512.18 | 515.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 12:00:00 | 512.75 | 512.18 | 515.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 14:15:00 | 519.65 | 514.17 | 515.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 15:00:00 | 519.65 | 514.17 | 515.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 520.20 | 515.38 | 516.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:15:00 | 521.05 | 515.38 | 516.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2024-03-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 09:15:00 | 522.00 | 516.70 | 516.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 11:15:00 | 524.25 | 518.85 | 517.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 09:15:00 | 527.80 | 527.91 | 524.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-04 09:45:00 | 527.60 | 527.91 | 524.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 10:15:00 | 523.25 | 526.98 | 524.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 11:00:00 | 523.25 | 526.98 | 524.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 11:15:00 | 528.60 | 527.31 | 524.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-04 12:15:00 | 530.15 | 527.31 | 524.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-06 10:30:00 | 529.90 | 536.35 | 533.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-06 12:30:00 | 530.45 | 533.92 | 533.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-07 09:45:00 | 529.80 | 533.73 | 533.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-07 10:15:00 | 529.95 | 532.97 | 533.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2024-03-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-07 10:15:00 | 529.95 | 532.97 | 533.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-07 11:15:00 | 525.35 | 531.45 | 532.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 497.50 | 496.14 | 502.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 10:45:00 | 497.05 | 496.14 | 502.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 14:15:00 | 502.50 | 498.70 | 501.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 15:00:00 | 502.50 | 498.70 | 501.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 15:15:00 | 502.90 | 499.54 | 501.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 09:15:00 | 497.35 | 499.54 | 501.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 11:15:00 | 472.48 | 488.54 | 495.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-03-19 09:15:00 | 447.62 | 461.82 | 473.11 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 65 — BUY (started 2024-03-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 11:15:00 | 466.60 | 461.17 | 460.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 12:15:00 | 469.90 | 462.92 | 461.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 12:15:00 | 470.45 | 471.13 | 468.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-27 13:00:00 | 470.45 | 471.13 | 468.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 13:15:00 | 468.55 | 470.61 | 468.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 13:45:00 | 468.20 | 470.61 | 468.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 468.00 | 470.09 | 468.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 14:45:00 | 466.60 | 470.09 | 468.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 467.45 | 469.56 | 468.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 09:15:00 | 468.30 | 469.56 | 468.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 463.95 | 468.44 | 468.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 10:00:00 | 463.95 | 468.44 | 468.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2024-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 10:15:00 | 464.95 | 467.74 | 467.78 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-02 12:15:00 | 469.40 | 466.83 | 466.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 15:15:00 | 473.75 | 468.91 | 467.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-03 10:15:00 | 468.55 | 468.87 | 467.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-03 10:15:00 | 468.55 | 468.87 | 467.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 10:15:00 | 468.55 | 468.87 | 467.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 10:45:00 | 467.80 | 468.87 | 467.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 13:15:00 | 468.55 | 468.90 | 468.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 14:00:00 | 468.55 | 468.90 | 468.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 14:15:00 | 468.65 | 468.85 | 468.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 14:30:00 | 469.80 | 468.85 | 468.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 15:15:00 | 469.00 | 468.88 | 468.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-04 09:15:00 | 474.30 | 468.88 | 468.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-04 09:15:00 | 466.75 | 468.45 | 468.16 | SL hit (close<static) qty=1.00 sl=468.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 10:15:00 | 464.40 | 467.64 | 467.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 11:15:00 | 459.30 | 465.97 | 467.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-05 09:15:00 | 465.00 | 463.05 | 464.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 09:15:00 | 465.00 | 463.05 | 464.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 465.00 | 463.05 | 464.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-05 10:00:00 | 465.00 | 463.05 | 464.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 10:15:00 | 465.80 | 463.60 | 464.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-05 11:00:00 | 465.80 | 463.60 | 464.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 11:15:00 | 466.85 | 464.25 | 465.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-05 11:30:00 | 468.55 | 464.25 | 465.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 13:15:00 | 466.60 | 464.91 | 465.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-05 14:00:00 | 466.60 | 464.91 | 465.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2024-04-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 14:15:00 | 468.70 | 465.67 | 465.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 09:15:00 | 468.95 | 466.81 | 466.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-09 10:15:00 | 468.90 | 469.70 | 468.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 10:15:00 | 468.90 | 469.70 | 468.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 10:15:00 | 468.90 | 469.70 | 468.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 10:45:00 | 468.00 | 469.70 | 468.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 12:15:00 | 466.70 | 469.00 | 468.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 13:00:00 | 466.70 | 469.00 | 468.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 13:15:00 | 464.95 | 468.19 | 468.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 14:00:00 | 464.95 | 468.19 | 468.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 15:15:00 | 469.35 | 468.69 | 468.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-10 09:15:00 | 478.90 | 468.69 | 468.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-15 09:45:00 | 474.20 | 481.52 | 479.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-15 14:15:00 | 472.65 | 477.75 | 478.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 14:15:00 | 472.65 | 477.75 | 478.21 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-04-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 09:15:00 | 482.00 | 478.03 | 477.78 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-04-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 14:15:00 | 472.90 | 477.62 | 477.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 09:15:00 | 469.40 | 476.07 | 477.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 12:15:00 | 473.25 | 472.85 | 475.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-19 12:45:00 | 472.30 | 472.85 | 475.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 13:15:00 | 475.15 | 473.31 | 475.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 14:00:00 | 475.15 | 473.31 | 475.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 14:15:00 | 473.65 | 473.38 | 474.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 15:15:00 | 472.25 | 473.38 | 474.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 09:15:00 | 490.90 | 476.70 | 476.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2024-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 09:15:00 | 490.90 | 476.70 | 476.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 14:15:00 | 499.65 | 490.25 | 487.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 14:15:00 | 492.90 | 493.75 | 491.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-26 15:00:00 | 492.90 | 493.75 | 491.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 15:15:00 | 491.00 | 493.20 | 491.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 09:15:00 | 502.15 | 493.20 | 491.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 14:00:00 | 494.35 | 494.98 | 492.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 15:00:00 | 495.30 | 495.05 | 493.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-03 13:15:00 | 499.45 | 505.74 | 505.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-05-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 13:15:00 | 499.45 | 505.74 | 505.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 09:15:00 | 489.80 | 501.96 | 504.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 481.50 | 479.93 | 485.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 09:30:00 | 481.05 | 479.93 | 485.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 13:15:00 | 479.45 | 479.89 | 483.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 14:15:00 | 479.15 | 479.89 | 483.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 10:30:00 | 477.60 | 479.66 | 482.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 12:30:00 | 478.10 | 479.40 | 481.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 13:45:00 | 478.70 | 479.48 | 481.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 11:15:00 | 478.90 | 478.48 | 480.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 11:30:00 | 480.30 | 478.48 | 480.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 12:15:00 | 478.15 | 478.41 | 480.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 12:30:00 | 479.00 | 478.41 | 480.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 13:15:00 | 480.15 | 478.76 | 480.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 13:45:00 | 480.85 | 478.76 | 480.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 14:15:00 | 480.80 | 479.17 | 480.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 14:45:00 | 480.90 | 479.17 | 480.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 15:15:00 | 480.95 | 479.52 | 480.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 09:15:00 | 480.75 | 479.52 | 480.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 477.00 | 479.02 | 479.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 10:45:00 | 472.40 | 477.34 | 478.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 11:15:00 | 472.30 | 477.34 | 478.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 12:15:00 | 472.65 | 476.44 | 478.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 12:45:00 | 472.55 | 475.68 | 477.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 15:15:00 | 476.30 | 475.10 | 476.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-15 09:15:00 | 478.00 | 475.10 | 476.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 09:15:00 | 466.45 | 473.37 | 475.85 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-16 09:15:00 | 502.80 | 478.66 | 476.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2024-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 09:15:00 | 502.80 | 478.66 | 476.79 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 13:15:00 | 483.80 | 485.91 | 486.09 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 11:15:00 | 490.45 | 486.63 | 486.24 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 11:15:00 | 482.65 | 486.77 | 486.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 15:15:00 | 481.30 | 483.90 | 484.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 15:15:00 | 468.00 | 467.83 | 470.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 09:15:00 | 473.00 | 467.83 | 470.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 469.10 | 468.08 | 470.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:30:00 | 471.85 | 468.08 | 470.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 473.25 | 469.12 | 470.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 10:45:00 | 472.95 | 469.12 | 470.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 471.85 | 469.66 | 470.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 11:45:00 | 472.10 | 469.66 | 470.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 466.35 | 469.00 | 470.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 14:15:00 | 465.80 | 469.00 | 470.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 472.55 | 468.01 | 469.43 | SL hit (close>static) qty=1.00 sl=472.20 alert=retest2 |

### Cycle 79 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 475.90 | 471.07 | 470.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 15:15:00 | 479.75 | 473.81 | 472.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 463.90 | 471.83 | 471.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 463.90 | 471.83 | 471.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 463.90 | 471.83 | 471.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 460.85 | 471.83 | 471.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 452.45 | 467.95 | 469.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 437.30 | 461.82 | 466.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 458.65 | 457.03 | 462.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 10:00:00 | 458.65 | 457.03 | 462.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 466.70 | 458.96 | 462.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 466.70 | 458.96 | 462.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 470.60 | 461.29 | 463.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:00:00 | 470.60 | 461.29 | 463.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 14:15:00 | 470.55 | 465.44 | 464.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 476.60 | 471.34 | 468.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 09:15:00 | 484.55 | 484.71 | 480.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 09:30:00 | 483.80 | 484.71 | 480.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 481.55 | 484.01 | 481.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 15:00:00 | 481.55 | 484.01 | 481.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 482.90 | 483.79 | 481.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 483.70 | 483.79 | 481.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 11:15:00 | 478.40 | 482.43 | 481.79 | SL hit (close<static) qty=1.00 sl=480.40 alert=retest2 |

### Cycle 82 — SELL (started 2024-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 13:15:00 | 477.65 | 480.66 | 481.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 09:15:00 | 473.70 | 477.14 | 478.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-14 14:15:00 | 476.55 | 476.20 | 477.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-14 14:15:00 | 476.55 | 476.20 | 477.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 476.55 | 476.20 | 477.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 14:45:00 | 476.50 | 476.20 | 477.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 483.90 | 477.80 | 478.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:45:00 | 483.45 | 477.80 | 478.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2024-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 10:15:00 | 486.10 | 479.46 | 478.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 09:15:00 | 490.40 | 485.09 | 483.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 09:15:00 | 491.70 | 497.81 | 493.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 491.70 | 497.81 | 493.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 491.70 | 497.81 | 493.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 10:15:00 | 501.60 | 498.14 | 495.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 12:00:00 | 501.50 | 498.89 | 496.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 12:30:00 | 501.80 | 499.63 | 497.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-03 15:15:00 | 533.40 | 534.86 | 534.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 15:15:00 | 533.40 | 534.86 | 534.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 09:15:00 | 527.95 | 533.48 | 534.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 15:15:00 | 529.90 | 529.12 | 531.21 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:15:00 | 527.00 | 529.12 | 531.21 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 529.50 | 529.19 | 531.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:45:00 | 527.70 | 529.19 | 531.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 529.20 | 526.66 | 528.46 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-08 09:15:00 | 529.20 | 526.66 | 528.46 | SL hit (close>ema400) qty=1.00 sl=528.46 alert=retest1 |

### Cycle 85 — BUY (started 2024-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 12:15:00 | 536.20 | 528.89 | 528.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 13:15:00 | 538.95 | 530.90 | 529.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 529.20 | 532.40 | 530.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 09:15:00 | 529.20 | 532.40 | 530.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 529.20 | 532.40 | 530.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 529.20 | 532.40 | 530.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 524.30 | 530.78 | 529.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 522.40 | 530.78 | 529.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 527.85 | 530.20 | 529.60 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2024-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 14:15:00 | 527.00 | 528.86 | 529.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 09:15:00 | 523.70 | 527.04 | 528.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 09:15:00 | 541.30 | 526.07 | 526.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 09:15:00 | 541.30 | 526.07 | 526.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 541.30 | 526.07 | 526.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:45:00 | 544.95 | 526.07 | 526.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2024-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 10:15:00 | 545.80 | 530.02 | 528.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 11:15:00 | 550.50 | 542.20 | 536.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 546.50 | 547.73 | 542.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-18 10:45:00 | 546.75 | 547.73 | 542.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 522.80 | 543.85 | 543.09 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 528.15 | 540.71 | 541.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 09:15:00 | 517.90 | 524.99 | 529.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 532.80 | 522.91 | 525.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 532.80 | 522.91 | 525.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 532.80 | 522.91 | 525.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 532.80 | 522.91 | 525.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 532.35 | 524.80 | 526.37 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 12:15:00 | 538.10 | 529.08 | 528.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 14:15:00 | 539.95 | 532.49 | 529.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 11:15:00 | 534.25 | 534.63 | 531.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-25 12:00:00 | 534.25 | 534.63 | 531.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 542.15 | 537.16 | 534.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 12:00:00 | 553.05 | 541.92 | 537.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 13:30:00 | 551.40 | 545.14 | 539.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 14:00:00 | 551.45 | 545.14 | 539.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 15:00:00 | 551.40 | 546.40 | 540.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 15:15:00 | 549.10 | 550.05 | 546.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 09:15:00 | 551.50 | 550.05 | 546.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 552.60 | 550.56 | 546.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 10:15:00 | 557.20 | 550.56 | 546.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:45:00 | 555.75 | 556.84 | 555.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 12:15:00 | 548.85 | 553.22 | 553.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2024-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 12:15:00 | 548.85 | 553.22 | 553.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 546.30 | 550.58 | 552.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 528.05 | 527.28 | 534.99 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 10:30:00 | 522.65 | 526.80 | 534.07 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 14:00:00 | 521.65 | 525.96 | 531.91 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 523.85 | 522.15 | 526.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:30:00 | 523.85 | 522.15 | 526.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 09:15:00 | 496.52 | 510.76 | 516.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 09:15:00 | 495.57 | 510.76 | 516.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-08-12 12:15:00 | 496.75 | 496.53 | 503.08 | SL hit (close>ema200) qty=0.50 sl=496.53 alert=retest1 |

### Cycle 91 — BUY (started 2024-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 11:15:00 | 489.15 | 486.01 | 485.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 12:15:00 | 497.20 | 488.25 | 486.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 11:15:00 | 507.00 | 507.37 | 503.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 12:00:00 | 507.00 | 507.37 | 503.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 11:15:00 | 504.95 | 506.93 | 505.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 12:00:00 | 504.95 | 506.93 | 505.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 504.55 | 506.46 | 505.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 12:45:00 | 504.65 | 506.46 | 505.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 505.00 | 506.16 | 505.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 10:00:00 | 506.60 | 505.59 | 505.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 12:00:00 | 506.95 | 506.31 | 505.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 13:15:00 | 506.90 | 506.25 | 505.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 14:15:00 | 506.80 | 506.21 | 505.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 14:15:00 | 505.25 | 506.02 | 505.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 14:45:00 | 505.00 | 506.02 | 505.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 504.00 | 505.61 | 505.52 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-27 15:15:00 | 504.00 | 505.61 | 505.52 | SL hit (close<static) qty=1.00 sl=504.20 alert=retest2 |

### Cycle 92 — SELL (started 2024-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 09:15:00 | 500.80 | 504.65 | 505.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 14:15:00 | 497.35 | 501.90 | 503.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 496.80 | 495.66 | 498.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 496.80 | 495.66 | 498.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 496.80 | 495.66 | 498.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:00:00 | 496.80 | 495.66 | 498.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 494.80 | 494.39 | 496.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:30:00 | 496.50 | 494.39 | 496.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 496.80 | 494.80 | 496.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 13:00:00 | 496.80 | 494.80 | 496.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 498.70 | 495.58 | 496.33 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2024-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 15:15:00 | 499.55 | 496.92 | 496.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 09:15:00 | 500.85 | 497.71 | 497.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 503.45 | 509.40 | 507.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 503.45 | 509.40 | 507.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 503.45 | 509.40 | 507.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 503.45 | 509.40 | 507.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 502.35 | 507.99 | 506.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 11:30:00 | 505.55 | 507.42 | 506.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 14:30:00 | 505.00 | 507.36 | 506.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 09:45:00 | 506.00 | 507.51 | 506.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 13:15:00 | 520.25 | 522.32 | 522.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2024-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 13:15:00 | 520.25 | 522.32 | 522.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 14:15:00 | 518.20 | 521.49 | 521.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 13:15:00 | 514.00 | 512.18 | 514.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 14:00:00 | 514.00 | 512.18 | 514.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 518.65 | 513.47 | 515.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 15:00:00 | 518.65 | 513.47 | 515.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 517.90 | 514.36 | 515.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 09:15:00 | 515.90 | 514.36 | 515.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 10:15:00 | 516.10 | 515.16 | 515.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 11:15:00 | 520.55 | 516.64 | 516.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2024-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 11:15:00 | 520.55 | 516.64 | 516.40 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-09-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-20 15:15:00 | 514.75 | 516.25 | 516.33 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 520.05 | 517.01 | 516.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 521.85 | 519.88 | 518.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 09:15:00 | 539.15 | 539.63 | 533.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-26 09:30:00 | 538.80 | 539.63 | 533.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 543.00 | 547.82 | 544.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:45:00 | 541.60 | 547.82 | 544.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 543.30 | 546.91 | 544.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:30:00 | 543.05 | 546.91 | 544.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 550.05 | 547.54 | 544.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:45:00 | 547.45 | 547.54 | 544.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 547.90 | 548.03 | 546.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 09:30:00 | 549.90 | 548.52 | 546.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 11:15:00 | 549.70 | 548.11 | 546.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 09:15:00 | 531.15 | 545.37 | 546.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 09:15:00 | 531.15 | 545.37 | 546.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 10:15:00 | 530.00 | 542.29 | 544.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 506.35 | 506.16 | 514.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 10:30:00 | 505.85 | 506.16 | 514.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 515.65 | 509.35 | 512.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:45:00 | 516.95 | 509.35 | 512.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 11:15:00 | 515.05 | 510.98 | 512.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 12:00:00 | 515.05 | 510.98 | 512.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 12:15:00 | 510.40 | 510.86 | 512.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 14:00:00 | 509.55 | 510.60 | 512.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 13:30:00 | 509.15 | 506.49 | 507.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 10:15:00 | 512.95 | 506.96 | 506.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 10:15:00 | 512.95 | 506.96 | 506.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 11:15:00 | 514.20 | 508.41 | 507.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 10:15:00 | 510.45 | 512.57 | 510.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 10:15:00 | 510.45 | 512.57 | 510.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 510.45 | 512.57 | 510.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:00:00 | 510.45 | 512.57 | 510.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 509.95 | 512.05 | 510.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 14:30:00 | 513.65 | 512.89 | 511.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 13:15:00 | 505.55 | 510.34 | 510.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2024-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 13:15:00 | 505.55 | 510.34 | 510.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 14:15:00 | 504.70 | 509.21 | 510.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 10:15:00 | 512.60 | 507.23 | 507.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 10:15:00 | 512.60 | 507.23 | 507.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 512.60 | 507.23 | 507.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 11:00:00 | 512.60 | 507.23 | 507.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 507.75 | 507.34 | 507.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:30:00 | 507.40 | 507.52 | 507.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 13:15:00 | 507.50 | 507.52 | 507.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 14:15:00 | 511.90 | 508.63 | 508.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2024-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 14:15:00 | 511.90 | 508.63 | 508.37 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-10-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 09:15:00 | 506.05 | 508.13 | 508.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 11:15:00 | 503.60 | 506.77 | 507.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 502.40 | 500.95 | 503.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 11:00:00 | 502.40 | 500.95 | 503.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 500.75 | 500.91 | 503.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:45:00 | 504.00 | 500.91 | 503.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 501.45 | 501.02 | 503.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:30:00 | 502.70 | 501.02 | 503.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 488.65 | 483.76 | 487.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 10:00:00 | 488.65 | 483.76 | 487.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 488.05 | 484.62 | 487.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 12:30:00 | 485.60 | 485.55 | 487.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 13:00:00 | 486.20 | 485.55 | 487.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 14:00:00 | 486.20 | 485.68 | 487.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 09:30:00 | 482.65 | 485.62 | 487.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 11:15:00 | 490.10 | 486.12 | 487.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 11:45:00 | 488.90 | 486.12 | 487.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 12:15:00 | 491.65 | 487.22 | 487.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-29 12:15:00 | 491.65 | 487.22 | 487.49 | SL hit (close>static) qty=1.00 sl=490.60 alert=retest2 |

### Cycle 103 — BUY (started 2024-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 13:15:00 | 493.50 | 488.48 | 488.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 14:15:00 | 499.15 | 490.61 | 489.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 14:15:00 | 497.20 | 497.72 | 494.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 15:00:00 | 497.20 | 497.72 | 494.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 499.05 | 498.03 | 495.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 14:45:00 | 503.95 | 500.64 | 497.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:00:00 | 504.75 | 501.68 | 498.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 487.75 | 498.85 | 497.84 | SL hit (close<static) qty=1.00 sl=494.65 alert=retest2 |

### Cycle 104 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 489.80 | 497.04 | 497.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 12:15:00 | 484.00 | 492.81 | 495.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 483.50 | 483.09 | 486.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 09:15:00 | 483.50 | 483.09 | 486.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 483.50 | 483.09 | 486.69 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 09:15:00 | 494.85 | 488.82 | 488.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 15:15:00 | 498.40 | 494.85 | 491.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 10:15:00 | 491.50 | 494.50 | 492.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 10:15:00 | 491.50 | 494.50 | 492.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 491.50 | 494.50 | 492.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 11:00:00 | 491.50 | 494.50 | 492.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 489.45 | 493.49 | 491.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:00:00 | 489.45 | 493.49 | 491.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 492.25 | 491.90 | 491.53 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 486.85 | 491.01 | 491.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 11:15:00 | 485.60 | 489.80 | 490.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 474.70 | 464.84 | 471.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 474.70 | 464.84 | 471.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 474.70 | 464.84 | 471.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 474.70 | 464.84 | 471.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 474.80 | 466.83 | 471.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:15:00 | 470.50 | 472.88 | 473.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 12:45:00 | 472.55 | 472.40 | 472.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 13:15:00 | 472.55 | 472.40 | 472.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 13:15:00 | 475.65 | 473.05 | 472.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2024-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 13:15:00 | 475.65 | 473.05 | 472.97 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 14:15:00 | 472.40 | 472.92 | 472.92 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 481.05 | 474.39 | 473.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 11:15:00 | 487.65 | 478.20 | 475.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 09:15:00 | 480.00 | 482.69 | 479.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 09:15:00 | 480.00 | 482.69 | 479.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 480.00 | 482.69 | 479.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:30:00 | 480.80 | 482.69 | 479.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 484.35 | 483.02 | 479.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 11:15:00 | 485.15 | 483.02 | 479.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 13:15:00 | 485.80 | 483.62 | 480.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 09:15:00 | 485.80 | 482.81 | 480.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 09:45:00 | 486.55 | 483.88 | 481.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 510.55 | 512.25 | 508.87 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-29 11:15:00 | 506.45 | 508.72 | 508.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 11:15:00 | 506.45 | 508.72 | 508.75 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-11-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 15:15:00 | 509.25 | 508.81 | 508.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 10:15:00 | 512.90 | 509.97 | 509.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 15:15:00 | 540.40 | 541.11 | 536.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-09 09:15:00 | 542.00 | 541.11 | 536.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 548.60 | 542.61 | 537.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 10:30:00 | 553.80 | 544.67 | 539.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 14:00:00 | 549.50 | 547.33 | 541.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 14:45:00 | 549.95 | 547.75 | 542.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 10:15:00 | 540.25 | 541.61 | 541.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2024-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 10:15:00 | 540.25 | 541.61 | 541.69 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 09:15:00 | 542.75 | 541.66 | 541.63 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 10:15:00 | 541.10 | 541.55 | 541.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 540.00 | 540.93 | 541.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 11:15:00 | 541.70 | 540.98 | 541.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 11:15:00 | 541.70 | 540.98 | 541.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 541.70 | 540.98 | 541.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:00:00 | 541.70 | 540.98 | 541.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2024-12-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 12:15:00 | 544.10 | 541.61 | 541.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 14:15:00 | 547.10 | 543.06 | 542.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 10:15:00 | 544.25 | 544.65 | 543.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-16 10:45:00 | 544.25 | 544.65 | 543.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 542.65 | 544.25 | 543.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 12:00:00 | 542.65 | 544.25 | 543.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 542.45 | 543.89 | 543.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 12:30:00 | 542.00 | 543.89 | 543.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2024-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 14:15:00 | 539.95 | 542.53 | 542.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 09:15:00 | 538.15 | 541.42 | 542.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 09:15:00 | 531.50 | 530.16 | 532.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 09:15:00 | 531.50 | 530.16 | 532.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 531.50 | 530.16 | 532.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:30:00 | 531.35 | 530.16 | 532.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 535.00 | 531.13 | 532.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 11:00:00 | 535.00 | 531.13 | 532.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 534.70 | 531.85 | 532.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 11:45:00 | 535.15 | 531.85 | 532.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 532.05 | 531.91 | 532.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 09:45:00 | 530.00 | 531.60 | 532.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-23 10:15:00 | 533.65 | 532.01 | 532.31 | SL hit (close>static) qty=1.00 sl=533.60 alert=retest2 |

### Cycle 117 — BUY (started 2024-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 09:15:00 | 534.00 | 532.34 | 532.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 11:15:00 | 539.30 | 534.26 | 533.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-24 13:15:00 | 533.70 | 534.28 | 533.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 13:15:00 | 533.70 | 534.28 | 533.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 13:15:00 | 533.70 | 534.28 | 533.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-24 13:45:00 | 533.15 | 534.28 | 533.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 14:15:00 | 534.90 | 534.40 | 533.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-26 09:15:00 | 537.40 | 534.43 | 533.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 10:15:00 | 537.10 | 538.89 | 538.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 13:15:00 | 533.00 | 537.46 | 537.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 13:15:00 | 533.00 | 537.46 | 537.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 14:15:00 | 529.60 | 534.03 | 535.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 14:15:00 | 527.95 | 527.23 | 530.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-01 15:00:00 | 527.95 | 527.23 | 530.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 520.35 | 525.65 | 529.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 12:15:00 | 516.15 | 521.93 | 525.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 13:30:00 | 516.45 | 519.49 | 523.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 09:15:00 | 490.34 | 497.74 | 504.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 09:15:00 | 490.63 | 497.74 | 504.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-10 09:15:00 | 464.53 | 475.81 | 484.43 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 119 — BUY (started 2025-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 12:15:00 | 457.95 | 457.08 | 456.99 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 15:15:00 | 455.00 | 456.68 | 456.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 10:15:00 | 452.35 | 455.46 | 456.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 10:15:00 | 454.00 | 452.60 | 454.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 10:15:00 | 454.00 | 452.60 | 454.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 454.00 | 452.60 | 454.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 11:00:00 | 454.00 | 452.60 | 454.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 11:15:00 | 450.00 | 452.08 | 453.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 12:15:00 | 449.25 | 452.08 | 453.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 10:15:00 | 448.10 | 451.53 | 452.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 12:00:00 | 449.25 | 450.50 | 452.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 14:30:00 | 447.65 | 450.15 | 451.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 441.20 | 448.16 | 450.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 10:15:00 | 440.90 | 448.16 | 450.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 10:45:00 | 440.55 | 446.36 | 449.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 10:30:00 | 440.25 | 440.25 | 443.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 13:15:00 | 426.79 | 433.00 | 437.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 13:15:00 | 426.79 | 433.00 | 437.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 14:15:00 | 425.69 | 431.04 | 436.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 14:15:00 | 425.27 | 431.04 | 436.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 418.85 | 426.26 | 433.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 418.52 | 426.26 | 433.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 418.24 | 426.26 | 433.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-27 14:15:00 | 419.95 | 419.81 | 426.67 | SL hit (close>ema200) qty=0.50 sl=419.81 alert=retest2 |

### Cycle 121 — BUY (started 2025-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 12:15:00 | 429.50 | 425.76 | 425.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 14:15:00 | 433.25 | 427.86 | 426.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 13:15:00 | 432.85 | 433.54 | 430.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 14:00:00 | 432.85 | 433.54 | 430.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 432.50 | 433.91 | 431.75 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 14:15:00 | 426.30 | 430.47 | 430.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 423.50 | 428.63 | 429.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 419.60 | 418.68 | 422.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 419.60 | 418.68 | 422.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 419.60 | 418.68 | 422.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 12:30:00 | 414.75 | 418.65 | 420.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 13:00:00 | 415.30 | 418.65 | 420.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 09:15:00 | 426.85 | 419.40 | 420.10 | SL hit (close>static) qty=1.00 sl=423.35 alert=retest2 |

### Cycle 123 — BUY (started 2025-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 10:15:00 | 428.10 | 421.14 | 420.83 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 415.45 | 422.33 | 422.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-12 09:15:00 | 409.85 | 417.17 | 419.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 422.75 | 416.22 | 417.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 422.75 | 416.22 | 417.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 422.75 | 416.22 | 417.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 423.30 | 416.22 | 417.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 421.75 | 417.33 | 417.99 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2025-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 12:15:00 | 424.60 | 419.63 | 418.98 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 10:15:00 | 414.25 | 418.56 | 418.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 13:15:00 | 410.35 | 415.45 | 417.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 12:15:00 | 414.25 | 413.50 | 415.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 12:15:00 | 414.25 | 413.50 | 415.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 414.25 | 413.50 | 415.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 12:45:00 | 414.90 | 413.50 | 415.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 414.50 | 413.70 | 415.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 417.35 | 413.70 | 415.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 416.55 | 414.27 | 415.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 416.55 | 414.27 | 415.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 417.40 | 414.90 | 415.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 410.50 | 414.90 | 415.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 411.25 | 411.55 | 413.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 15:00:00 | 411.25 | 411.55 | 413.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 414.55 | 411.99 | 413.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:45:00 | 414.85 | 411.99 | 413.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 412.40 | 412.07 | 413.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 413.50 | 412.07 | 413.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 11:15:00 | 414.40 | 412.54 | 413.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 12:00:00 | 414.40 | 412.54 | 413.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 12:15:00 | 411.50 | 412.33 | 413.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 09:15:00 | 410.60 | 412.78 | 413.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-20 10:15:00 | 414.70 | 413.10 | 413.18 | SL hit (close>static) qty=1.00 sl=414.55 alert=retest2 |

### Cycle 127 — BUY (started 2025-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 11:15:00 | 415.25 | 413.53 | 413.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 13:15:00 | 416.40 | 414.06 | 413.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 10:15:00 | 410.95 | 414.71 | 414.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 10:15:00 | 410.95 | 414.71 | 414.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 410.95 | 414.71 | 414.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 410.95 | 414.71 | 414.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2025-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 11:15:00 | 409.50 | 413.66 | 413.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 12:15:00 | 408.35 | 412.60 | 413.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 11:15:00 | 379.30 | 378.31 | 384.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 12:00:00 | 379.30 | 378.31 | 384.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 385.15 | 379.68 | 384.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 12:45:00 | 383.65 | 379.68 | 384.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 387.00 | 381.14 | 385.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 14:00:00 | 387.00 | 381.14 | 385.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 387.50 | 382.42 | 385.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 15:15:00 | 387.45 | 382.42 | 385.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 381.70 | 382.84 | 384.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:45:00 | 382.50 | 382.84 | 384.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 383.80 | 381.38 | 383.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:00:00 | 383.80 | 381.38 | 383.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 386.40 | 382.38 | 383.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:30:00 | 387.40 | 382.38 | 383.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 11:15:00 | 385.65 | 383.04 | 383.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:30:00 | 386.60 | 383.04 | 383.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 13:15:00 | 386.00 | 384.01 | 383.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 389.95 | 385.20 | 384.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 14:15:00 | 408.25 | 410.15 | 405.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 15:00:00 | 408.25 | 410.15 | 405.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 403.60 | 408.21 | 405.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:30:00 | 402.55 | 408.21 | 405.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 403.95 | 407.35 | 405.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 11:45:00 | 406.85 | 407.09 | 405.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 14:15:00 | 401.55 | 404.74 | 404.69 | SL hit (close<static) qty=1.00 sl=402.30 alert=retest2 |

### Cycle 130 — SELL (started 2025-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 15:15:00 | 402.90 | 404.37 | 404.52 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 09:15:00 | 408.45 | 405.18 | 404.88 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 12:15:00 | 403.90 | 405.37 | 405.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 13:15:00 | 403.80 | 405.06 | 405.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 13:15:00 | 401.05 | 400.37 | 402.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-17 14:00:00 | 401.05 | 400.37 | 402.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 402.45 | 400.79 | 402.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:45:00 | 402.75 | 400.79 | 402.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 401.80 | 400.99 | 402.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:15:00 | 408.00 | 400.99 | 402.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 413.25 | 403.44 | 403.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 417.65 | 410.36 | 407.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 11:15:00 | 422.95 | 424.24 | 420.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 11:45:00 | 423.00 | 424.24 | 420.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 425.70 | 427.79 | 425.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 425.70 | 427.79 | 425.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 430.00 | 428.23 | 425.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 13:45:00 | 433.30 | 429.25 | 426.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 14:45:00 | 431.65 | 429.65 | 427.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:45:00 | 433.25 | 430.10 | 427.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 10:15:00 | 431.40 | 430.10 | 427.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 429.80 | 431.61 | 429.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 15:00:00 | 429.80 | 431.61 | 429.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 15:15:00 | 427.80 | 430.85 | 429.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:15:00 | 425.25 | 430.85 | 429.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 428.30 | 430.34 | 429.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 10:30:00 | 429.75 | 430.19 | 429.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 11:15:00 | 430.55 | 430.19 | 429.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 12:30:00 | 432.00 | 429.99 | 429.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 14:45:00 | 430.40 | 429.19 | 429.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 15:15:00 | 426.10 | 428.57 | 428.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 15:15:00 | 426.10 | 428.57 | 428.86 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 430.00 | 429.08 | 429.04 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 12:15:00 | 427.20 | 428.70 | 428.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 13:15:00 | 423.75 | 427.71 | 428.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 09:15:00 | 427.75 | 427.14 | 427.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 427.75 | 427.14 | 427.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 427.75 | 427.14 | 427.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:00:00 | 427.75 | 427.14 | 427.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 421.90 | 426.10 | 427.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:30:00 | 421.30 | 425.72 | 427.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 420.95 | 423.86 | 425.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 15:15:00 | 428.70 | 426.35 | 426.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 428.70 | 426.35 | 426.14 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 09:15:00 | 423.40 | 425.76 | 425.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 09:15:00 | 412.20 | 421.83 | 423.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 398.55 | 396.38 | 403.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 398.55 | 396.38 | 403.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 403.85 | 399.54 | 403.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 14:00:00 | 403.85 | 399.54 | 403.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 401.65 | 399.96 | 403.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 14:30:00 | 403.40 | 399.96 | 403.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 11:15:00 | 400.80 | 399.21 | 401.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 11:30:00 | 400.60 | 399.21 | 401.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 12:15:00 | 402.70 | 399.91 | 401.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 12:30:00 | 401.55 | 399.91 | 401.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 13:15:00 | 405.00 | 400.92 | 402.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 14:00:00 | 405.00 | 400.92 | 402.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 14:15:00 | 407.65 | 402.27 | 402.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 15:00:00 | 407.65 | 402.27 | 402.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2025-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 15:15:00 | 409.25 | 403.67 | 403.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 414.30 | 405.79 | 404.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 14:15:00 | 461.80 | 462.27 | 456.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 15:00:00 | 461.80 | 462.27 | 456.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 459.25 | 460.80 | 457.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 11:30:00 | 459.00 | 460.80 | 457.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 459.55 | 460.36 | 458.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 12:30:00 | 464.65 | 460.58 | 459.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 13:15:00 | 464.45 | 460.58 | 459.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 12:45:00 | 465.65 | 463.19 | 461.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 14:15:00 | 463.25 | 462.44 | 461.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 460.40 | 462.03 | 461.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 15:15:00 | 461.00 | 462.03 | 461.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 15:15:00 | 461.00 | 461.83 | 461.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-28 09:15:00 | 450.65 | 461.83 | 461.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-28 09:15:00 | 455.85 | 460.63 | 460.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-28 09:15:00 | 455.85 | 460.63 | 460.93 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 10:15:00 | 467.90 | 459.98 | 459.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 11:15:00 | 472.70 | 462.52 | 460.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 12:15:00 | 485.80 | 486.17 | 481.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 13:00:00 | 485.80 | 486.17 | 481.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 482.95 | 485.12 | 481.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 15:15:00 | 480.00 | 485.12 | 481.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 480.00 | 484.09 | 481.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 486.70 | 484.09 | 481.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 482.05 | 483.68 | 481.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 12:45:00 | 492.25 | 486.03 | 483.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 473.75 | 486.82 | 487.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 473.75 | 486.82 | 487.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 10:15:00 | 468.70 | 483.19 | 485.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 476.10 | 472.42 | 477.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 476.10 | 472.42 | 477.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 476.10 | 472.42 | 477.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-13 12:15:00 | 471.15 | 476.37 | 477.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-15 09:15:00 | 483.30 | 476.72 | 476.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2025-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 09:15:00 | 483.30 | 476.72 | 476.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 504.50 | 484.03 | 480.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 12:15:00 | 492.50 | 493.54 | 489.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 13:00:00 | 492.50 | 493.54 | 489.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 489.10 | 492.09 | 489.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:45:00 | 488.65 | 492.09 | 489.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 488.30 | 491.34 | 489.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 486.00 | 491.34 | 489.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 490.65 | 491.20 | 489.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 487.70 | 491.20 | 489.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 488.05 | 490.57 | 489.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:00:00 | 488.05 | 490.57 | 489.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 489.50 | 490.35 | 489.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:30:00 | 491.00 | 489.59 | 489.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 11:15:00 | 486.25 | 488.92 | 489.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 486.25 | 488.92 | 489.05 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 491.30 | 488.86 | 488.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 14:15:00 | 492.20 | 489.95 | 489.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 14:15:00 | 495.20 | 495.78 | 493.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 15:15:00 | 496.85 | 495.78 | 493.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 495.70 | 495.93 | 494.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 12:30:00 | 498.60 | 496.45 | 494.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 13:30:00 | 498.35 | 496.27 | 495.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 15:15:00 | 493.00 | 494.92 | 494.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 15:15:00 | 493.00 | 494.92 | 494.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 09:15:00 | 489.05 | 493.74 | 494.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 09:15:00 | 461.90 | 461.61 | 465.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 09:15:00 | 461.90 | 461.61 | 465.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 461.90 | 461.61 | 465.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:30:00 | 463.65 | 461.61 | 465.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 464.65 | 462.85 | 464.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:15:00 | 460.40 | 462.85 | 464.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 465.45 | 463.37 | 464.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:15:00 | 466.95 | 463.37 | 464.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 471.45 | 464.99 | 465.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:00:00 | 471.45 | 464.99 | 465.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 11:15:00 | 469.25 | 465.84 | 465.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 12:15:00 | 473.95 | 467.46 | 466.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 09:15:00 | 469.50 | 469.62 | 467.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 10:15:00 | 469.75 | 469.65 | 468.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 469.75 | 469.65 | 468.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:30:00 | 468.50 | 469.65 | 468.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 11:15:00 | 469.20 | 469.56 | 468.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 12:00:00 | 469.20 | 469.56 | 468.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 469.70 | 471.38 | 469.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:00:00 | 469.70 | 471.38 | 469.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 468.90 | 470.88 | 469.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:45:00 | 468.75 | 470.88 | 469.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 472.00 | 471.11 | 469.98 | EMA400 retest candle locked (from upside) |

### Cycle 148 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 463.70 | 468.95 | 469.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 459.50 | 465.20 | 467.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 454.65 | 453.80 | 457.84 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:15:00 | 446.00 | 453.80 | 457.84 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 446.35 | 445.43 | 447.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:45:00 | 447.20 | 445.43 | 447.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 446.20 | 445.58 | 447.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:45:00 | 447.25 | 445.58 | 447.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 446.60 | 445.79 | 447.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 12:30:00 | 445.55 | 445.35 | 446.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 14:15:00 | 447.90 | 445.49 | 446.56 | SL hit (close>ema400) qty=1.00 sl=446.56 alert=retest1 |

### Cycle 149 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 446.60 | 443.90 | 443.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 447.80 | 445.26 | 444.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 447.70 | 448.58 | 446.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 09:15:00 | 447.70 | 448.58 | 446.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 447.70 | 448.58 | 446.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:45:00 | 447.95 | 448.58 | 446.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 447.20 | 448.31 | 446.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:30:00 | 447.35 | 448.31 | 446.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 446.85 | 448.01 | 446.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:45:00 | 446.25 | 448.01 | 446.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 446.95 | 447.80 | 446.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 13:00:00 | 446.95 | 447.80 | 446.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 446.70 | 447.58 | 446.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 14:00:00 | 446.70 | 447.58 | 446.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 446.75 | 447.41 | 446.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 14:30:00 | 446.70 | 447.41 | 446.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 447.00 | 447.33 | 446.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 447.60 | 447.33 | 446.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 15:15:00 | 445.95 | 448.12 | 447.72 | SL hit (close<static) qty=1.00 sl=446.10 alert=retest2 |

### Cycle 150 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 446.05 | 448.16 | 448.25 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 09:15:00 | 462.45 | 450.71 | 449.32 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 457.20 | 459.44 | 459.65 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 11:15:00 | 460.15 | 459.53 | 459.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 12:15:00 | 465.40 | 460.70 | 460.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 471.05 | 471.99 | 468.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 10:00:00 | 471.05 | 471.99 | 468.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 459.90 | 469.58 | 469.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 459.90 | 469.58 | 469.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 458.85 | 467.44 | 468.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 09:15:00 | 456.10 | 460.35 | 463.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 458.80 | 455.94 | 459.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 458.80 | 455.94 | 459.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 458.80 | 455.94 | 459.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 12:00:00 | 455.00 | 457.30 | 458.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 15:15:00 | 454.95 | 456.37 | 457.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 09:15:00 | 469.95 | 458.86 | 458.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 09:15:00 | 469.95 | 458.86 | 458.60 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 456.35 | 459.04 | 459.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 10:15:00 | 455.80 | 458.39 | 458.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 13:15:00 | 452.35 | 450.66 | 452.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 13:15:00 | 452.35 | 450.66 | 452.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 452.35 | 450.66 | 452.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:00:00 | 452.35 | 450.66 | 452.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 450.25 | 450.58 | 452.25 | EMA400 retest candle locked (from downside) |

### Cycle 157 — BUY (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 13:15:00 | 456.20 | 452.67 | 452.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 14:15:00 | 458.05 | 453.75 | 453.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 14:15:00 | 458.20 | 458.72 | 456.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 15:00:00 | 458.20 | 458.72 | 456.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 455.70 | 458.13 | 456.55 | EMA400 retest candle locked (from upside) |

### Cycle 158 — SELL (started 2025-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 15:15:00 | 455.45 | 455.80 | 455.82 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 09:15:00 | 458.70 | 456.38 | 456.08 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 12:15:00 | 453.00 | 455.44 | 455.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 14:15:00 | 448.40 | 453.80 | 454.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 456.00 | 453.67 | 454.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 09:15:00 | 456.00 | 453.67 | 454.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 456.00 | 453.67 | 454.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:00:00 | 456.00 | 453.67 | 454.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 455.25 | 453.99 | 454.67 | EMA400 retest candle locked (from downside) |

### Cycle 161 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 457.30 | 455.22 | 455.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 10:15:00 | 458.85 | 456.11 | 455.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 454.85 | 457.91 | 456.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 454.85 | 457.91 | 456.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 454.85 | 457.91 | 456.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:00:00 | 457.95 | 457.92 | 457.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 14:15:00 | 448.70 | 455.76 | 456.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 448.70 | 455.76 | 456.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 446.90 | 451.64 | 454.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 15:15:00 | 442.65 | 441.95 | 445.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-05 09:15:00 | 444.15 | 441.95 | 445.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 442.35 | 442.03 | 445.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 13:15:00 | 440.35 | 441.84 | 444.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 14:00:00 | 439.50 | 441.37 | 444.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:45:00 | 439.20 | 441.02 | 443.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 14:15:00 | 439.00 | 434.95 | 434.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 439.00 | 434.95 | 434.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 12:15:00 | 443.45 | 438.67 | 436.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 442.20 | 443.64 | 441.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 442.20 | 443.64 | 441.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 442.20 | 443.64 | 441.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:45:00 | 441.55 | 443.64 | 441.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 441.50 | 443.22 | 441.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:45:00 | 441.65 | 443.22 | 441.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 439.15 | 442.40 | 441.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:30:00 | 439.30 | 442.40 | 441.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 438.10 | 441.54 | 441.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 12:30:00 | 438.80 | 441.54 | 441.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 434.80 | 440.19 | 440.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 14:15:00 | 432.95 | 438.74 | 439.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 445.35 | 438.91 | 439.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 445.35 | 438.91 | 439.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 445.35 | 438.91 | 439.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:30:00 | 444.85 | 438.91 | 439.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 456.10 | 442.35 | 441.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 12:15:00 | 461.60 | 448.23 | 444.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 462.75 | 465.74 | 461.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 09:45:00 | 462.85 | 465.74 | 461.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 462.15 | 465.02 | 461.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:30:00 | 462.05 | 465.02 | 461.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 462.35 | 464.49 | 461.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:15:00 | 462.05 | 464.49 | 461.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 462.05 | 464.00 | 461.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 13:45:00 | 463.50 | 463.86 | 462.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 14:30:00 | 463.80 | 464.09 | 462.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 09:15:00 | 458.00 | 462.99 | 462.14 | SL hit (close<static) qty=1.00 sl=461.70 alert=retest2 |

### Cycle 166 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 462.20 | 463.69 | 463.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 455.75 | 461.98 | 463.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 11:15:00 | 460.70 | 459.87 | 461.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 11:15:00 | 460.70 | 459.87 | 461.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 460.70 | 459.87 | 461.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:45:00 | 460.20 | 459.87 | 461.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 460.70 | 460.04 | 461.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 14:45:00 | 458.60 | 459.66 | 461.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 14:15:00 | 462.30 | 459.16 | 459.87 | SL hit (close>static) qty=1.00 sl=461.50 alert=retest2 |

### Cycle 167 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 466.00 | 461.08 | 460.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 485.00 | 469.56 | 465.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 11:15:00 | 484.00 | 484.75 | 478.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 12:00:00 | 484.00 | 484.75 | 478.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 485.90 | 485.67 | 481.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:45:00 | 488.45 | 484.94 | 483.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 14:15:00 | 479.95 | 484.05 | 483.82 | SL hit (close<static) qty=1.00 sl=481.05 alert=retest2 |

### Cycle 168 — SELL (started 2025-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 15:15:00 | 480.00 | 483.24 | 483.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 09:15:00 | 475.40 | 480.01 | 481.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 12:15:00 | 479.60 | 479.48 | 480.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 12:15:00 | 479.60 | 479.48 | 480.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 479.60 | 479.48 | 480.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:30:00 | 479.40 | 479.48 | 480.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 479.95 | 479.58 | 480.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:30:00 | 480.40 | 479.58 | 480.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 479.00 | 479.46 | 480.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 15:00:00 | 479.00 | 479.46 | 480.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 480.00 | 479.57 | 480.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 476.50 | 479.57 | 480.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:30:00 | 476.00 | 478.34 | 479.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 13:30:00 | 477.30 | 478.07 | 479.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 15:00:00 | 476.90 | 477.84 | 479.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 474.95 | 476.97 | 478.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 11:30:00 | 472.25 | 475.23 | 477.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 10:15:00 | 473.05 | 475.03 | 476.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 12:15:00 | 480.10 | 477.31 | 477.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 480.10 | 477.31 | 477.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 487.85 | 480.39 | 478.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 483.80 | 485.68 | 483.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 13:15:00 | 483.80 | 485.68 | 483.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 483.80 | 485.68 | 483.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 483.80 | 485.68 | 483.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 483.40 | 485.22 | 483.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:30:00 | 483.00 | 485.22 | 483.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 482.90 | 484.76 | 483.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 482.10 | 484.76 | 483.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 484.00 | 484.60 | 483.68 | EMA400 retest candle locked (from upside) |

### Cycle 170 — SELL (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 09:15:00 | 479.85 | 483.13 | 483.33 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 10:15:00 | 487.25 | 483.54 | 483.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 11:15:00 | 488.60 | 484.55 | 483.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 15:15:00 | 485.50 | 485.65 | 484.57 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 09:15:00 | 490.95 | 485.65 | 484.57 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 485.40 | 488.85 | 487.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-24 10:15:00 | 485.40 | 488.85 | 487.64 | SL hit (close<ema400) qty=1.00 sl=487.64 alert=retest1 |

### Cycle 172 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 481.65 | 489.04 | 489.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 477.85 | 486.80 | 488.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 14:15:00 | 472.30 | 470.98 | 474.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 15:00:00 | 472.30 | 470.98 | 474.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 466.70 | 467.70 | 470.43 | EMA400 retest candle locked (from downside) |

### Cycle 173 — BUY (started 2025-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 15:15:00 | 472.00 | 471.48 | 471.45 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 09:15:00 | 470.55 | 471.29 | 471.37 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 475.05 | 472.04 | 471.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 11:15:00 | 479.15 | 476.32 | 474.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 09:15:00 | 483.10 | 484.47 | 481.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-09 10:00:00 | 483.10 | 484.47 | 481.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 481.85 | 483.95 | 481.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:45:00 | 481.50 | 483.95 | 481.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 480.40 | 483.24 | 481.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:45:00 | 480.25 | 483.24 | 481.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 482.00 | 482.99 | 481.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 14:30:00 | 483.15 | 483.16 | 481.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 10:15:00 | 479.05 | 484.38 | 483.92 | SL hit (close<static) qty=1.00 sl=480.40 alert=retest2 |

### Cycle 176 — SELL (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 13:15:00 | 478.25 | 483.26 | 483.85 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 487.30 | 483.99 | 483.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 489.90 | 485.58 | 484.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 12:15:00 | 485.25 | 486.31 | 485.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 13:00:00 | 485.25 | 486.31 | 485.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 487.65 | 486.58 | 485.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 14:15:00 | 488.70 | 486.58 | 485.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 15:00:00 | 491.35 | 493.03 | 490.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 15:15:00 | 500.60 | 504.73 | 504.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — SELL (started 2025-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 15:15:00 | 500.60 | 504.73 | 504.99 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2025-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 10:15:00 | 506.90 | 505.42 | 505.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 516.45 | 507.80 | 506.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 12:15:00 | 508.60 | 509.25 | 507.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 13:00:00 | 508.60 | 509.25 | 507.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 508.20 | 509.04 | 507.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:00:00 | 508.20 | 509.04 | 507.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 505.40 | 508.31 | 507.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 505.40 | 508.31 | 507.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 502.50 | 507.15 | 507.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 507.40 | 507.15 | 507.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 503.20 | 506.36 | 506.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 503.20 | 506.36 | 506.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 10:15:00 | 502.25 | 505.54 | 506.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 12:15:00 | 505.55 | 505.49 | 506.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 12:15:00 | 505.55 | 505.49 | 506.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 505.55 | 505.49 | 506.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:30:00 | 506.00 | 505.49 | 506.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 506.25 | 505.65 | 506.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 11:00:00 | 504.60 | 505.42 | 505.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 13:00:00 | 504.55 | 505.20 | 505.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 10:00:00 | 504.60 | 504.21 | 504.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 12:15:00 | 504.15 | 504.04 | 504.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 503.40 | 503.91 | 504.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 13:15:00 | 502.90 | 503.91 | 504.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 14:15:00 | 502.65 | 503.73 | 504.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 15:00:00 | 501.00 | 503.18 | 504.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 506.45 | 503.52 | 504.11 | SL hit (close>static) qty=1.00 sl=505.00 alert=retest2 |

### Cycle 181 — BUY (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 11:15:00 | 505.10 | 504.52 | 504.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 14:15:00 | 509.80 | 505.69 | 505.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 513.90 | 518.46 | 514.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 513.90 | 518.46 | 514.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 513.90 | 518.46 | 514.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 513.90 | 518.46 | 514.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 513.00 | 517.37 | 514.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 513.15 | 517.37 | 514.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 508.70 | 515.64 | 513.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:00:00 | 508.70 | 515.64 | 513.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — SELL (started 2025-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 14:15:00 | 506.20 | 511.72 | 512.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 15:15:00 | 505.05 | 510.39 | 511.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 513.65 | 508.44 | 509.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 12:15:00 | 513.65 | 508.44 | 509.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 513.65 | 508.44 | 509.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 513.65 | 508.44 | 509.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 514.00 | 509.56 | 510.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:15:00 | 516.25 | 509.56 | 510.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — BUY (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 14:15:00 | 516.10 | 510.86 | 510.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 12:15:00 | 521.25 | 516.34 | 513.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 516.70 | 518.41 | 515.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 516.70 | 518.41 | 515.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 516.70 | 518.41 | 515.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 516.70 | 518.41 | 515.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 516.10 | 517.95 | 515.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 516.10 | 517.95 | 515.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 517.75 | 517.91 | 516.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 12:30:00 | 521.10 | 518.18 | 516.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 13:15:00 | 519.70 | 518.18 | 516.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 14:15:00 | 518.40 | 527.74 | 528.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — SELL (started 2025-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 14:15:00 | 518.40 | 527.74 | 528.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 12:15:00 | 516.40 | 521.71 | 524.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 13:15:00 | 515.00 | 513.68 | 517.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-18 14:00:00 | 515.00 | 513.68 | 517.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 523.00 | 515.54 | 518.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 523.00 | 515.54 | 518.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 525.70 | 517.57 | 519.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 518.75 | 517.57 | 519.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 09:45:00 | 522.60 | 517.88 | 519.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 10:45:00 | 522.65 | 518.97 | 519.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 11:15:00 | 522.80 | 518.97 | 519.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 522.00 | 519.58 | 519.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 12:30:00 | 520.35 | 519.58 | 519.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 13:45:00 | 520.65 | 519.72 | 519.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 14:15:00 | 523.00 | 520.37 | 520.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — BUY (started 2025-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 14:15:00 | 523.00 | 520.37 | 520.05 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 15:15:00 | 517.50 | 519.73 | 520.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 12:15:00 | 515.70 | 518.36 | 519.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 519.85 | 516.50 | 517.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 519.85 | 516.50 | 517.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 519.85 | 516.50 | 517.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 519.85 | 516.50 | 517.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 517.90 | 516.78 | 517.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:30:00 | 518.60 | 516.78 | 517.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 516.05 | 516.64 | 517.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 13:30:00 | 515.35 | 516.45 | 517.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 527.45 | 518.65 | 518.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — BUY (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 14:15:00 | 527.45 | 518.65 | 518.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 15:15:00 | 529.00 | 520.72 | 519.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 12:15:00 | 522.15 | 522.24 | 521.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 12:15:00 | 522.15 | 522.24 | 521.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 522.15 | 522.24 | 521.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:00:00 | 522.15 | 522.24 | 521.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 522.90 | 522.50 | 521.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 15:00:00 | 522.90 | 522.50 | 521.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 520.00 | 522.00 | 521.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:15:00 | 518.05 | 522.00 | 521.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 519.00 | 521.40 | 521.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:30:00 | 517.35 | 521.40 | 521.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 519.50 | 521.02 | 521.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 12:15:00 | 517.70 | 519.87 | 520.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 14:15:00 | 519.55 | 519.51 | 520.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-27 15:00:00 | 519.55 | 519.51 | 520.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 520.40 | 519.40 | 520.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 520.40 | 519.40 | 520.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 518.30 | 519.18 | 519.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 11:15:00 | 516.75 | 519.18 | 519.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 521.25 | 517.25 | 518.28 | SL hit (close>static) qty=1.00 sl=520.35 alert=retest2 |

### Cycle 189 — BUY (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 09:15:00 | 526.10 | 519.01 | 518.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 13:15:00 | 530.65 | 524.60 | 522.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 11:15:00 | 529.45 | 530.33 | 526.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-04 12:00:00 | 529.45 | 530.33 | 526.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 529.50 | 530.56 | 527.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:30:00 | 526.25 | 529.47 | 527.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 524.85 | 528.54 | 527.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 524.90 | 528.54 | 527.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — SELL (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 14:15:00 | 521.65 | 525.72 | 526.24 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 527.80 | 525.30 | 525.00 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 10:15:00 | 519.65 | 524.17 | 524.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 11:15:00 | 518.55 | 523.05 | 523.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 09:15:00 | 510.45 | 508.31 | 511.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 510.45 | 508.31 | 511.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 510.45 | 508.31 | 511.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 512.25 | 508.31 | 511.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 510.45 | 509.07 | 510.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:30:00 | 510.10 | 509.07 | 510.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 509.95 | 509.24 | 510.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:45:00 | 510.75 | 509.24 | 510.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 510.05 | 509.40 | 510.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:45:00 | 509.55 | 509.40 | 510.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 511.45 | 509.81 | 510.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:00:00 | 511.45 | 509.81 | 510.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 510.00 | 509.85 | 510.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 507.30 | 509.85 | 510.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 511.00 | 504.37 | 503.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 511.00 | 504.37 | 503.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 514.00 | 509.91 | 507.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 510.65 | 511.57 | 509.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 510.65 | 511.57 | 509.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 510.65 | 511.57 | 509.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:30:00 | 507.60 | 511.57 | 509.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 509.50 | 511.16 | 509.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:45:00 | 509.10 | 511.16 | 509.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 510.10 | 510.95 | 509.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:15:00 | 511.30 | 510.95 | 509.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 14:00:00 | 510.85 | 511.03 | 510.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 14:45:00 | 511.05 | 511.25 | 510.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 09:15:00 | 507.15 | 510.43 | 510.02 | SL hit (close<static) qty=1.00 sl=508.95 alert=retest2 |

### Cycle 194 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 507.00 | 509.74 | 509.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 506.35 | 508.10 | 508.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 508.50 | 507.68 | 508.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 508.50 | 507.68 | 508.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 508.50 | 507.68 | 508.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 10:15:00 | 506.45 | 507.68 | 508.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:15:00 | 505.45 | 507.61 | 508.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 504.20 | 498.96 | 498.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — BUY (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 09:15:00 | 504.20 | 498.96 | 498.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 09:15:00 | 509.55 | 504.79 | 502.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 14:15:00 | 517.90 | 518.17 | 513.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 15:00:00 | 517.90 | 518.17 | 513.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 516.00 | 518.74 | 516.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 14:30:00 | 522.00 | 519.19 | 516.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 12:15:00 | 513.90 | 517.04 | 516.45 | SL hit (close<static) qty=1.00 sl=514.55 alert=retest2 |

### Cycle 196 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 507.05 | 515.04 | 515.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 14:15:00 | 504.50 | 512.94 | 514.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 503.45 | 501.25 | 506.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 15:00:00 | 503.45 | 501.25 | 506.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 507.60 | 503.14 | 506.37 | EMA400 retest candle locked (from downside) |

### Cycle 197 — BUY (started 2026-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 14:15:00 | 519.00 | 509.78 | 508.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 15:15:00 | 520.10 | 511.84 | 509.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 522.75 | 523.53 | 518.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 522.75 | 523.53 | 518.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 522.75 | 523.53 | 518.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:00:00 | 522.75 | 523.53 | 518.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 517.90 | 521.74 | 518.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:45:00 | 517.80 | 521.74 | 518.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 512.65 | 519.92 | 517.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:00:00 | 512.65 | 519.92 | 517.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 509.80 | 517.90 | 517.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:30:00 | 509.95 | 517.90 | 517.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — SELL (started 2026-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 14:15:00 | 507.90 | 515.90 | 516.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 11:15:00 | 502.00 | 508.60 | 511.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-20 14:15:00 | 507.00 | 505.78 | 509.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-20 15:00:00 | 507.00 | 505.78 | 509.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 503.00 | 505.22 | 508.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 09:15:00 | 495.20 | 505.22 | 508.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 10:30:00 | 501.95 | 503.98 | 507.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 12:15:00 | 501.80 | 504.21 | 507.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 13:30:00 | 501.55 | 503.35 | 506.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 509.40 | 504.11 | 505.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 514.20 | 504.11 | 505.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 506.25 | 504.54 | 505.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 505.25 | 504.54 | 505.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:00:00 | 506.10 | 503.88 | 504.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 15:15:00 | 507.15 | 503.46 | 503.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 507.15 | 503.46 | 503.23 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 11:15:00 | 501.80 | 503.01 | 503.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 13:15:00 | 501.10 | 502.58 | 502.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 14:15:00 | 492.05 | 490.84 | 493.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 14:15:00 | 492.05 | 490.84 | 493.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 492.05 | 490.84 | 493.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 15:00:00 | 492.05 | 490.84 | 493.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 492.00 | 491.07 | 493.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 489.00 | 491.07 | 493.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 10:15:00 | 497.95 | 486.91 | 487.69 | SL hit (close>static) qty=1.00 sl=493.65 alert=retest2 |

### Cycle 201 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 503.00 | 490.13 | 489.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 15:15:00 | 505.25 | 498.37 | 493.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 10:15:00 | 505.50 | 508.94 | 503.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 11:00:00 | 505.50 | 508.94 | 503.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 504.85 | 508.12 | 503.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 14:00:00 | 507.30 | 507.24 | 504.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 10:15:00 | 502.90 | 505.90 | 504.51 | SL hit (close<static) qty=1.00 sl=503.00 alert=retest2 |

### Cycle 202 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 501.60 | 507.21 | 507.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 500.00 | 505.77 | 506.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 09:15:00 | 481.00 | 479.88 | 484.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 10:00:00 | 481.00 | 479.88 | 484.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 455.65 | 451.10 | 456.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:15:00 | 456.60 | 451.10 | 456.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 453.70 | 451.62 | 456.67 | EMA400 retest candle locked (from downside) |

### Cycle 203 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 461.65 | 457.07 | 456.86 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 454.50 | 458.32 | 458.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 12:15:00 | 453.25 | 457.30 | 458.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 434.80 | 434.57 | 440.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 13:15:00 | 437.70 | 435.84 | 439.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 437.70 | 435.84 | 439.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:30:00 | 439.20 | 435.84 | 439.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 443.75 | 437.42 | 439.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 443.15 | 437.42 | 439.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 441.90 | 438.32 | 439.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:30:00 | 440.65 | 438.65 | 439.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 418.62 | 435.61 | 438.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 13:15:00 | 428.80 | 428.43 | 433.28 | SL hit (close>ema200) qty=0.50 sl=428.43 alert=retest2 |

### Cycle 205 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 436.85 | 434.31 | 434.23 | EMA200 above EMA400 |

### Cycle 206 — SELL (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 13:15:00 | 430.00 | 434.31 | 434.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 414.50 | 429.51 | 432.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 11:15:00 | 405.35 | 403.42 | 410.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 11:45:00 | 405.30 | 403.42 | 410.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 407.55 | 404.25 | 410.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 13:00:00 | 407.55 | 404.25 | 410.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 406.85 | 404.77 | 410.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:15:00 | 412.00 | 404.77 | 410.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 412.90 | 406.40 | 410.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 413.40 | 406.40 | 410.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 412.80 | 407.68 | 410.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 09:15:00 | 410.70 | 407.68 | 410.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 09:15:00 | 414.45 | 409.03 | 410.98 | SL hit (close>static) qty=1.00 sl=414.25 alert=retest2 |

### Cycle 207 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 419.15 | 413.23 | 412.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 14:15:00 | 423.60 | 415.30 | 413.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 14:15:00 | 419.70 | 422.18 | 418.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 14:15:00 | 419.70 | 422.18 | 418.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 419.70 | 422.18 | 418.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 15:00:00 | 419.70 | 422.18 | 418.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 417.70 | 420.93 | 418.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 15:15:00 | 422.00 | 418.21 | 417.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 11:30:00 | 419.65 | 418.66 | 418.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 12:30:00 | 419.10 | 419.00 | 418.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 403.85 | 416.67 | 417.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 403.85 | 416.67 | 417.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 401.95 | 413.73 | 416.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 406.20 | 404.98 | 409.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 13:15:00 | 407.75 | 406.09 | 408.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 407.75 | 406.09 | 408.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 409.10 | 406.09 | 408.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 411.85 | 406.85 | 408.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:15:00 | 412.60 | 406.85 | 408.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 415.55 | 408.59 | 409.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 11:00:00 | 415.55 | 408.59 | 409.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 209 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 416.60 | 410.19 | 409.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 417.55 | 411.66 | 410.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 409.40 | 414.20 | 412.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 409.40 | 414.20 | 412.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 409.40 | 414.20 | 412.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 409.40 | 414.20 | 412.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 412.90 | 413.94 | 412.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 11:45:00 | 413.50 | 413.93 | 412.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 405.50 | 411.47 | 411.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 210 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 405.50 | 411.47 | 411.92 | EMA200 below EMA400 |

### Cycle 211 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 418.65 | 411.95 | 411.44 | EMA200 above EMA400 |

### Cycle 212 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 403.25 | 411.27 | 412.02 | EMA200 below EMA400 |

### Cycle 213 — BUY (started 2026-04-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 13:15:00 | 411.30 | 409.50 | 409.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 14:15:00 | 412.45 | 410.09 | 409.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 429.55 | 430.25 | 423.16 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 11:30:00 | 434.60 | 432.01 | 425.22 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 14:30:00 | 436.30 | 434.04 | 427.87 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 430.45 | 436.16 | 433.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 430.45 | 436.16 | 433.21 | SL hit (close<ema400) qty=1.00 sl=433.21 alert=retest1 |

### Cycle 214 — SELL (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 15:15:00 | 442.00 | 442.63 | 442.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 439.50 | 442.00 | 442.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 428.75 | 427.33 | 430.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:30:00 | 428.65 | 427.33 | 430.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 426.70 | 427.59 | 429.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:45:00 | 427.10 | 427.59 | 429.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 428.00 | 427.88 | 429.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:15:00 | 429.95 | 427.88 | 429.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 431.40 | 428.59 | 429.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:15:00 | 430.65 | 428.59 | 429.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 431.65 | 429.20 | 430.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:00:00 | 431.65 | 429.20 | 430.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 429.00 | 426.79 | 428.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 10:30:00 | 426.50 | 426.85 | 428.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 09:15:00 | 405.17 | 409.94 | 414.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 414.10 | 406.65 | 409.74 | SL hit (close>ema200) qty=0.50 sl=406.65 alert=retest2 |

### Cycle 215 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 413.15 | 410.64 | 410.42 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 409.90 | 410.66 | 410.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 15:15:00 | 408.65 | 410.02 | 410.38 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-06-02 09:15:00 | 391.75 | 2023-06-02 10:15:00 | 389.00 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2023-06-02 12:00:00 | 391.75 | 2023-06-06 11:15:00 | 390.80 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2023-06-05 09:15:00 | 394.75 | 2023-06-19 15:15:00 | 407.95 | STOP_HIT | 1.00 | 3.34% |
| BUY | retest2 | 2023-06-06 09:15:00 | 392.20 | 2023-06-19 15:15:00 | 407.95 | STOP_HIT | 1.00 | 4.02% |
| BUY | retest2 | 2023-06-06 11:15:00 | 392.95 | 2023-06-19 15:15:00 | 407.95 | STOP_HIT | 1.00 | 3.82% |
| BUY | retest2 | 2023-06-07 09:15:00 | 395.05 | 2023-06-19 15:15:00 | 407.95 | STOP_HIT | 1.00 | 3.27% |
| BUY | retest2 | 2023-06-23 09:45:00 | 419.65 | 2023-06-26 09:15:00 | 412.50 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2023-06-23 12:30:00 | 417.40 | 2023-06-26 09:15:00 | 412.50 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2023-06-23 15:15:00 | 417.20 | 2023-06-26 09:15:00 | 412.50 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2023-07-18 11:30:00 | 416.50 | 2023-07-19 15:15:00 | 420.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2023-07-18 14:15:00 | 418.95 | 2023-07-19 15:15:00 | 420.00 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2023-07-18 15:15:00 | 418.00 | 2023-07-19 15:15:00 | 420.00 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2023-07-19 14:00:00 | 419.35 | 2023-07-19 15:15:00 | 420.00 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2023-08-01 15:00:00 | 427.40 | 2023-08-02 11:15:00 | 425.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2023-08-02 09:30:00 | 428.50 | 2023-08-02 11:15:00 | 425.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2023-08-24 10:30:00 | 394.70 | 2023-08-24 12:15:00 | 399.05 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2023-08-24 12:00:00 | 395.00 | 2023-08-24 12:15:00 | 399.05 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2023-09-01 14:30:00 | 387.00 | 2023-09-08 10:15:00 | 384.65 | STOP_HIT | 1.00 | 0.61% |
| SELL | retest2 | 2023-09-05 09:45:00 | 387.40 | 2023-09-08 10:15:00 | 384.65 | STOP_HIT | 1.00 | 0.71% |
| SELL | retest2 | 2023-09-05 10:15:00 | 387.35 | 2023-09-08 10:15:00 | 384.65 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2023-09-05 11:00:00 | 386.40 | 2023-09-08 10:15:00 | 384.65 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2023-09-12 09:45:00 | 380.60 | 2023-09-13 11:15:00 | 386.05 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2023-09-12 11:30:00 | 381.70 | 2023-09-13 11:15:00 | 386.05 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2023-09-12 14:45:00 | 381.55 | 2023-09-13 11:15:00 | 386.05 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2023-09-13 10:15:00 | 381.90 | 2023-09-13 11:15:00 | 386.05 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2023-09-21 09:15:00 | 371.10 | 2023-09-25 11:15:00 | 375.90 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2023-09-29 14:45:00 | 370.00 | 2023-10-05 12:15:00 | 370.80 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2023-10-04 09:15:00 | 369.20 | 2023-10-05 12:15:00 | 370.80 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2023-10-05 10:15:00 | 369.65 | 2023-10-05 12:15:00 | 370.80 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2023-10-05 11:00:00 | 371.30 | 2023-10-05 12:15:00 | 370.80 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2023-10-13 12:15:00 | 380.35 | 2023-10-19 09:15:00 | 379.80 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2023-10-16 11:15:00 | 379.90 | 2023-10-19 09:15:00 | 379.80 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2023-10-16 12:00:00 | 380.05 | 2023-10-19 09:15:00 | 379.80 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2023-10-27 12:45:00 | 374.75 | 2023-10-31 09:15:00 | 378.40 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2023-10-27 15:15:00 | 374.50 | 2023-10-31 09:15:00 | 378.40 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2023-11-06 09:15:00 | 386.05 | 2023-11-07 11:15:00 | 383.60 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2023-11-07 11:45:00 | 385.10 | 2023-11-07 12:15:00 | 383.65 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2023-12-08 09:15:00 | 461.80 | 2023-12-11 11:15:00 | 456.70 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2023-12-08 12:30:00 | 457.20 | 2023-12-11 11:15:00 | 456.70 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2023-12-08 14:30:00 | 457.30 | 2023-12-11 11:15:00 | 456.70 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2023-12-12 12:30:00 | 455.10 | 2023-12-14 13:15:00 | 461.55 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-01-09 09:15:00 | 469.30 | 2024-01-20 12:15:00 | 516.23 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-26 12:15:00 | 526.00 | 2024-02-27 11:15:00 | 521.75 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-02-26 12:45:00 | 525.70 | 2024-02-27 11:15:00 | 521.75 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-02-26 13:45:00 | 526.95 | 2024-02-27 11:15:00 | 521.75 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-02-27 09:15:00 | 526.40 | 2024-02-27 11:15:00 | 521.75 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-03-04 12:15:00 | 530.15 | 2024-03-07 10:15:00 | 529.95 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2024-03-06 10:30:00 | 529.90 | 2024-03-07 10:15:00 | 529.95 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2024-03-06 12:30:00 | 530.45 | 2024-03-07 10:15:00 | 529.95 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2024-03-07 09:45:00 | 529.80 | 2024-03-07 10:15:00 | 529.95 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2024-03-15 09:15:00 | 497.35 | 2024-03-15 11:15:00 | 472.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-15 09:15:00 | 497.35 | 2024-03-19 09:15:00 | 447.62 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-04-04 09:15:00 | 474.30 | 2024-04-04 09:15:00 | 466.75 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-04-10 09:15:00 | 478.90 | 2024-04-15 14:15:00 | 472.65 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-04-15 09:45:00 | 474.20 | 2024-04-15 14:15:00 | 472.65 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2024-04-19 15:15:00 | 472.25 | 2024-04-22 09:15:00 | 490.90 | STOP_HIT | 1.00 | -3.95% |
| BUY | retest2 | 2024-04-29 09:15:00 | 502.15 | 2024-05-03 13:15:00 | 499.45 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-04-29 14:00:00 | 494.35 | 2024-05-03 13:15:00 | 499.45 | STOP_HIT | 1.00 | 1.03% |
| BUY | retest2 | 2024-04-29 15:00:00 | 495.30 | 2024-05-03 13:15:00 | 499.45 | STOP_HIT | 1.00 | 0.84% |
| SELL | retest2 | 2024-05-08 14:15:00 | 479.15 | 2024-05-16 09:15:00 | 502.80 | STOP_HIT | 1.00 | -4.94% |
| SELL | retest2 | 2024-05-09 10:30:00 | 477.60 | 2024-05-16 09:15:00 | 502.80 | STOP_HIT | 1.00 | -5.28% |
| SELL | retest2 | 2024-05-09 12:30:00 | 478.10 | 2024-05-16 09:15:00 | 502.80 | STOP_HIT | 1.00 | -5.17% |
| SELL | retest2 | 2024-05-09 13:45:00 | 478.70 | 2024-05-16 09:15:00 | 502.80 | STOP_HIT | 1.00 | -5.03% |
| SELL | retest2 | 2024-05-14 10:45:00 | 472.40 | 2024-05-16 09:15:00 | 502.80 | STOP_HIT | 1.00 | -6.44% |
| SELL | retest2 | 2024-05-14 11:15:00 | 472.30 | 2024-05-16 09:15:00 | 502.80 | STOP_HIT | 1.00 | -6.46% |
| SELL | retest2 | 2024-05-14 12:15:00 | 472.65 | 2024-05-16 09:15:00 | 502.80 | STOP_HIT | 1.00 | -6.38% |
| SELL | retest2 | 2024-05-14 12:45:00 | 472.55 | 2024-05-16 09:15:00 | 502.80 | STOP_HIT | 1.00 | -6.40% |
| SELL | retest2 | 2024-05-31 14:15:00 | 465.80 | 2024-06-03 09:15:00 | 472.55 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-06-12 09:15:00 | 483.70 | 2024-06-12 11:15:00 | 478.40 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-06-25 10:15:00 | 501.60 | 2024-07-03 15:15:00 | 533.40 | STOP_HIT | 1.00 | 6.34% |
| BUY | retest2 | 2024-06-25 12:00:00 | 501.50 | 2024-07-03 15:15:00 | 533.40 | STOP_HIT | 1.00 | 6.36% |
| BUY | retest2 | 2024-06-25 12:30:00 | 501.80 | 2024-07-03 15:15:00 | 533.40 | STOP_HIT | 1.00 | 6.30% |
| SELL | retest1 | 2024-07-05 09:15:00 | 527.00 | 2024-07-08 09:15:00 | 529.20 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2024-07-08 13:15:00 | 521.95 | 2024-07-09 10:15:00 | 530.45 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-07-26 12:00:00 | 553.05 | 2024-08-01 12:15:00 | 548.85 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-07-26 13:30:00 | 551.40 | 2024-08-01 12:15:00 | 548.85 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-07-26 14:00:00 | 551.45 | 2024-08-01 12:15:00 | 548.85 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-07-26 15:00:00 | 551.40 | 2024-08-01 12:15:00 | 548.85 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-07-30 10:15:00 | 557.20 | 2024-08-01 12:15:00 | 548.85 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-08-01 09:45:00 | 555.75 | 2024-08-01 12:15:00 | 548.85 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest1 | 2024-08-06 10:30:00 | 522.65 | 2024-08-09 09:15:00 | 496.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-08-06 14:00:00 | 521.65 | 2024-08-09 09:15:00 | 495.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-08-06 10:30:00 | 522.65 | 2024-08-12 12:15:00 | 496.75 | STOP_HIT | 0.50 | 4.96% |
| SELL | retest1 | 2024-08-06 14:00:00 | 521.65 | 2024-08-12 12:15:00 | 496.75 | STOP_HIT | 0.50 | 4.77% |
| SELL | retest2 | 2024-08-16 10:15:00 | 481.00 | 2024-08-20 11:15:00 | 489.15 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-08-16 10:45:00 | 480.65 | 2024-08-20 11:15:00 | 489.15 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-08-27 10:00:00 | 506.60 | 2024-08-27 15:15:00 | 504.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-08-27 12:00:00 | 506.95 | 2024-08-27 15:15:00 | 504.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-08-27 13:15:00 | 506.90 | 2024-08-27 15:15:00 | 504.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-08-27 14:15:00 | 506.80 | 2024-08-27 15:15:00 | 504.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-09-06 11:30:00 | 505.55 | 2024-09-17 13:15:00 | 520.25 | STOP_HIT | 1.00 | 2.91% |
| BUY | retest2 | 2024-09-06 14:30:00 | 505.00 | 2024-09-17 13:15:00 | 520.25 | STOP_HIT | 1.00 | 3.02% |
| BUY | retest2 | 2024-09-09 09:45:00 | 506.00 | 2024-09-17 13:15:00 | 520.25 | STOP_HIT | 1.00 | 2.82% |
| SELL | retest2 | 2024-09-20 09:15:00 | 515.90 | 2024-09-20 11:15:00 | 520.55 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-09-20 10:15:00 | 516.10 | 2024-09-20 11:15:00 | 520.55 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-10-01 09:30:00 | 549.90 | 2024-10-03 09:15:00 | 531.15 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2024-10-01 11:15:00 | 549.70 | 2024-10-03 09:15:00 | 531.15 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2024-10-09 14:00:00 | 509.55 | 2024-10-15 10:15:00 | 512.95 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-10-11 13:30:00 | 509.15 | 2024-10-15 10:15:00 | 512.95 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-10-16 14:30:00 | 513.65 | 2024-10-17 13:15:00 | 505.55 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2024-10-21 12:30:00 | 507.40 | 2024-10-21 14:15:00 | 511.90 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-10-21 13:15:00 | 507.50 | 2024-10-21 14:15:00 | 511.90 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-10-28 12:30:00 | 485.60 | 2024-10-29 12:15:00 | 491.65 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-10-28 13:00:00 | 486.20 | 2024-10-29 12:15:00 | 491.65 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-10-28 14:00:00 | 486.20 | 2024-10-29 12:15:00 | 491.65 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-10-29 09:30:00 | 482.65 | 2024-10-29 12:15:00 | 491.65 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-10-31 14:45:00 | 503.95 | 2024-11-04 09:15:00 | 487.75 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2024-11-01 18:00:00 | 504.75 | 2024-11-04 09:15:00 | 487.75 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2024-11-18 09:15:00 | 470.50 | 2024-11-18 13:15:00 | 475.65 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-11-18 12:45:00 | 472.55 | 2024-11-18 13:15:00 | 475.65 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-11-18 13:15:00 | 472.55 | 2024-11-18 13:15:00 | 475.65 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-11-21 11:15:00 | 485.15 | 2024-11-29 11:15:00 | 506.45 | STOP_HIT | 1.00 | 4.39% |
| BUY | retest2 | 2024-11-21 13:15:00 | 485.80 | 2024-11-29 11:15:00 | 506.45 | STOP_HIT | 1.00 | 4.25% |
| BUY | retest2 | 2024-11-22 09:15:00 | 485.80 | 2024-11-29 11:15:00 | 506.45 | STOP_HIT | 1.00 | 4.25% |
| BUY | retest2 | 2024-11-22 09:45:00 | 486.55 | 2024-11-29 11:15:00 | 506.45 | STOP_HIT | 1.00 | 4.09% |
| BUY | retest2 | 2024-12-09 10:30:00 | 553.80 | 2024-12-11 10:15:00 | 540.25 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2024-12-09 14:00:00 | 549.50 | 2024-12-11 10:15:00 | 540.25 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-12-09 14:45:00 | 549.95 | 2024-12-11 10:15:00 | 540.25 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-12-23 09:45:00 | 530.00 | 2024-12-23 10:15:00 | 533.65 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-12-23 13:15:00 | 530.30 | 2024-12-24 09:15:00 | 534.00 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-12-23 13:45:00 | 530.00 | 2024-12-24 09:15:00 | 534.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-12-26 09:15:00 | 537.40 | 2024-12-30 13:15:00 | 533.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-12-30 10:15:00 | 537.10 | 2024-12-30 13:15:00 | 533.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-01-03 12:15:00 | 516.15 | 2025-01-08 09:15:00 | 490.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-03 13:30:00 | 516.45 | 2025-01-08 09:15:00 | 490.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-03 12:15:00 | 516.15 | 2025-01-10 09:15:00 | 464.53 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-03 13:30:00 | 516.45 | 2025-01-10 09:15:00 | 464.81 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-20 12:15:00 | 449.25 | 2025-01-24 13:15:00 | 426.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 10:15:00 | 448.10 | 2025-01-24 13:15:00 | 426.79 | PARTIAL | 0.50 | 4.76% |
| SELL | retest2 | 2025-01-21 12:00:00 | 449.25 | 2025-01-24 14:15:00 | 425.69 | PARTIAL | 0.50 | 5.24% |
| SELL | retest2 | 2025-01-21 14:30:00 | 447.65 | 2025-01-24 14:15:00 | 425.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-22 10:15:00 | 440.90 | 2025-01-27 09:15:00 | 418.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-22 10:45:00 | 440.55 | 2025-01-27 09:15:00 | 418.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 10:30:00 | 440.25 | 2025-01-27 09:15:00 | 418.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-20 12:15:00 | 449.25 | 2025-01-27 14:15:00 | 419.95 | STOP_HIT | 0.50 | 6.52% |
| SELL | retest2 | 2025-01-21 10:15:00 | 448.10 | 2025-01-27 14:15:00 | 419.95 | STOP_HIT | 0.50 | 6.28% |
| SELL | retest2 | 2025-01-21 12:00:00 | 449.25 | 2025-01-27 14:15:00 | 419.95 | STOP_HIT | 0.50 | 6.52% |
| SELL | retest2 | 2025-01-21 14:30:00 | 447.65 | 2025-01-27 14:15:00 | 419.95 | STOP_HIT | 0.50 | 6.19% |
| SELL | retest2 | 2025-01-22 10:15:00 | 440.90 | 2025-01-27 14:15:00 | 419.95 | STOP_HIT | 0.50 | 4.75% |
| SELL | retest2 | 2025-01-22 10:45:00 | 440.55 | 2025-01-27 14:15:00 | 419.95 | STOP_HIT | 0.50 | 4.68% |
| SELL | retest2 | 2025-01-23 10:30:00 | 440.25 | 2025-01-27 14:15:00 | 419.95 | STOP_HIT | 0.50 | 4.61% |
| SELL | retest2 | 2025-02-06 12:30:00 | 414.75 | 2025-02-07 09:15:00 | 426.85 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-02-06 13:00:00 | 415.30 | 2025-02-07 09:15:00 | 426.85 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-02-20 09:15:00 | 410.60 | 2025-02-20 10:15:00 | 414.70 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-03-11 11:45:00 | 406.85 | 2025-03-11 14:15:00 | 401.55 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-03-25 13:45:00 | 433.30 | 2025-03-27 15:15:00 | 426.10 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-03-25 14:45:00 | 431.65 | 2025-03-27 15:15:00 | 426.10 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-03-26 09:45:00 | 433.25 | 2025-03-27 15:15:00 | 426.10 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-03-26 10:15:00 | 431.40 | 2025-03-27 15:15:00 | 426.10 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-03-27 10:30:00 | 429.75 | 2025-03-27 15:15:00 | 426.10 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-03-27 11:15:00 | 430.55 | 2025-03-27 15:15:00 | 426.10 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-03-27 12:30:00 | 432.00 | 2025-03-27 15:15:00 | 426.10 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-03-27 14:45:00 | 430.40 | 2025-03-27 15:15:00 | 426.10 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-04-01 11:30:00 | 421.30 | 2025-04-02 15:15:00 | 428.70 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-04-02 09:15:00 | 420.95 | 2025-04-02 15:15:00 | 428.70 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-04-24 12:30:00 | 464.65 | 2025-04-28 09:15:00 | 455.85 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-04-24 13:15:00 | 464.45 | 2025-04-28 09:15:00 | 455.85 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-04-25 12:45:00 | 465.65 | 2025-04-28 09:15:00 | 455.85 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-04-25 14:15:00 | 463.25 | 2025-04-28 09:15:00 | 455.85 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-05-07 12:45:00 | 492.25 | 2025-05-09 09:15:00 | 473.75 | STOP_HIT | 1.00 | -3.76% |
| SELL | retest2 | 2025-05-13 12:15:00 | 471.15 | 2025-05-15 09:15:00 | 483.30 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-05-21 10:30:00 | 491.00 | 2025-05-21 11:15:00 | 486.25 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-05-27 12:30:00 | 498.60 | 2025-05-28 15:15:00 | 493.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-05-27 13:30:00 | 498.35 | 2025-05-28 15:15:00 | 493.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest1 | 2025-06-16 09:15:00 | 446.00 | 2025-06-19 14:15:00 | 447.90 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-06-19 12:30:00 | 445.55 | 2025-06-19 14:15:00 | 447.90 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-06-20 09:15:00 | 445.65 | 2025-06-24 11:15:00 | 446.60 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-06-20 10:30:00 | 445.55 | 2025-06-24 11:15:00 | 446.60 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-06-20 14:15:00 | 445.25 | 2025-06-24 11:15:00 | 446.60 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-06-27 09:15:00 | 447.60 | 2025-06-27 15:15:00 | 445.95 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-06-30 09:15:00 | 450.40 | 2025-07-01 11:15:00 | 446.05 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-07-01 11:00:00 | 447.25 | 2025-07-01 11:15:00 | 446.05 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-07-16 12:00:00 | 455.00 | 2025-07-17 09:15:00 | 469.95 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2025-07-16 15:15:00 | 454.95 | 2025-07-17 09:15:00 | 469.95 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2025-07-31 11:00:00 | 457.95 | 2025-07-31 14:15:00 | 448.70 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-08-05 13:15:00 | 440.35 | 2025-08-11 14:15:00 | 439.00 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2025-08-05 14:00:00 | 439.50 | 2025-08-11 14:15:00 | 439.00 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-08-06 09:45:00 | 439.20 | 2025-08-11 14:15:00 | 439.00 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-08-21 13:45:00 | 463.50 | 2025-08-22 09:15:00 | 458.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-08-21 14:30:00 | 463.80 | 2025-08-22 09:15:00 | 458.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-08-22 11:30:00 | 464.05 | 2025-08-26 09:15:00 | 461.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-08-26 11:15:00 | 465.05 | 2025-08-26 11:15:00 | 462.20 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-08-28 14:45:00 | 458.60 | 2025-08-29 14:15:00 | 462.30 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-09-08 09:45:00 | 488.45 | 2025-09-08 14:15:00 | 479.95 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-09-11 09:15:00 | 476.50 | 2025-09-15 12:15:00 | 480.10 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-09-11 11:30:00 | 476.00 | 2025-09-15 12:15:00 | 480.10 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-09-11 13:30:00 | 477.30 | 2025-09-15 12:15:00 | 480.10 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-09-11 15:00:00 | 476.90 | 2025-09-15 12:15:00 | 480.10 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-09-12 11:30:00 | 472.25 | 2025-09-15 12:15:00 | 480.10 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-09-15 10:15:00 | 473.05 | 2025-09-15 12:15:00 | 480.10 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest1 | 2025-09-23 09:15:00 | 490.95 | 2025-09-24 10:15:00 | 485.40 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-09-25 14:00:00 | 494.75 | 2025-09-26 09:15:00 | 484.55 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-09-25 15:00:00 | 493.60 | 2025-09-26 09:15:00 | 484.55 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-10-09 14:30:00 | 483.15 | 2025-10-13 10:15:00 | 479.05 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-10-13 11:15:00 | 485.65 | 2025-10-14 13:15:00 | 478.25 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-10-16 14:15:00 | 488.70 | 2025-10-24 15:15:00 | 500.60 | STOP_HIT | 1.00 | 2.44% |
| BUY | retest2 | 2025-10-17 15:00:00 | 491.35 | 2025-10-24 15:15:00 | 500.60 | STOP_HIT | 1.00 | 1.88% |
| BUY | retest2 | 2025-10-29 09:15:00 | 507.40 | 2025-10-29 09:15:00 | 503.20 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-10-30 11:00:00 | 504.60 | 2025-11-03 09:15:00 | 506.45 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-10-30 13:00:00 | 504.55 | 2025-11-03 09:15:00 | 506.45 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-10-31 10:00:00 | 504.60 | 2025-11-03 09:15:00 | 506.45 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-10-31 12:15:00 | 504.15 | 2025-11-03 11:15:00 | 505.10 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-10-31 13:15:00 | 502.90 | 2025-11-03 11:15:00 | 505.10 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-10-31 14:15:00 | 502.65 | 2025-11-03 11:15:00 | 505.10 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-10-31 15:00:00 | 501.00 | 2025-11-03 11:15:00 | 505.10 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-11-11 12:30:00 | 521.10 | 2025-11-14 14:15:00 | 518.40 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-11-11 13:15:00 | 519.70 | 2025-11-14 14:15:00 | 518.40 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-11-19 09:15:00 | 518.75 | 2025-11-19 14:15:00 | 523.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-11-19 09:45:00 | 522.60 | 2025-11-19 14:15:00 | 523.00 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-11-19 10:45:00 | 522.65 | 2025-11-19 14:15:00 | 523.00 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-11-19 11:15:00 | 522.80 | 2025-11-19 14:15:00 | 523.00 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-11-19 12:30:00 | 520.35 | 2025-11-19 14:15:00 | 523.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-11-19 13:45:00 | 520.65 | 2025-11-19 14:15:00 | 523.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-11-24 13:30:00 | 515.35 | 2025-11-24 14:15:00 | 527.45 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-11-28 11:15:00 | 516.75 | 2025-12-01 09:15:00 | 521.25 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-12-01 14:30:00 | 515.10 | 2025-12-02 09:15:00 | 526.10 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-12-17 09:15:00 | 507.30 | 2025-12-19 13:15:00 | 511.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-12-24 12:15:00 | 511.30 | 2025-12-26 09:15:00 | 507.15 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-12-24 14:00:00 | 510.85 | 2025-12-26 09:15:00 | 507.15 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-12-24 14:45:00 | 511.05 | 2025-12-26 09:15:00 | 507.15 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-12-29 10:15:00 | 506.45 | 2026-01-05 09:15:00 | 504.20 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2025-12-29 11:15:00 | 505.45 | 2026-01-05 09:15:00 | 504.20 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2026-01-08 14:30:00 | 522.00 | 2026-01-09 12:15:00 | 513.90 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-01-21 09:15:00 | 495.20 | 2026-01-27 15:15:00 | 507.15 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2026-01-21 10:30:00 | 501.95 | 2026-01-27 15:15:00 | 507.15 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2026-01-21 12:15:00 | 501.80 | 2026-01-27 15:15:00 | 507.15 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-01-21 13:30:00 | 501.55 | 2026-01-27 15:15:00 | 507.15 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-01-22 11:15:00 | 505.25 | 2026-01-27 15:15:00 | 507.15 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2026-01-23 11:00:00 | 506.10 | 2026-01-27 15:15:00 | 507.15 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2026-02-01 09:15:00 | 489.00 | 2026-02-03 10:15:00 | 497.95 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2026-02-05 14:00:00 | 507.30 | 2026-02-06 10:15:00 | 502.90 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-02-06 13:00:00 | 507.55 | 2026-02-06 15:15:00 | 502.85 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2026-02-09 09:15:00 | 508.25 | 2026-02-12 09:15:00 | 505.85 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2026-02-10 09:15:00 | 512.00 | 2026-02-12 10:15:00 | 501.60 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2026-02-11 09:15:00 | 517.80 | 2026-02-12 10:15:00 | 501.60 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2026-03-06 09:30:00 | 440.65 | 2026-03-09 09:15:00 | 418.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 09:30:00 | 440.65 | 2026-03-09 13:15:00 | 428.80 | STOP_HIT | 0.50 | 2.69% |
| SELL | retest2 | 2026-03-17 09:15:00 | 410.70 | 2026-03-17 09:15:00 | 414.45 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-03-19 15:15:00 | 422.00 | 2026-03-23 09:15:00 | 403.85 | STOP_HIT | 1.00 | -4.30% |
| BUY | retest2 | 2026-03-20 11:30:00 | 419.65 | 2026-03-23 09:15:00 | 403.85 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest2 | 2026-03-20 12:30:00 | 419.10 | 2026-03-23 09:15:00 | 403.85 | STOP_HIT | 1.00 | -3.64% |
| BUY | retest2 | 2026-03-27 11:45:00 | 413.50 | 2026-03-30 09:15:00 | 405.50 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest1 | 2026-04-09 11:30:00 | 434.60 | 2026-04-13 09:15:00 | 430.45 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest1 | 2026-04-09 14:30:00 | 436.30 | 2026-04-13 09:15:00 | 430.45 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2026-04-13 10:45:00 | 433.25 | 2026-04-21 15:15:00 | 442.00 | STOP_HIT | 1.00 | 2.02% |
| BUY | retest2 | 2026-04-13 14:15:00 | 433.35 | 2026-04-21 15:15:00 | 442.00 | STOP_HIT | 1.00 | 2.00% |
| BUY | retest2 | 2026-04-15 09:15:00 | 440.80 | 2026-04-21 15:15:00 | 442.00 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2026-04-29 10:30:00 | 426.50 | 2026-05-05 09:15:00 | 405.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-29 10:30:00 | 426.50 | 2026-05-06 09:15:00 | 414.10 | STOP_HIT | 0.50 | 2.91% |
