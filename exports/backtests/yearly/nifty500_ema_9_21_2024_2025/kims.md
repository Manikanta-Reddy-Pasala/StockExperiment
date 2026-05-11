# Krishna Institute of Medical Sciences Ltd. (KIMS)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 715.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 135 |
| ALERT1 | 90 |
| ALERT2 | 90 |
| ALERT2_SKIP | 53 |
| ALERT3 | 239 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 132 |
| PARTIAL | 18 |
| TARGET_HIT | 16 |
| STOP_HIT | 123 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 157 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 70 / 87
- **Target hits / Stop hits / Partials:** 16 / 123 / 18
- **Avg / median % per leg:** 1.28% / -0.38%
- **Sum % (uncompounded):** 201.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 68 | 29 | 42.6% | 14 | 54 | 0 | 1.57% | 106.8% |
| BUY @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.30% | -11.5% |
| BUY @ 3rd Alert (retest2) | 63 | 29 | 46.0% | 14 | 49 | 0 | 1.88% | 118.4% |
| SELL (all) | 89 | 41 | 46.1% | 2 | 69 | 18 | 1.06% | 94.2% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.39% | -7.2% |
| SELL @ 3rd Alert (retest2) | 86 | 41 | 47.7% | 2 | 66 | 18 | 1.18% | 101.4% |
| retest1 (combined) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.34% | -18.7% |
| retest2 (combined) | 149 | 70 | 47.0% | 16 | 115 | 18 | 1.48% | 219.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 15:15:00 | 367.63 | 365.59 | 365.48 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 09:15:00 | 361.40 | 364.75 | 365.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 356.88 | 363.18 | 364.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 12:15:00 | 366.85 | 362.97 | 364.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 12:15:00 | 366.85 | 362.97 | 364.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 366.85 | 362.97 | 364.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 13:00:00 | 366.85 | 362.97 | 364.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 13:15:00 | 372.32 | 364.84 | 364.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 09:15:00 | 375.23 | 368.24 | 366.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 15:15:00 | 376.79 | 376.98 | 374.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 15:15:00 | 376.79 | 376.98 | 374.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 15:15:00 | 376.79 | 376.98 | 374.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 09:15:00 | 381.73 | 376.98 | 374.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 12:15:00 | 378.62 | 379.58 | 377.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-10 14:15:00 | 374.00 | 378.36 | 377.75 | SL hit (close<static) qty=1.00 sl=374.06 alert=retest2 |

### Cycle 4 — SELL (started 2024-06-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-10 15:15:00 | 371.68 | 377.02 | 377.20 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-11 09:15:00 | 381.97 | 378.01 | 377.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 10:15:00 | 384.99 | 379.41 | 378.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-14 15:15:00 | 405.41 | 405.76 | 400.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-18 09:15:00 | 405.92 | 405.76 | 400.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 13:15:00 | 411.21 | 415.53 | 412.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 13:45:00 | 411.61 | 415.53 | 412.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 416.40 | 415.70 | 412.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 14:30:00 | 411.16 | 415.70 | 412.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 412.21 | 415.21 | 413.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:00:00 | 412.21 | 415.21 | 413.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 410.53 | 414.28 | 412.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:45:00 | 410.05 | 414.28 | 412.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 411.38 | 413.21 | 412.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 12:30:00 | 410.63 | 413.21 | 412.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 410.90 | 412.54 | 412.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 15:00:00 | 410.90 | 412.54 | 412.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 15:15:00 | 410.01 | 412.03 | 412.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 09:15:00 | 409.49 | 411.52 | 411.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 11:15:00 | 411.66 | 411.28 | 411.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 11:15:00 | 411.66 | 411.28 | 411.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 411.66 | 411.28 | 411.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:45:00 | 411.14 | 411.28 | 411.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 15:15:00 | 411.00 | 411.11 | 411.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 09:15:00 | 426.20 | 411.11 | 411.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 427.37 | 414.36 | 412.95 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 14:15:00 | 416.92 | 420.20 | 420.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 15:15:00 | 415.59 | 419.27 | 420.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 09:15:00 | 421.08 | 419.64 | 420.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 421.08 | 419.64 | 420.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 421.08 | 419.64 | 420.17 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 423.07 | 420.78 | 420.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 12:15:00 | 426.10 | 421.85 | 421.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 421.37 | 423.38 | 422.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 09:15:00 | 421.37 | 423.38 | 422.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 421.37 | 423.38 | 422.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:30:00 | 422.55 | 423.38 | 422.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 421.20 | 422.95 | 422.14 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 417.70 | 421.44 | 421.57 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 11:15:00 | 424.87 | 421.75 | 421.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 428.12 | 423.90 | 422.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 11:15:00 | 422.24 | 424.13 | 423.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 11:15:00 | 422.24 | 424.13 | 423.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 422.24 | 424.13 | 423.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 12:00:00 | 422.24 | 424.13 | 423.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 12:15:00 | 428.94 | 425.09 | 423.56 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2024-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 12:15:00 | 421.49 | 424.68 | 425.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 14:15:00 | 420.46 | 423.21 | 424.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 15:15:00 | 421.00 | 420.97 | 422.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-10 09:15:00 | 435.87 | 420.97 | 422.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 428.09 | 422.39 | 422.79 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 10:15:00 | 428.98 | 423.71 | 423.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 11:15:00 | 430.03 | 424.98 | 423.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 11:15:00 | 428.49 | 430.64 | 428.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 11:15:00 | 428.49 | 430.64 | 428.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 428.49 | 430.64 | 428.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:00:00 | 428.49 | 430.64 | 428.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 429.53 | 431.10 | 429.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 11:30:00 | 429.82 | 431.10 | 429.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 428.68 | 430.61 | 429.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:45:00 | 426.97 | 430.61 | 429.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 13:15:00 | 427.66 | 430.02 | 429.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 14:00:00 | 427.66 | 430.02 | 429.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 428.40 | 429.44 | 429.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:15:00 | 433.21 | 429.44 | 429.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 434.43 | 430.44 | 429.72 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2024-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 12:15:00 | 426.89 | 428.87 | 429.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 13:15:00 | 424.60 | 428.02 | 428.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 09:15:00 | 432.39 | 427.01 | 427.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 09:15:00 | 432.39 | 427.01 | 427.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 432.39 | 427.01 | 427.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:45:00 | 432.72 | 427.01 | 427.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 427.37 | 427.08 | 427.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 11:15:00 | 426.80 | 427.08 | 427.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 13:30:00 | 426.84 | 426.76 | 427.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 14:00:00 | 426.80 | 424.97 | 425.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-25 09:15:00 | 425.93 | 424.10 | 424.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 09:15:00 | 425.93 | 424.10 | 424.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 13:15:00 | 427.94 | 425.82 | 424.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 15:15:00 | 426.06 | 426.35 | 425.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 09:15:00 | 426.23 | 426.35 | 425.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 426.62 | 426.40 | 425.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 13:00:00 | 430.62 | 427.62 | 426.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 09:15:00 | 429.96 | 430.14 | 429.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 13:15:00 | 429.22 | 431.85 | 431.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-31 15:15:00 | 430.20 | 431.38 | 431.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 15:15:00 | 430.20 | 431.38 | 431.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 09:15:00 | 428.46 | 430.80 | 431.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 429.59 | 420.68 | 423.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 429.59 | 420.68 | 423.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 429.59 | 420.68 | 423.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:30:00 | 429.60 | 420.68 | 423.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 429.26 | 422.39 | 423.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:45:00 | 431.23 | 422.39 | 423.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2024-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 12:15:00 | 428.38 | 424.47 | 424.46 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 14:15:00 | 423.48 | 424.42 | 424.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-07 09:15:00 | 422.37 | 424.04 | 424.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 12:15:00 | 425.32 | 423.84 | 424.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 12:15:00 | 425.32 | 423.84 | 424.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 425.32 | 423.84 | 424.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 13:00:00 | 425.32 | 423.84 | 424.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 13:15:00 | 426.60 | 424.39 | 424.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 14:15:00 | 430.00 | 425.51 | 424.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 11:15:00 | 444.01 | 444.44 | 439.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-12 11:45:00 | 443.93 | 444.44 | 439.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 440.78 | 445.58 | 442.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 11:15:00 | 453.99 | 446.21 | 444.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 13:30:00 | 450.30 | 448.27 | 445.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 15:15:00 | 450.09 | 448.30 | 446.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-21 10:15:00 | 495.33 | 476.59 | 470.42 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 13:15:00 | 493.19 | 496.89 | 497.26 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 507.51 | 498.73 | 497.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 10:15:00 | 514.98 | 506.50 | 502.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 12:15:00 | 506.38 | 506.68 | 503.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 13:00:00 | 506.38 | 506.68 | 503.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 525.11 | 530.32 | 524.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 525.11 | 530.32 | 524.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 530.00 | 530.26 | 524.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 12:00:00 | 530.21 | 530.25 | 525.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 13:45:00 | 530.28 | 529.38 | 525.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 15:15:00 | 535.75 | 529.28 | 526.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 09:15:00 | 520.47 | 526.28 | 526.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 09:15:00 | 520.47 | 526.28 | 526.60 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 15:15:00 | 534.20 | 526.50 | 526.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 09:15:00 | 561.60 | 538.18 | 532.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 13:15:00 | 538.62 | 541.97 | 536.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 13:15:00 | 538.62 | 541.97 | 536.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 538.62 | 541.97 | 536.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:30:00 | 537.21 | 541.97 | 536.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 547.99 | 543.17 | 537.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 09:15:00 | 562.65 | 543.94 | 538.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 12:00:00 | 553.25 | 550.76 | 546.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 11:15:00 | 543.35 | 554.08 | 554.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 11:15:00 | 543.35 | 554.08 | 554.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 541.00 | 551.46 | 553.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 549.80 | 548.59 | 551.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 549.80 | 548.59 | 551.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 549.80 | 548.59 | 551.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:15:00 | 560.90 | 548.59 | 551.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 556.75 | 550.22 | 551.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:45:00 | 560.35 | 550.22 | 551.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 558.95 | 551.97 | 552.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 12:00:00 | 558.95 | 551.97 | 552.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 551.00 | 551.77 | 552.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 13:45:00 | 548.20 | 551.27 | 552.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 12:45:00 | 549.50 | 551.65 | 551.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:30:00 | 549.20 | 551.31 | 551.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 15:00:00 | 546.20 | 550.28 | 551.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 543.75 | 548.61 | 550.34 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-24 14:15:00 | 553.00 | 548.96 | 548.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 14:15:00 | 553.00 | 548.96 | 548.84 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 09:15:00 | 546.40 | 548.77 | 548.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 11:15:00 | 542.60 | 547.02 | 547.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 14:15:00 | 545.80 | 544.90 | 546.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 14:15:00 | 545.80 | 544.90 | 546.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 545.80 | 544.90 | 546.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 15:00:00 | 545.80 | 544.90 | 546.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 544.80 | 544.88 | 546.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 09:15:00 | 541.10 | 544.88 | 546.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 10:45:00 | 542.05 | 543.78 | 545.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 15:15:00 | 546.95 | 545.06 | 544.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 15:15:00 | 546.95 | 545.06 | 544.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 10:15:00 | 548.45 | 546.05 | 545.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 15:15:00 | 556.20 | 557.03 | 553.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 15:15:00 | 556.20 | 557.03 | 553.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 556.20 | 557.03 | 553.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 09:15:00 | 528.20 | 557.03 | 553.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 532.55 | 552.14 | 551.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 09:45:00 | 527.00 | 552.14 | 551.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 10:15:00 | 535.40 | 548.79 | 550.31 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 11:15:00 | 553.85 | 548.23 | 548.04 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 539.90 | 548.10 | 548.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 12:15:00 | 535.35 | 544.23 | 546.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 548.35 | 540.65 | 543.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 10:15:00 | 548.35 | 540.65 | 543.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 548.35 | 540.65 | 543.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:45:00 | 547.40 | 540.65 | 543.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 544.35 | 541.39 | 543.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 12:15:00 | 540.90 | 541.39 | 543.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 13:30:00 | 538.85 | 539.83 | 541.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-11 09:15:00 | 548.25 | 541.66 | 540.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 09:15:00 | 548.25 | 541.66 | 540.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 12:15:00 | 553.80 | 546.14 | 543.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 09:15:00 | 543.25 | 546.74 | 544.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 09:15:00 | 543.25 | 546.74 | 544.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 543.25 | 546.74 | 544.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:00:00 | 543.25 | 546.74 | 544.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 546.20 | 546.63 | 544.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:30:00 | 545.50 | 546.63 | 544.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 12:15:00 | 547.95 | 546.64 | 545.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 12:45:00 | 545.60 | 546.64 | 545.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 14:15:00 | 545.85 | 546.44 | 545.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 14:45:00 | 545.65 | 546.44 | 545.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 15:15:00 | 544.00 | 545.96 | 545.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 09:15:00 | 559.00 | 545.96 | 545.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 11:15:00 | 546.30 | 550.83 | 550.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 11:15:00 | 546.30 | 550.83 | 550.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 541.65 | 548.37 | 549.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 543.90 | 543.56 | 546.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 13:15:00 | 543.90 | 543.56 | 546.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 543.90 | 543.56 | 546.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:30:00 | 543.00 | 543.56 | 546.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 542.45 | 543.34 | 545.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:45:00 | 544.55 | 543.34 | 545.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 542.90 | 542.35 | 544.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 11:45:00 | 544.10 | 542.35 | 544.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 12:15:00 | 543.35 | 542.55 | 544.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 13:00:00 | 543.35 | 542.55 | 544.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 13:15:00 | 544.25 | 542.89 | 544.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:45:00 | 540.05 | 543.03 | 544.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 09:15:00 | 538.00 | 543.67 | 544.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 09:15:00 | 513.05 | 518.08 | 522.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 09:15:00 | 511.10 | 518.08 | 522.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-28 12:15:00 | 517.15 | 516.46 | 520.80 | SL hit (close>ema200) qty=0.50 sl=516.46 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 527.50 | 520.49 | 520.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 530.05 | 523.56 | 521.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-01 17:15:00 | 539.95 | 542.42 | 536.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 539.95 | 542.42 | 536.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 539.95 | 542.42 | 536.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 539.95 | 542.42 | 536.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 556.60 | 545.31 | 539.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 09:15:00 | 563.75 | 552.35 | 548.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 09:30:00 | 564.30 | 557.66 | 553.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 14:00:00 | 562.70 | 559.68 | 556.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 14:45:00 | 563.15 | 560.55 | 556.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 564.95 | 562.14 | 558.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 10:45:00 | 569.45 | 563.72 | 559.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 12:15:00 | 568.05 | 564.30 | 559.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 13:00:00 | 568.15 | 565.07 | 560.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-12 09:15:00 | 571.90 | 572.52 | 569.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 567.75 | 571.20 | 569.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:00:00 | 567.75 | 571.20 | 569.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 566.95 | 570.35 | 569.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:45:00 | 567.40 | 570.35 | 569.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 12:15:00 | 564.25 | 569.13 | 568.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 12:45:00 | 563.65 | 569.13 | 568.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-12 13:15:00 | 564.55 | 568.21 | 568.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 13:15:00 | 564.55 | 568.21 | 568.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 14:15:00 | 557.75 | 566.12 | 567.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 557.90 | 557.81 | 561.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 557.90 | 557.81 | 561.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 557.90 | 557.81 | 561.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 10:15:00 | 554.40 | 559.39 | 560.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 569.90 | 560.01 | 559.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 569.90 | 560.01 | 559.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-21 09:15:00 | 576.60 | 567.50 | 564.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-22 15:15:00 | 592.00 | 592.27 | 584.00 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 09:15:00 | 604.50 | 592.27 | 584.00 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 15:15:00 | 599.85 | 597.99 | 591.42 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 598.75 | 599.18 | 595.39 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-27 09:15:00 | 587.55 | 596.83 | 594.97 | SL hit (close<ema400) qty=1.00 sl=594.97 alert=retest1 |

### Cycle 36 — SELL (started 2024-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 11:15:00 | 586.50 | 593.08 | 593.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 12:15:00 | 584.70 | 591.41 | 592.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 15:15:00 | 590.00 | 589.94 | 591.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-28 09:15:00 | 578.60 | 589.94 | 591.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 578.15 | 577.52 | 582.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:00:00 | 578.15 | 577.52 | 582.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 581.50 | 578.55 | 582.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:45:00 | 581.40 | 578.55 | 582.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 581.20 | 579.08 | 582.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:30:00 | 582.35 | 579.08 | 582.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 581.50 | 579.57 | 582.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 13:30:00 | 582.50 | 579.57 | 582.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 588.55 | 581.36 | 582.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 15:00:00 | 588.55 | 581.36 | 582.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 585.00 | 582.09 | 582.91 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2024-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 09:15:00 | 596.55 | 584.98 | 584.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 11:15:00 | 601.50 | 590.04 | 586.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 12:15:00 | 613.40 | 613.94 | 606.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 13:00:00 | 613.40 | 613.94 | 606.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 609.95 | 613.44 | 607.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 15:00:00 | 609.95 | 613.44 | 607.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 607.10 | 612.17 | 607.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 09:15:00 | 614.95 | 612.17 | 607.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 10:00:00 | 612.75 | 615.75 | 612.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 13:45:00 | 613.15 | 613.34 | 612.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 15:15:00 | 612.95 | 612.93 | 612.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 15:15:00 | 612.95 | 612.94 | 612.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 09:30:00 | 616.35 | 613.35 | 612.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 10:15:00 | 611.90 | 616.18 | 616.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 10:15:00 | 611.90 | 616.18 | 616.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 11:15:00 | 609.20 | 614.79 | 615.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 11:15:00 | 597.15 | 595.29 | 599.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-16 11:30:00 | 596.75 | 595.29 | 599.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 595.90 | 593.53 | 596.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:00:00 | 595.90 | 593.53 | 596.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 599.00 | 594.63 | 596.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:00:00 | 599.00 | 594.63 | 596.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 596.95 | 595.09 | 596.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 15:15:00 | 594.50 | 595.53 | 596.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 10:15:00 | 599.55 | 596.51 | 596.83 | SL hit (close>static) qty=1.00 sl=599.10 alert=retest2 |

### Cycle 39 — BUY (started 2024-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 10:15:00 | 596.25 | 593.58 | 593.29 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 10:15:00 | 591.20 | 593.12 | 593.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 09:15:00 | 589.75 | 591.35 | 592.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 12:15:00 | 594.85 | 591.54 | 592.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 12:15:00 | 594.85 | 591.54 | 592.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 12:15:00 | 594.85 | 591.54 | 592.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 13:00:00 | 594.85 | 591.54 | 592.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2024-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 13:15:00 | 596.95 | 592.62 | 592.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 14:15:00 | 601.20 | 594.34 | 593.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 10:15:00 | 591.30 | 594.30 | 593.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 10:15:00 | 591.30 | 594.30 | 593.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 591.30 | 594.30 | 593.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 10:45:00 | 590.50 | 594.30 | 593.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 11:15:00 | 590.60 | 593.56 | 593.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 11:30:00 | 588.45 | 593.56 | 593.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2024-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 12:15:00 | 589.90 | 592.83 | 593.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 14:15:00 | 588.00 | 591.28 | 592.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 09:15:00 | 591.30 | 590.76 | 591.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 09:15:00 | 591.30 | 590.76 | 591.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 591.30 | 590.76 | 591.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:30:00 | 591.00 | 590.76 | 591.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 592.00 | 591.01 | 591.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:30:00 | 593.00 | 591.01 | 591.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 592.05 | 591.22 | 591.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:45:00 | 592.65 | 591.22 | 591.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 596.00 | 592.17 | 592.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:00:00 | 596.00 | 592.17 | 592.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 13:15:00 | 594.85 | 592.71 | 592.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 14:15:00 | 596.85 | 593.54 | 592.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 15:15:00 | 591.00 | 593.03 | 592.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 15:15:00 | 591.00 | 593.03 | 592.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 591.00 | 593.03 | 592.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:15:00 | 590.05 | 593.03 | 592.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 595.00 | 593.42 | 592.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:30:00 | 591.40 | 593.42 | 592.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 595.40 | 593.82 | 593.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:45:00 | 594.05 | 593.82 | 593.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 11:15:00 | 595.10 | 594.08 | 593.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 11:45:00 | 595.35 | 594.08 | 593.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 12:15:00 | 595.70 | 594.40 | 593.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 12:30:00 | 594.55 | 594.40 | 593.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 605.00 | 596.69 | 594.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 605.00 | 596.69 | 594.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 600.00 | 598.58 | 596.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 09:30:00 | 611.25 | 602.87 | 599.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 10:00:00 | 610.40 | 602.87 | 599.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 11:30:00 | 609.50 | 605.02 | 601.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 12:30:00 | 609.05 | 606.05 | 602.12 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 14:15:00 | 602.65 | 605.98 | 602.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 15:00:00 | 602.65 | 605.98 | 602.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 15:15:00 | 602.50 | 605.29 | 602.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 09:15:00 | 608.05 | 605.29 | 602.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-01-08 09:15:00 | 668.86 | 644.51 | 635.74 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 12:15:00 | 629.05 | 645.95 | 647.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 13:15:00 | 625.30 | 641.82 | 645.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 13:15:00 | 619.25 | 618.11 | 624.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 14:00:00 | 619.25 | 618.11 | 624.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 616.65 | 617.91 | 623.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 14:15:00 | 611.95 | 616.61 | 620.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 15:00:00 | 612.10 | 615.71 | 619.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 15:00:00 | 610.40 | 610.29 | 614.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 10:15:00 | 628.60 | 615.10 | 615.76 | SL hit (close>static) qty=1.00 sl=628.45 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 11:15:00 | 622.10 | 616.50 | 616.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-21 11:15:00 | 630.10 | 624.10 | 620.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 14:15:00 | 621.80 | 624.90 | 622.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 14:15:00 | 621.80 | 624.90 | 622.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 621.80 | 624.90 | 622.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 15:00:00 | 621.80 | 624.90 | 622.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 621.50 | 624.22 | 622.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 616.50 | 624.22 | 622.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 614.50 | 622.27 | 621.46 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 615.15 | 620.85 | 620.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 612.40 | 619.16 | 620.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 10:15:00 | 617.70 | 615.02 | 617.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 10:15:00 | 617.70 | 615.02 | 617.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 617.70 | 615.02 | 617.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:00:00 | 617.70 | 615.02 | 617.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 619.95 | 616.01 | 617.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:30:00 | 618.70 | 616.01 | 617.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 618.25 | 616.46 | 617.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:15:00 | 608.10 | 617.39 | 617.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 577.70 | 590.74 | 599.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 13:15:00 | 590.15 | 587.96 | 595.06 | SL hit (close>ema200) qty=0.50 sl=587.96 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 606.15 | 597.72 | 597.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 609.85 | 600.15 | 598.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 599.70 | 603.81 | 601.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 13:15:00 | 599.70 | 603.81 | 601.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 599.70 | 603.81 | 601.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 599.70 | 603.81 | 601.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 601.05 | 603.26 | 601.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:30:00 | 602.40 | 603.26 | 601.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 604.00 | 603.41 | 601.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:15:00 | 606.40 | 603.41 | 601.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 607.05 | 604.14 | 602.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 15:00:00 | 609.85 | 605.99 | 603.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 11:15:00 | 610.25 | 606.29 | 604.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 13:15:00 | 610.20 | 606.06 | 604.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 10:15:00 | 620.10 | 639.30 | 641.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 620.10 | 639.30 | 641.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 613.90 | 634.22 | 639.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 611.35 | 606.91 | 615.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:00:00 | 611.35 | 606.91 | 615.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 604.50 | 605.13 | 611.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 12:30:00 | 596.20 | 602.84 | 608.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:30:00 | 597.10 | 601.40 | 607.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 566.39 | 578.39 | 589.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 567.25 | 578.39 | 589.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-19 09:15:00 | 536.58 | 548.98 | 559.93 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 49 — BUY (started 2025-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 10:15:00 | 525.50 | 513.69 | 512.58 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-03-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 14:15:00 | 512.95 | 516.43 | 516.44 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-10 11:15:00 | 523.00 | 517.33 | 516.74 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 512.90 | 516.48 | 516.52 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 10:15:00 | 519.70 | 517.08 | 516.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 13:15:00 | 522.65 | 519.11 | 517.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 11:15:00 | 522.40 | 524.24 | 521.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 11:15:00 | 522.40 | 524.24 | 521.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 522.40 | 524.24 | 521.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:30:00 | 523.00 | 524.24 | 521.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 521.00 | 523.59 | 521.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:30:00 | 520.20 | 523.59 | 521.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 538.70 | 526.61 | 522.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 14:15:00 | 538.95 | 526.61 | 522.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 13:15:00 | 538.75 | 538.80 | 532.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 14:15:00 | 549.65 | 538.53 | 532.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-17 11:15:00 | 592.85 | 565.44 | 549.52 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 15:15:00 | 616.00 | 621.89 | 622.45 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 11:15:00 | 632.85 | 622.89 | 622.65 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 15:15:00 | 616.60 | 622.83 | 622.98 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 13:15:00 | 632.00 | 623.71 | 623.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 09:15:00 | 641.80 | 631.49 | 627.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 11:15:00 | 631.20 | 631.67 | 627.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-01 12:00:00 | 631.20 | 631.67 | 627.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 12:15:00 | 628.80 | 631.10 | 628.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 12:30:00 | 628.30 | 631.10 | 628.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 631.10 | 631.10 | 628.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 14:15:00 | 627.15 | 631.10 | 628.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 14:15:00 | 616.00 | 628.08 | 627.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 15:00:00 | 616.00 | 628.08 | 627.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 15:15:00 | 616.00 | 625.66 | 626.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 09:15:00 | 612.00 | 622.93 | 624.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 15:15:00 | 608.40 | 607.78 | 612.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-04 09:15:00 | 600.45 | 607.78 | 612.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 585.45 | 575.13 | 585.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 10:00:00 | 585.45 | 575.13 | 585.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 580.20 | 576.15 | 584.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 577.25 | 581.93 | 584.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:00:00 | 576.80 | 580.90 | 583.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 12:15:00 | 589.80 | 584.03 | 584.69 | SL hit (close>static) qty=1.00 sl=586.25 alert=retest2 |

### Cycle 59 — BUY (started 2025-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 13:15:00 | 596.15 | 586.45 | 585.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 605.00 | 592.32 | 588.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 13:15:00 | 669.50 | 673.85 | 661.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 14:00:00 | 669.50 | 673.85 | 661.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 684.00 | 686.97 | 678.36 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 14:15:00 | 666.95 | 674.52 | 674.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 11:15:00 | 662.15 | 669.10 | 671.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 13:15:00 | 668.75 | 667.71 | 670.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 14:00:00 | 668.75 | 667.71 | 670.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 670.20 | 668.21 | 670.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 15:00:00 | 670.20 | 668.21 | 670.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 669.00 | 668.37 | 670.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 09:15:00 | 666.25 | 668.37 | 670.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-29 10:15:00 | 676.65 | 670.19 | 670.67 | SL hit (close>static) qty=1.00 sl=673.35 alert=retest2 |

### Cycle 61 — BUY (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 11:15:00 | 680.05 | 672.16 | 671.52 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 11:15:00 | 666.35 | 671.01 | 671.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 12:15:00 | 663.75 | 669.55 | 670.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 11:15:00 | 662.80 | 662.70 | 666.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-02 11:45:00 | 662.80 | 662.70 | 666.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 669.60 | 662.24 | 664.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 669.60 | 662.24 | 664.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 663.40 | 662.47 | 664.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 12:15:00 | 658.75 | 662.55 | 664.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:15:00 | 660.05 | 661.44 | 662.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:45:00 | 658.50 | 661.07 | 662.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 13:15:00 | 659.55 | 661.01 | 662.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 663.50 | 661.14 | 662.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 15:00:00 | 663.50 | 661.14 | 662.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 662.10 | 661.34 | 662.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 09:15:00 | 649.10 | 661.34 | 662.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 625.81 | 635.19 | 642.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 627.05 | 635.19 | 642.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 625.57 | 635.19 | 642.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 626.57 | 635.19 | 642.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 616.64 | 635.19 | 642.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 13:15:00 | 633.10 | 632.91 | 639.20 | SL hit (close>ema200) qty=0.50 sl=632.91 alert=retest2 |

### Cycle 63 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 646.85 | 641.08 | 640.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 649.85 | 643.82 | 642.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 652.15 | 653.71 | 649.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 652.15 | 653.71 | 649.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 652.15 | 653.71 | 649.70 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 14:15:00 | 640.00 | 647.48 | 647.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-14 15:15:00 | 639.80 | 645.94 | 647.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-19 14:15:00 | 645.45 | 635.69 | 638.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 14:15:00 | 645.45 | 635.69 | 638.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 645.45 | 635.69 | 638.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 15:00:00 | 645.45 | 635.69 | 638.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 643.00 | 637.15 | 638.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 647.20 | 637.15 | 638.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 636.00 | 638.42 | 638.92 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 660.30 | 643.17 | 640.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 09:15:00 | 669.70 | 658.28 | 650.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 10:15:00 | 672.35 | 672.99 | 667.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 11:00:00 | 672.35 | 672.99 | 667.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 665.50 | 670.65 | 668.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 664.50 | 670.65 | 668.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 662.00 | 668.92 | 667.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:45:00 | 661.25 | 668.92 | 667.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 10:15:00 | 657.90 | 666.72 | 666.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 11:15:00 | 654.90 | 664.35 | 665.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 10:15:00 | 660.00 | 657.42 | 661.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 10:15:00 | 660.00 | 657.42 | 661.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 660.00 | 657.42 | 661.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:00:00 | 660.00 | 657.42 | 661.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 653.00 | 655.29 | 658.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 13:30:00 | 648.20 | 654.26 | 657.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 667.50 | 655.76 | 656.89 | SL hit (close>static) qty=1.00 sl=659.80 alert=retest2 |

### Cycle 67 — BUY (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 10:15:00 | 666.70 | 657.95 | 657.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 11:15:00 | 672.95 | 660.95 | 659.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 11:15:00 | 674.20 | 674.76 | 668.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 11:45:00 | 673.80 | 674.76 | 668.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 667.80 | 673.63 | 669.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 667.80 | 673.63 | 669.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 665.50 | 672.01 | 669.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 669.10 | 672.01 | 669.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 14:00:00 | 669.15 | 671.89 | 671.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 14:15:00 | 669.00 | 671.31 | 671.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 14:15:00 | 669.00 | 671.31 | 671.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 09:15:00 | 664.00 | 669.64 | 670.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 09:15:00 | 661.00 | 660.65 | 664.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 661.00 | 660.65 | 664.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 661.00 | 660.65 | 664.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 09:15:00 | 643.10 | 656.70 | 659.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 11:15:00 | 651.15 | 654.67 | 658.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 12:15:00 | 650.75 | 654.17 | 657.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 650.30 | 652.19 | 655.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 648.30 | 651.41 | 654.69 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 664.65 | 656.26 | 656.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 13:15:00 | 664.65 | 656.26 | 656.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 14:15:00 | 672.00 | 659.41 | 657.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 658.50 | 664.39 | 661.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 13:15:00 | 658.50 | 664.39 | 661.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 658.50 | 664.39 | 661.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 658.50 | 664.39 | 661.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 660.80 | 663.67 | 661.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 09:45:00 | 669.70 | 665.15 | 662.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 13:15:00 | 662.20 | 669.81 | 670.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 13:15:00 | 662.20 | 669.81 | 670.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 14:15:00 | 660.00 | 667.85 | 669.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 15:15:00 | 666.30 | 662.57 | 665.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 15:15:00 | 666.30 | 662.57 | 665.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 666.30 | 662.57 | 665.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:30:00 | 659.00 | 661.40 | 664.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-23 09:15:00 | 626.05 | 637.72 | 646.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 632.50 | 629.28 | 636.71 | SL hit (close>ema200) qty=0.50 sl=629.28 alert=retest2 |

### Cycle 71 — BUY (started 2025-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 12:15:00 | 644.35 | 636.06 | 635.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 14:15:00 | 651.60 | 640.34 | 637.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 665.70 | 667.06 | 658.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 13:00:00 | 665.70 | 667.06 | 658.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 667.30 | 670.84 | 666.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:00:00 | 667.30 | 670.84 | 666.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 662.15 | 669.10 | 666.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 662.15 | 669.10 | 666.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 668.55 | 668.99 | 666.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 13:00:00 | 671.95 | 669.58 | 667.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 13:30:00 | 670.75 | 674.24 | 671.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 10:00:00 | 671.15 | 672.87 | 671.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 12:15:00 | 667.10 | 670.56 | 670.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 12:15:00 | 667.10 | 670.56 | 670.93 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 14:15:00 | 674.90 | 671.37 | 671.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 09:15:00 | 683.40 | 673.72 | 672.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 09:15:00 | 675.85 | 680.31 | 677.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 675.85 | 680.31 | 677.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 675.85 | 680.31 | 677.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 678.05 | 680.31 | 677.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 674.60 | 679.17 | 676.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 12:30:00 | 676.95 | 678.15 | 676.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 13:15:00 | 676.90 | 678.15 | 676.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 14:00:00 | 682.20 | 678.96 | 677.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-16 13:15:00 | 744.65 | 730.30 | 721.36 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 10:15:00 | 754.20 | 765.89 | 767.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 12:15:00 | 751.05 | 760.81 | 764.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 09:15:00 | 756.85 | 755.57 | 760.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 09:15:00 | 756.85 | 755.57 | 760.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 756.85 | 755.57 | 760.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 11:00:00 | 747.90 | 752.04 | 755.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 12:15:00 | 760.75 | 756.31 | 755.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 12:15:00 | 760.75 | 756.31 | 755.91 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 13:15:00 | 750.50 | 755.19 | 755.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 14:15:00 | 748.40 | 753.83 | 755.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 09:15:00 | 756.60 | 745.69 | 748.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 756.60 | 745.69 | 748.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 756.60 | 745.69 | 748.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:15:00 | 757.20 | 745.69 | 748.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 744.90 | 745.53 | 747.98 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2025-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 13:15:00 | 763.60 | 751.45 | 750.24 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 734.90 | 747.02 | 748.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 09:15:00 | 711.50 | 736.09 | 741.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 11:15:00 | 721.80 | 716.43 | 725.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 11:45:00 | 721.50 | 716.43 | 725.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 726.30 | 719.50 | 724.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 15:00:00 | 726.30 | 719.50 | 724.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 726.00 | 720.80 | 724.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:15:00 | 721.35 | 720.80 | 724.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 727.00 | 722.04 | 724.92 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 738.45 | 728.76 | 727.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 13:15:00 | 742.45 | 731.50 | 728.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 732.70 | 733.39 | 730.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 10:15:00 | 731.50 | 733.39 | 730.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 729.40 | 732.60 | 730.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:00:00 | 729.40 | 732.60 | 730.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 731.75 | 732.43 | 730.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 10:00:00 | 736.00 | 732.25 | 731.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 12:00:00 | 737.60 | 732.88 | 731.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 11:15:00 | 745.25 | 756.96 | 757.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 11:15:00 | 745.25 | 756.96 | 757.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 742.55 | 754.08 | 756.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 12:15:00 | 744.85 | 741.01 | 744.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 12:15:00 | 744.85 | 741.01 | 744.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 744.85 | 741.01 | 744.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:00:00 | 744.85 | 741.01 | 744.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 756.05 | 744.02 | 745.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:30:00 | 757.50 | 744.02 | 745.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 756.65 | 746.55 | 746.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 14:45:00 | 756.60 | 746.55 | 746.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 15:15:00 | 756.15 | 748.47 | 747.64 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 09:15:00 | 739.85 | 747.02 | 747.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 13:15:00 | 737.65 | 742.93 | 745.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 736.25 | 731.75 | 736.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 736.25 | 731.75 | 736.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 736.25 | 731.75 | 736.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 736.25 | 731.75 | 736.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 732.00 | 731.80 | 736.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:30:00 | 729.00 | 731.44 | 735.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 13:00:00 | 730.25 | 731.20 | 735.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 13:30:00 | 729.55 | 731.52 | 734.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 14:15:00 | 727.80 | 731.52 | 734.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 730.65 | 730.10 | 733.25 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-02 15:15:00 | 741.90 | 734.81 | 734.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 15:15:00 | 741.90 | 734.81 | 734.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 742.60 | 736.36 | 735.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 13:15:00 | 737.20 | 738.48 | 736.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 13:15:00 | 737.20 | 738.48 | 736.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 737.20 | 738.48 | 736.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:45:00 | 734.60 | 738.48 | 736.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 739.40 | 738.66 | 737.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 15:00:00 | 739.40 | 738.66 | 737.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 741.05 | 743.11 | 740.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 741.05 | 743.11 | 740.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 737.00 | 741.89 | 740.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:45:00 | 738.40 | 741.89 | 740.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 736.15 | 740.74 | 739.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:45:00 | 741.20 | 741.91 | 740.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 11:00:00 | 740.50 | 741.63 | 740.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 12:30:00 | 740.85 | 740.80 | 740.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:00:00 | 740.20 | 740.80 | 740.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 741.55 | 740.95 | 740.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:30:00 | 741.10 | 740.95 | 740.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 743.00 | 741.36 | 740.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:15:00 | 740.05 | 741.36 | 740.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 740.05 | 741.10 | 740.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 737.80 | 741.10 | 740.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 742.20 | 741.32 | 740.66 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-08 11:15:00 | 733.20 | 738.89 | 739.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 11:15:00 | 733.20 | 738.89 | 739.62 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 10:15:00 | 741.15 | 739.53 | 739.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 14:15:00 | 747.60 | 741.94 | 740.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 12:15:00 | 760.55 | 762.13 | 757.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 13:00:00 | 760.55 | 762.13 | 757.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 758.10 | 761.32 | 757.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:15:00 | 758.05 | 761.32 | 757.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 756.95 | 760.45 | 757.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 15:00:00 | 756.95 | 760.45 | 757.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 760.00 | 760.36 | 757.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 748.35 | 760.36 | 757.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 753.95 | 759.08 | 757.54 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 12:15:00 | 754.10 | 756.19 | 756.44 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 15:15:00 | 758.15 | 756.25 | 756.11 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 09:15:00 | 754.30 | 755.86 | 755.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 11:15:00 | 746.25 | 753.68 | 754.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 09:15:00 | 741.95 | 740.59 | 744.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 741.95 | 740.59 | 744.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 741.95 | 740.59 | 744.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:45:00 | 739.00 | 740.59 | 744.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 743.15 | 741.10 | 744.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:30:00 | 744.55 | 741.10 | 744.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 745.15 | 742.17 | 744.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:45:00 | 744.20 | 742.17 | 744.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 746.60 | 743.05 | 744.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:45:00 | 746.10 | 743.05 | 744.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 745.00 | 743.44 | 744.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:30:00 | 745.90 | 743.44 | 744.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 745.00 | 743.75 | 744.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:15:00 | 741.65 | 743.75 | 744.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 742.80 | 743.56 | 744.60 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 11:15:00 | 747.55 | 745.61 | 745.43 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 12:15:00 | 739.15 | 744.32 | 744.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 15:15:00 | 737.10 | 741.59 | 743.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 15:15:00 | 736.25 | 735.92 | 738.95 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 09:15:00 | 729.45 | 735.92 | 738.95 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 736.10 | 730.56 | 733.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 736.10 | 730.56 | 733.76 | SL hit (close>ema400) qty=1.00 sl=733.76 alert=retest1 |

### Cycle 91 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 716.45 | 697.46 | 697.18 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 705.25 | 707.26 | 707.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 10:15:00 | 704.30 | 706.61 | 707.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 11:15:00 | 697.05 | 696.51 | 700.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 12:00:00 | 697.05 | 696.51 | 700.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 700.15 | 697.48 | 700.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:15:00 | 700.50 | 697.48 | 700.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 700.15 | 698.01 | 700.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:15:00 | 701.45 | 698.01 | 700.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 701.45 | 698.70 | 700.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 697.10 | 698.70 | 700.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 10:15:00 | 698.50 | 699.23 | 700.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 10:15:00 | 704.65 | 700.31 | 700.83 | SL hit (close>static) qty=1.00 sl=703.55 alert=retest2 |

### Cycle 93 — BUY (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 11:15:00 | 711.80 | 702.61 | 701.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 13:15:00 | 715.30 | 706.96 | 704.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 11:15:00 | 708.30 | 710.79 | 707.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 11:15:00 | 708.30 | 710.79 | 707.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 708.30 | 710.79 | 707.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:45:00 | 706.55 | 710.79 | 707.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 712.40 | 711.11 | 707.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 14:30:00 | 714.90 | 711.24 | 708.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 714.50 | 711.05 | 708.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 14:15:00 | 717.50 | 724.84 | 725.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 14:15:00 | 717.50 | 724.84 | 725.18 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 727.00 | 725.33 | 725.31 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 719.70 | 724.21 | 724.80 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 726.70 | 722.13 | 721.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 13:15:00 | 730.15 | 725.29 | 723.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 725.65 | 727.62 | 725.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 725.65 | 727.62 | 725.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 725.65 | 727.62 | 725.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 725.65 | 727.62 | 725.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 730.00 | 728.10 | 725.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:30:00 | 733.60 | 727.61 | 726.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 10:15:00 | 725.20 | 727.13 | 726.17 | SL hit (close<static) qty=1.00 sl=725.25 alert=retest2 |

### Cycle 98 — SELL (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 12:15:00 | 720.45 | 725.35 | 725.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 720.00 | 723.80 | 724.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 726.10 | 723.78 | 724.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 726.10 | 723.78 | 724.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 726.10 | 723.78 | 724.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:45:00 | 727.00 | 723.78 | 724.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 723.30 | 723.68 | 724.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:30:00 | 724.45 | 723.68 | 724.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 720.05 | 722.96 | 724.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:30:00 | 723.40 | 722.96 | 724.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 724.25 | 723.22 | 724.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:00:00 | 724.25 | 723.22 | 724.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 729.25 | 724.42 | 724.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:45:00 | 730.35 | 724.42 | 724.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 726.95 | 724.93 | 724.74 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 12:15:00 | 721.65 | 724.26 | 724.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 13:15:00 | 718.20 | 723.05 | 724.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 718.30 | 715.15 | 718.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 13:00:00 | 718.30 | 715.15 | 718.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 720.50 | 716.22 | 719.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 14:00:00 | 720.50 | 716.22 | 719.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 720.10 | 717.00 | 719.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 14:30:00 | 720.90 | 717.00 | 719.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 720.90 | 717.78 | 719.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:15:00 | 714.60 | 717.78 | 719.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 10:15:00 | 729.65 | 721.09 | 720.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 11:15:00 | 731.65 | 723.20 | 721.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 15:15:00 | 725.10 | 726.00 | 723.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 09:15:00 | 717.65 | 726.00 | 723.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 708.15 | 722.43 | 722.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:00:00 | 708.15 | 722.43 | 722.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 10:15:00 | 704.50 | 718.85 | 720.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 11:15:00 | 698.80 | 714.84 | 718.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 683.00 | 678.51 | 690.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 10:00:00 | 683.00 | 678.51 | 690.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 689.00 | 683.58 | 688.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 684.00 | 683.58 | 688.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 14:15:00 | 671.85 | 666.69 | 666.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 14:15:00 | 671.85 | 666.69 | 666.54 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 12:15:00 | 664.70 | 666.94 | 666.94 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 13:15:00 | 670.80 | 667.71 | 667.29 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 10:15:00 | 664.85 | 667.05 | 667.18 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 11:15:00 | 669.10 | 667.46 | 667.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 14:15:00 | 681.20 | 670.10 | 668.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 09:15:00 | 665.15 | 670.69 | 669.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 665.15 | 670.69 | 669.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 665.15 | 670.69 | 669.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 665.15 | 670.69 | 669.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 670.00 | 670.56 | 669.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:30:00 | 665.10 | 670.56 | 669.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 678.25 | 672.09 | 670.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 13:45:00 | 679.10 | 673.56 | 671.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 14:45:00 | 679.60 | 676.32 | 673.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 10:15:00 | 695.45 | 699.93 | 700.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 695.45 | 699.93 | 700.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 686.60 | 696.95 | 698.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 681.55 | 679.26 | 685.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 09:15:00 | 681.55 | 679.26 | 685.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 681.55 | 679.26 | 685.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 14:15:00 | 668.00 | 677.06 | 682.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 667.20 | 665.74 | 665.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 09:15:00 | 667.20 | 665.74 | 665.69 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 662.40 | 665.04 | 665.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 660.80 | 664.19 | 664.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 640.95 | 638.87 | 643.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 13:15:00 | 640.95 | 638.87 | 643.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 640.95 | 638.87 | 643.80 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2025-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 12:15:00 | 649.30 | 645.53 | 645.26 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 13:15:00 | 642.00 | 644.82 | 644.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-22 15:15:00 | 640.50 | 643.70 | 644.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 15:15:00 | 642.00 | 641.73 | 642.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:15:00 | 638.75 | 641.73 | 642.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 634.10 | 640.21 | 642.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 10:15:00 | 633.25 | 640.21 | 642.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 13:30:00 | 632.70 | 636.79 | 639.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 11:15:00 | 601.59 | 612.25 | 620.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 11:15:00 | 601.07 | 612.25 | 620.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 609.60 | 607.28 | 614.21 | SL hit (close>ema200) qty=0.50 sl=607.28 alert=retest2 |

### Cycle 113 — BUY (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 13:15:00 | 615.95 | 613.06 | 612.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 627.80 | 616.79 | 614.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 642.55 | 648.36 | 639.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 11:00:00 | 642.55 | 648.36 | 639.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 635.00 | 645.69 | 638.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:30:00 | 634.95 | 645.69 | 638.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 638.85 | 644.32 | 638.85 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 09:15:00 | 627.85 | 635.17 | 635.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 621.80 | 626.84 | 630.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 10:15:00 | 620.55 | 616.58 | 621.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 10:45:00 | 620.80 | 616.58 | 621.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 621.70 | 617.61 | 621.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:45:00 | 622.50 | 617.61 | 621.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 620.25 | 618.13 | 621.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 09:15:00 | 608.00 | 618.97 | 621.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:30:00 | 615.60 | 614.13 | 616.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:00:00 | 615.00 | 614.13 | 616.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:45:00 | 615.40 | 614.43 | 616.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 620.45 | 615.63 | 616.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:00:00 | 620.45 | 615.63 | 616.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 620.00 | 616.51 | 617.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:45:00 | 621.00 | 616.51 | 617.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-13 14:15:00 | 622.20 | 617.92 | 617.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2026-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 14:15:00 | 622.20 | 617.92 | 617.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 09:15:00 | 625.25 | 620.07 | 618.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 14:15:00 | 623.10 | 624.43 | 621.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-14 15:00:00 | 623.10 | 624.43 | 621.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 626.40 | 624.83 | 622.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:30:00 | 622.10 | 624.86 | 622.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 622.25 | 624.34 | 622.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:00:00 | 622.25 | 624.34 | 622.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 618.65 | 623.20 | 622.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:45:00 | 617.85 | 623.20 | 622.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 616.25 | 621.81 | 621.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:00:00 | 616.25 | 621.81 | 621.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2026-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 13:15:00 | 613.75 | 620.20 | 620.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 15:15:00 | 612.95 | 617.90 | 619.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-20 12:15:00 | 612.20 | 611.92 | 614.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-20 13:00:00 | 612.20 | 611.92 | 614.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 617.05 | 612.73 | 614.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 15:00:00 | 617.05 | 612.73 | 614.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 615.05 | 613.20 | 614.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 09:15:00 | 603.10 | 613.20 | 614.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 13:15:00 | 611.95 | 602.92 | 602.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2026-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 13:15:00 | 611.95 | 602.92 | 602.01 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 13:15:00 | 602.00 | 604.58 | 604.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 15:15:00 | 601.05 | 603.80 | 604.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 13:15:00 | 602.75 | 601.92 | 603.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 13:15:00 | 602.75 | 601.92 | 603.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 602.75 | 601.92 | 603.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:45:00 | 602.75 | 601.92 | 603.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 606.00 | 602.73 | 603.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 15:00:00 | 606.00 | 602.73 | 603.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 601.00 | 602.39 | 603.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 598.80 | 602.39 | 603.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 11:45:00 | 598.50 | 601.09 | 602.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 13:15:00 | 598.30 | 600.69 | 601.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 14:45:00 | 596.20 | 599.52 | 601.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 595.95 | 597.49 | 599.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:00:00 | 595.95 | 597.49 | 599.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 596.70 | 593.29 | 596.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 596.70 | 593.29 | 596.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 590.00 | 592.63 | 595.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 613.55 | 592.63 | 595.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 608.75 | 595.85 | 597.04 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 608.75 | 595.85 | 597.04 | SL hit (close>static) qty=1.00 sl=608.45 alert=retest2 |

### Cycle 119 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 610.50 | 598.78 | 598.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 621.15 | 607.03 | 602.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 605.10 | 609.12 | 605.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 10:15:00 | 605.10 | 609.12 | 605.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 605.10 | 609.12 | 605.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:00:00 | 605.10 | 609.12 | 605.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 607.05 | 608.71 | 605.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:30:00 | 606.20 | 608.71 | 605.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 605.50 | 608.07 | 605.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:00:00 | 605.50 | 608.07 | 605.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 608.20 | 608.09 | 605.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 15:00:00 | 609.70 | 608.42 | 606.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 11:15:00 | 603.35 | 606.69 | 605.98 | SL hit (close<static) qty=1.00 sl=604.85 alert=retest2 |

### Cycle 120 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 598.45 | 604.84 | 605.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 11:15:00 | 596.90 | 602.21 | 604.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 604.80 | 599.63 | 601.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 604.80 | 599.63 | 601.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 604.80 | 599.63 | 601.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 604.80 | 599.63 | 601.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 630.35 | 605.78 | 604.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 646.50 | 613.92 | 608.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 14:15:00 | 648.30 | 651.92 | 637.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 14:30:00 | 648.85 | 651.92 | 637.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 15:15:00 | 678.00 | 691.45 | 684.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 10:00:00 | 698.10 | 688.55 | 685.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:30:00 | 697.90 | 702.65 | 701.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 14:15:00 | 695.00 | 699.60 | 700.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 14:15:00 | 695.00 | 699.60 | 700.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 15:15:00 | 688.00 | 697.28 | 699.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 710.70 | 699.96 | 700.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 710.70 | 699.96 | 700.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 710.70 | 699.96 | 700.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:00:00 | 710.70 | 699.96 | 700.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 720.95 | 704.16 | 702.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 09:15:00 | 725.45 | 712.98 | 707.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 09:15:00 | 721.60 | 723.44 | 716.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 10:00:00 | 721.60 | 723.44 | 716.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 744.10 | 739.57 | 731.90 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 721.25 | 732.67 | 733.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 13:15:00 | 716.85 | 727.53 | 730.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 15:15:00 | 707.00 | 703.72 | 709.44 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 09:15:00 | 699.25 | 703.72 | 709.44 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 09:45:00 | 700.60 | 703.01 | 708.59 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 721.85 | 702.44 | 705.53 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-06 14:15:00 | 721.85 | 702.44 | 705.53 | SL hit (close>ema400) qty=1.00 sl=705.53 alert=retest1 |

### Cycle 125 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 673.60 | 656.42 | 654.40 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 650.70 | 658.98 | 659.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 09:15:00 | 643.05 | 654.49 | 657.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 14:15:00 | 649.50 | 647.59 | 652.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 15:00:00 | 649.50 | 647.59 | 652.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 15:15:00 | 626.00 | 630.86 | 638.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:15:00 | 645.50 | 630.86 | 638.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 630.45 | 630.78 | 638.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 628.40 | 630.67 | 637.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 14:15:00 | 649.85 | 638.05 | 637.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2026-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 14:15:00 | 649.85 | 638.05 | 637.31 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 634.45 | 637.22 | 637.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 632.75 | 635.74 | 636.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 632.25 | 624.84 | 628.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 632.25 | 624.84 | 628.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 632.25 | 624.84 | 628.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 629.70 | 624.84 | 628.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 627.45 | 625.36 | 628.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 12:15:00 | 624.15 | 626.54 | 628.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:45:00 | 624.65 | 627.11 | 628.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 619.25 | 626.83 | 628.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 617.30 | 625.15 | 626.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 617.20 | 623.56 | 625.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-07 10:15:00 | 631.80 | 624.87 | 624.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 10:15:00 | 631.80 | 624.87 | 624.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 11:15:00 | 635.45 | 626.99 | 625.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 661.00 | 661.30 | 654.37 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 666.65 | 661.30 | 654.37 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:45:00 | 664.65 | 661.88 | 655.84 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 12:00:00 | 665.00 | 662.50 | 656.68 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 650.65 | 661.66 | 658.78 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 650.65 | 661.66 | 658.78 | SL hit (close<ema400) qty=1.00 sl=658.78 alert=retest1 |

### Cycle 130 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 651.45 | 674.75 | 676.78 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 674.65 | 661.89 | 661.39 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 15:15:00 | 659.90 | 664.26 | 664.44 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 667.00 | 664.84 | 664.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 669.20 | 665.71 | 665.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 665.00 | 666.46 | 665.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 665.00 | 666.46 | 665.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 665.00 | 666.46 | 665.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 665.00 | 666.46 | 665.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 665.05 | 666.18 | 665.54 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 659.55 | 664.83 | 665.03 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 666.00 | 665.16 | 665.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 673.25 | 666.78 | 665.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 12:15:00 | 675.80 | 676.30 | 672.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 12:15:00 | 675.80 | 676.30 | 672.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 675.80 | 676.30 | 672.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:45:00 | 673.75 | 676.30 | 672.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 679.45 | 676.93 | 673.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 12:45:00 | 682.45 | 678.21 | 675.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-15 15:15:00 | 389.60 | 2024-05-27 09:15:00 | 370.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-18 12:15:00 | 390.00 | 2024-05-27 09:15:00 | 370.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-15 15:15:00 | 389.60 | 2024-05-28 09:15:00 | 373.99 | STOP_HIT | 0.50 | 4.01% |
| SELL | retest2 | 2024-05-18 12:15:00 | 390.00 | 2024-05-28 09:15:00 | 373.99 | STOP_HIT | 0.50 | 4.11% |
| BUY | retest2 | 2024-06-07 09:15:00 | 381.73 | 2024-06-10 14:15:00 | 374.00 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2024-06-10 12:15:00 | 378.62 | 2024-06-10 14:15:00 | 374.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-07-18 11:15:00 | 426.80 | 2024-07-25 09:15:00 | 425.93 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2024-07-18 13:30:00 | 426.84 | 2024-07-25 09:15:00 | 425.93 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2024-07-19 14:00:00 | 426.80 | 2024-07-25 09:15:00 | 425.93 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2024-07-26 13:00:00 | 430.62 | 2024-07-31 15:15:00 | 430.20 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2024-07-30 09:15:00 | 429.96 | 2024-07-31 15:15:00 | 430.20 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2024-07-31 13:15:00 | 429.22 | 2024-07-31 15:15:00 | 430.20 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2024-08-14 11:15:00 | 453.99 | 2024-08-21 10:15:00 | 495.33 | TARGET_HIT | 1.00 | 9.11% |
| BUY | retest2 | 2024-08-14 13:30:00 | 450.30 | 2024-08-21 10:15:00 | 495.10 | TARGET_HIT | 1.00 | 9.95% |
| BUY | retest2 | 2024-08-14 15:15:00 | 450.09 | 2024-08-23 09:15:00 | 499.39 | TARGET_HIT | 1.00 | 10.95% |
| BUY | retest2 | 2024-09-06 12:00:00 | 530.21 | 2024-09-10 09:15:00 | 520.47 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-09-06 13:45:00 | 530.28 | 2024-09-10 09:15:00 | 520.47 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-09-06 15:15:00 | 535.75 | 2024-09-10 09:15:00 | 520.47 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2024-09-13 09:15:00 | 562.65 | 2024-09-18 11:15:00 | 543.35 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest2 | 2024-09-16 12:00:00 | 553.25 | 2024-09-18 11:15:00 | 543.35 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-09-19 13:45:00 | 548.20 | 2024-09-24 14:15:00 | 553.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-09-20 12:45:00 | 549.50 | 2024-09-24 14:15:00 | 553.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-09-20 13:30:00 | 549.20 | 2024-09-24 14:15:00 | 553.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-09-20 15:00:00 | 546.20 | 2024-09-24 14:15:00 | 553.00 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-09-26 09:15:00 | 541.10 | 2024-09-27 15:15:00 | 546.95 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-09-26 10:45:00 | 542.05 | 2024-09-27 15:15:00 | 546.95 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-10-08 12:15:00 | 540.90 | 2024-10-11 09:15:00 | 548.25 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-10-09 13:30:00 | 538.85 | 2024-10-11 09:15:00 | 548.25 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-10-15 09:15:00 | 559.00 | 2024-10-17 11:15:00 | 546.30 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2024-10-21 14:45:00 | 540.05 | 2024-10-28 09:15:00 | 513.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 09:15:00 | 538.00 | 2024-10-28 09:15:00 | 511.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 14:45:00 | 540.05 | 2024-10-28 12:15:00 | 517.15 | STOP_HIT | 0.50 | 4.24% |
| SELL | retest2 | 2024-10-22 09:15:00 | 538.00 | 2024-10-28 12:15:00 | 517.15 | STOP_HIT | 0.50 | 3.88% |
| BUY | retest2 | 2024-11-06 09:15:00 | 563.75 | 2024-11-12 13:15:00 | 564.55 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2024-11-07 09:30:00 | 564.30 | 2024-11-12 13:15:00 | 564.55 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2024-11-07 14:00:00 | 562.70 | 2024-11-12 13:15:00 | 564.55 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2024-11-07 14:45:00 | 563.15 | 2024-11-12 13:15:00 | 564.55 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2024-11-08 10:45:00 | 569.45 | 2024-11-12 13:15:00 | 564.55 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-11-08 12:15:00 | 568.05 | 2024-11-12 13:15:00 | 564.55 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2024-11-08 13:00:00 | 568.15 | 2024-11-12 13:15:00 | 564.55 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-11-12 09:15:00 | 571.90 | 2024-11-12 13:15:00 | 564.55 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-11-18 10:15:00 | 554.40 | 2024-11-19 09:15:00 | 569.90 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest1 | 2024-11-25 09:15:00 | 604.50 | 2024-11-27 09:15:00 | 587.55 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest1 | 2024-11-25 15:15:00 | 599.85 | 2024-11-27 09:15:00 | 587.55 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2024-12-05 09:15:00 | 614.95 | 2024-12-11 10:15:00 | 611.90 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-12-06 10:00:00 | 612.75 | 2024-12-11 10:15:00 | 611.90 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2024-12-06 13:45:00 | 613.15 | 2024-12-11 10:15:00 | 611.90 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2024-12-06 15:15:00 | 612.95 | 2024-12-11 10:15:00 | 611.90 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2024-12-09 09:30:00 | 616.35 | 2024-12-11 10:15:00 | 611.90 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-12-17 15:15:00 | 594.50 | 2024-12-18 10:15:00 | 599.55 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-12-18 12:00:00 | 594.50 | 2024-12-20 10:15:00 | 596.25 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2024-12-19 14:15:00 | 594.35 | 2024-12-20 10:15:00 | 596.25 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2024-12-20 09:15:00 | 594.00 | 2024-12-20 10:15:00 | 596.25 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-01-01 09:30:00 | 611.25 | 2025-01-08 09:15:00 | 668.86 | TARGET_HIT | 1.00 | 9.42% |
| BUY | retest2 | 2025-01-01 10:00:00 | 610.40 | 2025-01-13 09:15:00 | 672.38 | TARGET_HIT | 1.00 | 10.15% |
| BUY | retest2 | 2025-01-01 11:30:00 | 609.50 | 2025-01-13 09:15:00 | 671.44 | TARGET_HIT | 1.00 | 10.16% |
| BUY | retest2 | 2025-01-01 12:30:00 | 609.05 | 2025-01-13 09:15:00 | 670.45 | TARGET_HIT | 1.00 | 10.08% |
| BUY | retest2 | 2025-01-02 09:15:00 | 608.05 | 2025-01-13 09:15:00 | 669.96 | TARGET_HIT | 1.00 | 10.18% |
| SELL | retest2 | 2025-01-16 14:15:00 | 611.95 | 2025-01-20 10:15:00 | 628.60 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2025-01-16 15:00:00 | 612.10 | 2025-01-20 10:15:00 | 628.60 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-01-17 15:00:00 | 610.40 | 2025-01-20 10:15:00 | 628.60 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2025-01-24 09:15:00 | 608.10 | 2025-01-28 09:15:00 | 577.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:15:00 | 608.10 | 2025-01-28 13:15:00 | 590.15 | STOP_HIT | 0.50 | 2.95% |
| BUY | retest2 | 2025-01-31 15:00:00 | 609.85 | 2025-02-10 10:15:00 | 620.10 | STOP_HIT | 1.00 | 1.68% |
| BUY | retest2 | 2025-02-01 11:15:00 | 610.25 | 2025-02-10 10:15:00 | 620.10 | STOP_HIT | 1.00 | 1.61% |
| BUY | retest2 | 2025-02-01 13:15:00 | 610.20 | 2025-02-10 10:15:00 | 620.10 | STOP_HIT | 1.00 | 1.62% |
| SELL | retest2 | 2025-02-13 12:30:00 | 596.20 | 2025-02-17 09:15:00 | 566.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:30:00 | 597.10 | 2025-02-17 09:15:00 | 567.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 12:30:00 | 596.20 | 2025-02-19 09:15:00 | 536.58 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-13 13:30:00 | 597.10 | 2025-02-19 09:15:00 | 537.39 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-03-12 14:15:00 | 538.95 | 2025-03-17 11:15:00 | 592.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-13 13:15:00 | 538.75 | 2025-03-17 11:15:00 | 592.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-13 14:15:00 | 549.65 | 2025-03-21 14:15:00 | 604.62 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-09 09:15:00 | 577.25 | 2025-04-09 12:15:00 | 589.80 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-04-09 10:00:00 | 576.80 | 2025-04-09 12:15:00 | 589.80 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-04-29 09:15:00 | 666.25 | 2025-04-29 10:15:00 | 676.65 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-05-05 12:15:00 | 658.75 | 2025-05-09 09:15:00 | 625.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 09:15:00 | 660.05 | 2025-05-09 09:15:00 | 627.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 09:45:00 | 658.50 | 2025-05-09 09:15:00 | 625.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 13:15:00 | 659.55 | 2025-05-09 09:15:00 | 626.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-07 09:15:00 | 649.10 | 2025-05-09 09:15:00 | 616.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-05 12:15:00 | 658.75 | 2025-05-09 13:15:00 | 633.10 | STOP_HIT | 0.50 | 3.89% |
| SELL | retest2 | 2025-05-06 09:15:00 | 660.05 | 2025-05-09 13:15:00 | 633.10 | STOP_HIT | 0.50 | 4.08% |
| SELL | retest2 | 2025-05-06 09:45:00 | 658.50 | 2025-05-09 13:15:00 | 633.10 | STOP_HIT | 0.50 | 3.86% |
| SELL | retest2 | 2025-05-06 13:15:00 | 659.55 | 2025-05-09 13:15:00 | 633.10 | STOP_HIT | 0.50 | 4.01% |
| SELL | retest2 | 2025-05-07 09:15:00 | 649.10 | 2025-05-09 13:15:00 | 633.10 | STOP_HIT | 0.50 | 2.46% |
| SELL | retest2 | 2025-05-29 13:30:00 | 648.20 | 2025-05-30 09:15:00 | 667.50 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2025-06-03 09:15:00 | 669.10 | 2025-06-04 14:15:00 | 669.00 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2025-06-04 14:00:00 | 669.15 | 2025-06-04 14:15:00 | 669.00 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-06-10 09:15:00 | 643.10 | 2025-06-11 13:15:00 | 664.65 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2025-06-10 11:15:00 | 651.15 | 2025-06-11 13:15:00 | 664.65 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-06-10 12:15:00 | 650.75 | 2025-06-11 13:15:00 | 664.65 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-06-11 09:15:00 | 650.30 | 2025-06-11 13:15:00 | 664.65 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-06-13 09:45:00 | 669.70 | 2025-06-17 13:15:00 | 662.20 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-06-19 09:30:00 | 659.00 | 2025-06-23 09:15:00 | 626.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-19 09:30:00 | 659.00 | 2025-06-24 09:15:00 | 632.50 | STOP_HIT | 0.50 | 4.02% |
| BUY | retest2 | 2025-07-01 13:00:00 | 671.95 | 2025-07-03 12:15:00 | 667.10 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-07-02 13:30:00 | 670.75 | 2025-07-03 12:15:00 | 667.10 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-07-03 10:00:00 | 671.15 | 2025-07-03 12:15:00 | 667.10 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-07-07 12:30:00 | 676.95 | 2025-07-16 13:15:00 | 744.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-07 13:15:00 | 676.90 | 2025-07-16 13:15:00 | 744.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-07 14:00:00 | 682.20 | 2025-07-18 09:15:00 | 750.42 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-31 11:00:00 | 747.90 | 2025-08-01 12:15:00 | 760.75 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-08-14 10:00:00 | 736.00 | 2025-08-21 11:15:00 | 745.25 | STOP_HIT | 1.00 | 1.26% |
| BUY | retest2 | 2025-08-14 12:00:00 | 737.60 | 2025-08-21 11:15:00 | 745.25 | STOP_HIT | 1.00 | 1.04% |
| SELL | retest2 | 2025-09-01 11:30:00 | 729.00 | 2025-09-02 15:15:00 | 741.90 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-09-01 13:00:00 | 730.25 | 2025-09-02 15:15:00 | 741.90 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-09-01 13:30:00 | 729.55 | 2025-09-02 15:15:00 | 741.90 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-09-01 14:15:00 | 727.80 | 2025-09-02 15:15:00 | 741.90 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-09-05 09:45:00 | 741.20 | 2025-09-08 11:15:00 | 733.20 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-09-05 11:00:00 | 740.50 | 2025-09-08 11:15:00 | 733.20 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-09-05 12:30:00 | 740.85 | 2025-09-08 11:15:00 | 733.20 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-09-05 13:00:00 | 740.20 | 2025-09-08 11:15:00 | 733.20 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest1 | 2025-09-24 09:15:00 | 729.45 | 2025-09-25 09:15:00 | 736.10 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-09-25 15:15:00 | 728.10 | 2025-10-01 09:15:00 | 691.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-29 10:00:00 | 730.20 | 2025-10-01 09:15:00 | 693.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 15:15:00 | 728.10 | 2025-10-03 13:15:00 | 691.35 | STOP_HIT | 0.50 | 5.05% |
| SELL | retest2 | 2025-09-29 10:00:00 | 730.20 | 2025-10-03 13:15:00 | 691.35 | STOP_HIT | 0.50 | 5.32% |
| SELL | retest2 | 2025-10-13 09:15:00 | 697.10 | 2025-10-13 10:15:00 | 704.65 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-10-13 10:15:00 | 698.50 | 2025-10-13 10:15:00 | 704.65 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-10-14 14:30:00 | 714.90 | 2025-10-20 14:15:00 | 717.50 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2025-10-15 09:15:00 | 714.50 | 2025-10-20 14:15:00 | 717.50 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2025-10-31 09:30:00 | 733.60 | 2025-10-31 10:15:00 | 725.20 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-11-13 09:15:00 | 684.00 | 2025-11-20 14:15:00 | 671.85 | STOP_HIT | 1.00 | 1.78% |
| BUY | retest2 | 2025-11-25 13:45:00 | 679.10 | 2025-12-08 10:15:00 | 695.45 | STOP_HIT | 1.00 | 2.41% |
| BUY | retest2 | 2025-11-26 14:45:00 | 679.60 | 2025-12-08 10:15:00 | 695.45 | STOP_HIT | 1.00 | 2.33% |
| SELL | retest2 | 2025-12-10 14:15:00 | 668.00 | 2025-12-16 09:15:00 | 667.20 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2025-12-24 10:15:00 | 633.25 | 2025-12-30 11:15:00 | 601.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 13:30:00 | 632.70 | 2025-12-30 11:15:00 | 601.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 10:15:00 | 633.25 | 2025-12-31 09:15:00 | 609.60 | STOP_HIT | 0.50 | 3.73% |
| SELL | retest2 | 2025-12-24 13:30:00 | 632.70 | 2025-12-31 09:15:00 | 609.60 | STOP_HIT | 0.50 | 3.65% |
| SELL | retest2 | 2026-01-12 09:15:00 | 608.00 | 2026-01-13 14:15:00 | 622.20 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2026-01-13 09:30:00 | 615.60 | 2026-01-13 14:15:00 | 622.20 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-01-13 10:00:00 | 615.00 | 2026-01-13 14:15:00 | 622.20 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2026-01-13 10:45:00 | 615.40 | 2026-01-13 14:15:00 | 622.20 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-01-21 09:15:00 | 603.10 | 2026-01-27 13:15:00 | 611.95 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-02-01 09:15:00 | 598.80 | 2026-02-03 09:15:00 | 608.75 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-02-01 11:45:00 | 598.50 | 2026-02-03 09:15:00 | 608.75 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-02-01 13:15:00 | 598.30 | 2026-02-03 09:15:00 | 608.75 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2026-02-01 14:45:00 | 596.20 | 2026-02-03 09:15:00 | 608.75 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2026-02-04 15:00:00 | 609.70 | 2026-02-05 11:15:00 | 603.35 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2026-02-17 10:00:00 | 698.10 | 2026-02-20 14:15:00 | 695.00 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2026-02-20 09:30:00 | 697.90 | 2026-02-20 14:15:00 | 695.00 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-03-06 09:15:00 | 699.25 | 2026-03-06 14:15:00 | 721.85 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest1 | 2026-03-06 09:45:00 | 700.60 | 2026-03-06 14:15:00 | 721.85 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2026-03-09 09:15:00 | 680.40 | 2026-03-13 10:15:00 | 646.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-09 09:15:00 | 680.40 | 2026-03-16 11:15:00 | 645.25 | STOP_HIT | 0.50 | 5.17% |
| SELL | retest2 | 2026-03-24 10:30:00 | 628.40 | 2026-03-25 14:15:00 | 649.85 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2026-04-01 12:15:00 | 624.15 | 2026-04-07 10:15:00 | 631.80 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-04-01 13:45:00 | 624.65 | 2026-04-07 10:15:00 | 631.80 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-04-02 09:15:00 | 619.25 | 2026-04-07 10:15:00 | 631.80 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2026-04-06 09:15:00 | 617.30 | 2026-04-07 10:15:00 | 631.80 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest1 | 2026-04-10 09:15:00 | 666.65 | 2026-04-13 09:15:00 | 650.65 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest1 | 2026-04-10 10:45:00 | 664.65 | 2026-04-13 09:15:00 | 650.65 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest1 | 2026-04-10 12:00:00 | 665.00 | 2026-04-13 09:15:00 | 650.65 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2026-04-13 10:30:00 | 655.65 | 2026-04-23 09:15:00 | 651.45 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2026-04-15 09:15:00 | 665.90 | 2026-04-23 09:15:00 | 651.45 | STOP_HIT | 1.00 | -2.17% |
