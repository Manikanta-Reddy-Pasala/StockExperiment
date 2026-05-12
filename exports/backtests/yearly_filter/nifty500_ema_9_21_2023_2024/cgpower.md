# CG Power and Industrial Solutions Ltd. (CGPOWER)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 875.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 230 |
| ALERT1 | 156 |
| ALERT2 | 153 |
| ALERT2_SKIP | 78 |
| ALERT3 | 388 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 150 |
| PARTIAL | 16 |
| TARGET_HIT | 13 |
| STOP_HIT | 143 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 171 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 64 / 107
- **Target hits / Stop hits / Partials:** 13 / 142 / 16
- **Avg / median % per leg:** 0.47% / -0.84%
- **Sum % (uncompounded):** 80.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 68 | 20 | 29.4% | 9 | 59 | 0 | 0.58% | 39.7% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 3 | 0 | -0.79% | -2.4% |
| BUY @ 3rd Alert (retest2) | 65 | 19 | 29.2% | 9 | 56 | 0 | 0.65% | 42.1% |
| SELL (all) | 103 | 44 | 42.7% | 4 | 83 | 16 | 0.39% | 40.4% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.83% | -5.5% |
| SELL @ 3rd Alert (retest2) | 100 | 44 | 44.0% | 4 | 80 | 16 | 0.46% | 45.9% |
| retest1 (combined) | 6 | 1 | 16.7% | 0 | 6 | 0 | -1.31% | -7.9% |
| retest2 (combined) | 165 | 63 | 38.2% | 13 | 136 | 16 | 0.53% | 88.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 12:15:00 | 334.85 | 335.66 | 335.69 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 14:15:00 | 337.20 | 335.94 | 335.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-19 15:15:00 | 341.60 | 337.07 | 336.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 14:15:00 | 383.20 | 383.93 | 378.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-29 14:45:00 | 383.45 | 383.93 | 378.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 09:15:00 | 377.90 | 382.27 | 378.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 09:45:00 | 378.90 | 382.27 | 378.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 10:15:00 | 376.85 | 381.18 | 378.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 11:30:00 | 378.40 | 380.17 | 378.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 12:45:00 | 378.00 | 379.74 | 378.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-02 13:15:00 | 381.50 | 386.06 | 386.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2023-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-02 13:15:00 | 381.50 | 386.06 | 386.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-02 15:15:00 | 379.85 | 384.19 | 385.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-08 09:15:00 | 373.95 | 372.55 | 375.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 09:15:00 | 373.95 | 372.55 | 375.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 09:15:00 | 373.95 | 372.55 | 375.29 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2023-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-09 09:15:00 | 384.50 | 375.97 | 375.76 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2023-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-13 12:15:00 | 377.15 | 378.71 | 378.73 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 14:15:00 | 383.50 | 379.48 | 379.06 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-14 15:15:00 | 376.75 | 379.19 | 379.31 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-15 10:15:00 | 380.00 | 379.40 | 379.39 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-06-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 11:15:00 | 377.25 | 378.97 | 379.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-19 09:15:00 | 364.35 | 375.60 | 377.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 14:15:00 | 367.15 | 366.48 | 369.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-20 15:00:00 | 367.15 | 366.48 | 369.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 377.35 | 368.78 | 369.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 10:00:00 | 377.35 | 368.78 | 369.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 10:15:00 | 376.80 | 370.38 | 370.61 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 11:15:00 | 378.50 | 372.01 | 371.33 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 09:15:00 | 367.35 | 372.12 | 372.60 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 12:15:00 | 374.00 | 371.63 | 371.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 09:15:00 | 374.75 | 372.98 | 372.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-27 12:15:00 | 372.75 | 373.13 | 372.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 12:15:00 | 372.75 | 373.13 | 372.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 12:15:00 | 372.75 | 373.13 | 372.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 13:00:00 | 372.75 | 373.13 | 372.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 13:15:00 | 371.60 | 372.83 | 372.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 14:00:00 | 371.60 | 372.83 | 372.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 14:15:00 | 373.40 | 372.94 | 372.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 14:30:00 | 372.55 | 372.94 | 372.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 11:15:00 | 373.80 | 373.73 | 373.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-28 15:00:00 | 374.85 | 373.96 | 373.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-30 09:15:00 | 375.85 | 374.12 | 373.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-07-18 11:15:00 | 412.34 | 408.80 | 406.69 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2023-07-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 15:15:00 | 411.00 | 413.46 | 413.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 09:15:00 | 405.05 | 411.78 | 412.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-21 14:15:00 | 409.00 | 408.43 | 410.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-21 15:00:00 | 409.00 | 408.43 | 410.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 409.80 | 408.54 | 410.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-24 10:15:00 | 406.90 | 408.54 | 410.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-28 11:15:00 | 404.55 | 400.92 | 400.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2023-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-28 11:15:00 | 404.55 | 400.92 | 400.85 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-31 13:15:00 | 400.25 | 401.27 | 401.34 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 14:15:00 | 402.50 | 401.51 | 401.45 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 09:15:00 | 399.65 | 401.06 | 401.25 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-01 12:15:00 | 402.00 | 401.49 | 401.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 13:15:00 | 405.70 | 402.33 | 401.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 13:15:00 | 403.55 | 406.00 | 404.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 13:15:00 | 403.55 | 406.00 | 404.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 13:15:00 | 403.55 | 406.00 | 404.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 14:00:00 | 403.55 | 406.00 | 404.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 14:15:00 | 407.85 | 406.37 | 404.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 09:15:00 | 410.50 | 406.69 | 405.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 10:15:00 | 408.80 | 407.00 | 405.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-04 09:15:00 | 410.80 | 407.42 | 406.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-04 14:30:00 | 408.70 | 407.74 | 407.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 09:15:00 | 407.85 | 407.67 | 407.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 09:45:00 | 406.45 | 407.67 | 407.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 10:15:00 | 406.85 | 407.51 | 407.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 10:45:00 | 406.65 | 407.51 | 407.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 11:15:00 | 404.95 | 407.00 | 406.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 11:45:00 | 405.45 | 407.00 | 406.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-08-07 12:15:00 | 405.35 | 406.67 | 406.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2023-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-07 12:15:00 | 405.35 | 406.67 | 406.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-07 13:15:00 | 403.90 | 406.11 | 406.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-08 14:15:00 | 408.20 | 405.65 | 405.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 14:15:00 | 408.20 | 405.65 | 405.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 14:15:00 | 408.20 | 405.65 | 405.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-08 15:00:00 | 408.20 | 405.65 | 405.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — BUY (started 2023-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 15:15:00 | 408.75 | 406.27 | 406.19 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 10:15:00 | 404.50 | 405.83 | 405.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-09 15:15:00 | 403.00 | 404.47 | 405.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-11 09:15:00 | 404.00 | 402.37 | 403.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 09:15:00 | 404.00 | 402.37 | 403.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 09:15:00 | 404.00 | 402.37 | 403.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-11 09:45:00 | 403.75 | 402.37 | 403.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 10:15:00 | 403.80 | 402.66 | 403.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-11 10:30:00 | 404.00 | 402.66 | 403.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 11:15:00 | 402.30 | 402.59 | 403.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-11 11:30:00 | 403.95 | 402.59 | 403.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 13:15:00 | 403.55 | 402.74 | 403.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-11 13:30:00 | 403.90 | 402.74 | 403.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 14:15:00 | 404.15 | 403.02 | 403.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-11 15:00:00 | 404.15 | 403.02 | 403.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 15:15:00 | 404.50 | 403.32 | 403.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-14 09:15:00 | 404.50 | 403.32 | 403.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 09:15:00 | 401.25 | 402.90 | 403.28 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 10:15:00 | 406.50 | 403.77 | 403.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-16 11:15:00 | 408.45 | 404.71 | 403.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-17 14:15:00 | 412.50 | 413.59 | 410.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-17 15:00:00 | 412.50 | 413.59 | 410.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 15:15:00 | 413.00 | 413.47 | 410.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-18 09:15:00 | 419.70 | 413.47 | 410.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-23 14:15:00 | 417.65 | 422.22 | 422.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2023-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 14:15:00 | 417.65 | 422.22 | 422.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-23 15:15:00 | 414.95 | 420.77 | 421.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-25 13:15:00 | 419.50 | 416.01 | 417.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 13:15:00 | 419.50 | 416.01 | 417.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 13:15:00 | 419.50 | 416.01 | 417.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-25 14:00:00 | 419.50 | 416.01 | 417.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 14:15:00 | 417.05 | 416.22 | 417.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-28 09:30:00 | 414.70 | 416.27 | 417.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-31 09:15:00 | 418.80 | 411.19 | 411.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2023-08-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 09:15:00 | 418.80 | 411.19 | 411.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 10:15:00 | 422.70 | 413.50 | 412.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 15:15:00 | 419.90 | 423.50 | 418.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 09:15:00 | 423.00 | 423.40 | 418.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 09:15:00 | 423.00 | 423.40 | 418.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 09:30:00 | 418.00 | 423.40 | 418.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 12:15:00 | 419.30 | 422.37 | 419.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 13:00:00 | 419.30 | 422.37 | 419.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 13:15:00 | 419.60 | 421.81 | 419.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-01 14:15:00 | 422.40 | 421.81 | 419.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 10:15:00 | 439.95 | 446.57 | 447.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2023-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 10:15:00 | 439.95 | 446.57 | 447.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 09:15:00 | 422.85 | 438.00 | 442.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 14:15:00 | 446.00 | 435.01 | 438.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 14:15:00 | 446.00 | 435.01 | 438.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 14:15:00 | 446.00 | 435.01 | 438.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 15:00:00 | 446.00 | 435.01 | 438.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 15:15:00 | 448.00 | 437.61 | 439.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 09:15:00 | 440.60 | 437.61 | 439.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-14 13:15:00 | 444.40 | 440.44 | 440.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2023-09-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 13:15:00 | 444.40 | 440.44 | 440.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 14:15:00 | 445.00 | 441.35 | 440.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 13:15:00 | 441.90 | 445.47 | 443.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 13:15:00 | 441.90 | 445.47 | 443.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 13:15:00 | 441.90 | 445.47 | 443.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-15 13:30:00 | 445.10 | 445.47 | 443.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 14:15:00 | 434.80 | 443.34 | 442.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-15 15:00:00 | 434.80 | 443.34 | 442.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2023-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-15 15:15:00 | 432.80 | 441.23 | 441.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 12:15:00 | 431.30 | 436.28 | 438.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 13:15:00 | 430.40 | 427.65 | 430.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 13:15:00 | 430.40 | 427.65 | 430.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 13:15:00 | 430.40 | 427.65 | 430.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 13:30:00 | 429.50 | 427.65 | 430.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 14:15:00 | 441.00 | 430.32 | 431.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 15:00:00 | 441.00 | 430.32 | 431.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 15:15:00 | 437.80 | 431.82 | 431.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-25 09:15:00 | 436.00 | 431.82 | 431.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-25 09:15:00 | 435.00 | 432.45 | 432.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2023-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 09:15:00 | 435.00 | 432.45 | 432.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-28 09:15:00 | 445.00 | 439.17 | 437.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 14:15:00 | 436.95 | 440.82 | 439.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 14:15:00 | 436.95 | 440.82 | 439.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 436.95 | 440.82 | 439.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 15:00:00 | 436.95 | 440.82 | 439.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 435.60 | 439.78 | 438.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 11:00:00 | 441.10 | 439.98 | 439.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-04 10:15:00 | 436.25 | 440.90 | 441.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2023-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 10:15:00 | 436.25 | 440.90 | 441.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 11:15:00 | 434.75 | 439.67 | 440.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 416.30 | 415.90 | 420.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-10 09:45:00 | 416.05 | 415.90 | 420.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 13:15:00 | 395.00 | 394.26 | 396.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 13:45:00 | 395.65 | 394.26 | 396.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 14:15:00 | 392.50 | 393.91 | 396.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-17 15:15:00 | 392.00 | 393.91 | 396.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 11:00:00 | 392.35 | 393.17 | 395.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 11:45:00 | 392.10 | 393.27 | 395.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 12:45:00 | 390.65 | 392.90 | 394.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-19 12:15:00 | 372.40 | 383.10 | 388.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-19 12:15:00 | 372.73 | 383.10 | 388.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-19 12:15:00 | 372.50 | 383.10 | 388.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-19 12:15:00 | 371.12 | 383.10 | 388.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-19 13:15:00 | 384.75 | 383.43 | 387.96 | SL hit (close>ema200) qty=0.50 sl=383.43 alert=retest2 |

### Cycle 30 — BUY (started 2023-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-23 09:15:00 | 402.10 | 388.76 | 387.32 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 09:15:00 | 379.75 | 387.88 | 388.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 10:15:00 | 374.60 | 385.22 | 387.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 14:15:00 | 373.55 | 370.09 | 375.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-26 14:15:00 | 373.55 | 370.09 | 375.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 14:15:00 | 373.55 | 370.09 | 375.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-26 14:30:00 | 374.70 | 370.09 | 375.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 383.50 | 373.24 | 375.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:45:00 | 383.60 | 373.24 | 375.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 387.35 | 376.06 | 376.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 11:00:00 | 387.35 | 376.06 | 376.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2023-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 11:15:00 | 384.85 | 377.82 | 377.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 09:15:00 | 392.00 | 383.32 | 380.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 14:15:00 | 389.30 | 392.23 | 388.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-31 14:15:00 | 389.30 | 392.23 | 388.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 14:15:00 | 389.30 | 392.23 | 388.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 15:00:00 | 389.30 | 392.23 | 388.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 09:15:00 | 381.45 | 389.63 | 388.29 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-11-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 12:15:00 | 381.80 | 386.38 | 386.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 15:15:00 | 380.70 | 384.38 | 385.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 09:15:00 | 386.45 | 384.79 | 385.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 09:15:00 | 386.45 | 384.79 | 385.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 09:15:00 | 386.45 | 384.79 | 385.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-02 11:15:00 | 383.45 | 384.63 | 385.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-02 13:45:00 | 382.70 | 383.99 | 385.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-02 14:30:00 | 383.35 | 383.82 | 384.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-07 09:30:00 | 382.95 | 377.90 | 379.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 10:15:00 | 382.85 | 378.89 | 379.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-07 10:45:00 | 382.60 | 378.89 | 379.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-11-07 13:15:00 | 382.95 | 380.28 | 380.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2023-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-07 13:15:00 | 382.95 | 380.28 | 380.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-07 14:15:00 | 388.50 | 381.92 | 380.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-08 09:15:00 | 383.20 | 383.29 | 381.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-08 10:00:00 | 383.20 | 383.29 | 381.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 10:15:00 | 382.45 | 383.13 | 381.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 11:00:00 | 382.45 | 383.13 | 381.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 11:15:00 | 384.00 | 383.30 | 382.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 11:30:00 | 383.80 | 383.30 | 382.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 12:15:00 | 382.00 | 383.04 | 382.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 13:00:00 | 382.00 | 383.04 | 382.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 13:15:00 | 382.20 | 382.87 | 382.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 13:30:00 | 381.95 | 382.87 | 382.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 14:15:00 | 383.05 | 382.91 | 382.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 14:30:00 | 381.90 | 382.91 | 382.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 15:15:00 | 382.10 | 382.75 | 382.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 09:15:00 | 383.35 | 382.75 | 382.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 381.20 | 382.44 | 382.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 09:45:00 | 381.60 | 382.44 | 382.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 10:15:00 | 381.15 | 382.18 | 381.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 10:30:00 | 381.00 | 382.18 | 381.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2023-11-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 11:15:00 | 380.20 | 381.78 | 381.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 12:15:00 | 379.20 | 381.27 | 381.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-09 14:15:00 | 384.10 | 381.66 | 381.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 14:15:00 | 384.10 | 381.66 | 381.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 14:15:00 | 384.10 | 381.66 | 381.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-09 15:00:00 | 384.10 | 381.66 | 381.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2023-11-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-09 15:15:00 | 383.80 | 382.09 | 381.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-10 10:15:00 | 385.55 | 382.85 | 382.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-13 09:15:00 | 385.40 | 385.86 | 384.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-13 09:15:00 | 385.40 | 385.86 | 384.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 09:15:00 | 385.40 | 385.86 | 384.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-13 09:45:00 | 384.70 | 385.86 | 384.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 10:15:00 | 383.05 | 385.30 | 384.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-13 10:30:00 | 382.85 | 385.30 | 384.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 11:15:00 | 381.85 | 384.61 | 384.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-13 12:00:00 | 381.85 | 384.61 | 384.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2023-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 14:15:00 | 382.50 | 383.67 | 383.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-13 15:15:00 | 382.05 | 383.35 | 383.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 13:15:00 | 387.95 | 383.21 | 383.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-15 13:15:00 | 387.95 | 383.21 | 383.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 13:15:00 | 387.95 | 383.21 | 383.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-15 14:00:00 | 387.95 | 383.21 | 383.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2023-11-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 14:15:00 | 395.90 | 385.75 | 384.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 12:15:00 | 398.20 | 391.24 | 387.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-17 09:15:00 | 394.50 | 394.55 | 390.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-17 10:15:00 | 391.90 | 394.55 | 390.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 13:15:00 | 390.05 | 393.66 | 391.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-17 13:45:00 | 389.20 | 393.66 | 391.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 14:15:00 | 389.15 | 392.76 | 391.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-17 15:00:00 | 389.15 | 392.76 | 391.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 15:15:00 | 389.90 | 392.18 | 391.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 09:15:00 | 390.10 | 392.18 | 391.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 10:15:00 | 388.50 | 390.82 | 390.71 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 11:15:00 | 388.75 | 390.40 | 390.53 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 14:15:00 | 394.85 | 391.12 | 390.80 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2023-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 09:15:00 | 387.65 | 390.45 | 390.76 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-22 11:15:00 | 411.90 | 394.14 | 392.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 12:15:00 | 469.35 | 409.18 | 399.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-24 09:15:00 | 457.15 | 458.88 | 442.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-24 09:45:00 | 454.65 | 458.88 | 442.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 12:15:00 | 438.00 | 452.40 | 443.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 13:00:00 | 438.00 | 452.40 | 443.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 13:15:00 | 431.20 | 448.16 | 442.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 14:00:00 | 431.20 | 448.16 | 442.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2023-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 09:15:00 | 413.15 | 435.64 | 437.47 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-11-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 13:15:00 | 438.20 | 434.45 | 434.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 14:15:00 | 438.75 | 435.31 | 434.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 11:15:00 | 464.50 | 466.23 | 460.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-05 11:45:00 | 464.80 | 466.23 | 460.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 09:15:00 | 464.15 | 468.24 | 463.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 10:00:00 | 464.15 | 468.24 | 463.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 10:15:00 | 465.70 | 467.73 | 463.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 10:45:00 | 464.60 | 467.73 | 463.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 11:15:00 | 460.05 | 466.20 | 463.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 12:00:00 | 460.05 | 466.20 | 463.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 12:15:00 | 459.80 | 464.92 | 463.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 13:00:00 | 459.80 | 464.92 | 463.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2023-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 14:15:00 | 453.30 | 461.69 | 461.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-07 09:15:00 | 452.00 | 458.66 | 460.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-07 14:15:00 | 457.05 | 455.77 | 458.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-07 14:15:00 | 457.05 | 455.77 | 458.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 14:15:00 | 457.05 | 455.77 | 458.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-07 14:45:00 | 456.25 | 455.77 | 458.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 15:15:00 | 454.50 | 455.52 | 457.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-08 09:15:00 | 449.45 | 455.52 | 457.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 09:15:00 | 449.55 | 454.32 | 456.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-08 13:30:00 | 447.25 | 450.97 | 454.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-11 09:30:00 | 447.05 | 452.07 | 454.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-12 09:30:00 | 446.75 | 449.72 | 451.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-12 12:30:00 | 448.00 | 449.10 | 450.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 12:15:00 | 443.00 | 444.54 | 447.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 13:00:00 | 443.00 | 444.54 | 447.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 14:15:00 | 462.75 | 448.06 | 448.58 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-12-13 14:15:00 | 462.75 | 448.06 | 448.58 | SL hit (close>static) qty=1.00 sl=458.10 alert=retest2 |

### Cycle 46 — BUY (started 2023-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 15:15:00 | 465.20 | 451.49 | 450.09 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2023-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 09:15:00 | 451.35 | 456.03 | 456.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 15:15:00 | 446.00 | 450.70 | 452.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 11:15:00 | 449.90 | 449.25 | 451.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-21 12:00:00 | 449.90 | 449.25 | 451.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 12:15:00 | 455.85 | 450.57 | 451.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 13:00:00 | 455.85 | 450.57 | 451.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 13:15:00 | 454.55 | 451.37 | 452.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 13:30:00 | 454.20 | 451.37 | 452.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 14:15:00 | 456.55 | 452.41 | 452.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 15:00:00 | 456.55 | 452.41 | 452.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2023-12-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-21 15:15:00 | 453.85 | 452.69 | 452.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 09:15:00 | 462.15 | 454.59 | 453.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-26 13:15:00 | 464.95 | 465.18 | 461.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-26 13:45:00 | 464.55 | 465.18 | 461.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 14:15:00 | 466.55 | 465.45 | 462.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-26 14:30:00 | 461.75 | 465.45 | 462.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 15:15:00 | 463.65 | 465.09 | 462.20 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2023-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-27 14:15:00 | 458.95 | 461.33 | 461.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-28 14:15:00 | 451.65 | 458.69 | 460.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-29 11:15:00 | 457.70 | 456.43 | 458.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-29 12:00:00 | 457.70 | 456.43 | 458.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 14:15:00 | 454.50 | 455.99 | 457.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-29 15:00:00 | 454.50 | 455.99 | 457.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 09:15:00 | 453.20 | 454.79 | 456.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-01 15:00:00 | 449.60 | 452.18 | 454.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-02 09:30:00 | 450.10 | 450.84 | 453.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-02 15:15:00 | 449.50 | 450.94 | 452.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-03 12:15:00 | 454.50 | 453.27 | 453.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2024-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 12:15:00 | 454.50 | 453.27 | 453.13 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2024-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 14:15:00 | 451.20 | 453.04 | 453.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-03 15:15:00 | 450.90 | 452.61 | 452.86 | Break + close below crossover candle low |

### Cycle 52 — BUY (started 2024-01-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 09:15:00 | 458.65 | 453.82 | 453.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 14:15:00 | 463.60 | 457.84 | 455.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 14:15:00 | 468.80 | 471.35 | 467.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-08 14:15:00 | 468.80 | 471.35 | 467.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 14:15:00 | 468.80 | 471.35 | 467.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 15:00:00 | 468.80 | 471.35 | 467.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 15:15:00 | 469.95 | 471.07 | 467.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 09:15:00 | 475.30 | 471.07 | 467.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 14:30:00 | 472.45 | 472.15 | 469.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 09:30:00 | 471.20 | 471.63 | 470.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 10:00:00 | 471.10 | 471.63 | 470.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 11:15:00 | 469.30 | 471.16 | 470.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 12:00:00 | 469.30 | 471.16 | 470.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 12:15:00 | 468.95 | 470.72 | 469.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 13:15:00 | 468.80 | 470.72 | 469.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 15:15:00 | 465.50 | 469.66 | 469.66 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-01-10 15:15:00 | 465.50 | 469.66 | 469.66 | SL hit (close<static) qty=1.00 sl=466.55 alert=retest2 |

### Cycle 53 — SELL (started 2024-01-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-11 09:15:00 | 468.70 | 469.47 | 469.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-11 11:15:00 | 464.15 | 468.00 | 468.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-12 09:15:00 | 465.80 | 465.70 | 467.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-12 09:30:00 | 465.40 | 465.70 | 467.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 13:15:00 | 461.45 | 464.70 | 466.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-12 15:15:00 | 460.30 | 464.02 | 465.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-15 09:30:00 | 457.55 | 461.55 | 464.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-17 14:15:00 | 463.30 | 458.91 | 458.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2024-01-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-17 14:15:00 | 463.30 | 458.91 | 458.59 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-01-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 12:15:00 | 455.40 | 458.13 | 458.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 14:15:00 | 452.95 | 456.84 | 457.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 457.25 | 456.38 | 457.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 09:15:00 | 457.25 | 456.38 | 457.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 457.25 | 456.38 | 457.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 11:00:00 | 454.35 | 455.98 | 457.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 14:45:00 | 454.30 | 454.67 | 456.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-20 09:30:00 | 454.10 | 454.77 | 455.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-20 10:45:00 | 454.50 | 454.81 | 455.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 12:15:00 | 454.15 | 454.62 | 455.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-20 12:45:00 | 455.00 | 454.62 | 455.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 15:15:00 | 449.85 | 453.70 | 454.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-23 09:15:00 | 464.80 | 453.70 | 454.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 462.60 | 455.48 | 455.57 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-01-23 09:15:00 | 462.60 | 455.48 | 455.57 | SL hit (close>static) qty=1.00 sl=460.00 alert=retest2 |

### Cycle 56 — BUY (started 2024-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-23 10:15:00 | 464.45 | 457.27 | 456.37 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 14:15:00 | 447.00 | 454.79 | 455.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-24 09:15:00 | 434.55 | 449.54 | 452.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 12:15:00 | 449.00 | 448.51 | 451.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 12:15:00 | 449.00 | 448.51 | 451.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 12:15:00 | 449.00 | 448.51 | 451.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 12:45:00 | 449.40 | 448.51 | 451.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 450.00 | 447.80 | 450.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 15:00:00 | 450.00 | 447.80 | 450.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 15:15:00 | 450.00 | 448.24 | 450.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:15:00 | 454.00 | 448.24 | 450.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 454.15 | 449.43 | 450.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 10:45:00 | 448.30 | 449.63 | 450.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 12:00:00 | 446.60 | 449.02 | 450.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 13:00:00 | 448.00 | 448.82 | 450.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-25 14:15:00 | 461.65 | 451.67 | 451.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2024-01-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 14:15:00 | 461.65 | 451.67 | 451.30 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 14:15:00 | 457.15 | 460.14 | 460.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 15:15:00 | 455.40 | 459.19 | 460.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 10:15:00 | 443.80 | 442.91 | 447.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-06 11:00:00 | 443.80 | 442.91 | 447.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 12:15:00 | 444.15 | 443.75 | 446.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 09:15:00 | 440.35 | 444.18 | 446.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-14 15:15:00 | 438.55 | 431.07 | 430.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2024-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 15:15:00 | 438.55 | 431.07 | 430.93 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-02-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 13:15:00 | 431.50 | 432.33 | 432.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-16 15:15:00 | 429.20 | 431.54 | 432.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-19 10:15:00 | 434.75 | 432.06 | 432.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 10:15:00 | 434.75 | 432.06 | 432.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 10:15:00 | 434.75 | 432.06 | 432.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-19 11:00:00 | 434.75 | 432.06 | 432.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2024-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 11:15:00 | 435.00 | 432.65 | 432.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-19 13:15:00 | 438.40 | 433.88 | 433.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 15:15:00 | 433.10 | 434.15 | 433.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 15:15:00 | 433.10 | 434.15 | 433.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 15:15:00 | 433.10 | 434.15 | 433.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 09:45:00 | 430.05 | 433.14 | 432.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2024-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 10:15:00 | 430.70 | 432.65 | 432.75 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-20 11:15:00 | 436.70 | 433.46 | 433.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-20 12:15:00 | 438.15 | 434.40 | 433.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-20 14:15:00 | 434.75 | 434.96 | 433.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 14:15:00 | 434.75 | 434.96 | 433.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 14:15:00 | 434.75 | 434.96 | 433.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 15:00:00 | 434.75 | 434.96 | 433.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 15:15:00 | 432.00 | 434.37 | 433.81 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2024-02-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 12:15:00 | 432.35 | 433.32 | 433.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 13:15:00 | 431.25 | 432.90 | 433.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 15:15:00 | 427.05 | 426.68 | 428.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-23 09:15:00 | 427.65 | 426.68 | 428.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 427.75 | 426.89 | 428.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:30:00 | 429.15 | 426.89 | 428.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 10:15:00 | 427.25 | 426.96 | 428.67 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 14:15:00 | 434.00 | 429.72 | 429.58 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 10:15:00 | 424.30 | 428.92 | 429.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 11:15:00 | 423.00 | 427.73 | 428.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-26 14:15:00 | 429.50 | 427.97 | 428.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 14:15:00 | 429.50 | 427.97 | 428.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 14:15:00 | 429.50 | 427.97 | 428.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-26 14:30:00 | 430.10 | 427.97 | 428.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 15:15:00 | 426.80 | 427.74 | 428.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 09:15:00 | 431.20 | 427.74 | 428.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 430.70 | 428.33 | 428.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 09:45:00 | 432.00 | 428.33 | 428.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2024-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 10:15:00 | 430.95 | 428.85 | 428.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-27 11:15:00 | 439.20 | 430.92 | 429.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 10:15:00 | 435.75 | 435.94 | 433.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 10:15:00 | 435.75 | 435.94 | 433.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 10:15:00 | 435.75 | 435.94 | 433.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 10:45:00 | 433.60 | 435.94 | 433.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 12:15:00 | 433.30 | 435.40 | 433.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 13:00:00 | 433.30 | 435.40 | 433.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 13:15:00 | 434.85 | 435.29 | 433.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-29 14:15:00 | 437.90 | 434.34 | 433.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-03-01 09:15:00 | 481.69 | 445.71 | 439.28 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 10:15:00 | 452.10 | 460.16 | 461.24 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-06 14:15:00 | 469.90 | 461.84 | 461.52 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-03-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-07 14:15:00 | 458.80 | 461.39 | 461.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 09:15:00 | 450.40 | 458.64 | 460.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-11 10:15:00 | 459.20 | 458.76 | 460.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-11 11:00:00 | 459.20 | 458.76 | 460.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 11:15:00 | 461.00 | 459.20 | 460.34 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2024-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 12:15:00 | 469.45 | 461.25 | 461.17 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 11:15:00 | 455.15 | 460.95 | 461.45 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-03-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 09:15:00 | 463.00 | 459.41 | 459.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-14 10:15:00 | 467.05 | 460.94 | 459.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-15 14:15:00 | 469.50 | 472.72 | 468.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-15 15:00:00 | 469.50 | 472.72 | 468.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 15:15:00 | 466.90 | 471.55 | 468.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-18 09:15:00 | 472.70 | 471.55 | 468.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-18 09:45:00 | 471.75 | 470.74 | 468.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-18 10:30:00 | 473.50 | 470.10 | 468.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-18 12:15:00 | 465.00 | 468.74 | 467.82 | SL hit (close<static) qty=1.00 sl=465.30 alert=retest2 |

### Cycle 75 — SELL (started 2024-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 12:15:00 | 465.50 | 467.35 | 467.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 13:15:00 | 464.85 | 466.85 | 467.27 | Break + close below crossover candle low |

### Cycle 76 — BUY (started 2024-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-19 14:15:00 | 480.00 | 469.48 | 468.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-20 10:15:00 | 495.55 | 477.18 | 472.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 11:15:00 | 523.05 | 526.16 | 512.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-22 12:00:00 | 523.05 | 526.16 | 512.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 14:15:00 | 543.00 | 544.19 | 540.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 14:30:00 | 540.50 | 544.19 | 540.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 15:15:00 | 537.00 | 542.75 | 539.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-01 10:00:00 | 535.80 | 541.36 | 539.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 10:15:00 | 535.00 | 540.09 | 538.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-01 10:45:00 | 534.55 | 540.09 | 538.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2024-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-01 12:15:00 | 524.60 | 535.70 | 537.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-01 13:15:00 | 519.25 | 532.41 | 535.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-04 12:15:00 | 513.95 | 511.44 | 516.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-04 12:15:00 | 513.95 | 511.44 | 516.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 12:15:00 | 513.95 | 511.44 | 516.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-04 12:45:00 | 516.00 | 511.44 | 516.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 513.75 | 511.75 | 514.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-08 12:45:00 | 507.55 | 511.93 | 513.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 482.17 | 497.29 | 500.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-15 14:15:00 | 495.20 | 493.49 | 496.97 | SL hit (close>ema200) qty=0.50 sl=493.49 alert=retest2 |

### Cycle 78 — BUY (started 2024-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 11:15:00 | 504.40 | 498.59 | 498.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-16 13:15:00 | 509.10 | 501.67 | 499.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 14:15:00 | 532.30 | 535.66 | 531.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-23 15:00:00 | 532.30 | 535.66 | 531.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 15:15:00 | 533.60 | 535.25 | 531.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 09:15:00 | 539.60 | 535.25 | 531.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 11:00:00 | 537.55 | 535.91 | 532.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-02 12:15:00 | 551.45 | 553.31 | 553.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2024-05-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 12:15:00 | 551.45 | 553.31 | 553.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-02 14:15:00 | 548.90 | 552.17 | 552.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-03 09:15:00 | 553.00 | 551.92 | 552.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-03 09:15:00 | 553.00 | 551.92 | 552.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 09:15:00 | 553.00 | 551.92 | 552.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 10:45:00 | 550.85 | 551.58 | 552.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 09:30:00 | 547.10 | 548.64 | 550.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 11:45:00 | 549.80 | 549.09 | 550.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 12:30:00 | 550.50 | 548.84 | 550.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 13:15:00 | 538.00 | 546.68 | 548.95 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-07 09:15:00 | 578.75 | 554.16 | 551.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-07 09:15:00 | 578.75 | 554.16 | 551.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 14:15:00 | 583.70 | 572.31 | 565.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 15:15:00 | 577.65 | 579.28 | 573.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-10 09:15:00 | 578.00 | 579.28 | 573.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 576.85 | 578.79 | 573.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-10 09:30:00 | 575.50 | 578.79 | 573.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 15:15:00 | 579.00 | 579.91 | 576.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 09:15:00 | 590.45 | 579.91 | 576.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 588.30 | 581.59 | 577.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-13 12:15:00 | 597.95 | 585.99 | 580.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-14 12:45:00 | 597.95 | 590.42 | 585.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-17 09:15:00 | 657.75 | 635.70 | 625.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 15:15:00 | 643.00 | 644.62 | 644.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 10:15:00 | 638.60 | 642.79 | 643.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 09:15:00 | 639.00 | 637.91 | 640.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 639.00 | 637.91 | 640.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 639.00 | 637.91 | 640.40 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 14:15:00 | 648.05 | 641.35 | 641.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 15:15:00 | 650.10 | 643.10 | 641.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 09:15:00 | 642.25 | 642.93 | 641.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 09:15:00 | 642.25 | 642.93 | 641.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 642.25 | 642.93 | 641.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:45:00 | 640.70 | 642.93 | 641.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 650.15 | 644.37 | 642.70 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 632.00 | 644.16 | 644.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 09:15:00 | 625.50 | 634.53 | 639.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 13:15:00 | 634.00 | 632.52 | 636.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 13:15:00 | 634.00 | 632.52 | 636.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 634.00 | 632.52 | 636.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 13:45:00 | 633.05 | 632.52 | 636.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 632.50 | 632.52 | 636.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 15:00:00 | 632.50 | 632.52 | 636.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 650.20 | 635.65 | 637.00 | EMA400 retest candle locked (from downside) |

### Cycle 84 — BUY (started 2024-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 10:15:00 | 656.45 | 639.81 | 638.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 674.40 | 652.87 | 648.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 653.25 | 672.76 | 663.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 653.25 | 672.76 | 663.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 653.25 | 672.76 | 663.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 652.80 | 672.76 | 663.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 600.95 | 658.40 | 658.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 600.95 | 658.40 | 658.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 556.70 | 638.06 | 648.85 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 662.60 | 637.89 | 636.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-12 09:15:00 | 671.75 | 663.18 | 659.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 14:15:00 | 665.40 | 667.91 | 663.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 15:00:00 | 665.40 | 667.91 | 663.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 15:15:00 | 669.30 | 668.82 | 666.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 11:30:00 | 674.00 | 669.28 | 667.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 09:15:00 | 675.25 | 678.54 | 678.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 09:15:00 | 675.25 | 678.54 | 678.73 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 13:15:00 | 680.00 | 679.01 | 678.90 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 10:15:00 | 676.90 | 678.86 | 678.93 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 11:15:00 | 681.00 | 679.29 | 679.12 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 12:15:00 | 674.90 | 678.41 | 678.73 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-06-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 13:15:00 | 687.00 | 680.13 | 679.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 10:15:00 | 690.05 | 682.29 | 680.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 10:15:00 | 695.00 | 701.07 | 694.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 10:15:00 | 695.00 | 701.07 | 694.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 695.00 | 701.07 | 694.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 11:00:00 | 695.00 | 701.07 | 694.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 695.35 | 699.93 | 694.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 11:30:00 | 695.25 | 699.93 | 694.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 12:15:00 | 694.55 | 698.85 | 694.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 12:30:00 | 694.40 | 698.85 | 694.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 13:15:00 | 695.50 | 698.18 | 694.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 13:30:00 | 691.35 | 698.18 | 694.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 698.90 | 698.33 | 694.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 15:00:00 | 698.90 | 698.33 | 694.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 691.00 | 696.86 | 694.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 09:15:00 | 699.30 | 696.86 | 694.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 09:15:00 | 686.10 | 694.71 | 693.63 | SL hit (close<static) qty=1.00 sl=689.05 alert=retest2 |

### Cycle 93 — SELL (started 2024-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 12:15:00 | 691.00 | 692.73 | 692.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 14:15:00 | 685.05 | 690.68 | 691.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 11:15:00 | 689.95 | 688.44 | 690.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 11:15:00 | 689.95 | 688.44 | 690.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 689.95 | 688.44 | 690.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 12:00:00 | 689.95 | 688.44 | 690.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 694.35 | 689.62 | 690.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 12:45:00 | 692.75 | 689.62 | 690.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 13:15:00 | 698.80 | 691.46 | 691.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 13:15:00 | 707.55 | 698.49 | 695.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 10:15:00 | 713.65 | 714.62 | 708.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 10:30:00 | 713.45 | 714.62 | 708.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 719.75 | 718.88 | 715.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 14:30:00 | 716.60 | 718.88 | 715.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 755.85 | 764.52 | 756.48 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 733.55 | 753.59 | 754.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 12:15:00 | 722.55 | 736.75 | 741.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 09:15:00 | 730.30 | 728.27 | 732.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 730.30 | 728.27 | 732.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 730.30 | 728.27 | 732.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:30:00 | 732.80 | 728.27 | 732.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 730.25 | 728.30 | 731.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:30:00 | 729.10 | 728.30 | 731.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 730.00 | 728.64 | 731.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 14:15:00 | 726.65 | 728.64 | 731.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 14:45:00 | 728.60 | 729.15 | 731.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 09:15:00 | 715.65 | 729.32 | 730.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 10:15:00 | 690.32 | 715.98 | 724.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 10:15:00 | 692.17 | 715.98 | 724.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 09:15:00 | 679.87 | 692.02 | 707.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-07-19 11:15:00 | 653.99 | 680.83 | 699.20 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 96 — BUY (started 2024-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 15:15:00 | 699.95 | 686.17 | 684.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 09:15:00 | 721.25 | 693.19 | 687.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 09:15:00 | 719.65 | 720.43 | 707.79 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 09:15:00 | 738.35 | 727.77 | 717.51 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 732.95 | 739.16 | 735.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-31 09:15:00 | 732.95 | 739.16 | 735.08 | SL hit (close<ema400) qty=1.00 sl=735.08 alert=retest1 |

### Cycle 97 — SELL (started 2024-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 12:15:00 | 730.15 | 734.58 | 734.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 14:15:00 | 727.85 | 732.45 | 733.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 14:15:00 | 728.20 | 721.35 | 726.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 14:15:00 | 728.20 | 721.35 | 726.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 14:15:00 | 728.20 | 721.35 | 726.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 14:45:00 | 738.60 | 721.35 | 726.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 15:15:00 | 724.85 | 722.05 | 726.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 09:15:00 | 692.50 | 722.05 | 726.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 11:15:00 | 704.90 | 689.83 | 688.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 11:15:00 | 704.90 | 689.83 | 688.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 12:15:00 | 707.05 | 693.27 | 690.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 700.90 | 702.19 | 696.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-12 10:00:00 | 700.90 | 702.19 | 696.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 12:15:00 | 702.90 | 701.91 | 697.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 14:00:00 | 704.55 | 702.44 | 698.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 15:00:00 | 706.00 | 703.15 | 698.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 11:30:00 | 704.50 | 703.91 | 700.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 14:15:00 | 695.15 | 700.88 | 699.92 | SL hit (close<static) qty=1.00 sl=696.45 alert=retest2 |

### Cycle 99 — SELL (started 2024-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 09:15:00 | 690.90 | 697.85 | 698.64 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 14:15:00 | 708.70 | 698.22 | 698.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 13:15:00 | 710.00 | 703.55 | 700.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 12:15:00 | 707.80 | 709.04 | 705.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 12:30:00 | 706.50 | 709.04 | 705.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 741.75 | 746.57 | 741.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:30:00 | 741.95 | 746.57 | 741.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 11:15:00 | 738.80 | 745.02 | 741.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:45:00 | 739.00 | 745.02 | 741.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 739.15 | 743.84 | 741.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 12:45:00 | 737.65 | 743.84 | 741.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 09:15:00 | 731.55 | 738.93 | 739.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 14:15:00 | 729.75 | 733.47 | 736.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 10:15:00 | 707.95 | 706.00 | 713.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-30 11:00:00 | 707.95 | 706.00 | 713.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 13:15:00 | 692.15 | 689.09 | 694.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 15:00:00 | 689.50 | 689.17 | 694.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 12:30:00 | 688.55 | 688.88 | 692.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 09:30:00 | 687.55 | 688.46 | 690.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 13:15:00 | 679.40 | 673.81 | 673.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2024-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 13:15:00 | 679.40 | 673.81 | 673.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 14:15:00 | 689.40 | 676.93 | 675.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 12:15:00 | 719.00 | 722.18 | 708.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 12:45:00 | 720.40 | 722.18 | 708.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 14:15:00 | 712.50 | 717.08 | 713.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 09:15:00 | 720.65 | 716.47 | 713.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-25 09:15:00 | 792.72 | 777.80 | 765.90 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2024-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 13:15:00 | 762.00 | 768.68 | 768.70 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 14:15:00 | 772.35 | 769.42 | 769.03 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 09:15:00 | 761.00 | 768.79 | 768.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 15:15:00 | 759.45 | 763.29 | 765.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 10:15:00 | 766.60 | 763.82 | 765.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 10:15:00 | 766.60 | 763.82 | 765.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 766.60 | 763.82 | 765.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:00:00 | 766.60 | 763.82 | 765.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 761.00 | 763.25 | 765.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 13:30:00 | 758.95 | 761.79 | 764.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 721.00 | 741.01 | 748.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-07 12:15:00 | 735.00 | 726.92 | 734.26 | SL hit (close>ema200) qty=0.50 sl=726.92 alert=retest2 |

### Cycle 106 — BUY (started 2024-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-07 15:15:00 | 756.35 | 740.99 | 739.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 09:15:00 | 762.90 | 745.37 | 741.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 09:15:00 | 842.45 | 848.64 | 830.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-14 10:00:00 | 842.45 | 848.64 | 830.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 833.00 | 848.71 | 839.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:00:00 | 833.00 | 848.71 | 839.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 840.00 | 846.97 | 839.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:30:00 | 840.65 | 846.97 | 839.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 839.60 | 845.49 | 839.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 09:15:00 | 842.50 | 837.67 | 837.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 14:15:00 | 834.15 | 838.30 | 838.20 | SL hit (close<static) qty=1.00 sl=836.40 alert=retest2 |

### Cycle 107 — SELL (started 2024-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 15:15:00 | 834.50 | 837.54 | 837.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 828.80 | 835.79 | 837.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 11:15:00 | 835.00 | 834.58 | 836.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-17 12:00:00 | 835.00 | 834.58 | 836.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 108 — BUY (started 2024-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-17 12:15:00 | 849.50 | 837.56 | 837.41 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 09:15:00 | 820.10 | 834.51 | 836.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 15:15:00 | 816.55 | 823.77 | 829.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 09:15:00 | 853.25 | 829.66 | 831.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 09:15:00 | 853.25 | 829.66 | 831.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 853.25 | 829.66 | 831.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 10:00:00 | 853.25 | 829.66 | 831.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 844.45 | 832.62 | 832.66 | EMA400 retest candle locked (from downside) |

### Cycle 110 — BUY (started 2024-10-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 11:15:00 | 836.80 | 833.46 | 833.03 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 14:15:00 | 769.30 | 820.09 | 827.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 12:15:00 | 760.90 | 788.38 | 807.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-25 12:15:00 | 731.05 | 730.85 | 744.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-25 12:30:00 | 734.10 | 730.85 | 744.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 14:15:00 | 744.00 | 733.51 | 742.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-25 15:00:00 | 744.00 | 733.51 | 742.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 15:15:00 | 742.00 | 735.21 | 742.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 09:15:00 | 737.55 | 735.21 | 742.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 729.60 | 734.09 | 741.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 12:00:00 | 722.85 | 731.22 | 739.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 11:45:00 | 727.05 | 720.94 | 723.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 13:30:00 | 727.45 | 722.96 | 723.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 10:15:00 | 728.80 | 713.52 | 713.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 728.80 | 713.52 | 713.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 12:15:00 | 733.90 | 719.77 | 716.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 15:15:00 | 726.70 | 727.89 | 724.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 15:15:00 | 726.70 | 727.89 | 724.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 726.70 | 727.89 | 724.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 724.95 | 727.89 | 724.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 725.00 | 727.31 | 724.25 | EMA400 retest candle locked (from upside) |

### Cycle 113 — SELL (started 2024-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 15:15:00 | 714.00 | 721.77 | 722.73 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2024-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 11:15:00 | 726.85 | 723.31 | 723.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-11 14:15:00 | 728.30 | 724.27 | 723.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-11 15:15:00 | 724.20 | 724.26 | 723.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 15:15:00 | 724.20 | 724.26 | 723.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 15:15:00 | 724.20 | 724.26 | 723.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:15:00 | 726.95 | 724.26 | 723.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 725.95 | 724.60 | 723.95 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2024-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 14:15:00 | 712.20 | 722.10 | 723.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 700.65 | 716.13 | 720.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 704.50 | 701.53 | 708.99 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-14 14:00:00 | 692.15 | 699.23 | 705.53 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 702.95 | 698.71 | 703.61 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-18 11:15:00 | 709.85 | 701.22 | 703.92 | SL hit (close>ema400) qty=1.00 sl=703.92 alert=retest1 |

### Cycle 116 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 709.25 | 704.87 | 704.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 11:15:00 | 716.30 | 707.16 | 705.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 10:15:00 | 710.55 | 712.39 | 709.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 10:15:00 | 710.55 | 712.39 | 709.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 710.55 | 712.39 | 709.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:45:00 | 709.15 | 712.39 | 709.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 11:15:00 | 712.80 | 712.47 | 710.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 11:45:00 | 713.45 | 712.47 | 710.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 12:15:00 | 713.70 | 712.71 | 710.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 12:45:00 | 710.90 | 712.71 | 710.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 13:15:00 | 710.25 | 712.22 | 710.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 14:00:00 | 710.25 | 712.22 | 710.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 14:15:00 | 712.05 | 712.19 | 710.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 14:45:00 | 710.00 | 712.19 | 710.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 15:15:00 | 710.00 | 711.75 | 710.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 09:15:00 | 717.10 | 711.75 | 710.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 713.50 | 712.10 | 710.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 11:00:00 | 718.40 | 713.36 | 711.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 11:15:00 | 736.05 | 747.53 | 748.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2024-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 11:15:00 | 736.05 | 747.53 | 748.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 14:15:00 | 730.60 | 742.16 | 745.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 747.45 | 741.59 | 744.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 09:15:00 | 747.45 | 741.59 | 744.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 747.45 | 741.59 | 744.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:45:00 | 747.65 | 741.59 | 744.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 744.65 | 742.20 | 744.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:15:00 | 748.65 | 742.20 | 744.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 751.85 | 744.13 | 745.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 12:00:00 | 751.85 | 744.13 | 745.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2024-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 12:15:00 | 757.20 | 746.75 | 746.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 11:15:00 | 760.10 | 754.55 | 751.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 11:15:00 | 756.10 | 758.35 | 755.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 11:15:00 | 756.10 | 758.35 | 755.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 756.10 | 758.35 | 755.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:00:00 | 756.10 | 758.35 | 755.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 756.60 | 758.00 | 755.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 09:15:00 | 766.05 | 755.86 | 754.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 09:15:00 | 784.10 | 785.81 | 785.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 784.10 | 785.81 | 785.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 11:15:00 | 774.30 | 782.30 | 784.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 14:15:00 | 783.05 | 780.11 | 782.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 14:15:00 | 783.05 | 780.11 | 782.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 14:15:00 | 783.05 | 780.11 | 782.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 14:45:00 | 783.60 | 780.11 | 782.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 781.00 | 780.29 | 782.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:15:00 | 771.50 | 780.29 | 782.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 763.60 | 776.95 | 780.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 11:45:00 | 762.70 | 768.01 | 772.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 13:30:00 | 761.50 | 765.88 | 771.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 12:15:00 | 774.55 | 772.63 | 772.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 12:15:00 | 774.55 | 772.63 | 772.59 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2024-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 14:15:00 | 771.25 | 772.33 | 772.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 15:15:00 | 767.00 | 771.27 | 771.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 13:15:00 | 764.45 | 764.08 | 767.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-18 14:00:00 | 764.45 | 764.08 | 767.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 14:15:00 | 767.95 | 764.85 | 767.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 15:00:00 | 767.95 | 764.85 | 767.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 15:15:00 | 764.10 | 764.70 | 767.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:15:00 | 753.20 | 764.70 | 767.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 11:45:00 | 763.25 | 762.93 | 765.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 10:00:00 | 759.30 | 762.76 | 764.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 15:15:00 | 725.09 | 744.52 | 753.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 09:15:00 | 721.33 | 741.23 | 751.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 12:15:00 | 715.54 | 723.33 | 734.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-26 09:15:00 | 723.70 | 719.67 | 728.60 | SL hit (close>ema200) qty=0.50 sl=719.67 alert=retest2 |

### Cycle 122 — BUY (started 2024-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 14:15:00 | 745.15 | 731.68 | 731.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 747.85 | 736.73 | 734.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 09:15:00 | 740.15 | 743.86 | 740.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 09:15:00 | 740.15 | 743.86 | 740.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 740.15 | 743.86 | 740.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:00:00 | 740.15 | 743.86 | 740.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 739.20 | 742.93 | 739.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 11:00:00 | 739.20 | 742.93 | 739.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 11:15:00 | 737.30 | 741.80 | 739.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 12:00:00 | 737.30 | 741.80 | 739.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 12:15:00 | 740.85 | 741.61 | 739.81 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2024-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 09:15:00 | 736.30 | 738.54 | 738.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 12:15:00 | 727.50 | 735.44 | 737.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 13:15:00 | 730.45 | 729.09 | 732.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-01 14:00:00 | 730.45 | 729.09 | 732.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 14:15:00 | 742.05 | 731.68 | 733.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 15:00:00 | 742.05 | 731.68 | 733.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 15:15:00 | 741.00 | 733.54 | 733.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:15:00 | 745.55 | 733.54 | 733.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 09:15:00 | 746.30 | 736.09 | 734.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 10:15:00 | 750.50 | 738.98 | 736.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 09:15:00 | 744.55 | 745.02 | 741.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 10:00:00 | 744.55 | 745.02 | 741.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 743.85 | 744.78 | 741.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:45:00 | 741.75 | 744.78 | 741.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 744.75 | 744.77 | 742.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 12:30:00 | 745.10 | 744.77 | 742.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 739.10 | 743.64 | 741.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:00:00 | 739.10 | 743.64 | 741.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 737.65 | 742.44 | 741.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 737.65 | 742.44 | 741.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 733.80 | 740.71 | 740.70 | EMA400 retest candle locked (from upside) |

### Cycle 125 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 739.30 | 740.43 | 740.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 726.90 | 737.72 | 739.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 723.90 | 722.40 | 729.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 11:00:00 | 723.90 | 722.40 | 729.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 631.80 | 625.51 | 639.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:00:00 | 631.80 | 625.51 | 639.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 640.75 | 630.09 | 639.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 09:15:00 | 626.30 | 630.09 | 639.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-17 10:15:00 | 639.15 | 630.80 | 629.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2025-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 10:15:00 | 639.15 | 630.80 | 629.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 12:15:00 | 645.00 | 635.60 | 632.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 669.00 | 670.01 | 658.15 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-21 11:30:00 | 673.40 | 669.92 | 659.18 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 658.50 | 666.55 | 660.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-21 14:15:00 | 658.50 | 666.55 | 660.21 | SL hit (close<ema400) qty=1.00 sl=660.21 alert=retest1 |

### Cycle 127 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 630.80 | 653.47 | 655.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 618.65 | 646.50 | 652.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 640.80 | 640.33 | 647.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 640.80 | 640.33 | 647.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 645.05 | 640.74 | 646.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 645.05 | 640.74 | 646.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 641.00 | 640.79 | 645.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 11:30:00 | 638.35 | 639.66 | 644.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 606.43 | 619.12 | 628.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-28 09:15:00 | 574.51 | 592.55 | 608.49 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 128 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 623.00 | 601.22 | 598.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 629.65 | 606.91 | 601.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 11:15:00 | 614.20 | 614.86 | 607.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 12:00:00 | 614.20 | 614.86 | 607.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 627.45 | 632.66 | 625.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 624.50 | 632.66 | 625.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 603.55 | 626.84 | 623.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 611.70 | 626.84 | 623.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 622.85 | 626.04 | 623.78 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 15:15:00 | 610.00 | 620.82 | 621.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 589.30 | 614.51 | 618.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 600.15 | 593.38 | 603.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 600.15 | 593.38 | 603.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 600.15 | 593.38 | 603.09 | EMA400 retest candle locked (from downside) |

### Cycle 130 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 618.45 | 605.61 | 604.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 13:15:00 | 627.00 | 613.57 | 609.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 10:15:00 | 618.90 | 619.19 | 613.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 11:00:00 | 618.90 | 619.19 | 613.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 616.60 | 618.21 | 614.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 616.60 | 618.21 | 614.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 610.35 | 616.64 | 613.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:00:00 | 610.35 | 616.64 | 613.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 614.05 | 616.12 | 613.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:30:00 | 612.65 | 616.12 | 613.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 615.05 | 615.90 | 613.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 617.70 | 615.90 | 613.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 619.30 | 616.58 | 614.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:45:00 | 621.65 | 618.10 | 615.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 598.30 | 612.94 | 613.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 598.30 | 612.94 | 613.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 10:15:00 | 593.25 | 609.00 | 612.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 14:15:00 | 587.95 | 584.32 | 593.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-11 14:45:00 | 588.00 | 584.32 | 593.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 589.15 | 579.50 | 584.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 589.15 | 579.50 | 584.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 593.25 | 582.25 | 585.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 593.25 | 582.25 | 585.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 572.25 | 579.87 | 582.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:30:00 | 567.05 | 576.02 | 580.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 538.70 | 559.99 | 569.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 10:15:00 | 563.05 | 560.60 | 568.90 | SL hit (close>ema200) qty=0.50 sl=560.60 alert=retest2 |

### Cycle 132 — BUY (started 2025-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 15:15:00 | 585.00 | 572.00 | 571.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 11:15:00 | 592.15 | 581.10 | 576.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 09:15:00 | 584.85 | 584.89 | 580.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 09:15:00 | 584.85 | 584.89 | 580.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 584.85 | 584.89 | 580.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 10:15:00 | 589.45 | 584.89 | 580.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 10:30:00 | 587.95 | 588.92 | 585.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 11:15:00 | 587.15 | 588.92 | 585.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 13:45:00 | 586.70 | 589.08 | 586.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 587.00 | 588.90 | 587.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 575.00 | 588.90 | 587.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 575.25 | 586.17 | 586.02 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-24 10:15:00 | 579.50 | 584.84 | 585.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 579.50 | 584.84 | 585.42 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2025-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 12:15:00 | 600.15 | 587.61 | 586.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-24 14:15:00 | 602.00 | 592.09 | 588.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-25 13:15:00 | 598.70 | 598.72 | 594.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-25 14:00:00 | 598.70 | 598.72 | 594.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 586.80 | 596.16 | 594.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 10:00:00 | 586.80 | 596.16 | 594.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — SELL (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 10:15:00 | 577.25 | 592.38 | 592.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 572.35 | 582.18 | 586.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 572.50 | 568.38 | 574.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 12:15:00 | 572.50 | 568.38 | 574.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 572.50 | 568.38 | 574.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 12:45:00 | 570.25 | 568.38 | 574.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 587.50 | 572.84 | 575.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 15:00:00 | 587.50 | 572.84 | 575.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 585.00 | 575.27 | 576.68 | EMA400 retest candle locked (from downside) |

### Cycle 136 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 591.50 | 578.52 | 578.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 14:15:00 | 601.50 | 589.15 | 583.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 623.90 | 625.17 | 616.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 623.90 | 625.17 | 616.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 619.70 | 625.20 | 619.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 619.70 | 625.20 | 619.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 613.75 | 622.91 | 619.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 613.75 | 622.91 | 619.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 611.95 | 620.72 | 618.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 611.95 | 620.72 | 618.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2025-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 13:15:00 | 607.10 | 616.55 | 616.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 595.55 | 612.35 | 614.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 13:15:00 | 603.60 | 601.37 | 607.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 14:00:00 | 603.60 | 601.37 | 607.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 612.50 | 604.43 | 607.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 609.50 | 604.43 | 607.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 606.00 | 604.74 | 607.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:45:00 | 605.00 | 604.73 | 607.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 15:15:00 | 604.40 | 604.37 | 605.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 10:15:00 | 612.10 | 607.26 | 606.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — BUY (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 10:15:00 | 612.10 | 607.26 | 606.96 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 14:15:00 | 598.80 | 605.60 | 606.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 15:15:00 | 597.95 | 604.07 | 605.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 605.95 | 604.45 | 605.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 605.95 | 604.45 | 605.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 605.95 | 604.45 | 605.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:45:00 | 605.00 | 604.45 | 605.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 603.35 | 604.23 | 605.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:00:00 | 603.35 | 604.23 | 605.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 604.50 | 604.28 | 605.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:30:00 | 605.15 | 604.28 | 605.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 13:15:00 | 606.85 | 604.63 | 605.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 13:45:00 | 607.00 | 604.63 | 605.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 609.70 | 605.64 | 605.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 15:00:00 | 609.70 | 605.64 | 605.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — BUY (started 2025-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 15:15:00 | 613.55 | 607.23 | 606.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 623.80 | 610.54 | 607.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 650.25 | 650.86 | 637.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 650.25 | 650.86 | 637.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 11:15:00 | 646.45 | 651.23 | 646.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 11:30:00 | 645.30 | 651.23 | 646.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 12:15:00 | 639.45 | 648.88 | 645.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 13:00:00 | 639.45 | 648.88 | 645.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 638.00 | 646.70 | 644.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 13:45:00 | 636.75 | 646.70 | 644.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 640.95 | 644.15 | 643.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 10:00:00 | 640.95 | 644.15 | 643.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2025-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-24 10:15:00 | 642.10 | 643.74 | 643.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-24 14:15:00 | 633.35 | 640.16 | 641.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-25 09:15:00 | 644.55 | 639.99 | 641.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 09:15:00 | 644.55 | 639.99 | 641.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 644.55 | 639.99 | 641.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:30:00 | 647.25 | 639.99 | 641.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 643.70 | 640.73 | 641.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 11:15:00 | 635.85 | 640.73 | 641.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 09:30:00 | 635.10 | 635.16 | 637.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 14:45:00 | 636.50 | 638.49 | 638.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 12:00:00 | 639.45 | 636.68 | 637.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 640.70 | 637.48 | 637.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:00:00 | 640.70 | 637.48 | 637.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 636.10 | 636.59 | 637.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 636.10 | 636.59 | 637.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 628.00 | 634.88 | 636.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:30:00 | 643.35 | 635.90 | 636.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 641.00 | 636.92 | 637.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:30:00 | 645.50 | 636.92 | 637.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-28 11:15:00 | 645.25 | 638.59 | 637.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 645.25 | 638.59 | 637.99 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 09:15:00 | 630.05 | 636.65 | 637.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 10:15:00 | 616.50 | 632.62 | 635.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 10:15:00 | 620.00 | 619.08 | 625.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 11:00:00 | 620.00 | 619.08 | 625.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 624.20 | 620.10 | 625.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 624.20 | 620.10 | 625.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 623.05 | 620.76 | 624.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 623.05 | 620.76 | 624.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 623.35 | 621.28 | 624.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:15:00 | 624.95 | 621.28 | 624.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 615.75 | 620.17 | 623.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 09:15:00 | 604.90 | 617.19 | 620.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 14:15:00 | 574.65 | 593.08 | 605.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-07 09:15:00 | 544.41 | 580.92 | 597.62 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 144 — BUY (started 2025-04-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 12:15:00 | 571.40 | 557.30 | 556.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 14:15:00 | 572.00 | 562.04 | 559.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 12:15:00 | 638.35 | 639.47 | 627.68 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 13:15:00 | 640.40 | 639.47 | 627.68 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 644.10 | 651.44 | 646.72 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-24 13:15:00 | 644.10 | 651.44 | 646.72 | SL hit (close<ema400) qty=1.00 sl=646.72 alert=retest1 |

### Cycle 145 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 623.15 | 640.89 | 643.00 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 13:15:00 | 639.45 | 637.74 | 637.71 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 09:15:00 | 629.65 | 636.29 | 637.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 10:15:00 | 624.10 | 633.86 | 635.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 14:15:00 | 626.00 | 625.71 | 630.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-30 14:45:00 | 623.45 | 625.71 | 630.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 634.15 | 624.49 | 626.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 634.15 | 624.49 | 626.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 632.80 | 626.16 | 627.14 | EMA400 retest candle locked (from downside) |

### Cycle 148 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 635.05 | 627.93 | 627.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 09:15:00 | 645.10 | 634.50 | 631.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 11:15:00 | 631.40 | 634.80 | 632.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 11:15:00 | 631.40 | 634.80 | 632.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 631.40 | 634.80 | 632.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 12:00:00 | 631.40 | 634.80 | 632.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 632.15 | 634.27 | 632.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 13:30:00 | 636.30 | 634.08 | 632.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 14:15:00 | 605.05 | 628.27 | 629.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 14:15:00 | 605.05 | 628.27 | 629.72 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 633.35 | 616.46 | 614.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 640.35 | 624.70 | 619.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 11:15:00 | 665.90 | 667.05 | 657.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 12:00:00 | 665.90 | 667.05 | 657.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 681.35 | 692.99 | 686.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 681.45 | 692.99 | 686.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 681.70 | 690.73 | 686.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 13:45:00 | 685.05 | 685.28 | 684.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 13:15:00 | 693.00 | 696.46 | 696.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 693.00 | 696.46 | 696.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 12:15:00 | 686.20 | 693.03 | 694.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 10:15:00 | 691.25 | 690.11 | 692.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 10:15:00 | 691.25 | 690.11 | 692.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 691.25 | 690.11 | 692.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:00:00 | 691.25 | 690.11 | 692.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 698.50 | 691.79 | 693.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:00:00 | 698.50 | 691.79 | 693.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 697.65 | 692.96 | 693.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:45:00 | 697.35 | 692.96 | 693.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 13:15:00 | 699.60 | 694.29 | 694.01 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 688.70 | 693.13 | 693.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 11:15:00 | 687.70 | 692.04 | 693.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 14:15:00 | 689.00 | 688.70 | 691.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 14:15:00 | 689.00 | 688.70 | 691.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 689.00 | 688.70 | 691.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:45:00 | 690.15 | 688.70 | 691.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 686.10 | 681.66 | 684.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:30:00 | 685.50 | 681.66 | 684.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 686.70 | 682.67 | 685.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:00:00 | 686.70 | 682.67 | 685.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 683.35 | 682.81 | 684.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:30:00 | 685.80 | 682.81 | 684.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 683.20 | 682.89 | 684.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 12:30:00 | 687.90 | 682.89 | 684.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 683.35 | 682.98 | 684.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:30:00 | 684.50 | 682.98 | 684.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 685.30 | 682.10 | 683.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:00:00 | 685.30 | 682.10 | 683.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 686.50 | 682.98 | 683.92 | EMA400 retest candle locked (from downside) |

### Cycle 154 — BUY (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 13:15:00 | 685.75 | 684.42 | 684.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 14:15:00 | 689.10 | 685.36 | 684.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 09:15:00 | 681.95 | 691.17 | 689.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 681.95 | 691.17 | 689.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 681.95 | 691.17 | 689.14 | EMA400 retest candle locked (from upside) |

### Cycle 155 — SELL (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 11:15:00 | 681.90 | 687.36 | 687.64 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 14:15:00 | 695.10 | 687.62 | 686.82 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-06-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 11:15:00 | 686.10 | 688.92 | 689.05 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2025-06-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 12:15:00 | 692.40 | 689.62 | 689.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 14:15:00 | 696.00 | 690.72 | 689.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 687.35 | 690.58 | 690.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 687.35 | 690.58 | 690.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 687.35 | 690.58 | 690.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:00:00 | 687.35 | 690.58 | 690.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 685.30 | 689.53 | 689.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 682.55 | 688.13 | 688.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 674.95 | 673.73 | 678.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:15:00 | 672.95 | 673.73 | 678.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 679.50 | 674.05 | 677.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 679.50 | 674.05 | 677.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 680.65 | 675.37 | 677.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 680.65 | 675.37 | 677.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — BUY (started 2025-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 15:15:00 | 682.50 | 679.44 | 679.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 14:15:00 | 694.80 | 683.99 | 681.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 11:15:00 | 686.70 | 687.21 | 684.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 11:30:00 | 687.00 | 687.21 | 684.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 683.20 | 686.41 | 684.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:45:00 | 683.10 | 686.41 | 684.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 680.40 | 685.21 | 683.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 14:00:00 | 680.40 | 685.21 | 683.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 682.05 | 684.36 | 683.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 09:15:00 | 687.40 | 684.36 | 683.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 12:15:00 | 672.80 | 682.12 | 682.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 672.80 | 682.12 | 682.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 14:15:00 | 666.20 | 677.56 | 680.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 684.60 | 676.46 | 679.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 10:15:00 | 684.60 | 676.46 | 679.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 684.60 | 676.46 | 679.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 684.60 | 676.46 | 679.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 687.90 | 678.75 | 679.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 687.90 | 678.75 | 679.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — BUY (started 2025-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 13:15:00 | 688.40 | 682.13 | 681.36 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 11:15:00 | 674.40 | 681.95 | 682.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 09:15:00 | 670.45 | 676.87 | 679.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 12:15:00 | 676.35 | 675.37 | 678.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-26 13:00:00 | 676.35 | 675.37 | 678.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 686.45 | 677.10 | 677.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:00:00 | 686.45 | 677.10 | 677.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 681.00 | 677.88 | 678.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 11:15:00 | 680.30 | 677.88 | 678.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 12:45:00 | 680.45 | 678.22 | 678.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 13:15:00 | 679.25 | 678.43 | 678.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — BUY (started 2025-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 13:15:00 | 679.25 | 678.43 | 678.42 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 14:15:00 | 671.10 | 676.96 | 677.75 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-06-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 15:15:00 | 681.10 | 677.71 | 677.60 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 676.40 | 677.57 | 677.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 12:15:00 | 674.75 | 677.00 | 677.32 | Break + close below crossover candle low |

### Cycle 168 — BUY (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 13:15:00 | 683.00 | 678.20 | 677.84 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 673.70 | 677.41 | 677.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 11:15:00 | 667.65 | 675.46 | 676.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 672.00 | 670.57 | 673.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 10:00:00 | 672.00 | 670.57 | 673.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 674.40 | 671.34 | 673.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:00:00 | 674.40 | 671.34 | 673.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 673.75 | 671.82 | 673.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:45:00 | 671.95 | 671.66 | 673.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 12:15:00 | 669.95 | 672.02 | 672.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 14:15:00 | 677.45 | 673.54 | 673.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — BUY (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 14:15:00 | 677.45 | 673.54 | 673.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 11:15:00 | 681.50 | 676.08 | 674.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 670.30 | 676.77 | 675.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 09:15:00 | 670.30 | 676.77 | 675.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 670.30 | 676.77 | 675.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 670.30 | 676.77 | 675.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 666.50 | 674.72 | 674.88 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 13:15:00 | 676.55 | 673.05 | 672.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 15:15:00 | 677.35 | 674.55 | 673.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 11:15:00 | 673.50 | 674.78 | 674.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 11:15:00 | 673.50 | 674.78 | 674.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 673.50 | 674.78 | 674.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:15:00 | 672.05 | 674.78 | 674.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 670.85 | 673.99 | 673.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:00:00 | 670.85 | 673.99 | 673.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2025-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 13:15:00 | 667.35 | 672.66 | 673.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 15:15:00 | 666.50 | 670.71 | 672.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 671.90 | 670.95 | 672.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 671.90 | 670.95 | 672.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 671.90 | 670.95 | 672.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 670.85 | 670.95 | 672.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 675.15 | 671.79 | 672.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:45:00 | 675.35 | 671.79 | 672.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 670.10 | 671.45 | 672.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 12:15:00 | 668.35 | 671.45 | 672.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 13:15:00 | 675.40 | 672.16 | 672.39 | SL hit (close>static) qty=1.00 sl=675.20 alert=retest2 |

### Cycle 174 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 674.25 | 672.58 | 672.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 13:15:00 | 677.50 | 674.55 | 673.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 675.00 | 675.34 | 674.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 675.00 | 675.34 | 674.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 675.00 | 675.34 | 674.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 673.05 | 675.34 | 674.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 676.65 | 675.60 | 674.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:30:00 | 674.10 | 675.60 | 674.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 675.15 | 675.93 | 674.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:30:00 | 675.00 | 675.93 | 674.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 677.25 | 676.19 | 675.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:30:00 | 676.05 | 676.19 | 675.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 675.00 | 675.95 | 675.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 15:00:00 | 675.00 | 675.95 | 675.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 675.50 | 675.86 | 675.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 683.80 | 675.86 | 675.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 672.90 | 682.36 | 679.88 | SL hit (close<static) qty=1.00 sl=674.55 alert=retest2 |

### Cycle 175 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 672.00 | 678.54 | 678.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 14:15:00 | 668.05 | 675.40 | 677.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 674.25 | 673.83 | 676.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 10:00:00 | 674.25 | 673.83 | 676.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 677.70 | 674.60 | 676.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:30:00 | 677.65 | 674.60 | 676.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 677.30 | 675.14 | 676.26 | EMA400 retest candle locked (from downside) |

### Cycle 176 — BUY (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 15:15:00 | 679.95 | 677.33 | 677.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 10:15:00 | 682.40 | 679.01 | 677.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 10:15:00 | 681.95 | 682.76 | 680.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-23 10:30:00 | 682.25 | 682.76 | 680.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 673.30 | 680.80 | 680.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:00:00 | 673.30 | 680.80 | 680.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 670.15 | 678.67 | 679.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 666.00 | 676.14 | 678.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 14:15:00 | 679.30 | 674.21 | 676.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 14:15:00 | 679.30 | 674.21 | 676.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 679.30 | 674.21 | 676.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:45:00 | 672.60 | 674.21 | 676.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 682.00 | 675.76 | 677.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 677.40 | 675.76 | 677.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 664.70 | 659.62 | 661.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:45:00 | 654.30 | 659.63 | 661.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 13:15:00 | 664.85 | 662.04 | 662.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — BUY (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 13:15:00 | 664.85 | 662.04 | 662.01 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 660.55 | 661.74 | 661.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 659.80 | 661.15 | 661.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 12:15:00 | 659.90 | 659.77 | 660.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-01 13:00:00 | 659.90 | 659.77 | 660.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 660.50 | 659.92 | 660.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:45:00 | 661.00 | 659.92 | 660.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 652.05 | 658.34 | 659.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 10:45:00 | 649.05 | 655.18 | 657.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 12:15:00 | 661.95 | 656.25 | 657.93 | SL hit (close>static) qty=1.00 sl=661.00 alert=retest2 |

### Cycle 180 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 663.35 | 659.17 | 658.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 09:15:00 | 669.55 | 661.25 | 659.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 10:15:00 | 675.10 | 678.59 | 674.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-07 11:00:00 | 675.10 | 678.59 | 674.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 679.30 | 678.73 | 674.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:30:00 | 675.30 | 678.73 | 674.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 677.00 | 678.38 | 675.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 12:30:00 | 676.15 | 678.38 | 675.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 677.70 | 678.25 | 675.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:45:00 | 674.15 | 678.25 | 675.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 678.20 | 679.12 | 676.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:00:00 | 678.20 | 679.12 | 676.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 676.95 | 678.68 | 676.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:45:00 | 675.95 | 678.68 | 676.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 672.80 | 677.51 | 676.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:00:00 | 672.80 | 677.51 | 676.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 671.75 | 676.35 | 675.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:30:00 | 668.10 | 676.35 | 675.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — SELL (started 2025-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 13:15:00 | 668.45 | 674.77 | 675.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 666.10 | 673.04 | 674.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 661.20 | 660.17 | 663.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 661.20 | 660.17 | 663.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 661.20 | 660.17 | 663.38 | EMA400 retest candle locked (from downside) |

### Cycle 182 — BUY (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 13:15:00 | 670.75 | 665.54 | 665.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 09:15:00 | 671.80 | 668.16 | 666.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 11:15:00 | 668.00 | 668.34 | 666.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 11:15:00 | 668.00 | 668.34 | 666.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 668.00 | 668.34 | 666.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:45:00 | 667.80 | 668.34 | 666.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 666.30 | 667.93 | 666.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:00:00 | 666.30 | 667.93 | 666.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 666.25 | 667.60 | 666.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:15:00 | 665.20 | 667.60 | 666.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 666.30 | 667.34 | 666.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:30:00 | 664.85 | 667.34 | 666.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 665.45 | 666.96 | 666.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 676.05 | 666.96 | 666.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 14:15:00 | 671.15 | 678.40 | 678.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — SELL (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 14:15:00 | 671.15 | 678.40 | 678.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 662.00 | 673.75 | 676.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 09:15:00 | 675.75 | 668.34 | 671.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 09:15:00 | 675.75 | 668.34 | 671.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 675.75 | 668.34 | 671.44 | EMA400 retest candle locked (from downside) |

### Cycle 184 — BUY (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 11:15:00 | 695.45 | 676.27 | 674.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 09:15:00 | 720.10 | 693.83 | 684.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 736.50 | 737.66 | 725.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:00:00 | 736.50 | 737.66 | 725.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 740.55 | 740.99 | 737.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:30:00 | 737.90 | 740.99 | 737.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 735.85 | 739.96 | 737.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:00:00 | 735.85 | 739.96 | 737.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 738.00 | 739.57 | 737.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:30:00 | 742.70 | 740.14 | 737.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 10:15:00 | 735.35 | 741.73 | 740.68 | SL hit (close<static) qty=1.00 sl=735.40 alert=retest2 |

### Cycle 185 — SELL (started 2025-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 12:15:00 | 737.00 | 739.63 | 739.84 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 757.15 | 743.09 | 741.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 760.65 | 746.60 | 743.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 11:15:00 | 790.05 | 790.40 | 785.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 12:00:00 | 790.05 | 790.40 | 785.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 784.00 | 789.12 | 785.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 13:00:00 | 784.00 | 789.12 | 785.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 786.80 | 788.65 | 785.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 14:15:00 | 788.65 | 788.65 | 785.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 10:15:00 | 778.95 | 788.09 | 788.04 | SL hit (close<static) qty=1.00 sl=783.15 alert=retest2 |

### Cycle 187 — SELL (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 11:15:00 | 776.60 | 785.79 | 787.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 770.00 | 776.29 | 779.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 09:15:00 | 775.65 | 775.32 | 778.14 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 11:00:00 | 770.25 | 774.30 | 777.42 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 772.95 | 771.91 | 775.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:45:00 | 775.50 | 771.91 | 775.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 774.00 | 771.63 | 773.65 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-24 12:15:00 | 774.00 | 771.63 | 773.65 | SL hit (close>ema400) qty=1.00 sl=773.65 alert=retest1 |

### Cycle 188 — BUY (started 2025-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 15:15:00 | 748.35 | 743.33 | 742.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 10:15:00 | 750.30 | 745.85 | 744.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 14:15:00 | 747.15 | 747.34 | 745.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 15:00:00 | 747.15 | 747.34 | 745.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 742.80 | 746.60 | 745.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 742.80 | 746.60 | 745.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 742.05 | 745.69 | 745.39 | EMA400 retest candle locked (from upside) |

### Cycle 189 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 739.30 | 744.41 | 744.84 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 09:15:00 | 754.75 | 745.88 | 745.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 10:15:00 | 757.95 | 748.29 | 746.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 09:15:00 | 754.60 | 756.28 | 752.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 754.60 | 756.28 | 752.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 754.60 | 756.28 | 752.10 | EMA400 retest candle locked (from upside) |

### Cycle 191 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 740.90 | 749.65 | 750.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 11:15:00 | 740.25 | 747.77 | 749.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 14:15:00 | 747.10 | 746.86 | 748.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-13 14:45:00 | 747.00 | 746.86 | 748.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 749.80 | 747.61 | 748.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:45:00 | 749.25 | 747.61 | 748.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 742.30 | 746.55 | 748.15 | EMA400 retest candle locked (from downside) |

### Cycle 192 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 753.90 | 747.96 | 747.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 13:15:00 | 758.00 | 751.63 | 749.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 757.55 | 759.24 | 756.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 757.55 | 759.24 | 756.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 757.55 | 759.24 | 756.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 757.95 | 759.24 | 756.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 753.40 | 758.30 | 756.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 753.40 | 758.30 | 756.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 754.10 | 757.46 | 756.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 751.70 | 757.46 | 756.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 757.60 | 757.49 | 756.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 760.60 | 757.49 | 756.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 10:15:00 | 737.80 | 753.55 | 754.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — SELL (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 10:15:00 | 737.80 | 753.55 | 754.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 723.35 | 730.16 | 736.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 727.80 | 727.70 | 733.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 10:00:00 | 727.80 | 727.70 | 733.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 723.20 | 726.97 | 730.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 10:30:00 | 720.85 | 725.83 | 729.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 11:00:00 | 721.30 | 725.83 | 729.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 12:00:00 | 720.05 | 724.68 | 728.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 13:00:00 | 721.20 | 723.98 | 728.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 729.20 | 724.44 | 726.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 729.20 | 724.44 | 726.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 730.35 | 725.62 | 727.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:30:00 | 732.35 | 725.62 | 727.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 730.95 | 726.69 | 727.54 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-29 14:15:00 | 753.35 | 733.04 | 730.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — BUY (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 14:15:00 | 753.35 | 733.04 | 730.31 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 12:15:00 | 730.20 | 732.88 | 732.96 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 14:15:00 | 735.45 | 733.50 | 733.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 12:15:00 | 743.70 | 737.45 | 735.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 14:15:00 | 747.05 | 747.27 | 742.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 15:00:00 | 747.05 | 747.27 | 742.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 740.70 | 746.07 | 743.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:15:00 | 735.05 | 746.07 | 743.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 735.55 | 743.97 | 742.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 734.40 | 743.97 | 742.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — SELL (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 12:15:00 | 735.00 | 740.30 | 740.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 15:15:00 | 730.30 | 736.22 | 738.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 735.15 | 733.82 | 736.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:00:00 | 735.15 | 733.82 | 736.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 731.75 | 733.23 | 735.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 730.40 | 733.56 | 735.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 13:15:00 | 738.00 | 734.37 | 735.18 | SL hit (close>static) qty=1.00 sl=736.55 alert=retest2 |

### Cycle 198 — BUY (started 2025-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 15:15:00 | 738.70 | 735.85 | 735.75 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 733.40 | 735.27 | 735.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 11:15:00 | 732.85 | 734.79 | 735.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 10:15:00 | 734.00 | 732.90 | 733.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 10:15:00 | 734.00 | 732.90 | 733.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 734.00 | 732.90 | 733.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:00:00 | 734.00 | 732.90 | 733.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 732.00 | 732.72 | 733.71 | EMA400 retest candle locked (from downside) |

### Cycle 200 — BUY (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 14:15:00 | 740.85 | 734.50 | 734.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 10:15:00 | 748.00 | 738.89 | 736.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 12:15:00 | 743.85 | 745.05 | 742.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 13:00:00 | 743.85 | 745.05 | 742.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 741.40 | 744.32 | 742.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:45:00 | 742.35 | 744.32 | 742.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 743.65 | 744.19 | 742.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:15:00 | 744.00 | 744.19 | 742.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 12:00:00 | 745.85 | 744.01 | 742.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 13:15:00 | 740.95 | 743.48 | 743.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — SELL (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 13:15:00 | 740.95 | 743.48 | 743.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 733.50 | 741.48 | 742.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 09:15:00 | 722.90 | 722.40 | 726.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 09:30:00 | 722.05 | 722.40 | 726.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 693.10 | 687.29 | 692.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:45:00 | 693.00 | 687.29 | 692.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 689.50 | 687.73 | 692.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 14:15:00 | 688.50 | 688.15 | 692.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 654.07 | 662.65 | 664.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 14:15:00 | 661.90 | 660.90 | 662.89 | SL hit (close>ema200) qty=0.50 sl=660.90 alert=retest2 |

### Cycle 202 — BUY (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 13:15:00 | 666.05 | 657.92 | 657.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 14:15:00 | 668.90 | 660.12 | 658.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 13:15:00 | 667.50 | 668.57 | 664.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 14:00:00 | 667.50 | 668.57 | 664.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 666.05 | 668.06 | 664.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:45:00 | 666.25 | 668.06 | 664.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 663.65 | 667.18 | 664.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 667.50 | 667.18 | 664.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:45:00 | 666.80 | 666.51 | 664.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 12:00:00 | 666.30 | 666.47 | 664.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 14:15:00 | 666.50 | 666.28 | 665.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 667.35 | 666.49 | 665.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:45:00 | 665.50 | 666.49 | 665.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 665.00 | 666.19 | 665.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:30:00 | 669.95 | 666.46 | 665.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 10:15:00 | 670.70 | 666.46 | 665.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 661.45 | 666.77 | 666.56 | SL hit (close<static) qty=1.00 sl=663.40 alert=retest2 |

### Cycle 203 — SELL (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 10:15:00 | 661.90 | 665.80 | 666.13 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 12:15:00 | 669.00 | 666.50 | 666.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 14:15:00 | 670.70 | 667.79 | 667.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 664.85 | 667.36 | 666.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 664.85 | 667.36 | 666.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 664.85 | 667.36 | 666.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 664.65 | 667.36 | 666.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 666.50 | 667.18 | 666.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:30:00 | 665.10 | 667.18 | 666.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 668.65 | 667.48 | 667.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:30:00 | 665.00 | 667.48 | 667.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 671.20 | 668.63 | 667.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 14:45:00 | 673.50 | 669.72 | 668.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 688.30 | 670.04 | 668.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 14:15:00 | 672.00 | 673.88 | 671.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 15:15:00 | 667.00 | 672.01 | 671.07 | SL hit (close<static) qty=1.00 sl=667.60 alert=retest2 |

### Cycle 205 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 658.50 | 669.31 | 669.92 | EMA200 below EMA400 |

### Cycle 206 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 670.95 | 666.90 | 666.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 674.60 | 668.44 | 667.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 14:15:00 | 670.75 | 671.86 | 669.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 14:15:00 | 670.75 | 671.86 | 669.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 670.75 | 671.86 | 669.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 14:45:00 | 671.30 | 671.86 | 669.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 669.20 | 671.21 | 669.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 11:30:00 | 672.05 | 671.05 | 670.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 14:15:00 | 666.35 | 669.05 | 669.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 207 — SELL (started 2025-12-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 14:15:00 | 666.35 | 669.05 | 669.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 10:15:00 | 661.40 | 666.40 | 667.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 662.70 | 662.41 | 664.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 10:30:00 | 663.20 | 662.41 | 664.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 648.10 | 644.40 | 648.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:00:00 | 648.10 | 644.40 | 648.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 651.60 | 645.84 | 648.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:00:00 | 651.60 | 645.84 | 648.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 650.45 | 646.76 | 648.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 15:00:00 | 647.95 | 647.00 | 648.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:30:00 | 647.10 | 646.06 | 647.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 15:15:00 | 651.20 | 645.31 | 644.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — BUY (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 15:15:00 | 651.20 | 645.31 | 644.89 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 11:15:00 | 641.80 | 644.56 | 644.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 12:15:00 | 637.40 | 641.70 | 643.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 640.80 | 639.80 | 641.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 640.80 | 639.80 | 641.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 640.80 | 639.80 | 641.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:15:00 | 638.60 | 639.80 | 641.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 639.25 | 639.69 | 641.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 11:15:00 | 636.70 | 639.69 | 641.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 12:45:00 | 636.55 | 638.84 | 640.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 657.60 | 641.87 | 641.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 210 — BUY (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 09:15:00 | 657.60 | 641.87 | 641.36 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 14:15:00 | 614.90 | 637.42 | 639.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 592.90 | 611.89 | 624.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 590.05 | 589.53 | 600.86 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 11:30:00 | 584.60 | 587.48 | 597.94 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 598.95 | 574.29 | 578.37 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 598.95 | 574.29 | 578.37 | SL hit (close>ema400) qty=1.00 sl=578.37 alert=retest1 |

### Cycle 212 — BUY (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 11:15:00 | 589.60 | 581.16 | 581.02 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 575.60 | 581.22 | 581.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 569.20 | 578.02 | 580.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 573.35 | 571.27 | 575.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 13:00:00 | 573.35 | 571.27 | 575.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 574.85 | 571.25 | 574.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 15:00:00 | 574.85 | 571.25 | 574.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 575.80 | 572.16 | 574.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 579.00 | 572.16 | 574.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 574.30 | 571.39 | 572.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 15:00:00 | 574.30 | 571.39 | 572.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 574.00 | 571.91 | 573.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:15:00 | 569.75 | 571.91 | 573.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 558.00 | 545.10 | 552.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:45:00 | 558.00 | 545.10 | 552.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 570.15 | 550.11 | 553.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 570.15 | 550.11 | 553.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 214 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 577.05 | 559.33 | 557.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 581.65 | 566.62 | 561.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 11:15:00 | 579.80 | 583.98 | 576.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 12:00:00 | 579.80 | 583.98 | 576.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 577.40 | 582.66 | 576.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 13:30:00 | 580.95 | 582.19 | 577.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 12:30:00 | 586.75 | 593.59 | 585.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-03 09:15:00 | 639.05 | 611.03 | 599.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 215 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 11:15:00 | 678.70 | 680.00 | 680.14 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 12:15:00 | 682.50 | 680.50 | 680.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 09:15:00 | 688.80 | 682.58 | 681.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 10:15:00 | 685.65 | 687.37 | 685.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 10:15:00 | 685.65 | 687.37 | 685.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 685.65 | 687.37 | 685.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:00:00 | 685.65 | 687.37 | 685.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 685.25 | 686.95 | 685.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:45:00 | 685.80 | 686.95 | 685.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 686.00 | 686.76 | 685.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 14:30:00 | 689.00 | 687.26 | 685.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 11:15:00 | 707.10 | 718.33 | 719.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 217 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 707.10 | 718.33 | 719.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 704.40 | 715.55 | 718.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 11:15:00 | 691.00 | 689.42 | 697.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 12:00:00 | 691.00 | 689.42 | 697.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 699.60 | 692.46 | 697.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 699.60 | 692.46 | 697.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 700.70 | 694.11 | 697.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 707.25 | 694.11 | 697.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 218 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 711.70 | 700.22 | 699.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 11:15:00 | 720.65 | 704.30 | 701.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 692.40 | 708.31 | 705.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 692.40 | 708.31 | 705.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 692.40 | 708.31 | 705.61 | EMA400 retest candle locked (from upside) |

### Cycle 219 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 692.40 | 702.18 | 703.11 | EMA200 below EMA400 |

### Cycle 220 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 720.80 | 705.17 | 703.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 12:15:00 | 725.55 | 711.62 | 706.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 727.30 | 727.38 | 718.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 13:00:00 | 727.30 | 727.38 | 718.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 711.85 | 723.82 | 719.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 708.45 | 723.82 | 719.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 724.90 | 724.03 | 720.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:30:00 | 729.65 | 724.40 | 720.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:15:00 | 726.80 | 724.40 | 720.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 708.30 | 721.93 | 722.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 221 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 708.30 | 721.93 | 722.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 694.95 | 711.83 | 717.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 698.45 | 698.00 | 707.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 701.80 | 698.00 | 707.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 710.50 | 700.56 | 705.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 710.50 | 700.56 | 705.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 703.45 | 701.13 | 705.69 | EMA400 retest candle locked (from downside) |

### Cycle 222 — BUY (started 2026-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 14:15:00 | 707.55 | 707.11 | 707.05 | EMA200 above EMA400 |

### Cycle 223 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 691.35 | 703.93 | 705.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 13:15:00 | 682.70 | 690.53 | 695.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 14:15:00 | 664.55 | 663.38 | 675.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-23 14:45:00 | 664.50 | 663.38 | 675.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 665.75 | 663.70 | 673.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 663.00 | 663.70 | 673.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 14:30:00 | 664.60 | 666.90 | 671.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 683.00 | 670.52 | 672.57 | SL hit (close>static) qty=1.00 sl=680.95 alert=retest2 |

### Cycle 224 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 683.65 | 675.54 | 674.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 687.80 | 677.99 | 675.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 675.90 | 682.93 | 679.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 675.90 | 682.93 | 679.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 675.90 | 682.93 | 679.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:45:00 | 674.75 | 682.93 | 679.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 673.05 | 680.95 | 678.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 671.10 | 680.95 | 678.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 225 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 673.85 | 677.38 | 677.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 668.85 | 675.68 | 676.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 685.95 | 665.85 | 668.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 685.95 | 665.85 | 668.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 685.95 | 665.85 | 668.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 687.95 | 665.85 | 668.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 226 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 684.45 | 671.04 | 670.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 689.55 | 674.74 | 672.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 660.95 | 674.00 | 673.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 660.95 | 674.00 | 673.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 660.95 | 674.00 | 673.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 660.95 | 674.00 | 673.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 227 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 660.85 | 671.37 | 672.04 | EMA200 below EMA400 |

### Cycle 228 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 681.20 | 673.34 | 672.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 688.40 | 680.18 | 676.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 676.10 | 680.66 | 677.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 676.10 | 680.66 | 677.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 676.10 | 680.66 | 677.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 12:30:00 | 684.65 | 681.89 | 678.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 09:15:00 | 753.12 | 743.93 | 734.81 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 229 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 817.45 | 826.37 | 826.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 12:15:00 | 809.05 | 818.70 | 822.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 819.00 | 816.44 | 820.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 819.00 | 816.44 | 820.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 819.00 | 816.44 | 820.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 810.55 | 816.21 | 819.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:30:00 | 812.65 | 811.21 | 815.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 831.00 | 817.05 | 817.13 | SL hit (close>static) qty=1.00 sl=825.00 alert=retest2 |

### Cycle 230 — BUY (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 13:15:00 | 828.00 | 819.24 | 818.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 11:15:00 | 838.10 | 826.22 | 822.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 14:15:00 | 826.65 | 828.44 | 824.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 14:15:00 | 826.65 | 828.44 | 824.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 826.65 | 828.44 | 824.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 09:15:00 | 849.85 | 827.55 | 824.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-30 11:30:00 | 378.40 | 2023-06-02 13:15:00 | 381.50 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2023-05-30 12:45:00 | 378.00 | 2023-06-02 13:15:00 | 381.50 | STOP_HIT | 1.00 | 0.93% |
| BUY | retest2 | 2023-06-28 15:00:00 | 374.85 | 2023-07-18 11:15:00 | 412.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-30 09:15:00 | 375.85 | 2023-07-18 11:15:00 | 413.44 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-07-24 10:15:00 | 406.90 | 2023-07-28 11:15:00 | 404.55 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2023-08-03 09:15:00 | 410.50 | 2023-08-07 12:15:00 | 405.35 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2023-08-03 10:15:00 | 408.80 | 2023-08-07 12:15:00 | 405.35 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2023-08-04 09:15:00 | 410.80 | 2023-08-07 12:15:00 | 405.35 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2023-08-04 14:30:00 | 408.70 | 2023-08-07 12:15:00 | 405.35 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2023-08-18 09:15:00 | 419.70 | 2023-08-23 14:15:00 | 417.65 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2023-08-28 09:30:00 | 414.70 | 2023-08-31 09:15:00 | 418.80 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2023-09-01 14:15:00 | 422.40 | 2023-09-12 10:15:00 | 439.95 | STOP_HIT | 1.00 | 4.15% |
| SELL | retest2 | 2023-09-14 09:15:00 | 440.60 | 2023-09-14 13:15:00 | 444.40 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2023-09-25 09:15:00 | 436.00 | 2023-09-25 09:15:00 | 435.00 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2023-09-29 11:00:00 | 441.10 | 2023-10-04 10:15:00 | 436.25 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2023-10-17 15:15:00 | 392.00 | 2023-10-19 12:15:00 | 372.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-18 11:00:00 | 392.35 | 2023-10-19 12:15:00 | 372.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-18 11:45:00 | 392.10 | 2023-10-19 12:15:00 | 372.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-18 12:45:00 | 390.65 | 2023-10-19 12:15:00 | 371.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-17 15:15:00 | 392.00 | 2023-10-19 13:15:00 | 384.75 | STOP_HIT | 0.50 | 1.85% |
| SELL | retest2 | 2023-10-18 11:00:00 | 392.35 | 2023-10-19 13:15:00 | 384.75 | STOP_HIT | 0.50 | 1.94% |
| SELL | retest2 | 2023-10-18 11:45:00 | 392.10 | 2023-10-19 13:15:00 | 384.75 | STOP_HIT | 0.50 | 1.87% |
| SELL | retest2 | 2023-10-18 12:45:00 | 390.65 | 2023-10-19 13:15:00 | 384.75 | STOP_HIT | 0.50 | 1.51% |
| SELL | retest2 | 2023-10-20 10:15:00 | 379.55 | 2023-10-20 14:15:00 | 392.50 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2023-11-02 11:15:00 | 383.45 | 2023-11-07 13:15:00 | 382.95 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2023-11-02 13:45:00 | 382.70 | 2023-11-07 13:15:00 | 382.95 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2023-11-02 14:30:00 | 383.35 | 2023-11-07 13:15:00 | 382.95 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2023-11-07 09:30:00 | 382.95 | 2023-11-07 13:15:00 | 382.95 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2023-12-08 13:30:00 | 447.25 | 2023-12-13 14:15:00 | 462.75 | STOP_HIT | 1.00 | -3.47% |
| SELL | retest2 | 2023-12-11 09:30:00 | 447.05 | 2023-12-13 14:15:00 | 462.75 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2023-12-12 09:30:00 | 446.75 | 2023-12-13 14:15:00 | 462.75 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest2 | 2023-12-12 12:30:00 | 448.00 | 2023-12-13 14:15:00 | 462.75 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2024-01-01 15:00:00 | 449.60 | 2024-01-03 12:15:00 | 454.50 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-01-02 09:30:00 | 450.10 | 2024-01-03 12:15:00 | 454.50 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-01-02 15:15:00 | 449.50 | 2024-01-03 12:15:00 | 454.50 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-01-09 09:15:00 | 475.30 | 2024-01-10 15:15:00 | 465.50 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2024-01-09 14:30:00 | 472.45 | 2024-01-10 15:15:00 | 465.50 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-01-10 09:30:00 | 471.20 | 2024-01-10 15:15:00 | 465.50 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-01-10 10:00:00 | 471.10 | 2024-01-10 15:15:00 | 465.50 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-01-12 15:15:00 | 460.30 | 2024-01-17 14:15:00 | 463.30 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-01-15 09:30:00 | 457.55 | 2024-01-17 14:15:00 | 463.30 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-01-19 11:00:00 | 454.35 | 2024-01-23 09:15:00 | 462.60 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-01-19 14:45:00 | 454.30 | 2024-01-23 09:15:00 | 462.60 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-01-20 09:30:00 | 454.10 | 2024-01-23 09:15:00 | 462.60 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2024-01-20 10:45:00 | 454.50 | 2024-01-23 09:15:00 | 462.60 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-01-25 10:45:00 | 448.30 | 2024-01-25 14:15:00 | 461.65 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2024-01-25 12:00:00 | 446.60 | 2024-01-25 14:15:00 | 461.65 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2024-01-25 13:00:00 | 448.00 | 2024-01-25 14:15:00 | 461.65 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2024-02-07 09:15:00 | 440.35 | 2024-02-14 15:15:00 | 438.55 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest2 | 2024-02-29 14:15:00 | 437.90 | 2024-03-01 09:15:00 | 481.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-18 09:15:00 | 472.70 | 2024-03-18 12:15:00 | 465.00 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-03-18 09:45:00 | 471.75 | 2024-03-18 12:15:00 | 465.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-03-18 10:30:00 | 473.50 | 2024-03-18 12:15:00 | 465.00 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2024-03-18 15:15:00 | 471.40 | 2024-03-19 12:15:00 | 465.50 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-04-08 12:45:00 | 507.55 | 2024-04-15 09:15:00 | 482.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-08 12:45:00 | 507.55 | 2024-04-15 14:15:00 | 495.20 | STOP_HIT | 0.50 | 2.43% |
| BUY | retest2 | 2024-04-24 09:15:00 | 539.60 | 2024-05-02 12:15:00 | 551.45 | STOP_HIT | 1.00 | 2.20% |
| BUY | retest2 | 2024-04-24 11:00:00 | 537.55 | 2024-05-02 12:15:00 | 551.45 | STOP_HIT | 1.00 | 2.59% |
| SELL | retest2 | 2024-05-03 10:45:00 | 550.85 | 2024-05-07 09:15:00 | 578.75 | STOP_HIT | 1.00 | -5.06% |
| SELL | retest2 | 2024-05-06 09:30:00 | 547.10 | 2024-05-07 09:15:00 | 578.75 | STOP_HIT | 1.00 | -5.79% |
| SELL | retest2 | 2024-05-06 11:45:00 | 549.80 | 2024-05-07 09:15:00 | 578.75 | STOP_HIT | 1.00 | -5.27% |
| SELL | retest2 | 2024-05-06 12:30:00 | 550.50 | 2024-05-07 09:15:00 | 578.75 | STOP_HIT | 1.00 | -5.13% |
| BUY | retest2 | 2024-05-13 12:15:00 | 597.95 | 2024-05-17 09:15:00 | 657.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-14 12:45:00 | 597.95 | 2024-05-17 09:15:00 | 657.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-14 11:30:00 | 674.00 | 2024-06-20 09:15:00 | 675.25 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2024-06-26 09:15:00 | 699.30 | 2024-06-26 09:15:00 | 686.10 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-07-16 14:15:00 | 726.65 | 2024-07-18 10:15:00 | 690.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-16 14:45:00 | 728.60 | 2024-07-18 10:15:00 | 692.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-18 09:15:00 | 715.65 | 2024-07-19 09:15:00 | 679.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-16 14:15:00 | 726.65 | 2024-07-19 11:15:00 | 653.99 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-16 14:45:00 | 728.60 | 2024-07-19 11:15:00 | 655.74 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-18 09:15:00 | 715.65 | 2024-07-19 15:15:00 | 676.15 | STOP_HIT | 0.50 | 5.52% |
| BUY | retest1 | 2024-07-29 09:15:00 | 738.35 | 2024-07-31 09:15:00 | 732.95 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-08-05 09:15:00 | 692.50 | 2024-08-09 11:15:00 | 704.90 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-08-12 14:00:00 | 704.55 | 2024-08-13 14:15:00 | 695.15 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-08-12 15:00:00 | 706.00 | 2024-08-13 14:15:00 | 695.15 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-08-13 11:30:00 | 704.50 | 2024-08-13 14:15:00 | 695.15 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-09-03 15:00:00 | 689.50 | 2024-09-11 13:15:00 | 679.40 | STOP_HIT | 1.00 | 1.46% |
| SELL | retest2 | 2024-09-04 12:30:00 | 688.55 | 2024-09-11 13:15:00 | 679.40 | STOP_HIT | 1.00 | 1.33% |
| SELL | retest2 | 2024-09-05 09:30:00 | 687.55 | 2024-09-11 13:15:00 | 679.40 | STOP_HIT | 1.00 | 1.19% |
| BUY | retest2 | 2024-09-17 09:15:00 | 720.65 | 2024-09-25 09:15:00 | 792.72 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-30 13:30:00 | 758.95 | 2024-10-04 09:15:00 | 721.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 13:30:00 | 758.95 | 2024-10-07 12:15:00 | 735.00 | STOP_HIT | 0.50 | 3.16% |
| BUY | retest2 | 2024-10-16 09:15:00 | 842.50 | 2024-10-16 14:15:00 | 834.15 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-10-28 12:00:00 | 722.85 | 2024-11-06 10:15:00 | 728.80 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-10-30 11:45:00 | 727.05 | 2024-11-06 10:15:00 | 728.80 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2024-10-30 13:30:00 | 727.45 | 2024-11-06 10:15:00 | 728.80 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-11-14 14:00:00 | 692.15 | 2024-11-18 11:15:00 | 709.85 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2024-11-22 11:00:00 | 718.40 | 2024-11-29 11:15:00 | 736.05 | STOP_HIT | 1.00 | 2.46% |
| BUY | retest2 | 2024-12-05 09:15:00 | 766.05 | 2024-12-12 09:15:00 | 784.10 | STOP_HIT | 1.00 | 2.36% |
| SELL | retest2 | 2024-12-16 11:45:00 | 762.70 | 2024-12-17 12:15:00 | 774.55 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-12-16 13:30:00 | 761.50 | 2024-12-17 12:15:00 | 774.55 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-12-19 09:15:00 | 753.20 | 2024-12-20 15:15:00 | 725.09 | PARTIAL | 0.50 | 3.73% |
| SELL | retest2 | 2024-12-19 11:45:00 | 763.25 | 2024-12-23 09:15:00 | 721.33 | PARTIAL | 0.50 | 5.49% |
| SELL | retest2 | 2024-12-20 10:00:00 | 759.30 | 2024-12-24 12:15:00 | 715.54 | PARTIAL | 0.50 | 5.76% |
| SELL | retest2 | 2024-12-19 09:15:00 | 753.20 | 2024-12-26 09:15:00 | 723.70 | STOP_HIT | 0.50 | 3.92% |
| SELL | retest2 | 2024-12-19 11:45:00 | 763.25 | 2024-12-26 09:15:00 | 723.70 | STOP_HIT | 0.50 | 5.18% |
| SELL | retest2 | 2024-12-20 10:00:00 | 759.30 | 2024-12-26 09:15:00 | 723.70 | STOP_HIT | 0.50 | 4.69% |
| SELL | retest2 | 2025-01-15 09:15:00 | 626.30 | 2025-01-17 10:15:00 | 639.15 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest1 | 2025-01-21 11:30:00 | 673.40 | 2025-01-21 14:15:00 | 658.50 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-01-23 11:30:00 | 638.35 | 2025-01-27 09:15:00 | 606.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 11:30:00 | 638.35 | 2025-01-28 09:15:00 | 574.51 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-07 10:45:00 | 621.65 | 2025-02-10 09:15:00 | 598.30 | STOP_HIT | 1.00 | -3.76% |
| SELL | retest2 | 2025-02-14 10:30:00 | 567.05 | 2025-02-17 09:15:00 | 538.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 10:30:00 | 567.05 | 2025-02-17 10:15:00 | 563.05 | STOP_HIT | 0.50 | 0.71% |
| BUY | retest2 | 2025-02-20 10:15:00 | 589.45 | 2025-02-24 10:15:00 | 579.50 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-02-21 10:30:00 | 587.95 | 2025-02-24 10:15:00 | 579.50 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-02-21 11:15:00 | 587.15 | 2025-02-24 10:15:00 | 579.50 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-02-21 13:45:00 | 586.70 | 2025-02-24 10:15:00 | 579.50 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-03-12 10:45:00 | 605.00 | 2025-03-13 10:15:00 | 612.10 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-03-12 15:15:00 | 604.40 | 2025-03-13 10:15:00 | 612.10 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-03-25 11:15:00 | 635.85 | 2025-03-28 11:15:00 | 645.25 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-03-26 09:30:00 | 635.10 | 2025-03-28 11:15:00 | 645.25 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-03-26 14:45:00 | 636.50 | 2025-03-28 11:15:00 | 645.25 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-03-27 12:00:00 | 639.45 | 2025-03-28 11:15:00 | 645.25 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-04-04 09:15:00 | 604.90 | 2025-04-04 14:15:00 | 574.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 09:15:00 | 604.90 | 2025-04-07 09:15:00 | 544.41 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-04-22 13:15:00 | 640.40 | 2025-04-24 13:15:00 | 644.10 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2025-05-06 13:30:00 | 636.30 | 2025-05-06 14:15:00 | 605.05 | STOP_HIT | 1.00 | -4.91% |
| BUY | retest2 | 2025-05-20 13:45:00 | 685.05 | 2025-05-27 13:15:00 | 693.00 | STOP_HIT | 1.00 | 1.16% |
| BUY | retest2 | 2025-06-19 09:15:00 | 687.40 | 2025-06-19 12:15:00 | 672.80 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-06-27 11:15:00 | 680.30 | 2025-06-27 13:15:00 | 679.25 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-06-27 12:45:00 | 680.45 | 2025-06-27 13:15:00 | 679.25 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-07-03 12:45:00 | 671.95 | 2025-07-04 14:15:00 | 677.45 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-07-04 12:15:00 | 669.95 | 2025-07-04 14:15:00 | 677.45 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-07-14 12:15:00 | 668.35 | 2025-07-14 13:15:00 | 675.40 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-07-17 09:15:00 | 683.80 | 2025-07-18 09:15:00 | 672.90 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-07-18 11:00:00 | 676.65 | 2025-07-18 12:15:00 | 672.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-07-31 09:45:00 | 654.30 | 2025-07-31 13:15:00 | 664.85 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-08-04 10:45:00 | 649.05 | 2025-08-04 12:15:00 | 661.95 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-08-18 09:15:00 | 676.05 | 2025-08-26 14:15:00 | 671.15 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-09-08 09:30:00 | 742.70 | 2025-09-09 10:15:00 | 735.35 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-09-16 14:15:00 | 788.65 | 2025-09-18 10:15:00 | 778.95 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest1 | 2025-09-23 11:00:00 | 770.25 | 2025-09-24 12:15:00 | 774.00 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-09-24 14:15:00 | 771.80 | 2025-10-03 15:15:00 | 748.35 | STOP_HIT | 1.00 | 3.04% |
| SELL | retest2 | 2025-09-25 10:00:00 | 770.00 | 2025-10-03 15:15:00 | 748.35 | STOP_HIT | 1.00 | 2.81% |
| BUY | retest2 | 2025-10-20 09:15:00 | 760.60 | 2025-10-20 10:15:00 | 737.80 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-10-28 10:30:00 | 720.85 | 2025-10-29 14:15:00 | 753.35 | STOP_HIT | 1.00 | -4.51% |
| SELL | retest2 | 2025-10-28 11:00:00 | 721.30 | 2025-10-29 14:15:00 | 753.35 | STOP_HIT | 1.00 | -4.44% |
| SELL | retest2 | 2025-10-28 12:00:00 | 720.05 | 2025-10-29 14:15:00 | 753.35 | STOP_HIT | 1.00 | -4.62% |
| SELL | retest2 | 2025-10-28 13:00:00 | 721.20 | 2025-10-29 14:15:00 | 753.35 | STOP_HIT | 1.00 | -4.46% |
| SELL | retest2 | 2025-11-10 09:15:00 | 730.40 | 2025-11-10 13:15:00 | 738.00 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-11-14 15:15:00 | 744.00 | 2025-11-18 13:15:00 | 740.95 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-11-17 12:00:00 | 745.85 | 2025-11-18 13:15:00 | 740.95 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-11-26 14:15:00 | 688.50 | 2025-12-05 09:15:00 | 654.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 14:15:00 | 688.50 | 2025-12-05 14:15:00 | 661.90 | STOP_HIT | 0.50 | 3.86% |
| BUY | retest2 | 2025-12-11 09:15:00 | 667.50 | 2025-12-15 09:15:00 | 661.45 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-12-11 10:45:00 | 666.80 | 2025-12-15 09:15:00 | 661.45 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-12-11 12:00:00 | 666.30 | 2025-12-15 10:15:00 | 661.90 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-12-11 14:15:00 | 666.50 | 2025-12-15 10:15:00 | 661.90 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-12-12 09:30:00 | 669.95 | 2025-12-15 10:15:00 | 661.90 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-12-12 10:15:00 | 670.70 | 2025-12-15 10:15:00 | 661.90 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-12-16 14:45:00 | 673.50 | 2025-12-17 15:15:00 | 667.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-12-17 09:15:00 | 688.30 | 2025-12-17 15:15:00 | 667.00 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2025-12-17 14:15:00 | 672.00 | 2025-12-17 15:15:00 | 667.00 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-12-23 11:30:00 | 672.05 | 2025-12-23 14:15:00 | 666.35 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-12-31 15:00:00 | 647.95 | 2026-01-02 15:15:00 | 651.20 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2026-01-01 09:30:00 | 647.10 | 2026-01-02 15:15:00 | 651.20 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2026-01-07 11:15:00 | 636.70 | 2026-01-08 09:15:00 | 657.60 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2026-01-07 12:45:00 | 636.55 | 2026-01-08 09:15:00 | 657.60 | STOP_HIT | 1.00 | -3.31% |
| SELL | retest1 | 2026-01-13 11:30:00 | 584.60 | 2026-01-19 09:15:00 | 598.95 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2026-01-30 13:30:00 | 580.95 | 2026-02-03 09:15:00 | 639.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-01 12:30:00 | 586.75 | 2026-02-03 09:15:00 | 645.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-17 14:30:00 | 689.00 | 2026-03-02 11:15:00 | 707.10 | STOP_HIT | 1.00 | 2.63% |
| BUY | retest2 | 2026-03-12 11:30:00 | 729.65 | 2026-03-13 12:15:00 | 708.30 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2026-03-12 12:15:00 | 726.80 | 2026-03-13 12:15:00 | 708.30 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2026-03-24 10:15:00 | 663.00 | 2026-03-25 09:15:00 | 683.00 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2026-03-24 14:30:00 | 664.60 | 2026-03-25 09:15:00 | 683.00 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2026-04-07 12:30:00 | 684.65 | 2026-04-16 09:15:00 | 753.12 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-05-04 13:15:00 | 810.55 | 2026-05-05 12:15:00 | 831.00 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2026-05-05 10:30:00 | 812.65 | 2026-05-05 12:15:00 | 831.00 | STOP_HIT | 1.00 | -2.26% |
