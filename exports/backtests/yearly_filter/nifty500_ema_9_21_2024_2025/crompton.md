# Crompton Greaves Consumer Electricals Ltd. (CROMPTON)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 293.55
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 151 |
| ALERT1 | 103 |
| ALERT2 | 101 |
| ALERT2_SKIP | 53 |
| ALERT3 | 266 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 97 |
| PARTIAL | 20 |
| TARGET_HIT | 3 |
| STOP_HIT | 98 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 121 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 63 / 58
- **Target hits / Stop hits / Partials:** 3 / 98 / 20
- **Avg / median % per leg:** 1.28% / 0.18%
- **Sum % (uncompounded):** 155.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 15 | 39.5% | 2 | 36 | 0 | 0.33% | 12.6% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.45% | -4.3% |
| BUY @ 3rd Alert (retest2) | 35 | 15 | 42.9% | 2 | 33 | 0 | 0.49% | 17.0% |
| SELL (all) | 83 | 48 | 57.8% | 1 | 62 | 20 | 1.72% | 142.4% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.41% | 8.8% |
| SELL @ 3rd Alert (retest2) | 81 | 46 | 56.8% | 1 | 61 | 19 | 1.65% | 133.6% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 4 | 1 | 0.89% | 4.5% |
| retest2 (combined) | 116 | 61 | 52.6% | 3 | 94 | 19 | 1.30% | 150.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 15:15:00 | 333.50 | 329.30 | 329.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 10:15:00 | 334.45 | 330.76 | 329.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 09:15:00 | 334.80 | 335.38 | 333.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 09:15:00 | 334.80 | 335.38 | 333.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 334.80 | 335.38 | 333.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 09:45:00 | 333.80 | 335.38 | 333.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 394.25 | 392.94 | 390.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:00:00 | 394.25 | 392.94 | 390.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 11:15:00 | 391.50 | 392.57 | 391.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 12:00:00 | 391.50 | 392.57 | 391.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 12:15:00 | 389.30 | 391.91 | 390.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 13:00:00 | 389.30 | 391.91 | 390.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 13:15:00 | 391.20 | 391.77 | 390.89 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2024-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 15:15:00 | 387.10 | 390.30 | 390.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 09:15:00 | 385.75 | 389.39 | 389.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 10:15:00 | 391.55 | 389.82 | 390.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 10:15:00 | 391.55 | 389.82 | 390.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 391.55 | 389.82 | 390.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 10:30:00 | 390.05 | 389.82 | 390.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 389.60 | 389.78 | 390.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 12:15:00 | 387.85 | 389.78 | 390.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 15:15:00 | 388.50 | 389.92 | 390.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 09:15:00 | 395.90 | 390.89 | 390.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 09:15:00 | 395.90 | 390.89 | 390.42 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 10:15:00 | 391.40 | 392.13 | 392.22 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 395.15 | 392.61 | 392.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 11:15:00 | 396.90 | 393.47 | 392.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 394.30 | 394.66 | 393.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 394.30 | 394.66 | 393.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 394.30 | 394.66 | 393.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 389.25 | 394.66 | 393.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 377.35 | 391.20 | 392.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 368.95 | 386.75 | 390.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 384.80 | 381.87 | 385.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 10:00:00 | 384.80 | 381.87 | 385.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 397.80 | 385.05 | 387.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 397.80 | 385.05 | 387.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 398.15 | 387.67 | 388.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:30:00 | 397.50 | 387.67 | 388.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 401.05 | 390.35 | 389.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 09:15:00 | 412.70 | 401.34 | 398.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 13:15:00 | 412.40 | 412.63 | 408.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 14:00:00 | 412.40 | 412.63 | 408.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 413.80 | 412.41 | 409.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 10:15:00 | 415.10 | 412.41 | 409.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 09:15:00 | 420.50 | 426.45 | 426.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 09:15:00 | 420.50 | 426.45 | 426.95 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 12:15:00 | 429.55 | 427.21 | 427.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 14:15:00 | 430.75 | 428.36 | 427.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 10:15:00 | 429.35 | 429.40 | 428.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 10:15:00 | 429.35 | 429.40 | 428.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 429.35 | 429.40 | 428.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 11:15:00 | 428.65 | 429.40 | 428.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 428.45 | 429.21 | 428.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 12:15:00 | 428.90 | 429.21 | 428.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 12:15:00 | 428.85 | 429.14 | 428.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 14:45:00 | 430.75 | 429.34 | 428.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 09:15:00 | 426.90 | 428.83 | 428.58 | SL hit (close<static) qty=1.00 sl=427.45 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 11:15:00 | 424.95 | 427.86 | 428.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 12:15:00 | 421.85 | 426.66 | 427.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 13:15:00 | 428.45 | 427.02 | 427.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 13:15:00 | 428.45 | 427.02 | 427.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 428.45 | 427.02 | 427.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 14:00:00 | 428.45 | 427.02 | 427.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 14:15:00 | 428.65 | 427.34 | 427.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 15:00:00 | 428.65 | 427.34 | 427.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 15:15:00 | 429.50 | 427.78 | 427.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 09:15:00 | 426.95 | 427.78 | 427.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-28 14:15:00 | 405.60 | 412.18 | 417.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-01 14:15:00 | 416.55 | 410.37 | 413.71 | SL hit (close>ema200) qty=0.50 sl=410.37 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 09:15:00 | 421.40 | 413.02 | 412.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 09:15:00 | 428.90 | 420.17 | 416.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 420.50 | 425.32 | 421.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 09:15:00 | 420.50 | 425.32 | 421.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 420.50 | 425.32 | 421.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 420.50 | 425.32 | 421.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 418.90 | 424.03 | 421.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 418.35 | 424.03 | 421.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 12:15:00 | 426.70 | 423.77 | 421.83 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2024-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 12:15:00 | 418.55 | 421.03 | 421.27 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 15:15:00 | 425.00 | 422.09 | 421.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 09:15:00 | 428.75 | 423.42 | 422.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 09:15:00 | 429.75 | 430.35 | 427.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 09:30:00 | 429.40 | 430.35 | 427.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 11:15:00 | 430.85 | 432.10 | 430.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 11:45:00 | 429.85 | 432.10 | 430.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 431.40 | 431.96 | 430.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 13:15:00 | 431.15 | 431.96 | 430.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 432.20 | 432.01 | 430.44 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 422.50 | 429.63 | 429.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 10:15:00 | 417.90 | 427.29 | 428.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 12:15:00 | 430.30 | 427.59 | 428.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 12:15:00 | 430.30 | 427.59 | 428.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 430.30 | 427.59 | 428.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 13:00:00 | 430.30 | 427.59 | 428.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 430.00 | 428.07 | 428.66 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2024-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 15:15:00 | 430.50 | 429.19 | 429.10 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 426.30 | 428.61 | 428.85 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 12:15:00 | 431.55 | 428.81 | 428.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 13:15:00 | 433.10 | 429.67 | 428.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 09:15:00 | 426.50 | 430.36 | 429.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 09:15:00 | 426.50 | 430.36 | 429.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 426.50 | 430.36 | 429.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 10:00:00 | 426.50 | 430.36 | 429.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 429.75 | 430.24 | 429.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 11:15:00 | 430.55 | 430.24 | 429.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 12:15:00 | 426.25 | 429.16 | 429.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 12:15:00 | 426.25 | 429.16 | 429.18 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 14:15:00 | 430.85 | 429.26 | 429.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 09:15:00 | 433.20 | 430.17 | 429.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 15:15:00 | 443.80 | 444.04 | 439.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 09:15:00 | 437.50 | 444.04 | 439.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 444.20 | 444.07 | 440.10 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2024-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 10:15:00 | 435.90 | 439.88 | 439.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-29 12:15:00 | 428.95 | 437.03 | 438.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-29 15:15:00 | 435.00 | 434.64 | 436.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-30 09:15:00 | 456.75 | 434.64 | 436.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 450.55 | 437.82 | 438.12 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2024-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 10:15:00 | 448.45 | 439.95 | 439.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-01 09:15:00 | 465.20 | 450.28 | 446.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 12:15:00 | 451.90 | 452.21 | 448.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-01 13:00:00 | 451.90 | 452.21 | 448.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 449.80 | 451.42 | 448.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 15:15:00 | 453.05 | 451.42 | 448.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 09:15:00 | 447.00 | 450.80 | 448.75 | SL hit (close<static) qty=1.00 sl=448.30 alert=retest2 |

### Cycle 22 — SELL (started 2024-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 12:15:00 | 439.95 | 446.87 | 447.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 14:15:00 | 438.65 | 444.21 | 445.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 427.45 | 423.46 | 429.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 10:00:00 | 427.45 | 423.46 | 429.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 431.35 | 425.04 | 429.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:45:00 | 431.30 | 425.04 | 429.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 431.70 | 426.37 | 429.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:30:00 | 432.65 | 426.37 | 429.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 15:15:00 | 435.50 | 431.06 | 431.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 10:15:00 | 438.00 | 432.55 | 431.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 433.20 | 433.77 | 432.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 15:00:00 | 433.20 | 433.77 | 432.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 432.40 | 433.49 | 432.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 435.85 | 433.49 | 432.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 09:15:00 | 429.90 | 434.60 | 434.17 | SL hit (close<static) qty=1.00 sl=431.80 alert=retest2 |

### Cycle 24 — SELL (started 2024-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 12:15:00 | 432.20 | 433.91 | 433.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 14:15:00 | 430.90 | 433.25 | 433.64 | Break + close below crossover candle low |

### Cycle 25 — BUY (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 09:15:00 | 437.20 | 433.94 | 433.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-13 10:15:00 | 440.70 | 435.29 | 434.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 13:15:00 | 435.60 | 436.49 | 435.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 13:15:00 | 435.60 | 436.49 | 435.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 435.60 | 436.49 | 435.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 435.60 | 436.49 | 435.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 436.35 | 436.46 | 435.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:30:00 | 435.95 | 436.46 | 435.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 435.00 | 436.17 | 435.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:15:00 | 432.65 | 436.17 | 435.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2024-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 09:15:00 | 429.75 | 434.88 | 434.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 11:15:00 | 427.55 | 432.48 | 433.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 435.75 | 431.20 | 432.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 435.75 | 431.20 | 432.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 435.75 | 431.20 | 432.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:45:00 | 435.50 | 431.20 | 432.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 435.70 | 432.10 | 432.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 11:00:00 | 435.70 | 432.10 | 432.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 438.85 | 433.45 | 433.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 445.15 | 439.01 | 436.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 11:15:00 | 464.55 | 465.25 | 459.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 11:45:00 | 463.95 | 465.25 | 459.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 461.40 | 462.82 | 460.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 465.60 | 461.13 | 460.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 15:15:00 | 457.65 | 464.40 | 465.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 15:15:00 | 457.65 | 464.40 | 465.08 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 471.60 | 464.95 | 464.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 10:15:00 | 475.50 | 467.06 | 465.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 10:15:00 | 471.85 | 473.65 | 470.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 11:00:00 | 471.85 | 473.65 | 470.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 471.30 | 473.18 | 470.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:45:00 | 471.65 | 473.18 | 470.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 469.40 | 472.42 | 470.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:45:00 | 470.00 | 472.42 | 470.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 465.35 | 471.01 | 470.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 14:00:00 | 465.35 | 471.01 | 470.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 15:15:00 | 466.55 | 469.18 | 469.30 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 09:15:00 | 472.60 | 469.86 | 469.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 10:15:00 | 475.15 | 470.92 | 470.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 14:15:00 | 469.30 | 471.66 | 470.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 14:15:00 | 469.30 | 471.66 | 470.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 469.30 | 471.66 | 470.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 469.30 | 471.66 | 470.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 469.25 | 471.17 | 470.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 472.20 | 471.17 | 470.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 10:15:00 | 468.50 | 470.29 | 470.35 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 11:15:00 | 470.65 | 469.96 | 469.91 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 12:15:00 | 469.55 | 469.88 | 469.88 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 13:15:00 | 470.60 | 470.02 | 469.95 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 14:15:00 | 468.80 | 469.78 | 469.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 10:15:00 | 466.40 | 468.61 | 469.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 11:15:00 | 469.10 | 468.71 | 469.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 11:15:00 | 469.10 | 468.71 | 469.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 469.10 | 468.71 | 469.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 12:00:00 | 469.10 | 468.71 | 469.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 464.35 | 467.83 | 468.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 14:30:00 | 462.30 | 466.40 | 467.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 09:15:00 | 460.95 | 465.84 | 467.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 11:15:00 | 461.40 | 465.08 | 466.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 09:15:00 | 470.05 | 463.65 | 464.91 | SL hit (close>static) qty=1.00 sl=469.05 alert=retest2 |

### Cycle 37 — BUY (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 11:15:00 | 468.55 | 465.71 | 465.70 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 12:15:00 | 461.95 | 465.44 | 465.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 13:15:00 | 459.60 | 464.27 | 465.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 09:15:00 | 462.60 | 462.33 | 464.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 09:15:00 | 462.60 | 462.33 | 464.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 462.60 | 462.33 | 464.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 11:15:00 | 461.05 | 462.39 | 463.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 10:15:00 | 438.00 | 443.66 | 445.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-25 10:15:00 | 442.70 | 440.83 | 442.73 | SL hit (close>ema200) qty=0.50 sl=440.83 alert=retest2 |

### Cycle 39 — BUY (started 2024-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 11:15:00 | 430.05 | 424.87 | 424.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 15:15:00 | 433.00 | 428.87 | 426.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 13:15:00 | 428.35 | 430.47 | 428.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 13:15:00 | 428.35 | 430.47 | 428.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 428.35 | 430.47 | 428.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 14:00:00 | 428.35 | 430.47 | 428.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 427.35 | 429.84 | 428.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 15:15:00 | 427.00 | 429.84 | 428.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 15:15:00 | 427.00 | 429.27 | 428.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:15:00 | 425.60 | 429.27 | 428.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 428.55 | 429.21 | 428.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:30:00 | 429.15 | 429.21 | 428.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 428.05 | 428.97 | 428.49 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2024-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 13:15:00 | 423.85 | 427.45 | 427.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 14:15:00 | 421.95 | 426.35 | 427.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 13:15:00 | 424.45 | 424.26 | 425.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-07 13:45:00 | 424.30 | 424.26 | 425.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 14:15:00 | 424.85 | 424.38 | 425.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-07 14:45:00 | 425.40 | 424.38 | 425.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 15:15:00 | 425.25 | 424.55 | 425.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 09:15:00 | 424.95 | 424.55 | 425.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 427.30 | 425.10 | 425.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 09:45:00 | 428.75 | 425.10 | 425.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2024-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 10:15:00 | 431.70 | 426.42 | 426.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 13:15:00 | 433.85 | 429.23 | 427.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-08 15:15:00 | 429.30 | 429.42 | 428.05 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 09:30:00 | 434.90 | 430.78 | 428.79 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 430.85 | 437.38 | 435.73 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-11 10:15:00 | 430.85 | 437.38 | 435.73 | SL hit (close<ema400) qty=1.00 sl=435.73 alert=retest1 |

### Cycle 42 — SELL (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 13:15:00 | 429.15 | 434.10 | 434.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 09:15:00 | 425.65 | 431.67 | 433.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 14:15:00 | 429.60 | 428.32 | 430.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-14 15:00:00 | 429.60 | 428.32 | 430.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 15:15:00 | 430.15 | 428.68 | 430.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 09:15:00 | 427.95 | 428.68 | 430.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 424.65 | 427.88 | 430.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 11:30:00 | 421.50 | 427.03 | 428.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 09:45:00 | 423.65 | 423.66 | 426.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:15:00 | 402.47 | 406.24 | 410.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-22 09:15:00 | 406.90 | 406.24 | 410.23 | SL hit (close>static) qty=0.50 sl=406.24 alert=retest2 |

### Cycle 43 — BUY (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 14:15:00 | 393.85 | 391.35 | 391.06 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 384.40 | 391.12 | 391.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 09:15:00 | 379.50 | 384.95 | 387.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 383.90 | 382.27 | 385.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 13:30:00 | 382.85 | 382.27 | 385.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 385.55 | 382.92 | 385.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 385.55 | 382.92 | 385.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 384.90 | 383.32 | 385.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:15:00 | 388.25 | 383.32 | 385.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 390.85 | 384.83 | 385.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:00:00 | 390.85 | 384.83 | 385.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 392.50 | 386.36 | 386.41 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 11:15:00 | 397.35 | 388.56 | 387.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 12:15:00 | 397.55 | 390.36 | 388.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 393.80 | 395.59 | 391.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:00:00 | 393.80 | 395.59 | 391.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 392.90 | 395.05 | 391.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:45:00 | 392.35 | 395.05 | 391.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 391.30 | 394.30 | 391.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:45:00 | 391.50 | 394.30 | 391.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 390.85 | 393.61 | 391.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:15:00 | 389.65 | 393.61 | 391.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 390.35 | 392.96 | 391.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:30:00 | 389.75 | 392.96 | 391.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 391.00 | 392.02 | 391.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 389.30 | 392.02 | 391.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 399.75 | 396.45 | 394.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 10:30:00 | 405.55 | 397.55 | 395.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 15:15:00 | 390.20 | 394.18 | 394.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2024-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 15:15:00 | 390.20 | 394.18 | 394.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 15:15:00 | 389.10 | 391.02 | 392.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 09:15:00 | 395.00 | 380.71 | 383.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-18 09:15:00 | 395.00 | 380.71 | 383.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 395.00 | 380.71 | 383.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 10:00:00 | 395.00 | 380.71 | 383.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 10:15:00 | 393.35 | 383.24 | 384.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 11:15:00 | 396.30 | 383.24 | 384.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 15:15:00 | 384.00 | 384.44 | 384.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:15:00 | 379.65 | 384.44 | 384.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 386.60 | 384.87 | 384.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 386.60 | 384.87 | 384.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 390.20 | 385.93 | 385.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 11:15:00 | 392.15 | 387.18 | 385.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 09:15:00 | 387.90 | 389.08 | 387.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 09:15:00 | 387.90 | 389.08 | 387.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 387.90 | 389.08 | 387.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:30:00 | 385.85 | 389.08 | 387.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 388.00 | 388.87 | 387.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:45:00 | 386.35 | 388.87 | 387.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 11:15:00 | 386.60 | 388.41 | 387.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 12:00:00 | 386.60 | 388.41 | 387.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 12:15:00 | 385.80 | 387.89 | 387.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 13:00:00 | 385.80 | 387.89 | 387.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2024-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 13:15:00 | 381.65 | 386.64 | 386.88 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 12:15:00 | 389.95 | 386.32 | 386.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 14:15:00 | 393.05 | 388.28 | 387.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 10:15:00 | 404.45 | 406.12 | 402.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 11:00:00 | 404.45 | 406.12 | 402.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 406.65 | 407.72 | 405.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 406.65 | 407.72 | 405.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 406.05 | 407.39 | 405.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:15:00 | 405.45 | 407.39 | 405.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 405.25 | 406.96 | 405.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:30:00 | 403.35 | 406.96 | 405.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 405.80 | 406.73 | 405.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 11:15:00 | 407.20 | 406.73 | 405.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 12:45:00 | 406.95 | 406.67 | 405.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 13:15:00 | 407.80 | 406.67 | 405.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 11:00:00 | 407.00 | 412.07 | 410.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 14:15:00 | 408.55 | 410.06 | 410.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2024-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 14:15:00 | 408.55 | 410.06 | 410.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 15:15:00 | 408.00 | 409.30 | 409.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 15:15:00 | 406.20 | 405.97 | 407.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-06 09:15:00 | 407.65 | 405.97 | 407.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 405.70 | 405.92 | 407.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 10:45:00 | 404.10 | 405.44 | 406.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 09:15:00 | 411.60 | 406.92 | 406.96 | SL hit (close>static) qty=1.00 sl=408.65 alert=retest2 |

### Cycle 51 — BUY (started 2024-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 10:15:00 | 413.55 | 408.25 | 407.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 11:15:00 | 416.35 | 409.87 | 408.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 09:15:00 | 413.65 | 414.12 | 412.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-11 10:00:00 | 413.65 | 414.12 | 412.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 413.95 | 414.27 | 413.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 15:00:00 | 413.95 | 414.27 | 413.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 411.20 | 413.78 | 413.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:00:00 | 411.20 | 413.78 | 413.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 410.65 | 413.15 | 412.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 11:00:00 | 410.65 | 413.15 | 412.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 11:15:00 | 407.55 | 412.03 | 412.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 12:15:00 | 407.20 | 411.06 | 412.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 13:15:00 | 408.40 | 407.26 | 408.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 14:00:00 | 408.40 | 407.26 | 408.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 410.95 | 408.00 | 409.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 410.95 | 408.00 | 409.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 412.00 | 408.80 | 409.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 413.25 | 408.80 | 409.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 406.55 | 408.70 | 409.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 14:15:00 | 406.10 | 407.73 | 408.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:15:00 | 404.20 | 407.26 | 408.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 09:15:00 | 385.80 | 391.55 | 394.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 09:15:00 | 383.99 | 391.55 | 394.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-23 10:15:00 | 392.35 | 391.71 | 394.09 | SL hit (close>ema200) qty=0.50 sl=391.71 alert=retest2 |

### Cycle 53 — BUY (started 2024-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 14:15:00 | 395.35 | 393.57 | 393.56 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 10:15:00 | 391.65 | 393.22 | 393.42 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 397.70 | 393.91 | 393.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 09:15:00 | 399.40 | 397.04 | 395.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 13:15:00 | 393.85 | 396.88 | 396.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 13:15:00 | 393.85 | 396.88 | 396.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 393.85 | 396.88 | 396.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 14:00:00 | 393.85 | 396.88 | 396.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 394.10 | 396.33 | 395.91 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2024-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 09:15:00 | 393.10 | 395.61 | 395.66 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 10:15:00 | 396.05 | 395.70 | 395.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 11:15:00 | 397.05 | 395.97 | 395.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 12:15:00 | 395.30 | 395.84 | 395.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 12:15:00 | 395.30 | 395.84 | 395.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 395.30 | 395.84 | 395.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 12:45:00 | 394.50 | 395.84 | 395.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 398.25 | 396.32 | 395.99 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 09:15:00 | 387.90 | 394.46 | 395.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-01 11:15:00 | 382.15 | 390.88 | 393.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 11:15:00 | 373.35 | 373.13 | 378.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-03 12:00:00 | 373.35 | 373.13 | 378.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 376.45 | 371.43 | 375.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 10:45:00 | 371.25 | 371.22 | 374.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 12:15:00 | 371.45 | 371.39 | 374.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 15:00:00 | 370.50 | 372.40 | 374.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 10:15:00 | 371.85 | 369.62 | 371.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 370.75 | 369.84 | 371.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 11:30:00 | 370.00 | 369.32 | 370.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 14:30:00 | 369.85 | 369.28 | 370.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:30:00 | 369.30 | 369.31 | 370.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 13:15:00 | 369.60 | 369.14 | 369.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 13:15:00 | 369.55 | 369.22 | 369.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 365.60 | 369.73 | 369.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 352.69 | 360.95 | 364.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 352.88 | 360.95 | 364.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 351.97 | 360.95 | 364.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 353.26 | 360.95 | 364.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 351.50 | 360.95 | 364.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 351.36 | 360.95 | 364.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 350.83 | 360.95 | 364.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 351.12 | 360.95 | 364.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 10:15:00 | 356.75 | 354.48 | 358.30 | SL hit (close>ema200) qty=0.50 sl=354.48 alert=retest2 |

### Cycle 59 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 365.55 | 360.01 | 359.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 11:15:00 | 365.75 | 362.48 | 361.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 13:15:00 | 360.95 | 362.29 | 361.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 13:15:00 | 360.95 | 362.29 | 361.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 360.95 | 362.29 | 361.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:00:00 | 360.95 | 362.29 | 361.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 362.55 | 362.34 | 361.90 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 09:15:00 | 357.10 | 361.37 | 361.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 11:15:00 | 356.15 | 359.70 | 360.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 12:15:00 | 360.90 | 359.94 | 360.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 12:15:00 | 360.90 | 359.94 | 360.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 12:15:00 | 360.90 | 359.94 | 360.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 13:00:00 | 360.90 | 359.94 | 360.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 364.60 | 360.87 | 361.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:00:00 | 364.60 | 360.87 | 361.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 362.45 | 361.19 | 361.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 15:15:00 | 362.10 | 361.19 | 361.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 15:15:00 | 362.10 | 361.37 | 361.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 15:15:00 | 362.10 | 361.37 | 361.29 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 09:15:00 | 357.50 | 360.59 | 360.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 10:15:00 | 356.45 | 359.77 | 360.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 349.80 | 348.71 | 352.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:00:00 | 349.80 | 348.71 | 352.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 350.35 | 349.04 | 352.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 351.20 | 349.04 | 352.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 352.25 | 349.79 | 351.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 12:30:00 | 353.00 | 349.79 | 351.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 352.00 | 350.23 | 351.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:30:00 | 349.65 | 350.38 | 351.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:15:00 | 350.10 | 350.45 | 351.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 332.60 | 342.09 | 346.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-27 14:15:00 | 338.50 | 338.45 | 342.49 | SL hit (close>ema200) qty=0.50 sl=338.45 alert=retest2 |

### Cycle 63 — BUY (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 09:15:00 | 349.15 | 342.04 | 341.75 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-01-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 15:15:00 | 342.35 | 343.43 | 343.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 10:15:00 | 341.95 | 343.09 | 343.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 11:15:00 | 345.20 | 343.51 | 343.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 345.20 | 343.51 | 343.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 345.20 | 343.51 | 343.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:00:00 | 345.20 | 343.51 | 343.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 12:15:00 | 362.00 | 347.21 | 345.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 13:15:00 | 363.25 | 350.42 | 346.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-04 09:15:00 | 363.95 | 366.06 | 360.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-04 10:00:00 | 363.95 | 366.06 | 360.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 361.95 | 365.24 | 360.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:45:00 | 359.95 | 365.24 | 360.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 359.60 | 364.11 | 360.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 12:00:00 | 359.60 | 364.11 | 360.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 12:15:00 | 359.10 | 363.11 | 360.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 14:45:00 | 361.55 | 362.10 | 360.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 09:45:00 | 361.20 | 363.83 | 363.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 10:15:00 | 357.00 | 362.47 | 363.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 357.00 | 362.47 | 363.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 352.95 | 360.56 | 362.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 343.40 | 340.99 | 345.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 343.40 | 340.99 | 345.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 343.40 | 340.99 | 345.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:45:00 | 333.20 | 339.27 | 342.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-20 15:15:00 | 336.00 | 331.60 | 331.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 15:15:00 | 336.00 | 331.60 | 331.16 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 330.50 | 331.62 | 331.70 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 13:15:00 | 333.30 | 331.51 | 331.46 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 329.30 | 331.08 | 331.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 10:15:00 | 327.30 | 330.32 | 330.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 11:15:00 | 322.45 | 321.30 | 323.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 12:00:00 | 322.45 | 321.30 | 323.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 327.95 | 322.63 | 324.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:00:00 | 327.95 | 322.63 | 324.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 326.70 | 323.44 | 324.52 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2025-03-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 15:15:00 | 328.45 | 325.18 | 325.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 10:15:00 | 329.00 | 326.09 | 325.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-04 12:15:00 | 326.00 | 326.12 | 325.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 12:15:00 | 326.00 | 326.12 | 325.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 12:15:00 | 326.00 | 326.12 | 325.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-04 13:00:00 | 326.00 | 326.12 | 325.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 13:15:00 | 327.10 | 326.32 | 325.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-04 14:45:00 | 327.50 | 326.45 | 325.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-04 15:15:00 | 329.80 | 326.45 | 325.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 15:15:00 | 344.00 | 348.16 | 348.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 15:15:00 | 344.00 | 348.16 | 348.51 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 10:15:00 | 349.80 | 348.77 | 348.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 355.15 | 352.07 | 350.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 357.40 | 359.27 | 356.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 357.40 | 359.27 | 356.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 12:15:00 | 357.45 | 358.63 | 357.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 12:30:00 | 357.60 | 358.63 | 357.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 13:15:00 | 357.25 | 358.36 | 357.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 14:30:00 | 358.20 | 358.13 | 357.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 15:15:00 | 359.00 | 358.13 | 357.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 10:45:00 | 357.90 | 360.74 | 359.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 09:30:00 | 360.10 | 361.32 | 360.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 360.95 | 361.25 | 360.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:30:00 | 358.50 | 361.25 | 360.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 358.60 | 360.72 | 360.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 356.80 | 360.72 | 360.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-25 12:15:00 | 357.80 | 360.13 | 360.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 357.80 | 360.13 | 360.25 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 10:15:00 | 363.30 | 360.33 | 360.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-26 11:15:00 | 364.60 | 361.19 | 360.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-26 13:15:00 | 361.70 | 362.13 | 361.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-26 13:45:00 | 362.30 | 362.13 | 361.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 360.35 | 361.78 | 361.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 15:00:00 | 360.35 | 361.78 | 361.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 15:15:00 | 359.35 | 361.29 | 360.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:15:00 | 357.00 | 361.29 | 360.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2025-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 09:15:00 | 353.40 | 359.71 | 360.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 11:15:00 | 350.95 | 357.08 | 358.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 11:15:00 | 357.05 | 355.68 | 357.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 11:15:00 | 357.05 | 355.68 | 357.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 11:15:00 | 357.05 | 355.68 | 357.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 12:00:00 | 357.05 | 355.68 | 357.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 352.90 | 355.13 | 356.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:00:00 | 349.00 | 353.68 | 355.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 09:15:00 | 331.55 | 335.03 | 339.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-07 09:15:00 | 314.10 | 328.55 | 333.75 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 77 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 335.00 | 331.45 | 331.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 338.20 | 334.14 | 332.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 09:15:00 | 335.20 | 336.24 | 334.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-16 09:15:00 | 335.20 | 336.24 | 334.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 09:15:00 | 335.20 | 336.24 | 334.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 10:00:00 | 335.20 | 336.24 | 334.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 10:15:00 | 335.95 | 336.18 | 334.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 11:00:00 | 335.95 | 336.18 | 334.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 12:15:00 | 335.50 | 335.95 | 335.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 13:00:00 | 335.50 | 335.95 | 335.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 13:15:00 | 335.75 | 335.91 | 335.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 13:30:00 | 335.05 | 335.91 | 335.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 14:15:00 | 335.05 | 335.74 | 335.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 14:30:00 | 335.00 | 335.74 | 335.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 15:15:00 | 335.00 | 335.59 | 335.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:15:00 | 335.30 | 335.59 | 335.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 335.25 | 335.52 | 335.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:00:00 | 336.40 | 335.75 | 335.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-17 14:15:00 | 332.20 | 334.95 | 335.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-17 14:15:00 | 332.20 | 334.95 | 335.02 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 12:15:00 | 337.55 | 334.87 | 334.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 14:15:00 | 338.55 | 335.87 | 335.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 341.05 | 346.98 | 343.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-23 09:15:00 | 341.05 | 346.98 | 343.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 341.05 | 346.98 | 343.04 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2025-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 11:15:00 | 337.65 | 341.94 | 342.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 336.80 | 340.71 | 341.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 09:15:00 | 335.45 | 333.98 | 335.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 335.45 | 333.98 | 335.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 335.45 | 333.98 | 335.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:30:00 | 335.45 | 333.98 | 335.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 338.45 | 334.88 | 335.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 11:00:00 | 338.45 | 334.88 | 335.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 337.00 | 335.30 | 335.95 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2025-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 15:15:00 | 337.30 | 336.39 | 336.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 09:15:00 | 340.00 | 337.11 | 336.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 13:15:00 | 337.10 | 337.64 | 337.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 13:15:00 | 337.10 | 337.64 | 337.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 13:15:00 | 337.10 | 337.64 | 337.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 14:00:00 | 337.10 | 337.64 | 337.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 14:15:00 | 337.10 | 337.54 | 337.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 15:15:00 | 336.85 | 337.54 | 337.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 336.85 | 337.40 | 337.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:15:00 | 333.10 | 337.40 | 337.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 09:15:00 | 332.90 | 336.50 | 336.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 10:15:00 | 327.45 | 334.69 | 335.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 329.50 | 327.66 | 331.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-05 09:30:00 | 329.20 | 327.66 | 331.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 13:15:00 | 332.65 | 328.90 | 330.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:00:00 | 332.65 | 328.90 | 330.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 335.00 | 330.12 | 331.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:45:00 | 335.00 | 330.12 | 331.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 329.70 | 330.89 | 331.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:15:00 | 328.05 | 330.89 | 331.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 13:15:00 | 329.20 | 330.52 | 331.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 14:00:00 | 328.55 | 330.12 | 330.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 11:15:00 | 332.60 | 330.01 | 330.35 | SL hit (close>static) qty=1.00 sl=332.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 13:15:00 | 333.00 | 330.99 | 330.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 15:15:00 | 334.00 | 331.98 | 331.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 10:15:00 | 332.20 | 332.35 | 331.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 10:45:00 | 332.40 | 332.35 | 331.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 11:15:00 | 333.20 | 332.52 | 331.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 11:30:00 | 334.05 | 332.52 | 331.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 12:15:00 | 330.60 | 332.13 | 331.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:00:00 | 330.60 | 332.13 | 331.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 328.25 | 331.36 | 331.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:45:00 | 328.20 | 331.36 | 331.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 326.65 | 330.42 | 330.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 325.00 | 329.33 | 330.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 15:15:00 | 323.25 | 323.23 | 326.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-12 09:15:00 | 329.05 | 323.23 | 326.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 331.40 | 324.87 | 326.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:15:00 | 333.85 | 324.87 | 326.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2025-05-12 11:15:00)

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

### Cycle 86 — SELL (started 2025-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 13:15:00 | 329.65 | 330.45 | 330.54 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 15:15:00 | 331.80 | 330.84 | 330.71 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 09:15:00 | 327.00 | 330.07 | 330.37 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-05-16 09:15:00)

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

### Cycle 90 — SELL (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 13:15:00 | 345.60 | 346.60 | 346.74 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 348.55 | 346.81 | 346.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 10:15:00 | 349.95 | 347.44 | 347.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 354.55 | 355.06 | 352.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 10:00:00 | 354.55 | 355.06 | 352.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 353.75 | 354.80 | 352.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:00:00 | 353.75 | 354.80 | 352.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 357.80 | 355.40 | 353.18 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 352.60 | 353.63 | 353.65 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 13:15:00 | 358.80 | 354.56 | 353.99 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-05-30 13:15:00)

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

### Cycle 95 — BUY (started 2025-06-06 09:15:00)

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

### Cycle 96 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 352.80 | 353.20 | 353.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 348.05 | 352.11 | 352.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 344.05 | 344.04 | 346.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 344.05 | 344.04 | 346.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 344.55 | 344.13 | 346.22 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2025-06-16 15:15:00)

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

### Cycle 98 — SELL (started 2025-06-18 13:15:00)

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

### Cycle 99 — BUY (started 2025-06-23 13:15:00)

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

### Cycle 100 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 350.60 | 354.38 | 354.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 347.50 | 351.44 | 353.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 348.45 | 347.88 | 349.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 348.45 | 347.88 | 349.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 348.45 | 347.88 | 349.96 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 10:15:00 | 355.30 | 351.33 | 350.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 15:15:00 | 356.00 | 353.73 | 352.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 11:15:00 | 351.85 | 353.73 | 352.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 11:15:00 | 351.85 | 353.73 | 352.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 351.85 | 353.73 | 352.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:45:00 | 352.00 | 353.73 | 352.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 352.10 | 353.41 | 352.67 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 346.90 | 351.30 | 351.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 13:15:00 | 343.60 | 348.05 | 349.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 10:15:00 | 344.20 | 343.85 | 345.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 10:45:00 | 344.00 | 343.85 | 345.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 339.65 | 338.43 | 339.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 342.00 | 338.43 | 339.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 342.20 | 339.19 | 340.10 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2025-07-15 12:15:00)

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

### Cycle 104 — SELL (started 2025-07-18 13:15:00)

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

### Cycle 105 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 326.60 | 324.37 | 324.21 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-08-05 09:15:00)

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

### Cycle 107 — BUY (started 2025-08-14 09:15:00)

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

### Cycle 108 — SELL (started 2025-08-21 12:15:00)

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

### Cycle 109 — BUY (started 2025-08-26 09:15:00)

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

### Cycle 110 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 329.60 | 331.99 | 332.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 326.65 | 330.52 | 331.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 328.45 | 328.37 | 329.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 13:45:00 | 328.45 | 328.37 | 329.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 328.30 | 328.11 | 329.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 323.75 | 327.17 | 328.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 12:15:00 | 317.55 | 315.69 | 315.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 12:15:00 | 317.55 | 315.69 | 315.48 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 12:15:00 | 314.30 | 315.63 | 315.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 09:15:00 | 311.90 | 314.33 | 315.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 14:15:00 | 299.70 | 299.56 | 302.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 14:30:00 | 300.15 | 299.56 | 302.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 290.40 | 291.14 | 293.05 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 10:15:00 | 294.85 | 293.18 | 293.15 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-10-06 10:15:00)

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

### Cycle 115 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 289.80 | 287.03 | 286.97 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-10-13 10:15:00)

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

### Cycle 117 — BUY (started 2025-10-15 12:15:00)

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

### Cycle 118 — SELL (started 2025-10-28 10:15:00)

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

### Cycle 119 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 282.45 | 280.17 | 279.99 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 275.10 | 279.49 | 279.82 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 12:15:00 | 280.05 | 279.38 | 279.29 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-11-14 09:15:00)

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

### Cycle 123 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 269.00 | 267.47 | 267.40 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-11-27 12:15:00)

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

### Cycle 125 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 260.60 | 260.28 | 260.25 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-12-05 15:15:00)

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

### Cycle 127 — BUY (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 15:15:00 | 255.10 | 252.71 | 252.56 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 09:15:00 | 251.95 | 252.85 | 252.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 10:15:00 | 251.25 | 252.53 | 252.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 09:15:00 | 258.25 | 252.02 | 252.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 09:15:00 | 258.25 | 252.02 | 252.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 258.25 | 252.02 | 252.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:00:00 | 258.25 | 252.02 | 252.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 10:15:00 | 263.05 | 254.23 | 253.14 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 12:15:00 | 251.60 | 253.50 | 253.76 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-12-19 15:15:00)

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

### Cycle 132 — SELL (started 2025-12-26 14:15:00)

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

### Cycle 133 — BUY (started 2026-01-05 10:15:00)

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

### Cycle 134 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 257.50 | 259.14 | 259.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 255.95 | 258.50 | 258.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 251.35 | 251.11 | 253.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 252.65 | 251.11 | 253.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 249.50 | 250.73 | 252.78 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 11:15:00 | 254.45 | 252.73 | 252.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 255.85 | 253.35 | 252.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 12:15:00 | 251.50 | 253.81 | 253.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 12:15:00 | 251.50 | 253.81 | 253.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 251.50 | 253.81 | 253.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:00:00 | 251.50 | 253.81 | 253.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2026-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 13:15:00 | 250.40 | 253.13 | 253.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 249.60 | 251.91 | 252.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 233.55 | 233.24 | 237.21 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 10:45:00 | 232.25 | 232.93 | 236.71 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 12:15:00 | 220.64 | 224.33 | 227.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-27 15:15:00 | 223.40 | 223.21 | 226.42 | SL hit (close>ema200) qty=0.50 sl=223.21 alert=retest1 |

### Cycle 137 — BUY (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 09:15:00 | 230.29 | 223.28 | 223.22 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 220.14 | 223.61 | 223.90 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 232.56 | 225.40 | 224.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 237.80 | 229.43 | 226.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 243.30 | 244.60 | 240.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 254.78 | 246.85 | 243.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 254.78 | 246.85 | 243.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 10:15:00 | 256.38 | 246.85 | 243.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 14:15:00 | 265.58 | 268.66 | 269.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2026-02-19 14:15:00)

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

### Cycle 141 — BUY (started 2026-03-11 10:15:00)

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

### Cycle 142 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 240.50 | 246.72 | 246.97 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2026-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 09:15:00 | 251.40 | 246.17 | 245.83 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-03-19 10:15:00)

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

### Cycle 145 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 243.75 | 238.98 | 238.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 245.10 | 240.93 | 239.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 235.65 | 240.73 | 240.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 235.65 | 240.73 | 240.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 235.65 | 240.73 | 240.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 235.65 | 240.73 | 240.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2026-03-27 10:15:00)

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

### Cycle 147 — BUY (started 2026-04-06 12:15:00)

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

### Cycle 148 — SELL (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 11:15:00 | 238.70 | 240.01 | 240.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 14:15:00 | 237.62 | 239.34 | 239.69 | Break + close below crossover candle low |

### Cycle 149 — BUY (started 2026-04-15 09:15:00)

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

### Cycle 150 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 253.30 | 258.59 | 258.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 10:15:00 | 252.30 | 257.33 | 258.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 252.82 | 251.31 | 253.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 252.82 | 251.31 | 253.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 252.82 | 251.31 | 253.35 | EMA400 retest candle locked (from downside) |

### Cycle 151 — BUY (started 2026-04-27 13:15:00)

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
| SELL | retest2 | 2024-05-28 12:15:00 | 387.85 | 2024-05-29 09:15:00 | 395.90 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-05-28 15:15:00 | 388.50 | 2024-05-29 09:15:00 | 395.90 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-06-12 10:15:00 | 415.10 | 2024-06-24 09:15:00 | 420.50 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2024-06-25 14:45:00 | 430.75 | 2024-06-26 09:15:00 | 426.90 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-06-27 09:15:00 | 426.95 | 2024-06-28 14:15:00 | 405.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-27 09:15:00 | 426.95 | 2024-07-01 14:15:00 | 416.55 | STOP_HIT | 0.50 | 2.44% |
| BUY | retest2 | 2024-07-23 11:15:00 | 430.55 | 2024-07-23 12:15:00 | 426.25 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-08-01 15:15:00 | 453.05 | 2024-08-02 09:15:00 | 447.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-08-09 09:15:00 | 435.85 | 2024-08-12 09:15:00 | 429.90 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-08-12 11:00:00 | 434.50 | 2024-08-12 12:15:00 | 432.20 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-08-26 09:15:00 | 465.60 | 2024-08-28 15:15:00 | 457.65 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-09-06 14:30:00 | 462.30 | 2024-09-10 09:15:00 | 470.05 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-09-09 09:15:00 | 460.95 | 2024-09-10 09:15:00 | 470.05 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2024-09-09 11:15:00 | 461.40 | 2024-09-10 09:15:00 | 470.05 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2024-09-12 11:15:00 | 461.05 | 2024-09-24 10:15:00 | 438.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-12 11:15:00 | 461.05 | 2024-09-25 10:15:00 | 442.70 | STOP_HIT | 0.50 | 3.98% |
| BUY | retest1 | 2024-10-09 09:30:00 | 434.90 | 2024-10-11 10:15:00 | 430.85 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-10-16 11:30:00 | 421.50 | 2024-10-22 09:15:00 | 402.47 | PARTIAL | 0.50 | 4.52% |
| SELL | retest2 | 2024-10-16 11:30:00 | 421.50 | 2024-10-22 09:15:00 | 406.90 | STOP_HIT | 0.50 | 3.46% |
| SELL | retest2 | 2024-10-17 09:45:00 | 423.65 | 2024-10-22 12:15:00 | 400.42 | PARTIAL | 0.50 | 5.48% |
| SELL | retest2 | 2024-10-17 09:45:00 | 423.65 | 2024-10-23 10:15:00 | 402.90 | STOP_HIT | 0.50 | 4.90% |
| BUY | retest2 | 2024-11-11 10:30:00 | 405.55 | 2024-11-11 15:15:00 | 390.20 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest2 | 2024-11-29 11:15:00 | 407.20 | 2024-12-03 14:15:00 | 408.55 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2024-11-29 12:45:00 | 406.95 | 2024-12-03 14:15:00 | 408.55 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest2 | 2024-11-29 13:15:00 | 407.80 | 2024-12-03 14:15:00 | 408.55 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2024-12-03 11:00:00 | 407.00 | 2024-12-03 14:15:00 | 408.55 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2024-12-06 10:45:00 | 404.10 | 2024-12-09 09:15:00 | 411.60 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-12-16 14:15:00 | 406.10 | 2024-12-23 09:15:00 | 385.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 09:15:00 | 404.20 | 2024-12-23 09:15:00 | 383.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 14:15:00 | 406.10 | 2024-12-23 10:15:00 | 392.35 | STOP_HIT | 0.50 | 3.39% |
| SELL | retest2 | 2024-12-17 09:15:00 | 404.20 | 2024-12-23 10:15:00 | 392.35 | STOP_HIT | 0.50 | 2.93% |
| SELL | retest2 | 2025-01-06 10:45:00 | 371.25 | 2025-01-13 09:15:00 | 352.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 12:15:00 | 371.45 | 2025-01-13 09:15:00 | 352.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 15:00:00 | 370.50 | 2025-01-13 09:15:00 | 351.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 10:15:00 | 371.85 | 2025-01-13 09:15:00 | 353.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 11:30:00 | 370.00 | 2025-01-13 09:15:00 | 351.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 14:30:00 | 369.85 | 2025-01-13 09:15:00 | 351.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 10:30:00 | 369.30 | 2025-01-13 09:15:00 | 350.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 13:15:00 | 369.60 | 2025-01-13 09:15:00 | 351.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 10:45:00 | 371.25 | 2025-01-14 10:15:00 | 356.75 | STOP_HIT | 0.50 | 3.91% |
| SELL | retest2 | 2025-01-06 12:15:00 | 371.45 | 2025-01-14 10:15:00 | 356.75 | STOP_HIT | 0.50 | 3.96% |
| SELL | retest2 | 2025-01-06 15:00:00 | 370.50 | 2025-01-14 10:15:00 | 356.75 | STOP_HIT | 0.50 | 3.71% |
| SELL | retest2 | 2025-01-08 10:15:00 | 371.85 | 2025-01-14 10:15:00 | 356.75 | STOP_HIT | 0.50 | 4.06% |
| SELL | retest2 | 2025-01-08 11:30:00 | 370.00 | 2025-01-14 10:15:00 | 356.75 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2025-01-08 14:30:00 | 369.85 | 2025-01-14 10:15:00 | 356.75 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2025-01-09 10:30:00 | 369.30 | 2025-01-14 10:15:00 | 356.75 | STOP_HIT | 0.50 | 3.40% |
| SELL | retest2 | 2025-01-09 13:15:00 | 369.60 | 2025-01-14 10:15:00 | 356.75 | STOP_HIT | 0.50 | 3.48% |
| SELL | retest2 | 2025-01-10 09:15:00 | 365.60 | 2025-01-15 11:15:00 | 365.55 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-01-20 15:15:00 | 362.10 | 2025-01-20 15:15:00 | 362.10 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-01-23 14:30:00 | 349.65 | 2025-01-27 09:15:00 | 332.60 | PARTIAL | 0.50 | 4.88% |
| SELL | retest2 | 2025-01-23 14:30:00 | 349.65 | 2025-01-27 14:15:00 | 338.50 | STOP_HIT | 0.50 | 3.19% |
| SELL | retest2 | 2025-01-24 09:15:00 | 350.10 | 2025-01-28 09:15:00 | 332.17 | PARTIAL | 0.50 | 5.12% |
| SELL | retest2 | 2025-01-24 09:15:00 | 350.10 | 2025-01-28 11:15:00 | 339.50 | STOP_HIT | 0.50 | 3.03% |
| BUY | retest2 | 2025-02-04 14:45:00 | 361.55 | 2025-02-10 10:15:00 | 357.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-02-10 09:45:00 | 361.20 | 2025-02-10 10:15:00 | 357.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-02-14 10:45:00 | 333.20 | 2025-02-20 15:15:00 | 336.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-03-04 14:45:00 | 327.50 | 2025-03-13 15:15:00 | 344.00 | STOP_HIT | 1.00 | 5.04% |
| BUY | retest2 | 2025-03-04 15:15:00 | 329.80 | 2025-03-13 15:15:00 | 344.00 | STOP_HIT | 1.00 | 4.31% |
| BUY | retest2 | 2025-03-20 14:30:00 | 358.20 | 2025-03-25 12:15:00 | 357.80 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-03-20 15:15:00 | 359.00 | 2025-03-25 12:15:00 | 357.80 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-03-24 10:45:00 | 357.90 | 2025-03-25 12:15:00 | 357.80 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-03-25 09:30:00 | 360.10 | 2025-03-25 12:15:00 | 357.80 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-04-01 11:00:00 | 349.00 | 2025-04-04 09:15:00 | 331.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 11:00:00 | 349.00 | 2025-04-07 09:15:00 | 314.10 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-04-17 12:00:00 | 336.40 | 2025-04-17 14:15:00 | 332.20 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-05-06 11:15:00 | 328.05 | 2025-05-07 11:15:00 | 332.60 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-05-06 13:15:00 | 329.20 | 2025-05-07 11:15:00 | 332.60 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-05-06 14:00:00 | 328.55 | 2025-05-07 11:15:00 | 332.60 | STOP_HIT | 1.00 | -1.23% |
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
