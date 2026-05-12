# Laurus Labs Ltd. (LAURUSLABS)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1225.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 134 |
| ALERT1 | 88 |
| ALERT2 | 88 |
| ALERT2_SKIP | 41 |
| ALERT3 | 271 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 92 |
| PARTIAL | 9 |
| TARGET_HIT | 7 |
| STOP_HIT | 89 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 105 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 44 / 61
- **Target hits / Stop hits / Partials:** 7 / 89 / 9
- **Avg / median % per leg:** 0.62% / -0.40%
- **Sum % (uncompounded):** 65.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 51 | 26 | 51.0% | 6 | 45 | 0 | 0.78% | 39.8% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 3 | 0 | 0.60% | 1.8% |
| BUY @ 3rd Alert (retest2) | 48 | 24 | 50.0% | 6 | 42 | 0 | 0.79% | 38.0% |
| SELL (all) | 54 | 18 | 33.3% | 1 | 44 | 9 | 0.47% | 25.2% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.59% | -1.6% |
| SELL @ 3rd Alert (retest2) | 53 | 18 | 34.0% | 1 | 43 | 9 | 0.51% | 26.8% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 4 | 0 | 0.05% | 0.2% |
| retest2 (combined) | 101 | 42 | 41.6% | 7 | 85 | 9 | 0.64% | 64.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 10:15:00 | 427.55 | 432.21 | 432.80 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 13:15:00 | 433.00 | 432.57 | 432.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 434.75 | 433.01 | 432.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 09:15:00 | 432.00 | 433.13 | 432.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 09:15:00 | 432.00 | 433.13 | 432.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 09:15:00 | 432.00 | 433.13 | 432.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:00:00 | 432.00 | 433.13 | 432.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 439.00 | 434.30 | 433.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:30:00 | 434.20 | 434.30 | 433.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 12:15:00 | 439.30 | 440.57 | 438.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 13:00:00 | 439.30 | 440.57 | 438.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 13:15:00 | 438.75 | 440.20 | 438.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 13:30:00 | 438.45 | 440.20 | 438.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 14:15:00 | 439.85 | 440.13 | 438.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 14:30:00 | 439.75 | 440.13 | 438.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 450.35 | 453.82 | 450.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:00:00 | 450.35 | 453.82 | 450.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 450.25 | 453.11 | 450.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:30:00 | 445.75 | 453.11 | 450.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 450.60 | 452.60 | 450.36 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 09:15:00 | 441.65 | 449.13 | 449.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 13:15:00 | 430.45 | 436.36 | 439.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 11:15:00 | 421.90 | 421.73 | 426.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 13:15:00 | 426.50 | 422.90 | 426.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 13:15:00 | 426.50 | 422.90 | 426.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 14:00:00 | 426.50 | 422.90 | 426.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 14:15:00 | 427.95 | 423.91 | 426.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 15:00:00 | 427.95 | 423.91 | 426.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 15:15:00 | 427.85 | 424.70 | 426.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:15:00 | 417.40 | 424.70 | 426.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 407.00 | 411.36 | 417.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 423.50 | 413.79 | 418.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 428.35 | 416.70 | 419.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:45:00 | 429.50 | 416.70 | 419.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 430.95 | 422.15 | 421.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 437.80 | 428.10 | 424.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 14:15:00 | 440.80 | 442.74 | 439.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 15:00:00 | 440.80 | 442.74 | 439.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 441.00 | 442.39 | 439.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 09:15:00 | 441.20 | 442.39 | 439.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 437.55 | 441.42 | 439.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 10:00:00 | 437.55 | 441.42 | 439.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 10:15:00 | 438.65 | 440.87 | 439.56 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2024-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 09:15:00 | 434.70 | 438.18 | 438.57 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-06-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 13:15:00 | 439.90 | 438.86 | 438.78 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 09:15:00 | 437.10 | 438.57 | 438.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 12:15:00 | 434.50 | 437.08 | 437.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 10:15:00 | 431.55 | 430.17 | 432.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 10:15:00 | 431.55 | 430.17 | 432.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 431.55 | 430.17 | 432.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 11:15:00 | 432.80 | 430.17 | 432.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 433.25 | 430.78 | 432.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 11:30:00 | 434.70 | 430.78 | 432.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 433.85 | 431.40 | 432.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 14:00:00 | 432.50 | 431.62 | 432.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 09:15:00 | 436.95 | 432.35 | 432.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 436.95 | 432.35 | 432.07 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 14:15:00 | 429.75 | 432.18 | 432.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 09:15:00 | 428.80 | 431.08 | 431.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 14:15:00 | 430.10 | 429.97 | 430.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-24 15:00:00 | 430.10 | 429.97 | 430.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 430.35 | 429.86 | 430.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 09:45:00 | 430.30 | 429.86 | 430.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 427.90 | 429.47 | 430.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 10:30:00 | 430.05 | 429.47 | 430.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 429.35 | 428.34 | 429.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:00:00 | 429.35 | 428.34 | 429.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 429.20 | 428.51 | 429.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:30:00 | 431.40 | 428.51 | 429.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 429.75 | 428.76 | 429.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:45:00 | 430.35 | 428.76 | 429.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 428.70 | 428.75 | 429.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 14:45:00 | 428.10 | 429.17 | 429.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 10:15:00 | 428.00 | 428.56 | 429.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 09:30:00 | 428.30 | 425.30 | 426.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 11:15:00 | 425.25 | 425.95 | 426.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 426.05 | 425.97 | 426.84 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-01 13:15:00 | 428.90 | 426.98 | 426.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2024-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 13:15:00 | 428.90 | 426.98 | 426.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 14:15:00 | 430.75 | 427.73 | 427.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 14:15:00 | 436.95 | 437.01 | 434.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-03 14:45:00 | 437.25 | 437.01 | 434.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 471.60 | 475.70 | 471.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 471.60 | 475.70 | 471.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 469.25 | 474.41 | 471.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 467.20 | 474.41 | 471.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 472.35 | 474.00 | 471.32 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2024-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 09:15:00 | 467.55 | 471.09 | 471.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 11:15:00 | 465.75 | 469.57 | 470.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-12 13:15:00 | 470.00 | 469.33 | 470.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 13:15:00 | 470.00 | 469.33 | 470.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 13:15:00 | 470.00 | 469.33 | 470.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 14:00:00 | 470.00 | 469.33 | 470.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 14:15:00 | 463.90 | 468.24 | 469.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 15:15:00 | 461.90 | 468.24 | 469.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 09:15:00 | 462.75 | 466.20 | 467.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 15:15:00 | 439.61 | 446.93 | 453.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 09:15:00 | 438.80 | 446.09 | 452.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 438.65 | 434.73 | 439.94 | SL hit (close>ema200) qty=0.50 sl=434.73 alert=retest2 |

### Cycle 12 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 10:15:00 | 450.50 | 439.54 | 438.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 09:15:00 | 454.90 | 449.70 | 446.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 09:15:00 | 460.10 | 462.39 | 458.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-01 10:00:00 | 460.10 | 462.39 | 458.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 458.70 | 461.65 | 458.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:30:00 | 458.85 | 461.65 | 458.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 453.00 | 459.92 | 458.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:00:00 | 453.00 | 459.92 | 458.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 457.00 | 459.34 | 457.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:30:00 | 452.25 | 459.34 | 457.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2024-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 15:15:00 | 451.15 | 456.02 | 456.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 10:15:00 | 450.70 | 454.22 | 455.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 437.60 | 436.61 | 442.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 437.60 | 436.61 | 442.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 437.60 | 436.61 | 442.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:15:00 | 432.10 | 435.51 | 441.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:00:00 | 431.50 | 434.71 | 440.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 10:15:00 | 445.00 | 435.01 | 434.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 10:15:00 | 445.00 | 435.01 | 434.93 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 14:15:00 | 431.05 | 434.18 | 434.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 10:15:00 | 429.80 | 432.90 | 433.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 15:15:00 | 426.50 | 426.25 | 428.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-13 09:15:00 | 425.70 | 426.25 | 428.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 425.15 | 426.03 | 428.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 14:00:00 | 422.65 | 425.04 | 427.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 11:15:00 | 422.00 | 423.78 | 425.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 11:45:00 | 422.00 | 423.65 | 425.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 15:15:00 | 432.75 | 426.26 | 426.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2024-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 15:15:00 | 432.75 | 426.26 | 426.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 433.45 | 429.95 | 428.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 11:15:00 | 432.65 | 434.16 | 432.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 11:15:00 | 432.65 | 434.16 | 432.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 432.65 | 434.16 | 432.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:45:00 | 433.50 | 434.16 | 432.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 434.70 | 434.27 | 432.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 13:15:00 | 435.95 | 434.27 | 432.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-05 09:15:00 | 479.55 | 477.14 | 471.95 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 10:15:00 | 500.50 | 504.63 | 504.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 09:15:00 | 491.25 | 500.24 | 502.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 14:15:00 | 467.65 | 467.55 | 474.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 14:45:00 | 467.55 | 467.55 | 474.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 471.50 | 468.39 | 473.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 09:30:00 | 466.95 | 468.85 | 470.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 11:00:00 | 466.00 | 462.62 | 463.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 12:15:00 | 467.95 | 463.73 | 463.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2024-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 12:15:00 | 467.95 | 463.73 | 463.31 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2024-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 14:15:00 | 461.95 | 464.70 | 464.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 12:15:00 | 453.35 | 460.99 | 462.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 439.05 | 437.36 | 444.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 11:00:00 | 439.05 | 437.36 | 444.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 449.70 | 440.64 | 443.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 449.70 | 440.64 | 443.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 453.10 | 443.13 | 444.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 11:00:00 | 453.10 | 443.13 | 444.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 451.00 | 444.71 | 444.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 12:15:00 | 454.90 | 446.74 | 445.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 11:15:00 | 476.30 | 477.72 | 472.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-15 11:45:00 | 475.95 | 477.72 | 472.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 472.15 | 478.01 | 475.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:30:00 | 472.95 | 478.01 | 475.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 12:15:00 | 476.60 | 477.73 | 475.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 12:30:00 | 470.15 | 477.73 | 475.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 13:15:00 | 475.40 | 477.26 | 475.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 13:30:00 | 474.40 | 477.26 | 475.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 480.65 | 477.94 | 476.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 14:45:00 | 478.20 | 477.94 | 476.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 474.25 | 477.37 | 476.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 474.25 | 477.37 | 476.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 471.10 | 476.12 | 475.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:00:00 | 471.10 | 476.12 | 475.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 11:15:00 | 470.20 | 474.93 | 475.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 464.40 | 472.09 | 473.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 09:15:00 | 470.20 | 469.92 | 472.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 10:00:00 | 470.20 | 469.92 | 472.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 476.35 | 471.20 | 472.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 11:00:00 | 476.35 | 471.20 | 472.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 479.25 | 472.81 | 473.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 479.25 | 472.81 | 473.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2024-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 12:15:00 | 476.85 | 473.62 | 473.54 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 09:15:00 | 468.90 | 473.50 | 473.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 12:15:00 | 465.10 | 470.39 | 472.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 458.55 | 457.90 | 462.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 10:00:00 | 458.55 | 457.90 | 462.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 457.55 | 458.36 | 461.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:30:00 | 462.40 | 458.36 | 461.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 464.05 | 453.71 | 455.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-25 09:30:00 | 466.00 | 453.71 | 455.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 464.70 | 455.91 | 456.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-25 11:00:00 | 464.70 | 455.91 | 456.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2024-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-25 11:15:00 | 474.40 | 459.61 | 458.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 10:15:00 | 480.00 | 468.01 | 463.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-29 09:15:00 | 481.05 | 483.83 | 474.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-29 10:00:00 | 481.05 | 483.83 | 474.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 475.80 | 482.23 | 475.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 11:00:00 | 475.80 | 482.23 | 475.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 11:15:00 | 482.50 | 482.28 | 475.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 11:30:00 | 475.75 | 482.28 | 475.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 482.15 | 484.47 | 482.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:00:00 | 482.15 | 484.47 | 482.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 481.70 | 483.92 | 482.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 13:00:00 | 483.70 | 483.87 | 482.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 11:45:00 | 483.55 | 487.22 | 487.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 12:15:00 | 485.00 | 486.78 | 486.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2024-11-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 12:15:00 | 485.00 | 486.78 | 486.85 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2024-11-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 13:15:00 | 489.15 | 487.25 | 487.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 12:15:00 | 497.80 | 490.24 | 488.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 491.50 | 492.97 | 490.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 491.50 | 492.97 | 490.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 491.50 | 492.97 | 490.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 491.50 | 492.97 | 490.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 494.00 | 493.18 | 490.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 492.50 | 493.18 | 490.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 492.10 | 497.83 | 496.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:30:00 | 491.65 | 497.83 | 496.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 497.40 | 497.74 | 496.32 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2024-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 15:15:00 | 494.00 | 495.64 | 495.69 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2024-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 09:15:00 | 497.55 | 496.02 | 495.86 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2024-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 14:15:00 | 489.35 | 495.35 | 495.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 479.50 | 491.17 | 493.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 485.20 | 482.46 | 487.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 485.20 | 482.46 | 487.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 483.45 | 482.66 | 486.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 486.50 | 482.66 | 486.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 486.20 | 483.61 | 486.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:00:00 | 486.20 | 483.61 | 486.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 486.60 | 484.21 | 486.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:15:00 | 480.25 | 484.43 | 486.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 12:15:00 | 489.10 | 484.56 | 485.53 | SL hit (close>static) qty=1.00 sl=487.75 alert=retest2 |

### Cycle 30 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 489.40 | 486.31 | 486.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 12:15:00 | 492.55 | 488.32 | 487.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 488.50 | 489.01 | 487.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 15:00:00 | 488.50 | 489.01 | 487.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 486.80 | 488.57 | 487.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:15:00 | 485.05 | 488.57 | 487.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 486.50 | 488.15 | 487.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:30:00 | 486.80 | 488.15 | 487.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 489.20 | 488.36 | 487.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 11:30:00 | 490.35 | 488.89 | 487.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 09:15:00 | 492.15 | 489.06 | 488.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-26 12:15:00 | 539.39 | 530.11 | 519.12 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 557.30 | 582.66 | 583.25 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2024-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 15:15:00 | 572.50 | 570.07 | 569.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 12:15:00 | 577.00 | 572.26 | 570.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 10:15:00 | 572.45 | 572.98 | 571.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 10:15:00 | 572.45 | 572.98 | 571.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 572.45 | 572.98 | 571.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:30:00 | 573.30 | 572.98 | 571.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 572.20 | 572.82 | 571.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 11:30:00 | 573.00 | 572.82 | 571.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 572.25 | 572.71 | 571.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:45:00 | 572.15 | 572.71 | 571.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 570.30 | 572.23 | 571.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:00:00 | 570.30 | 572.23 | 571.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 572.15 | 572.21 | 571.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:45:00 | 570.20 | 572.21 | 571.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 15:15:00 | 572.45 | 572.26 | 571.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:15:00 | 573.00 | 572.26 | 571.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 571.85 | 572.18 | 571.88 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2024-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 12:15:00 | 569.85 | 571.36 | 571.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 565.40 | 568.75 | 570.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 10:15:00 | 571.25 | 569.25 | 570.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 10:15:00 | 571.25 | 569.25 | 570.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 10:15:00 | 571.25 | 569.25 | 570.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 11:00:00 | 571.25 | 569.25 | 570.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 572.00 | 569.80 | 570.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 11:45:00 | 570.65 | 569.80 | 570.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 570.50 | 569.94 | 570.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 12:30:00 | 572.00 | 569.94 | 570.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 572.50 | 570.45 | 570.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:00:00 | 572.50 | 570.45 | 570.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2024-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 14:15:00 | 572.60 | 570.88 | 570.79 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 12:15:00 | 567.00 | 570.95 | 571.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 13:15:00 | 565.30 | 569.82 | 570.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 10:15:00 | 565.50 | 564.97 | 567.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 11:00:00 | 565.50 | 564.97 | 567.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 566.50 | 565.28 | 567.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:30:00 | 567.95 | 565.28 | 567.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 12:15:00 | 566.85 | 565.59 | 567.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 13:00:00 | 566.85 | 565.59 | 567.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 13:15:00 | 565.30 | 565.53 | 567.21 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2024-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 11:15:00 | 569.05 | 568.04 | 567.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 12:15:00 | 572.00 | 568.83 | 568.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-24 14:15:00 | 567.25 | 568.92 | 568.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 14:15:00 | 567.25 | 568.92 | 568.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 14:15:00 | 567.25 | 568.92 | 568.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-24 15:00:00 | 567.25 | 568.92 | 568.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 15:15:00 | 565.00 | 568.14 | 568.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 09:15:00 | 565.70 | 568.14 | 568.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2024-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 09:15:00 | 562.00 | 566.91 | 567.57 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2024-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 15:15:00 | 572.10 | 567.56 | 567.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 582.00 | 570.45 | 568.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 13:15:00 | 584.75 | 586.25 | 580.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-30 14:00:00 | 584.75 | 586.25 | 580.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 607.60 | 610.22 | 606.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 12:00:00 | 607.60 | 610.22 | 606.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 603.15 | 608.80 | 606.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 13:00:00 | 603.15 | 608.80 | 606.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 607.05 | 608.45 | 606.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 14:30:00 | 611.75 | 609.14 | 607.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 10:15:00 | 597.65 | 606.40 | 606.37 | SL hit (close<static) qty=1.00 sl=600.70 alert=retest2 |

### Cycle 39 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 600.50 | 605.22 | 605.83 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2025-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 11:15:00 | 613.35 | 604.98 | 604.81 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 595.70 | 609.33 | 609.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 11:15:00 | 591.30 | 603.96 | 607.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 14:15:00 | 568.40 | 568.33 | 576.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 15:00:00 | 568.40 | 568.33 | 576.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 564.35 | 559.92 | 566.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 10:00:00 | 564.35 | 559.92 | 566.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 10:15:00 | 561.90 | 560.31 | 565.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 10:30:00 | 562.45 | 560.31 | 565.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 12:15:00 | 564.30 | 559.17 | 561.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 12:45:00 | 564.60 | 559.17 | 561.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 564.45 | 560.23 | 561.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 13:45:00 | 563.75 | 560.23 | 561.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2025-01-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 15:15:00 | 569.60 | 563.24 | 562.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 10:15:00 | 573.40 | 565.70 | 564.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 578.45 | 578.88 | 573.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 11:00:00 | 578.45 | 578.88 | 573.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 576.85 | 579.02 | 575.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 571.55 | 579.02 | 575.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 569.50 | 577.12 | 574.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:45:00 | 571.20 | 577.12 | 574.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 562.75 | 574.25 | 573.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:00:00 | 562.75 | 574.25 | 573.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 560.35 | 571.47 | 572.64 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 579.70 | 572.58 | 572.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 15:15:00 | 586.00 | 580.31 | 576.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 579.25 | 580.10 | 576.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 10:00:00 | 579.25 | 580.10 | 576.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 588.10 | 581.70 | 577.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 11:45:00 | 589.35 | 583.22 | 578.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 14:45:00 | 604.70 | 586.98 | 581.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 09:15:00 | 523.80 | 578.27 | 578.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 523.80 | 578.27 | 578.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 10:15:00 | 512.30 | 535.37 | 551.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 534.65 | 523.55 | 537.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 10:00:00 | 534.65 | 523.55 | 537.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 536.10 | 526.06 | 537.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 536.10 | 526.06 | 537.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 537.60 | 528.37 | 537.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:00:00 | 537.60 | 528.37 | 537.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 535.65 | 529.83 | 537.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 13:15:00 | 538.10 | 529.83 | 537.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 13:15:00 | 539.70 | 531.80 | 537.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 14:00:00 | 539.70 | 531.80 | 537.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 542.70 | 533.98 | 537.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 15:00:00 | 542.70 | 533.98 | 537.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 580.70 | 546.36 | 542.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 10:15:00 | 586.70 | 554.43 | 546.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 13:15:00 | 579.45 | 581.70 | 570.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 13:45:00 | 579.80 | 581.70 | 570.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 578.70 | 581.70 | 575.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 579.00 | 581.70 | 575.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 12:15:00 | 581.15 | 587.18 | 582.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 13:00:00 | 581.15 | 587.18 | 582.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 13:15:00 | 583.75 | 586.50 | 582.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:15:00 | 596.65 | 586.35 | 583.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 12:15:00 | 623.40 | 628.95 | 628.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2025-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 12:15:00 | 623.40 | 628.95 | 628.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 613.40 | 624.23 | 626.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 15:15:00 | 602.50 | 601.95 | 608.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 09:15:00 | 613.35 | 601.95 | 608.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 615.25 | 604.61 | 609.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 615.25 | 604.61 | 609.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 615.25 | 606.74 | 609.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:30:00 | 616.50 | 606.74 | 609.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 12:15:00 | 611.00 | 608.58 | 610.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:15:00 | 607.55 | 608.58 | 610.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 09:15:00 | 577.17 | 598.08 | 604.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-14 12:15:00 | 546.79 | 575.32 | 591.56 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 48 — BUY (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 09:15:00 | 544.60 | 536.03 | 535.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-25 11:15:00 | 550.75 | 540.70 | 538.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-27 10:15:00 | 541.60 | 545.69 | 542.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 10:15:00 | 541.60 | 545.69 | 542.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 10:15:00 | 541.60 | 545.69 | 542.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 11:00:00 | 541.60 | 545.69 | 542.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 11:15:00 | 536.90 | 543.93 | 541.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 11:45:00 | 535.10 | 543.93 | 541.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 544.25 | 542.64 | 541.74 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 534.35 | 541.11 | 541.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 09:15:00 | 522.85 | 529.17 | 534.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 532.85 | 528.24 | 532.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 12:15:00 | 532.85 | 528.24 | 532.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 532.85 | 528.24 | 532.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:00:00 | 532.85 | 528.24 | 532.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 529.65 | 528.52 | 532.00 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2025-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 11:15:00 | 542.30 | 534.67 | 533.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 09:15:00 | 546.80 | 541.59 | 537.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 10:15:00 | 572.05 | 575.01 | 566.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 10:45:00 | 571.90 | 575.01 | 566.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 567.85 | 571.20 | 567.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 571.00 | 571.20 | 567.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 570.00 | 570.96 | 568.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 13:30:00 | 572.45 | 571.41 | 568.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 14:15:00 | 558.50 | 568.83 | 568.03 | SL hit (close<static) qty=1.00 sl=564.70 alert=retest2 |

### Cycle 51 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 560.95 | 567.25 | 567.39 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 09:15:00 | 570.60 | 567.92 | 567.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 13:15:00 | 577.80 | 571.41 | 569.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 11:15:00 | 575.85 | 577.49 | 574.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 11:15:00 | 575.85 | 577.49 | 574.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 11:15:00 | 575.85 | 577.49 | 574.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 11:45:00 | 575.50 | 577.49 | 574.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 13:15:00 | 574.70 | 576.58 | 574.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 14:00:00 | 574.70 | 576.58 | 574.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 574.65 | 576.19 | 574.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 15:15:00 | 572.55 | 576.19 | 574.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 572.55 | 575.46 | 574.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:15:00 | 582.50 | 575.46 | 574.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 583.30 | 577.03 | 575.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 09:30:00 | 589.00 | 582.11 | 578.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 12:45:00 | 588.55 | 584.11 | 580.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 15:00:00 | 586.85 | 584.91 | 581.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-19 09:30:00 | 592.40 | 587.42 | 583.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 611.15 | 618.47 | 612.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 611.15 | 618.47 | 612.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 614.45 | 617.66 | 613.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 13:45:00 | 616.60 | 616.40 | 613.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 618.30 | 615.52 | 613.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 10:00:00 | 617.15 | 615.85 | 613.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 10:45:00 | 616.60 | 615.69 | 614.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 616.50 | 615.85 | 614.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:00:00 | 616.50 | 615.85 | 614.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 615.65 | 615.76 | 614.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 15:15:00 | 614.00 | 615.76 | 614.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 15:15:00 | 614.00 | 615.41 | 614.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:15:00 | 612.30 | 615.41 | 614.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 611.00 | 614.53 | 614.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:45:00 | 610.05 | 614.53 | 614.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 613.70 | 614.36 | 614.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:15:00 | 609.55 | 614.36 | 614.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-27 11:15:00 | 609.80 | 613.45 | 613.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 11:15:00 | 609.80 | 613.45 | 613.80 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 619.20 | 614.36 | 613.99 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 12:15:00 | 608.25 | 613.00 | 613.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 09:15:00 | 606.50 | 611.68 | 612.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 10:15:00 | 605.85 | 602.38 | 605.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 10:15:00 | 605.85 | 602.38 | 605.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 605.85 | 602.38 | 605.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:00:00 | 605.85 | 602.38 | 605.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 605.25 | 602.95 | 605.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 605.25 | 602.95 | 605.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 607.75 | 603.91 | 606.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:45:00 | 605.90 | 603.91 | 606.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 606.60 | 604.45 | 606.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:30:00 | 606.40 | 604.45 | 606.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 609.30 | 605.42 | 606.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 609.30 | 605.42 | 606.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 613.40 | 607.36 | 607.09 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 571.05 | 607.44 | 608.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 569.45 | 590.43 | 599.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 567.70 | 563.47 | 574.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 567.70 | 563.47 | 574.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 567.70 | 563.47 | 574.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 557.00 | 574.80 | 576.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 15:00:00 | 562.90 | 566.78 | 570.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 10:15:00 | 592.10 | 574.25 | 573.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 592.10 | 574.25 | 573.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 601.10 | 579.62 | 575.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 15:15:00 | 626.05 | 627.12 | 621.07 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:15:00 | 631.50 | 627.12 | 621.07 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 639.95 | 649.92 | 646.56 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-24 12:15:00 | 639.95 | 649.92 | 646.56 | SL hit (close<ema400) qty=1.00 sl=646.56 alert=retest1 |

### Cycle 59 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 620.30 | 641.21 | 643.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 610.65 | 635.09 | 640.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-25 12:15:00 | 636.75 | 633.99 | 638.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 12:15:00 | 636.75 | 633.99 | 638.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 12:15:00 | 636.75 | 633.99 | 638.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 13:00:00 | 636.75 | 633.99 | 638.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 13:15:00 | 632.15 | 633.62 | 638.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 13:45:00 | 638.00 | 633.62 | 638.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 637.05 | 631.75 | 636.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 09:30:00 | 633.90 | 631.75 | 636.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 634.00 | 632.20 | 635.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:45:00 | 636.45 | 632.20 | 635.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 633.00 | 632.36 | 635.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:00:00 | 633.00 | 632.36 | 635.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 620.60 | 627.51 | 631.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 10:00:00 | 616.00 | 620.97 | 626.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:00:00 | 614.65 | 612.49 | 617.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 627.85 | 619.16 | 619.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 627.85 | 619.16 | 619.07 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 609.75 | 617.60 | 618.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 606.55 | 612.71 | 615.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 614.00 | 612.97 | 615.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 10:45:00 | 612.70 | 612.97 | 615.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 610.00 | 612.37 | 614.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 13:00:00 | 608.45 | 611.59 | 614.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 09:15:00 | 608.05 | 611.10 | 613.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 578.03 | 592.97 | 601.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 577.65 | 592.97 | 601.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 14:15:00 | 588.35 | 587.28 | 595.08 | SL hit (close>ema200) qty=0.50 sl=587.28 alert=retest2 |

### Cycle 62 — BUY (started 2025-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 11:15:00 | 598.15 | 595.75 | 595.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 604.50 | 597.28 | 596.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 11:15:00 | 602.95 | 604.93 | 602.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 12:00:00 | 602.95 | 604.93 | 602.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 604.80 | 604.91 | 602.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 12:45:00 | 603.20 | 604.91 | 602.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 605.45 | 605.02 | 602.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:45:00 | 605.85 | 604.79 | 602.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 608.55 | 604.79 | 602.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 12:30:00 | 606.40 | 604.32 | 603.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 09:30:00 | 606.80 | 606.00 | 604.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 608.10 | 608.34 | 606.65 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 598.35 | 604.96 | 605.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 598.35 | 604.96 | 605.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 594.40 | 602.85 | 604.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 12:15:00 | 594.05 | 590.98 | 593.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 12:15:00 | 594.05 | 590.98 | 593.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 594.05 | 590.98 | 593.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:00:00 | 594.05 | 590.98 | 593.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 594.95 | 591.77 | 593.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 14:00:00 | 594.95 | 591.77 | 593.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 593.70 | 592.16 | 593.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 15:15:00 | 592.00 | 592.16 | 593.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 10:15:00 | 598.95 | 594.21 | 594.23 | SL hit (close>static) qty=1.00 sl=596.90 alert=retest2 |

### Cycle 64 — BUY (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 11:15:00 | 596.65 | 594.70 | 594.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 14:15:00 | 601.75 | 597.88 | 596.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 13:15:00 | 611.60 | 611.68 | 607.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 14:00:00 | 611.60 | 611.68 | 607.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 609.50 | 611.98 | 609.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:00:00 | 609.50 | 611.98 | 609.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 608.00 | 611.18 | 609.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:00:00 | 608.00 | 611.18 | 609.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 607.25 | 610.40 | 608.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:30:00 | 607.20 | 610.40 | 608.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 610.45 | 610.20 | 609.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:45:00 | 609.70 | 610.20 | 609.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 610.50 | 610.26 | 609.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 610.40 | 610.26 | 609.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 610.00 | 610.21 | 609.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 11:00:00 | 614.95 | 611.66 | 610.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-11 09:15:00 | 676.45 | 666.62 | 658.46 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2025-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 09:15:00 | 666.00 | 667.71 | 667.94 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2025-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 10:15:00 | 672.20 | 668.61 | 668.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 11:15:00 | 679.60 | 670.81 | 669.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 10:15:00 | 673.45 | 675.79 | 673.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 11:00:00 | 673.45 | 675.79 | 673.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 671.35 | 674.91 | 672.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:00:00 | 671.35 | 674.91 | 672.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 667.90 | 673.50 | 672.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:45:00 | 668.30 | 673.50 | 672.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2025-06-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 14:15:00 | 663.60 | 671.20 | 671.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 11:15:00 | 655.50 | 661.66 | 665.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 14:15:00 | 650.80 | 648.11 | 653.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 15:00:00 | 650.80 | 648.11 | 653.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 656.65 | 650.59 | 653.85 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 662.65 | 656.48 | 656.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 13:15:00 | 666.50 | 658.49 | 656.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 13:15:00 | 670.70 | 671.32 | 667.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 13:30:00 | 671.15 | 671.32 | 667.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 668.55 | 670.27 | 668.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:45:00 | 668.20 | 670.27 | 668.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 665.25 | 669.27 | 667.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 665.25 | 669.27 | 667.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 681.40 | 671.69 | 669.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:30:00 | 667.95 | 671.69 | 669.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 762.10 | 770.16 | 766.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 762.10 | 770.16 | 766.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 767.30 | 769.59 | 766.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 12:15:00 | 768.80 | 769.59 | 766.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 12:15:00 | 821.50 | 823.67 | 823.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 12:15:00 | 821.50 | 823.67 | 823.74 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 13:15:00 | 825.30 | 823.99 | 823.88 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 15:15:00 | 822.05 | 823.56 | 823.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 10:15:00 | 818.30 | 822.27 | 823.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 11:15:00 | 823.90 | 822.60 | 823.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 11:15:00 | 823.90 | 822.60 | 823.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 823.90 | 822.60 | 823.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:45:00 | 823.95 | 822.60 | 823.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 820.15 | 822.11 | 822.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 13:15:00 | 819.50 | 822.11 | 822.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 14:15:00 | 818.70 | 821.69 | 822.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 11:15:00 | 819.00 | 822.30 | 822.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 11:15:00 | 826.45 | 823.13 | 823.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 11:15:00 | 826.45 | 823.13 | 823.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 12:15:00 | 827.65 | 824.03 | 823.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 10:15:00 | 828.40 | 829.93 | 827.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 10:15:00 | 828.40 | 829.93 | 827.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 828.40 | 829.93 | 827.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:45:00 | 828.50 | 829.93 | 827.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 831.55 | 830.25 | 827.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 15:00:00 | 839.05 | 832.19 | 829.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 11:15:00 | 874.55 | 879.98 | 880.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 11:15:00 | 874.55 | 879.98 | 880.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 864.95 | 875.13 | 877.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 857.80 | 855.05 | 863.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 12:00:00 | 857.80 | 855.05 | 863.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 862.55 | 856.55 | 863.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:45:00 | 863.35 | 856.55 | 863.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 862.85 | 857.81 | 863.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:45:00 | 863.00 | 857.81 | 863.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 860.55 | 858.36 | 862.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:15:00 | 861.10 | 858.36 | 862.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 861.10 | 858.91 | 862.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 870.35 | 858.91 | 862.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 860.50 | 859.23 | 862.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 11:30:00 | 857.90 | 859.34 | 862.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 09:15:00 | 815.00 | 827.00 | 830.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 11:15:00 | 826.00 | 824.92 | 828.98 | SL hit (close>ema200) qty=0.50 sl=824.92 alert=retest2 |

### Cycle 74 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 842.95 | 831.52 | 830.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 10:15:00 | 846.90 | 834.59 | 832.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 15:15:00 | 857.60 | 859.40 | 852.33 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 09:15:00 | 867.75 | 859.40 | 852.33 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 13:45:00 | 863.20 | 863.20 | 857.34 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 867.50 | 875.90 | 873.34 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-21 09:15:00 | 867.50 | 875.90 | 873.34 | SL hit (close<ema400) qty=1.00 sl=873.34 alert=retest1 |

### Cycle 75 — SELL (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 12:15:00 | 874.45 | 879.76 | 879.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 13:15:00 | 872.15 | 878.24 | 879.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 10:15:00 | 867.75 | 867.70 | 871.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 11:00:00 | 867.75 | 867.70 | 871.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 875.60 | 869.28 | 871.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:45:00 | 873.85 | 869.28 | 871.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 868.40 | 869.11 | 871.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 14:45:00 | 866.95 | 868.02 | 870.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 09:45:00 | 866.45 | 866.02 | 869.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 867.45 | 867.61 | 869.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:00:00 | 866.45 | 862.77 | 865.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 862.00 | 862.62 | 865.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:15:00 | 860.75 | 862.62 | 865.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:45:00 | 859.25 | 861.77 | 864.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 870.10 | 863.58 | 864.25 | SL hit (close>static) qty=1.00 sl=867.90 alert=retest2 |

### Cycle 76 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 870.80 | 865.02 | 864.85 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 12:15:00 | 862.20 | 864.46 | 864.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 13:15:00 | 860.00 | 863.57 | 864.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 868.65 | 863.61 | 863.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 09:15:00 | 868.65 | 863.61 | 863.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 868.65 | 863.61 | 863.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:00:00 | 868.65 | 863.61 | 863.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 876.80 | 866.25 | 865.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 11:15:00 | 880.30 | 869.06 | 866.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 867.05 | 871.54 | 869.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 867.05 | 871.54 | 869.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 867.05 | 871.54 | 869.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:00:00 | 867.05 | 871.54 | 869.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 864.90 | 870.21 | 868.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:30:00 | 865.05 | 870.21 | 868.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 867.75 | 869.72 | 868.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 12:30:00 | 873.65 | 870.34 | 868.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 892.25 | 896.18 | 896.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 09:15:00 | 892.25 | 896.18 | 896.25 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 10:15:00 | 897.85 | 896.03 | 895.95 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 13:15:00 | 895.00 | 895.90 | 895.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 14:15:00 | 894.20 | 895.56 | 895.77 | Break + close below crossover candle low |

### Cycle 82 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 900.00 | 896.29 | 896.03 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 14:15:00 | 894.50 | 895.98 | 896.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 09:15:00 | 889.25 | 894.16 | 895.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 10:15:00 | 894.45 | 894.22 | 895.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 10:15:00 | 894.45 | 894.22 | 895.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 894.45 | 894.22 | 895.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:45:00 | 895.05 | 894.22 | 895.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 11:15:00 | 906.25 | 896.62 | 896.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 13:15:00 | 915.00 | 901.10 | 898.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 09:15:00 | 922.90 | 927.64 | 918.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 922.90 | 927.64 | 918.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 922.90 | 927.64 | 918.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:00:00 | 922.90 | 927.64 | 918.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 916.80 | 925.48 | 918.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:00:00 | 916.80 | 925.48 | 918.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 917.00 | 923.78 | 918.36 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 904.25 | 914.80 | 915.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 10:15:00 | 893.00 | 906.57 | 911.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 13:15:00 | 906.65 | 906.36 | 909.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 13:30:00 | 904.90 | 906.36 | 909.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 909.60 | 907.01 | 909.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:45:00 | 912.65 | 907.01 | 909.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 908.00 | 907.21 | 909.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:15:00 | 915.00 | 907.21 | 909.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 915.35 | 908.84 | 910.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:30:00 | 919.30 | 908.84 | 910.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 914.20 | 909.91 | 910.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:15:00 | 918.00 | 909.91 | 910.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 11:15:00 | 916.20 | 911.17 | 911.02 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 12:15:00 | 908.30 | 911.19 | 911.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 13:15:00 | 902.35 | 909.42 | 910.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 848.20 | 843.44 | 861.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:00:00 | 848.20 | 843.44 | 861.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 860.40 | 843.42 | 849.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 861.65 | 843.42 | 849.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 861.80 | 847.10 | 850.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:45:00 | 860.50 | 847.10 | 850.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 867.45 | 853.76 | 853.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 873.60 | 857.73 | 854.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 13:15:00 | 864.20 | 867.85 | 862.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 13:15:00 | 864.20 | 867.85 | 862.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 864.20 | 867.85 | 862.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 14:00:00 | 864.20 | 867.85 | 862.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 867.35 | 867.75 | 863.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 15:00:00 | 867.35 | 867.75 | 863.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 862.60 | 867.06 | 863.73 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 09:15:00 | 860.70 | 862.49 | 862.55 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 10:15:00 | 863.60 | 862.71 | 862.64 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 861.80 | 862.53 | 862.57 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2025-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 12:15:00 | 863.75 | 862.77 | 862.68 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 860.50 | 862.70 | 862.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 13:15:00 | 858.40 | 861.67 | 862.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 11:15:00 | 859.05 | 857.66 | 859.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 11:15:00 | 859.05 | 857.66 | 859.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 859.05 | 857.66 | 859.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 859.05 | 857.66 | 859.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 863.70 | 858.87 | 860.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 13:00:00 | 863.70 | 858.87 | 860.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 13:15:00 | 870.85 | 861.27 | 860.99 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 09:15:00 | 855.45 | 861.04 | 861.04 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 884.15 | 865.03 | 862.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 887.25 | 869.47 | 865.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 13:15:00 | 871.75 | 874.12 | 870.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 13:15:00 | 871.75 | 874.12 | 870.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 871.75 | 874.12 | 870.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:45:00 | 871.90 | 874.12 | 870.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 875.20 | 874.34 | 870.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:30:00 | 871.60 | 874.34 | 870.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 871.20 | 873.66 | 871.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:30:00 | 870.00 | 873.66 | 871.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 868.90 | 872.71 | 871.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 868.90 | 872.71 | 871.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 867.70 | 871.71 | 870.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:45:00 | 869.60 | 871.71 | 870.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 868.65 | 871.10 | 870.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:45:00 | 869.70 | 871.10 | 870.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 872.70 | 874.24 | 872.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:45:00 | 873.00 | 874.24 | 872.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 874.75 | 874.34 | 873.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:30:00 | 874.65 | 874.34 | 873.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 875.65 | 874.60 | 873.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 875.65 | 874.60 | 873.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 874.00 | 876.02 | 874.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:00:00 | 874.00 | 876.02 | 874.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 874.15 | 875.65 | 874.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:30:00 | 874.35 | 875.65 | 874.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 872.30 | 874.98 | 874.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 15:00:00 | 872.30 | 874.98 | 874.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 874.00 | 874.78 | 874.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 876.95 | 874.78 | 874.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 884.75 | 876.78 | 875.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 11:00:00 | 888.45 | 879.11 | 876.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 13:30:00 | 885.75 | 884.20 | 879.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-03 12:15:00 | 974.33 | 966.03 | 961.78 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 14:15:00 | 984.60 | 986.34 | 986.42 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 994.15 | 987.85 | 987.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 13:15:00 | 999.30 | 992.03 | 990.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 12:15:00 | 998.80 | 999.30 | 995.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 13:00:00 | 998.80 | 999.30 | 995.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 1000.30 | 999.61 | 996.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 14:45:00 | 993.40 | 999.61 | 996.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1010.00 | 1001.75 | 997.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 11:30:00 | 1014.25 | 1006.52 | 1000.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 12:30:00 | 1013.00 | 1015.46 | 1009.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 990.95 | 1008.19 | 1008.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 09:15:00 | 990.95 | 1008.19 | 1008.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 978.10 | 985.74 | 989.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 985.30 | 984.46 | 988.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 10:00:00 | 985.30 | 984.46 | 988.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 984.95 | 984.83 | 988.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:30:00 | 986.75 | 984.83 | 988.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 984.10 | 982.91 | 985.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 11:15:00 | 980.00 | 982.71 | 985.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 12:45:00 | 979.65 | 981.94 | 984.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 12:30:00 | 979.95 | 980.15 | 982.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 1006.80 | 987.08 | 984.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 1006.80 | 987.08 | 984.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 09:15:00 | 1021.90 | 1003.30 | 995.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 15:15:00 | 1028.20 | 1028.43 | 1019.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 09:15:00 | 1033.40 | 1028.43 | 1019.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1024.90 | 1027.08 | 1020.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:30:00 | 1017.20 | 1027.08 | 1020.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 1017.10 | 1025.08 | 1020.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:00:00 | 1017.10 | 1025.08 | 1020.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 1021.00 | 1024.27 | 1020.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 13:45:00 | 1022.90 | 1023.73 | 1020.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 1014.00 | 1020.64 | 1019.72 | SL hit (close<static) qty=1.00 sl=1016.50 alert=retest2 |

### Cycle 101 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 1009.30 | 1018.37 | 1018.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 09:15:00 | 1007.00 | 1015.26 | 1017.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 13:15:00 | 1013.90 | 1012.40 | 1014.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 13:15:00 | 1013.90 | 1012.40 | 1014.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 1013.90 | 1012.40 | 1014.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:00:00 | 1013.90 | 1012.40 | 1014.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 1015.00 | 1012.92 | 1014.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 1015.00 | 1012.92 | 1014.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 1011.00 | 1012.54 | 1014.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 1006.80 | 1012.54 | 1014.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1009.10 | 1011.85 | 1014.02 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2025-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 12:15:00 | 1026.20 | 1017.02 | 1016.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 15:15:00 | 1027.00 | 1021.68 | 1018.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 10:15:00 | 1020.30 | 1021.52 | 1019.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 10:15:00 | 1020.30 | 1021.52 | 1019.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 1020.30 | 1021.52 | 1019.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 1020.30 | 1021.52 | 1019.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 1013.90 | 1020.00 | 1018.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 1013.90 | 1020.00 | 1018.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 1006.70 | 1017.34 | 1017.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 1004.30 | 1014.73 | 1016.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 1010.70 | 1008.21 | 1012.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 10:15:00 | 1010.70 | 1008.21 | 1012.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 1010.70 | 1008.21 | 1012.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 1010.70 | 1008.21 | 1012.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 1016.40 | 1009.85 | 1012.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 1016.40 | 1009.85 | 1012.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1020.90 | 1012.06 | 1013.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 1020.90 | 1012.06 | 1013.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 1021.60 | 1015.75 | 1014.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 1029.60 | 1019.05 | 1016.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 14:15:00 | 1018.40 | 1021.28 | 1018.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 14:15:00 | 1018.40 | 1021.28 | 1018.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 1018.40 | 1021.28 | 1018.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 15:00:00 | 1018.40 | 1021.28 | 1018.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 1021.80 | 1021.39 | 1019.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 1022.90 | 1021.39 | 1019.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 1024.40 | 1021.99 | 1019.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:30:00 | 1027.40 | 1022.83 | 1020.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:45:00 | 1026.80 | 1021.96 | 1021.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 11:15:00 | 1014.10 | 1019.71 | 1020.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 11:15:00 | 1014.10 | 1019.71 | 1020.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 09:15:00 | 1002.50 | 1013.79 | 1017.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 14:15:00 | 1002.30 | 1000.87 | 1005.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-16 15:00:00 | 1002.30 | 1000.87 | 1005.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 1012.40 | 1003.15 | 1006.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:30:00 | 1014.60 | 1003.15 | 1006.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 1009.70 | 1004.46 | 1006.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 14:15:00 | 1007.50 | 1006.94 | 1007.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 14:15:00 | 1011.10 | 1007.77 | 1007.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2025-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 14:15:00 | 1011.10 | 1007.77 | 1007.59 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 1003.70 | 1006.77 | 1007.15 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2025-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 11:15:00 | 1009.70 | 1007.46 | 1007.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 12:15:00 | 1015.80 | 1009.12 | 1008.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 12:15:00 | 1081.60 | 1082.63 | 1074.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 12:30:00 | 1080.10 | 1082.63 | 1074.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1077.80 | 1081.70 | 1077.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:00:00 | 1077.80 | 1081.70 | 1077.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 1079.90 | 1081.34 | 1077.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 14:00:00 | 1086.60 | 1082.61 | 1078.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 13:45:00 | 1089.20 | 1083.58 | 1080.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 10:30:00 | 1089.30 | 1087.15 | 1083.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 14:15:00 | 1102.30 | 1114.75 | 1116.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 14:15:00 | 1102.30 | 1114.75 | 1116.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 1095.20 | 1108.91 | 1113.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 12:15:00 | 1043.80 | 1043.00 | 1061.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 12:30:00 | 1044.20 | 1043.00 | 1061.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 1061.20 | 1048.98 | 1059.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 1068.50 | 1048.98 | 1059.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 1089.10 | 1057.00 | 1062.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:00:00 | 1089.10 | 1057.00 | 1062.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 1096.20 | 1064.84 | 1065.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 1096.20 | 1064.84 | 1065.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 11:15:00 | 1090.10 | 1069.89 | 1067.52 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 1065.70 | 1073.43 | 1074.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 1043.10 | 1061.84 | 1067.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1004.70 | 1004.09 | 1022.94 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 10:15:00 | 998.10 | 1004.09 | 1022.94 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 1014.00 | 1001.71 | 1012.64 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-22 15:15:00 | 1014.00 | 1001.71 | 1012.64 | SL hit (close>ema400) qty=1.00 sl=1012.64 alert=retest1 |

### Cycle 112 — BUY (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 15:15:00 | 1030.00 | 1016.82 | 1016.74 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 999.00 | 1013.31 | 1015.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 993.80 | 1005.86 | 1011.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 10:15:00 | 1003.70 | 1002.55 | 1007.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 11:00:00 | 1003.70 | 1002.55 | 1007.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 997.00 | 998.25 | 1002.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:30:00 | 992.20 | 997.38 | 1001.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 12:00:00 | 990.00 | 995.91 | 1000.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 11:15:00 | 942.59 | 961.55 | 974.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 12:15:00 | 940.50 | 958.78 | 971.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-02 13:15:00 | 941.05 | 938.16 | 951.27 | SL hit (close>ema200) qty=0.50 sl=938.16 alert=retest2 |

### Cycle 114 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 980.45 | 961.07 | 958.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 992.00 | 974.87 | 966.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 15:15:00 | 977.20 | 979.05 | 972.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:15:00 | 963.35 | 979.05 | 972.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 962.85 | 975.81 | 972.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 962.65 | 975.81 | 972.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 959.75 | 972.60 | 970.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 959.75 | 972.60 | 970.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — SELL (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 12:15:00 | 963.85 | 968.63 | 969.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 947.90 | 962.40 | 965.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 967.85 | 954.96 | 958.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 967.85 | 954.96 | 958.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 967.85 | 954.96 | 958.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 967.85 | 954.96 | 958.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 981.80 | 960.33 | 960.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 981.80 | 960.33 | 960.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 987.10 | 965.68 | 963.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 10:15:00 | 988.70 | 976.10 | 973.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 1004.75 | 1011.63 | 1001.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-13 09:30:00 | 1008.00 | 1011.63 | 1001.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 1009.60 | 1011.23 | 1002.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 12:15:00 | 1018.45 | 1011.56 | 1003.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 13:00:00 | 1016.00 | 1012.45 | 1004.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 14:45:00 | 1017.15 | 1013.25 | 1006.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 1017.00 | 1012.60 | 1006.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 1015.05 | 1013.09 | 1007.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 12:30:00 | 1027.60 | 1020.57 | 1016.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:30:00 | 1025.00 | 1025.59 | 1024.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 11:15:00 | 1020.90 | 1023.15 | 1023.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 11:15:00 | 1020.90 | 1023.15 | 1023.34 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 1024.90 | 1023.45 | 1023.44 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 14:15:00 | 1018.45 | 1022.45 | 1022.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 10:15:00 | 1013.95 | 1020.87 | 1022.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 10:15:00 | 1021.25 | 1016.99 | 1018.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 10:15:00 | 1021.25 | 1016.99 | 1018.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 1021.25 | 1016.99 | 1018.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 11:00:00 | 1021.25 | 1016.99 | 1018.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 1019.85 | 1017.56 | 1019.04 | EMA400 retest candle locked (from downside) |

### Cycle 120 — BUY (started 2026-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 13:15:00 | 1027.40 | 1019.89 | 1019.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 15:15:00 | 1031.05 | 1023.34 | 1021.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 14:15:00 | 1076.50 | 1085.47 | 1075.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 14:15:00 | 1076.50 | 1085.47 | 1075.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 1076.50 | 1085.47 | 1075.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 1076.50 | 1085.47 | 1075.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 1071.95 | 1082.77 | 1074.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 1063.50 | 1082.77 | 1074.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1066.20 | 1079.46 | 1074.10 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 1062.40 | 1071.08 | 1071.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 1031.50 | 1061.54 | 1066.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 1048.00 | 1041.01 | 1050.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 1048.00 | 1041.01 | 1050.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 1048.00 | 1041.01 | 1050.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:30:00 | 1052.50 | 1041.01 | 1050.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1044.50 | 1041.85 | 1047.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 1046.70 | 1041.85 | 1047.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 1047.00 | 1042.88 | 1047.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 1049.50 | 1042.88 | 1047.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1040.40 | 1042.38 | 1046.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:45:00 | 1027.90 | 1036.46 | 1042.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 976.50 | 1027.07 | 1036.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 14:15:00 | 1021.70 | 1016.17 | 1026.27 | SL hit (close>ema200) qty=0.50 sl=1016.17 alert=retest2 |

### Cycle 122 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 1037.90 | 1030.00 | 1029.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 1041.90 | 1034.68 | 1032.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 13:15:00 | 1047.50 | 1049.48 | 1041.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 14:00:00 | 1047.50 | 1049.48 | 1041.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 1039.00 | 1047.38 | 1041.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 1039.00 | 1047.38 | 1041.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1042.00 | 1046.30 | 1041.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1026.90 | 1046.30 | 1041.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1020.30 | 1041.10 | 1039.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:30:00 | 1046.70 | 1041.44 | 1040.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 995.50 | 1035.73 | 1038.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 995.50 | 1035.73 | 1038.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 992.90 | 1027.16 | 1034.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 979.00 | 973.32 | 983.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 979.00 | 973.32 | 983.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 979.00 | 973.32 | 983.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 983.60 | 973.32 | 983.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 986.30 | 975.91 | 984.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:45:00 | 986.40 | 975.91 | 984.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 983.20 | 977.37 | 984.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:30:00 | 985.20 | 977.37 | 984.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 987.70 | 979.44 | 984.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:00:00 | 987.70 | 979.44 | 984.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 13:15:00 | 988.00 | 981.15 | 984.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 14:30:00 | 981.30 | 981.90 | 984.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 13:15:00 | 987.00 | 974.42 | 973.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 13:15:00 | 987.00 | 974.42 | 973.89 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 964.10 | 973.54 | 974.05 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 992.60 | 974.48 | 972.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 13:15:00 | 998.40 | 979.26 | 975.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1010.80 | 1013.16 | 1000.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 12:15:00 | 1012.60 | 1010.92 | 1002.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 1012.60 | 1010.92 | 1002.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 13:15:00 | 1014.70 | 1010.92 | 1002.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 14:45:00 | 1015.80 | 1010.58 | 1003.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 992.30 | 1006.51 | 1002.86 | SL hit (close<static) qty=1.00 sl=1000.50 alert=retest2 |

### Cycle 127 — SELL (started 2026-03-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 14:15:00 | 994.00 | 1000.04 | 1000.54 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 1033.60 | 1005.46 | 1002.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 1045.45 | 1029.10 | 1021.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 1081.90 | 1082.85 | 1072.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 1083.05 | 1082.85 | 1072.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1086.25 | 1091.55 | 1084.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1111.00 | 1088.58 | 1085.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 13:15:00 | 1116.10 | 1125.99 | 1127.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2026-04-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 13:15:00 | 1116.10 | 1125.99 | 1127.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 12:15:00 | 1106.30 | 1116.26 | 1121.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 09:15:00 | 1126.90 | 1111.31 | 1116.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 1126.90 | 1111.31 | 1116.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 1126.90 | 1111.31 | 1116.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:00:00 | 1126.90 | 1111.31 | 1116.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 1117.10 | 1112.46 | 1116.54 | EMA400 retest candle locked (from downside) |

### Cycle 130 — BUY (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 14:15:00 | 1126.90 | 1119.94 | 1119.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 09:15:00 | 1130.65 | 1123.69 | 1121.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 11:15:00 | 1123.70 | 1123.95 | 1121.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 11:15:00 | 1123.70 | 1123.95 | 1121.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 1123.70 | 1123.95 | 1121.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:00:00 | 1123.70 | 1123.95 | 1121.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 1125.40 | 1124.24 | 1122.02 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 15:15:00 | 1114.85 | 1120.44 | 1120.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-27 10:15:00 | 1109.30 | 1118.12 | 1119.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 11:15:00 | 1120.35 | 1118.57 | 1119.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 11:15:00 | 1120.35 | 1118.57 | 1119.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 1120.35 | 1118.57 | 1119.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:00:00 | 1120.35 | 1118.57 | 1119.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 1118.00 | 1118.46 | 1119.52 | EMA400 retest candle locked (from downside) |

### Cycle 132 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 1125.00 | 1120.20 | 1120.16 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 09:15:00 | 1115.15 | 1119.64 | 1119.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 13:15:00 | 1089.30 | 1110.08 | 1115.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 12:15:00 | 1115.40 | 1105.48 | 1109.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 12:15:00 | 1115.40 | 1105.48 | 1109.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 1115.40 | 1105.48 | 1109.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 13:00:00 | 1115.40 | 1105.48 | 1109.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 1109.40 | 1106.27 | 1109.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 15:00:00 | 1105.20 | 1106.05 | 1109.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1162.00 | 1112.68 | 1108.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 1162.00 | 1112.68 | 1108.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 11:15:00 | 1164.00 | 1130.02 | 1117.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 14:15:00 | 1174.70 | 1175.61 | 1162.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 15:00:00 | 1174.70 | 1175.61 | 1162.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-06-19 14:00:00 | 432.50 | 2024-06-21 09:15:00 | 436.95 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-06-26 14:45:00 | 428.10 | 2024-07-01 13:15:00 | 428.90 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2024-06-27 10:15:00 | 428.00 | 2024-07-01 13:15:00 | 428.90 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-06-28 09:30:00 | 428.30 | 2024-07-01 13:15:00 | 428.90 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2024-06-28 11:15:00 | 425.25 | 2024-07-01 13:15:00 | 428.90 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-07-12 15:15:00 | 461.90 | 2024-07-19 15:15:00 | 439.61 | PARTIAL | 0.50 | 4.83% |
| SELL | retest2 | 2024-07-18 09:15:00 | 462.75 | 2024-07-22 09:15:00 | 438.80 | PARTIAL | 0.50 | 5.17% |
| SELL | retest2 | 2024-07-12 15:15:00 | 461.90 | 2024-07-24 09:15:00 | 438.65 | STOP_HIT | 0.50 | 5.03% |
| SELL | retest2 | 2024-07-18 09:15:00 | 462.75 | 2024-07-24 09:15:00 | 438.65 | STOP_HIT | 0.50 | 5.21% |
| SELL | retest2 | 2024-08-06 12:15:00 | 432.10 | 2024-08-08 10:15:00 | 445.00 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2024-08-06 13:00:00 | 431.50 | 2024-08-08 10:15:00 | 445.00 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2024-08-13 14:00:00 | 422.65 | 2024-08-14 15:15:00 | 432.75 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2024-08-14 11:15:00 | 422.00 | 2024-08-14 15:15:00 | 432.75 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2024-08-14 11:45:00 | 422.00 | 2024-08-14 15:15:00 | 432.75 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2024-08-20 13:15:00 | 435.95 | 2024-09-05 09:15:00 | 479.55 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-25 09:30:00 | 466.95 | 2024-10-01 12:15:00 | 467.95 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-09-27 11:00:00 | 466.00 | 2024-10-01 12:15:00 | 467.95 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2024-10-31 13:00:00 | 483.70 | 2024-11-05 12:15:00 | 485.00 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2024-11-05 11:45:00 | 483.55 | 2024-11-05 12:15:00 | 485.00 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2024-11-18 09:15:00 | 480.25 | 2024-11-18 12:15:00 | 489.10 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-11-18 14:15:00 | 483.00 | 2024-11-19 09:15:00 | 490.15 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-11-18 15:15:00 | 483.00 | 2024-11-19 09:15:00 | 490.15 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2024-11-21 11:30:00 | 490.35 | 2024-11-26 12:15:00 | 539.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-22 09:15:00 | 492.15 | 2024-11-26 12:15:00 | 541.37 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-03 14:30:00 | 611.75 | 2025-01-06 10:15:00 | 597.65 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-01-24 11:45:00 | 589.35 | 2025-01-27 09:15:00 | 523.80 | STOP_HIT | 1.00 | -11.12% |
| BUY | retest2 | 2025-01-24 14:45:00 | 604.70 | 2025-01-27 09:15:00 | 523.80 | STOP_HIT | 1.00 | -13.38% |
| BUY | retest2 | 2025-02-04 09:15:00 | 596.65 | 2025-02-10 12:15:00 | 623.40 | STOP_HIT | 1.00 | 4.48% |
| SELL | retest2 | 2025-02-13 13:15:00 | 607.55 | 2025-02-14 09:15:00 | 577.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:15:00 | 607.55 | 2025-02-14 12:15:00 | 546.79 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-03-10 13:30:00 | 572.45 | 2025-03-10 14:15:00 | 558.50 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-03-18 09:30:00 | 589.00 | 2025-03-27 11:15:00 | 609.80 | STOP_HIT | 1.00 | 3.53% |
| BUY | retest2 | 2025-03-18 12:45:00 | 588.55 | 2025-03-27 11:15:00 | 609.80 | STOP_HIT | 1.00 | 3.61% |
| BUY | retest2 | 2025-03-18 15:00:00 | 586.85 | 2025-03-27 11:15:00 | 609.80 | STOP_HIT | 1.00 | 3.91% |
| BUY | retest2 | 2025-03-19 09:30:00 | 592.40 | 2025-03-27 11:15:00 | 609.80 | STOP_HIT | 1.00 | 2.94% |
| BUY | retest2 | 2025-03-25 13:45:00 | 616.60 | 2025-03-27 11:15:00 | 609.80 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-03-26 09:15:00 | 618.30 | 2025-03-27 11:15:00 | 609.80 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-03-26 10:00:00 | 617.15 | 2025-03-27 11:15:00 | 609.80 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-03-26 10:45:00 | 616.60 | 2025-03-27 11:15:00 | 609.80 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-04-09 09:15:00 | 557.00 | 2025-04-11 10:15:00 | 592.10 | STOP_HIT | 1.00 | -6.30% |
| SELL | retest2 | 2025-04-09 15:00:00 | 562.90 | 2025-04-11 10:15:00 | 592.10 | STOP_HIT | 1.00 | -5.19% |
| BUY | retest1 | 2025-04-21 09:15:00 | 631.50 | 2025-04-24 12:15:00 | 639.95 | STOP_HIT | 1.00 | 1.34% |
| SELL | retest2 | 2025-04-30 10:00:00 | 616.00 | 2025-05-05 09:15:00 | 627.85 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-05-02 11:00:00 | 614.65 | 2025-05-05 09:15:00 | 627.85 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-05-07 13:00:00 | 608.45 | 2025-05-09 09:15:00 | 578.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 09:15:00 | 608.05 | 2025-05-09 09:15:00 | 577.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-07 13:00:00 | 608.45 | 2025-05-09 14:15:00 | 588.35 | STOP_HIT | 0.50 | 3.30% |
| SELL | retest2 | 2025-05-08 09:15:00 | 608.05 | 2025-05-09 14:15:00 | 588.35 | STOP_HIT | 0.50 | 3.24% |
| BUY | retest2 | 2025-05-15 14:45:00 | 605.85 | 2025-05-20 13:15:00 | 598.35 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-05-16 09:15:00 | 608.55 | 2025-05-20 13:15:00 | 598.35 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-05-16 12:30:00 | 606.40 | 2025-05-20 13:15:00 | 598.35 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-05-19 09:30:00 | 606.80 | 2025-05-20 13:15:00 | 598.35 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-05-23 15:15:00 | 592.00 | 2025-05-26 10:15:00 | 598.95 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-06-03 11:00:00 | 614.95 | 2025-06-11 09:15:00 | 676.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-08 12:15:00 | 768.80 | 2025-07-22 12:15:00 | 821.50 | STOP_HIT | 1.00 | 6.85% |
| SELL | retest2 | 2025-07-23 13:15:00 | 819.50 | 2025-07-24 11:15:00 | 826.45 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-07-23 14:15:00 | 818.70 | 2025-07-24 11:15:00 | 826.45 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-07-24 11:15:00 | 819.00 | 2025-07-24 11:15:00 | 826.45 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-07-25 15:00:00 | 839.05 | 2025-07-31 11:15:00 | 874.55 | STOP_HIT | 1.00 | 4.23% |
| SELL | retest2 | 2025-08-05 11:30:00 | 857.90 | 2025-08-12 09:15:00 | 815.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 11:30:00 | 857.90 | 2025-08-12 11:15:00 | 826.00 | STOP_HIT | 0.50 | 3.72% |
| BUY | retest1 | 2025-08-18 09:15:00 | 867.75 | 2025-08-21 09:15:00 | 867.50 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest1 | 2025-08-18 13:45:00 | 863.20 | 2025-08-21 09:15:00 | 867.50 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2025-08-21 11:15:00 | 881.00 | 2025-08-25 12:15:00 | 874.45 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-08-25 09:15:00 | 885.65 | 2025-08-25 12:15:00 | 874.45 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-08-28 14:45:00 | 866.95 | 2025-09-02 10:15:00 | 870.10 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-08-29 09:45:00 | 866.45 | 2025-09-02 10:15:00 | 870.10 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-08-29 12:15:00 | 867.45 | 2025-09-02 11:15:00 | 870.80 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-09-01 10:00:00 | 866.45 | 2025-09-02 11:15:00 | 870.80 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-09-01 11:15:00 | 860.75 | 2025-09-02 11:15:00 | 870.80 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-09-01 11:45:00 | 859.25 | 2025-09-02 11:15:00 | 870.80 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-09-04 12:30:00 | 873.65 | 2025-09-15 09:15:00 | 892.25 | STOP_HIT | 1.00 | 2.13% |
| BUY | retest2 | 2025-10-17 11:00:00 | 888.45 | 2025-11-03 12:15:00 | 974.33 | TARGET_HIT | 1.00 | 9.67% |
| BUY | retest2 | 2025-10-17 13:30:00 | 885.75 | 2025-11-03 13:15:00 | 977.30 | TARGET_HIT | 1.00 | 10.34% |
| BUY | retest2 | 2025-11-17 11:30:00 | 1014.25 | 2025-11-19 09:15:00 | 990.95 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-11-18 12:30:00 | 1013.00 | 2025-11-19 09:15:00 | 990.95 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-11-25 11:15:00 | 980.00 | 2025-11-27 09:15:00 | 1006.80 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-11-25 12:45:00 | 979.65 | 2025-11-27 09:15:00 | 1006.80 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2025-11-26 12:30:00 | 979.95 | 2025-11-27 09:15:00 | 1006.80 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-12-02 13:45:00 | 1022.90 | 2025-12-03 09:15:00 | 1014.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-12-11 10:30:00 | 1027.40 | 2025-12-12 11:15:00 | 1014.10 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-12-12 09:45:00 | 1026.80 | 2025-12-12 11:15:00 | 1014.10 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-12-17 14:15:00 | 1007.50 | 2025-12-17 14:15:00 | 1011.10 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-12-29 14:00:00 | 1086.60 | 2026-01-08 14:15:00 | 1102.30 | STOP_HIT | 1.00 | 1.44% |
| BUY | retest2 | 2025-12-30 13:45:00 | 1089.20 | 2026-01-08 14:15:00 | 1102.30 | STOP_HIT | 1.00 | 1.20% |
| BUY | retest2 | 2025-12-31 10:30:00 | 1089.30 | 2026-01-08 14:15:00 | 1102.30 | STOP_HIT | 1.00 | 1.19% |
| SELL | retest1 | 2026-01-22 10:15:00 | 998.10 | 2026-01-22 15:15:00 | 1014.00 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-01-29 10:30:00 | 992.20 | 2026-02-01 11:15:00 | 942.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-29 12:00:00 | 990.00 | 2026-02-01 12:15:00 | 940.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-29 10:30:00 | 992.20 | 2026-02-02 13:15:00 | 941.05 | STOP_HIT | 0.50 | 5.16% |
| SELL | retest2 | 2026-01-29 12:00:00 | 990.00 | 2026-02-02 13:15:00 | 941.05 | STOP_HIT | 0.50 | 4.94% |
| BUY | retest2 | 2026-02-13 12:15:00 | 1018.45 | 2026-02-20 11:15:00 | 1020.90 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest2 | 2026-02-13 13:00:00 | 1016.00 | 2026-02-20 11:15:00 | 1020.90 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2026-02-13 14:45:00 | 1017.15 | 2026-02-20 11:15:00 | 1020.90 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2026-02-16 09:15:00 | 1017.00 | 2026-02-20 11:15:00 | 1020.90 | STOP_HIT | 1.00 | 0.38% |
| BUY | retest2 | 2026-02-18 12:30:00 | 1027.60 | 2026-02-20 11:15:00 | 1020.90 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2026-02-20 09:30:00 | 1025.00 | 2026-02-20 11:15:00 | 1020.90 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2026-03-06 14:45:00 | 1027.90 | 2026-03-09 09:15:00 | 976.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 14:45:00 | 1027.90 | 2026-03-09 14:15:00 | 1021.70 | STOP_HIT | 0.50 | 0.60% |
| BUY | retest2 | 2026-03-12 11:30:00 | 1046.70 | 2026-03-13 09:15:00 | 995.50 | STOP_HIT | 1.00 | -4.89% |
| SELL | retest2 | 2026-03-18 14:30:00 | 981.30 | 2026-03-20 13:15:00 | 987.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2026-03-27 13:15:00 | 1014.70 | 2026-03-30 09:15:00 | 992.30 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2026-03-27 14:45:00 | 1015.80 | 2026-03-30 09:15:00 | 992.30 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1111.00 | 2026-04-21 13:15:00 | 1116.10 | STOP_HIT | 1.00 | 0.46% |
| SELL | retest2 | 2026-04-29 15:00:00 | 1105.20 | 2026-05-04 09:15:00 | 1162.00 | STOP_HIT | 1.00 | -5.14% |
