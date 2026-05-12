# Cohance Lifesciences Ltd. (COHANCE)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 487.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 235 |
| ALERT1 | 153 |
| ALERT2 | 152 |
| ALERT2_SKIP | 94 |
| ALERT3 | 384 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 186 |
| PARTIAL | 39 |
| TARGET_HIT | 14 |
| STOP_HIT | 178 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 231 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 113 / 118
- **Target hits / Stop hits / Partials:** 14 / 178 / 39
- **Avg / median % per leg:** 1.20% / -0.03%
- **Sum % (uncompounded):** 277.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 96 | 31 | 32.3% | 4 | 91 | 1 | 0.02% | 2.1% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 1 | 4 | 1 | 1.97% | 11.8% |
| BUY @ 3rd Alert (retest2) | 90 | 29 | 32.2% | 3 | 87 | 0 | -0.11% | -9.7% |
| SELL (all) | 135 | 82 | 60.7% | 10 | 87 | 38 | 2.04% | 275.7% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.15% | -3.2% |
| SELL @ 3rd Alert (retest2) | 134 | 82 | 61.2% | 10 | 86 | 38 | 2.08% | 278.9% |
| retest1 (combined) | 7 | 2 | 28.6% | 1 | 5 | 1 | 1.24% | 8.7% |
| retest2 (combined) | 224 | 111 | 49.6% | 13 | 173 | 38 | 1.20% | 269.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 12:15:00 | 474.00 | 473.67 | 473.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-15 13:15:00 | 474.95 | 473.93 | 473.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-15 15:15:00 | 474.00 | 474.18 | 473.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-16 09:15:00 | 473.00 | 474.18 | 473.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 09:15:00 | 476.05 | 474.55 | 474.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-18 11:00:00 | 477.00 | 475.37 | 474.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-19 09:15:00 | 473.00 | 474.82 | 474.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2023-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 09:15:00 | 473.00 | 474.82 | 474.88 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 10:15:00 | 475.60 | 474.66 | 474.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 14:15:00 | 476.45 | 475.19 | 474.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 10:15:00 | 473.85 | 475.27 | 475.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 10:15:00 | 473.85 | 475.27 | 475.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 10:15:00 | 473.85 | 475.27 | 475.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 11:00:00 | 473.85 | 475.27 | 475.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 11:15:00 | 476.00 | 475.41 | 475.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 11:30:00 | 476.05 | 475.41 | 475.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 12:15:00 | 474.45 | 475.22 | 475.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 13:00:00 | 474.45 | 475.22 | 475.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 13:15:00 | 474.45 | 475.07 | 475.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 13:30:00 | 474.45 | 475.07 | 475.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2023-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 15:15:00 | 473.90 | 474.80 | 474.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 09:15:00 | 473.20 | 474.48 | 474.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-25 09:15:00 | 472.80 | 472.70 | 473.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 09:15:00 | 472.80 | 472.70 | 473.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 472.80 | 472.70 | 473.50 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-05-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 14:15:00 | 475.60 | 473.81 | 473.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 15:15:00 | 477.00 | 475.24 | 474.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 15:15:00 | 476.00 | 476.21 | 475.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-30 09:15:00 | 476.65 | 476.21 | 475.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 12:15:00 | 476.05 | 476.34 | 475.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 12:30:00 | 476.40 | 476.34 | 475.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 11:15:00 | 477.00 | 476.76 | 476.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 11:45:00 | 477.20 | 476.76 | 476.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 12:15:00 | 477.45 | 476.90 | 476.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-01 10:15:00 | 478.45 | 477.09 | 476.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-01 13:15:00 | 474.10 | 476.50 | 476.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2023-06-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-01 13:15:00 | 474.10 | 476.50 | 476.52 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 12:15:00 | 481.65 | 476.69 | 476.42 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-05 13:15:00 | 475.35 | 476.42 | 476.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-05 14:15:00 | 474.50 | 476.04 | 476.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-07 11:15:00 | 475.15 | 475.12 | 475.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-07 11:15:00 | 475.15 | 475.12 | 475.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 11:15:00 | 475.15 | 475.12 | 475.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-07 11:30:00 | 475.50 | 475.12 | 475.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 12:15:00 | 475.45 | 475.18 | 475.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-07 12:30:00 | 475.50 | 475.18 | 475.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 13:15:00 | 476.00 | 475.35 | 475.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-07 14:00:00 | 476.00 | 475.35 | 475.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2023-06-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 14:15:00 | 477.55 | 475.79 | 475.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-08 10:15:00 | 480.00 | 477.04 | 476.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 11:15:00 | 476.80 | 476.99 | 476.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 11:15:00 | 476.80 | 476.99 | 476.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 476.80 | 476.99 | 476.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:00:00 | 476.80 | 476.99 | 476.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 13:15:00 | 476.70 | 476.92 | 476.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 13:45:00 | 476.75 | 476.92 | 476.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 14:15:00 | 475.55 | 476.65 | 476.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 15:00:00 | 475.55 | 476.65 | 476.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 15:15:00 | 476.00 | 476.52 | 476.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 09:15:00 | 474.35 | 476.52 | 476.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2023-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 09:15:00 | 471.65 | 475.54 | 475.91 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 11:15:00 | 476.30 | 475.58 | 475.56 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2023-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-12 12:15:00 | 475.25 | 475.52 | 475.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-12 14:15:00 | 475.00 | 475.38 | 475.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-13 10:15:00 | 475.45 | 475.34 | 475.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 10:15:00 | 475.45 | 475.34 | 475.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 10:15:00 | 475.45 | 475.34 | 475.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 10:30:00 | 475.25 | 475.34 | 475.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 11:15:00 | 475.85 | 475.44 | 475.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 12:00:00 | 475.85 | 475.44 | 475.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2023-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 12:15:00 | 475.60 | 475.47 | 475.47 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-06-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-13 13:15:00 | 475.25 | 475.43 | 475.45 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 14:15:00 | 475.95 | 475.53 | 475.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-14 09:15:00 | 476.10 | 475.60 | 475.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-14 11:15:00 | 475.60 | 475.75 | 475.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-14 11:15:00 | 475.60 | 475.75 | 475.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 11:15:00 | 475.60 | 475.75 | 475.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-14 11:45:00 | 475.95 | 475.75 | 475.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 12:15:00 | 476.40 | 475.88 | 475.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-14 15:00:00 | 477.50 | 476.21 | 475.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-15 14:00:00 | 477.10 | 476.31 | 476.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-15 15:00:00 | 477.65 | 476.58 | 476.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-16 09:45:00 | 477.50 | 476.92 | 476.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 09:15:00 | 487.20 | 485.99 | 483.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-26 15:15:00 | 489.40 | 486.79 | 486.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-27 09:30:00 | 488.50 | 487.53 | 486.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-27 13:00:00 | 490.45 | 488.99 | 487.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-27 15:30:00 | 488.40 | 489.20 | 488.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 09:15:00 | 487.90 | 488.94 | 488.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 09:45:00 | 488.05 | 488.94 | 488.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 10:15:00 | 488.90 | 488.93 | 488.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 10:30:00 | 488.10 | 488.93 | 488.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 11:15:00 | 489.40 | 489.02 | 488.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 11:30:00 | 487.50 | 489.02 | 488.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 12:15:00 | 487.55 | 488.73 | 488.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 13:00:00 | 487.55 | 488.73 | 488.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 13:15:00 | 489.15 | 488.81 | 488.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 14:00:00 | 489.15 | 488.81 | 488.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 14:15:00 | 489.00 | 488.85 | 488.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 15:00:00 | 489.00 | 488.85 | 488.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 09:15:00 | 486.00 | 488.30 | 488.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-30 09:30:00 | 485.25 | 488.30 | 488.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-06-30 10:15:00 | 484.15 | 487.47 | 487.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2023-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 10:15:00 | 484.15 | 487.47 | 487.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-03 13:15:00 | 482.20 | 484.43 | 485.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-04 10:15:00 | 483.90 | 483.09 | 484.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-04 11:00:00 | 483.90 | 483.09 | 484.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 11:15:00 | 485.35 | 483.54 | 484.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-04 11:45:00 | 485.40 | 483.54 | 484.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 12:15:00 | 485.60 | 483.95 | 484.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-04 12:45:00 | 485.80 | 483.95 | 484.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 13:15:00 | 485.10 | 484.18 | 484.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-04 14:45:00 | 484.95 | 484.04 | 484.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-05 11:15:00 | 485.90 | 484.33 | 484.52 | SL hit (close>static) qty=1.00 sl=485.85 alert=retest2 |

### Cycle 17 — BUY (started 2023-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 13:15:00 | 485.85 | 484.86 | 484.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 09:15:00 | 487.65 | 485.52 | 485.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-06 12:15:00 | 486.20 | 486.20 | 485.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-06 13:00:00 | 486.20 | 486.20 | 485.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 14:15:00 | 487.00 | 486.38 | 485.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-06 14:45:00 | 485.60 | 486.38 | 485.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 15:15:00 | 485.05 | 486.11 | 485.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 09:15:00 | 485.50 | 486.11 | 485.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 488.00 | 486.49 | 485.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 09:45:00 | 486.05 | 486.49 | 485.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 10:15:00 | 484.50 | 486.09 | 485.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 10:45:00 | 484.85 | 486.09 | 485.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 11:15:00 | 483.65 | 485.60 | 485.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 11:45:00 | 483.75 | 485.60 | 485.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2023-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 12:15:00 | 483.75 | 485.23 | 485.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 13:15:00 | 483.00 | 484.79 | 485.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 09:15:00 | 486.00 | 484.59 | 484.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-10 09:15:00 | 486.00 | 484.59 | 484.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 486.00 | 484.59 | 484.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-10 09:45:00 | 485.80 | 484.59 | 484.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 10:15:00 | 486.00 | 484.87 | 485.05 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 12:15:00 | 486.90 | 485.43 | 485.29 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-11 15:15:00 | 483.50 | 485.28 | 485.45 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 13:15:00 | 487.20 | 485.62 | 485.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-14 09:15:00 | 487.50 | 486.22 | 485.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-17 14:15:00 | 488.30 | 488.99 | 488.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 14:15:00 | 488.30 | 488.99 | 488.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 14:15:00 | 488.30 | 488.99 | 488.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 14:45:00 | 488.20 | 488.99 | 488.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 15:15:00 | 487.75 | 488.74 | 488.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 09:15:00 | 489.40 | 488.74 | 488.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-18 11:15:00 | 485.50 | 487.61 | 487.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2023-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 11:15:00 | 485.50 | 487.61 | 487.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-18 12:15:00 | 483.45 | 486.77 | 487.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-19 09:15:00 | 486.90 | 486.14 | 486.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-19 09:15:00 | 486.90 | 486.14 | 486.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 09:15:00 | 486.90 | 486.14 | 486.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-19 10:00:00 | 486.90 | 486.14 | 486.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 10:15:00 | 486.00 | 486.12 | 486.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-19 11:15:00 | 487.90 | 486.12 | 486.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 11:15:00 | 486.20 | 486.13 | 486.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-19 15:15:00 | 484.75 | 486.41 | 486.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-20 09:30:00 | 485.05 | 485.81 | 486.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-20 10:00:00 | 484.75 | 485.81 | 486.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-20 11:15:00 | 485.00 | 485.74 | 486.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 13:15:00 | 482.75 | 482.10 | 483.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-24 09:15:00 | 480.35 | 483.00 | 483.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-25 09:15:00 | 485.70 | 483.68 | 483.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2023-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 09:15:00 | 485.70 | 483.68 | 483.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-26 14:15:00 | 488.05 | 486.59 | 485.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-28 10:15:00 | 489.05 | 490.84 | 489.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-28 10:15:00 | 489.05 | 490.84 | 489.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 10:15:00 | 489.05 | 490.84 | 489.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-28 11:00:00 | 489.05 | 490.84 | 489.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 11:15:00 | 489.50 | 490.57 | 489.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-28 11:30:00 | 489.05 | 490.57 | 489.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 12:15:00 | 493.65 | 491.19 | 489.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-28 12:30:00 | 493.65 | 491.19 | 489.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 11:15:00 | 490.45 | 492.52 | 491.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-31 12:00:00 | 490.45 | 492.52 | 491.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 12:15:00 | 493.45 | 492.71 | 491.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-31 12:30:00 | 492.25 | 492.71 | 491.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 09:15:00 | 492.30 | 493.34 | 492.20 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 13:15:00 | 489.20 | 491.24 | 491.48 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-02 10:15:00 | 492.95 | 491.77 | 491.65 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 11:15:00 | 490.45 | 491.51 | 491.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 13:15:00 | 486.60 | 490.53 | 491.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 09:15:00 | 491.50 | 490.36 | 490.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-03 09:15:00 | 491.50 | 490.36 | 490.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 09:15:00 | 491.50 | 490.36 | 490.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-03 09:45:00 | 491.55 | 490.36 | 490.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 10:15:00 | 490.60 | 490.41 | 490.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-03 10:30:00 | 491.00 | 490.41 | 490.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 11:15:00 | 492.00 | 490.73 | 490.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-03 11:30:00 | 491.00 | 490.73 | 490.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 12:15:00 | 491.45 | 490.87 | 490.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-03 12:30:00 | 492.70 | 490.87 | 490.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 14:15:00 | 489.20 | 490.40 | 490.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 10:15:00 | 488.50 | 489.94 | 490.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 12:30:00 | 486.95 | 489.36 | 490.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 13:00:00 | 488.00 | 489.36 | 490.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-07 09:15:00 | 494.50 | 490.06 | 490.09 | SL hit (close>static) qty=1.00 sl=490.90 alert=retest2 |

### Cycle 27 — BUY (started 2023-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 10:15:00 | 491.60 | 490.36 | 490.23 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-07 13:15:00 | 488.65 | 489.88 | 490.04 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 15:15:00 | 492.70 | 490.45 | 490.27 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 10:15:00 | 487.50 | 489.63 | 489.91 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 09:15:00 | 491.35 | 490.04 | 489.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-10 12:15:00 | 502.65 | 493.67 | 491.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-11 14:15:00 | 500.25 | 501.96 | 498.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-11 15:00:00 | 500.25 | 501.96 | 498.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 15:15:00 | 500.45 | 501.66 | 498.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-14 09:15:00 | 505.00 | 501.66 | 498.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-23 10:15:00 | 520.00 | 520.77 | 520.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2023-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 10:15:00 | 520.00 | 520.77 | 520.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-23 11:15:00 | 519.05 | 520.43 | 520.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-23 14:15:00 | 520.95 | 520.31 | 520.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 14:15:00 | 520.95 | 520.31 | 520.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 14:15:00 | 520.95 | 520.31 | 520.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 15:00:00 | 520.95 | 520.31 | 520.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2023-08-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 15:15:00 | 523.00 | 520.85 | 520.74 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-08-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 13:15:00 | 519.85 | 520.63 | 520.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-24 14:15:00 | 516.65 | 519.83 | 520.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-25 14:15:00 | 518.35 | 510.61 | 514.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 14:15:00 | 518.35 | 510.61 | 514.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 14:15:00 | 518.35 | 510.61 | 514.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-25 15:00:00 | 518.35 | 510.61 | 514.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 15:15:00 | 519.00 | 512.29 | 514.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-28 09:45:00 | 513.85 | 512.64 | 514.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-28 10:30:00 | 514.50 | 513.88 | 514.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-28 13:30:00 | 514.90 | 514.51 | 514.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-29 09:45:00 | 514.35 | 514.62 | 514.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 505.55 | 509.07 | 510.64 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-09-01 09:15:00 | 514.50 | 511.18 | 510.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2023-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 09:15:00 | 514.50 | 511.18 | 510.94 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 15:15:00 | 506.75 | 511.37 | 511.77 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-09-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 11:15:00 | 512.65 | 511.01 | 510.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 14:15:00 | 517.15 | 512.97 | 511.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-11 09:15:00 | 515.10 | 516.31 | 515.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 09:15:00 | 515.10 | 516.31 | 515.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 09:15:00 | 515.10 | 516.31 | 515.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-11 10:00:00 | 515.10 | 516.31 | 515.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 10:15:00 | 514.95 | 516.04 | 515.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-11 11:15:00 | 515.65 | 516.04 | 515.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 11:15:00 | 515.60 | 515.95 | 515.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 15:00:00 | 518.55 | 516.44 | 515.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-12 10:15:00 | 518.50 | 517.04 | 516.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 13:15:00 | 510.65 | 515.93 | 515.90 | SL hit (close<static) qty=1.00 sl=513.65 alert=retest2 |

### Cycle 38 — SELL (started 2023-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 14:15:00 | 514.70 | 515.68 | 515.79 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-09-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 12:15:00 | 518.60 | 516.24 | 515.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 09:15:00 | 529.65 | 519.71 | 517.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 15:15:00 | 552.00 | 552.48 | 544.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-20 09:15:00 | 548.40 | 552.48 | 544.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 543.00 | 550.59 | 544.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 10:00:00 | 543.00 | 550.59 | 544.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 10:15:00 | 543.10 | 549.09 | 544.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 10:45:00 | 542.40 | 549.09 | 544.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 13:15:00 | 543.65 | 547.01 | 544.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 14:00:00 | 543.65 | 547.01 | 544.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 14:15:00 | 543.90 | 546.39 | 544.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 15:15:00 | 540.10 | 546.39 | 544.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 15:15:00 | 540.10 | 545.13 | 544.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 09:15:00 | 538.00 | 545.13 | 544.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2023-09-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 09:15:00 | 533.70 | 542.85 | 543.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 09:15:00 | 529.00 | 537.19 | 539.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 11:15:00 | 530.70 | 530.05 | 533.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-25 12:00:00 | 530.70 | 530.05 | 533.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 12:15:00 | 528.20 | 529.68 | 533.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-25 15:15:00 | 526.00 | 529.48 | 532.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-26 09:30:00 | 526.90 | 528.67 | 531.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-26 11:15:00 | 537.25 | 530.40 | 531.86 | SL hit (close>static) qty=1.00 sl=533.55 alert=retest2 |

### Cycle 41 — BUY (started 2023-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 10:15:00 | 536.70 | 533.06 | 532.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 12:15:00 | 540.10 | 535.18 | 533.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-27 15:15:00 | 535.00 | 535.20 | 534.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-28 09:15:00 | 534.35 | 535.20 | 534.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 09:15:00 | 539.70 | 536.10 | 534.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-28 12:15:00 | 540.55 | 537.45 | 535.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-28 14:15:00 | 540.95 | 538.95 | 536.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-28 15:00:00 | 541.40 | 539.44 | 537.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-10 11:15:00 | 562.00 | 564.35 | 564.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2023-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-10 11:15:00 | 562.00 | 564.35 | 564.60 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2023-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 10:15:00 | 570.10 | 565.34 | 564.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 11:15:00 | 572.90 | 566.85 | 565.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-16 09:15:00 | 584.45 | 588.32 | 584.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-16 09:15:00 | 584.45 | 588.32 | 584.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 584.45 | 588.32 | 584.01 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2023-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 13:15:00 | 580.85 | 583.14 | 583.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 09:15:00 | 571.15 | 580.22 | 581.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-18 14:15:00 | 586.45 | 576.29 | 578.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 14:15:00 | 586.45 | 576.29 | 578.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 14:15:00 | 586.45 | 576.29 | 578.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-18 15:00:00 | 586.45 | 576.29 | 578.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 15:15:00 | 576.50 | 576.33 | 578.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-19 09:15:00 | 570.25 | 576.33 | 578.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-20 09:15:00 | 599.45 | 579.13 | 577.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2023-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-20 09:15:00 | 599.45 | 579.13 | 577.97 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2023-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 12:15:00 | 569.00 | 579.98 | 580.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 13:15:00 | 567.70 | 577.52 | 579.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 10:15:00 | 579.65 | 576.81 | 578.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 10:15:00 | 579.65 | 576.81 | 578.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 10:15:00 | 579.65 | 576.81 | 578.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-25 10:30:00 | 579.45 | 576.81 | 578.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 11:15:00 | 578.95 | 577.24 | 578.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-25 12:30:00 | 574.80 | 574.93 | 577.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 546.06 | 569.51 | 574.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-26 13:15:00 | 569.15 | 563.40 | 569.07 | SL hit (close>ema200) qty=0.50 sl=563.40 alert=retest2 |

### Cycle 47 — BUY (started 2023-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 13:15:00 | 570.00 | 567.30 | 566.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 14:15:00 | 576.45 | 569.13 | 567.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 15:15:00 | 570.60 | 571.77 | 570.28 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 09:15:00 | 575.95 | 571.77 | 570.28 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 10:15:00 | 575.50 | 576.61 | 574.24 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 10:15:00 | 571.65 | 575.62 | 574.00 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-11-03 10:15:00 | 571.65 | 575.62 | 574.00 | SL hit (close<ema400) qty=1.00 sl=574.00 alert=retest1 |

### Cycle 48 — SELL (started 2023-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 12:15:00 | 561.00 | 570.84 | 571.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 14:15:00 | 556.30 | 566.27 | 569.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 10:15:00 | 564.90 | 563.77 | 567.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-06 11:00:00 | 564.90 | 563.77 | 567.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 12:15:00 | 570.95 | 565.24 | 567.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 12:45:00 | 570.55 | 565.24 | 567.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 13:15:00 | 570.00 | 566.19 | 567.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 13:30:00 | 571.50 | 566.19 | 567.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2023-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-07 09:15:00 | 574.95 | 569.54 | 568.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-07 13:15:00 | 579.05 | 573.95 | 571.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 14:15:00 | 572.90 | 573.74 | 571.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-07 15:00:00 | 572.90 | 573.74 | 571.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 15:15:00 | 569.20 | 572.83 | 571.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 11:15:00 | 574.45 | 572.48 | 571.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-08 15:15:00 | 565.90 | 571.13 | 571.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2023-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 15:15:00 | 565.90 | 571.13 | 571.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 09:15:00 | 562.55 | 569.41 | 570.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 10:15:00 | 571.85 | 562.44 | 565.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 10:15:00 | 571.85 | 562.44 | 565.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 10:15:00 | 571.85 | 562.44 | 565.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 10:45:00 | 570.00 | 562.44 | 565.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 11:15:00 | 571.30 | 564.22 | 565.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 12:15:00 | 571.15 | 564.22 | 565.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2023-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 12:15:00 | 576.10 | 566.59 | 566.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-10 13:15:00 | 581.10 | 569.49 | 567.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-13 14:15:00 | 578.50 | 578.60 | 575.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-13 15:00:00 | 578.50 | 578.60 | 575.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 577.95 | 578.37 | 575.54 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2023-11-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-15 14:15:00 | 570.00 | 573.63 | 574.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-15 15:15:00 | 568.00 | 572.50 | 573.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-16 09:15:00 | 573.55 | 572.71 | 573.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 09:15:00 | 573.55 | 572.71 | 573.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 09:15:00 | 573.55 | 572.71 | 573.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-16 09:45:00 | 573.50 | 572.71 | 573.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 10:15:00 | 574.00 | 572.97 | 573.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-16 11:00:00 | 574.00 | 572.97 | 573.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 11:15:00 | 573.30 | 573.04 | 573.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-16 14:00:00 | 570.30 | 572.53 | 573.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-17 11:15:00 | 576.45 | 573.81 | 573.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2023-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 11:15:00 | 576.45 | 573.81 | 573.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-17 13:15:00 | 579.75 | 575.52 | 574.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-21 15:15:00 | 595.75 | 597.90 | 591.87 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-22 09:15:00 | 615.00 | 597.90 | 591.87 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 632.80 | 619.24 | 613.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 09:30:00 | 624.65 | 619.24 | 613.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-28 10:15:00 | 645.75 | 624.80 | 616.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2023-12-01 09:15:00 | 676.50 | 662.28 | 653.68 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 54 — SELL (started 2023-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 09:15:00 | 658.00 | 666.58 | 667.33 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2023-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 11:15:00 | 674.90 | 667.85 | 667.75 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2023-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 12:15:00 | 666.30 | 667.54 | 667.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-06 13:15:00 | 661.90 | 666.41 | 667.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-06 14:15:00 | 668.50 | 666.83 | 667.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-06 14:15:00 | 668.50 | 666.83 | 667.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 14:15:00 | 668.50 | 666.83 | 667.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-06 15:00:00 | 668.50 | 666.83 | 667.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 15:15:00 | 661.00 | 665.66 | 666.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-07 09:15:00 | 660.25 | 665.66 | 666.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-07 13:15:00 | 670.00 | 666.84 | 666.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2023-12-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 13:15:00 | 670.00 | 666.84 | 666.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 14:15:00 | 685.05 | 670.48 | 668.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 10:15:00 | 674.10 | 674.22 | 670.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-08 11:00:00 | 674.10 | 674.22 | 670.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 12:15:00 | 669.50 | 673.22 | 671.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 13:00:00 | 669.50 | 673.22 | 671.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 13:15:00 | 664.00 | 671.38 | 670.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 14:00:00 | 664.00 | 671.38 | 670.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2023-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 15:15:00 | 665.25 | 669.82 | 669.86 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2023-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 09:15:00 | 671.00 | 670.06 | 669.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 14:15:00 | 690.00 | 676.12 | 672.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 13:15:00 | 681.35 | 681.54 | 677.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-12 14:00:00 | 681.35 | 681.54 | 677.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 09:15:00 | 685.60 | 685.77 | 680.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-13 09:45:00 | 685.65 | 685.77 | 680.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 13:15:00 | 684.70 | 684.85 | 681.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-13 13:45:00 | 684.60 | 684.85 | 681.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 10:15:00 | 680.60 | 684.39 | 682.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-14 10:45:00 | 680.20 | 684.39 | 682.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 11:15:00 | 677.55 | 683.02 | 682.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-14 11:30:00 | 677.35 | 683.02 | 682.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2023-12-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-14 13:15:00 | 673.00 | 680.06 | 681.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-14 14:15:00 | 672.00 | 678.44 | 680.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-15 09:15:00 | 681.45 | 678.49 | 679.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-15 09:15:00 | 681.45 | 678.49 | 679.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 09:15:00 | 681.45 | 678.49 | 679.87 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2023-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 11:15:00 | 697.60 | 682.68 | 681.55 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2023-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 14:15:00 | 688.30 | 689.48 | 689.51 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2023-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 15:15:00 | 691.95 | 689.97 | 689.73 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2023-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 09:15:00 | 683.00 | 688.58 | 689.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 12:15:00 | 677.75 | 684.34 | 686.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 14:15:00 | 678.50 | 671.51 | 676.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-21 14:15:00 | 678.50 | 671.51 | 676.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 14:15:00 | 678.50 | 671.51 | 676.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 15:00:00 | 678.50 | 671.51 | 676.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 15:15:00 | 668.00 | 670.81 | 675.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 09:15:00 | 680.25 | 670.81 | 675.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 677.20 | 672.09 | 676.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 09:30:00 | 682.75 | 672.09 | 676.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 692.85 | 676.24 | 677.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 11:00:00 | 692.85 | 676.24 | 677.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2023-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 11:15:00 | 694.00 | 679.79 | 679.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 13:15:00 | 700.55 | 686.29 | 682.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-26 10:15:00 | 699.20 | 700.47 | 691.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-26 10:45:00 | 697.50 | 700.47 | 691.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 10:15:00 | 697.95 | 703.43 | 697.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 11:00:00 | 697.95 | 703.43 | 697.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 11:15:00 | 701.85 | 703.11 | 697.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 12:15:00 | 697.85 | 703.11 | 697.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 12:15:00 | 701.00 | 702.69 | 698.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 12:30:00 | 701.85 | 702.69 | 698.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 13:15:00 | 697.50 | 701.65 | 698.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 14:00:00 | 697.50 | 701.65 | 698.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 14:15:00 | 698.10 | 700.94 | 698.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 14:30:00 | 697.00 | 700.94 | 698.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 15:15:00 | 696.60 | 700.07 | 697.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-28 09:15:00 | 702.40 | 700.07 | 697.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-28 10:15:00 | 694.45 | 698.30 | 697.46 | SL hit (close<static) qty=1.00 sl=695.00 alert=retest2 |

### Cycle 66 — SELL (started 2024-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 10:15:00 | 729.40 | 739.97 | 740.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 12:15:00 | 723.00 | 734.79 | 737.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 10:15:00 | 730.95 | 728.59 | 732.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-09 11:00:00 | 730.95 | 728.59 | 732.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 13:15:00 | 735.00 | 728.97 | 731.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-09 14:00:00 | 735.00 | 728.97 | 731.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 14:15:00 | 728.85 | 728.94 | 731.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-10 12:15:00 | 723.40 | 728.97 | 730.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 14:00:00 | 726.00 | 729.50 | 730.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 14:30:00 | 721.80 | 728.65 | 729.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 15:00:00 | 725.25 | 728.65 | 729.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 15:15:00 | 722.90 | 727.50 | 729.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-12 09:15:00 | 720.10 | 727.50 | 729.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-12 10:30:00 | 716.75 | 723.51 | 727.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-16 10:15:00 | 687.23 | 701.03 | 709.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-16 10:15:00 | 689.70 | 701.03 | 709.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-16 10:15:00 | 685.71 | 701.03 | 709.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-16 10:15:00 | 688.99 | 701.03 | 709.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-16 12:15:00 | 684.10 | 694.36 | 704.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-16 12:15:00 | 680.91 | 694.36 | 704.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-01-17 15:15:00 | 653.40 | 680.25 | 689.25 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 67 — BUY (started 2024-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 12:15:00 | 688.55 | 685.73 | 685.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-20 09:15:00 | 707.45 | 690.91 | 688.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 09:15:00 | 691.10 | 700.95 | 696.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 09:15:00 | 691.10 | 700.95 | 696.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 691.10 | 700.95 | 696.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:00:00 | 691.10 | 700.95 | 696.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 690.00 | 698.76 | 695.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:45:00 | 691.80 | 698.76 | 695.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 11:15:00 | 684.00 | 695.81 | 694.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 12:00:00 | 684.00 | 695.81 | 694.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2024-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 14:15:00 | 689.80 | 693.41 | 693.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 15:15:00 | 675.50 | 689.83 | 692.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-25 09:15:00 | 688.80 | 687.04 | 689.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-25 09:15:00 | 688.80 | 687.04 | 689.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 688.80 | 687.04 | 689.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:30:00 | 686.00 | 687.04 | 689.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 10:15:00 | 682.55 | 686.14 | 688.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 12:30:00 | 677.15 | 684.82 | 687.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 15:15:00 | 675.00 | 685.04 | 687.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-29 12:45:00 | 679.85 | 683.29 | 685.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-29 13:15:00 | 679.35 | 683.29 | 685.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 13:15:00 | 679.50 | 682.54 | 684.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 14:00:00 | 679.50 | 682.54 | 684.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 14:15:00 | 685.00 | 683.03 | 684.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-29 15:15:00 | 677.45 | 683.03 | 684.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-30 11:15:00 | 692.65 | 684.95 | 685.02 | SL hit (close>static) qty=1.00 sl=688.90 alert=retest2 |

### Cycle 69 — BUY (started 2024-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 12:15:00 | 686.35 | 685.23 | 685.14 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 13:15:00 | 676.20 | 683.42 | 684.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-31 11:15:00 | 674.75 | 679.46 | 682.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-02 10:15:00 | 662.35 | 662.06 | 668.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-02 10:45:00 | 663.35 | 662.06 | 668.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 12:15:00 | 627.90 | 622.78 | 631.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-07 12:45:00 | 629.00 | 622.78 | 631.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 09:15:00 | 635.40 | 624.96 | 629.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 09:45:00 | 637.40 | 624.96 | 629.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 10:15:00 | 634.50 | 626.87 | 629.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 10:45:00 | 638.55 | 626.87 | 629.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 14:15:00 | 633.85 | 629.18 | 630.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 14:45:00 | 634.90 | 629.18 | 630.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 15:15:00 | 628.00 | 628.95 | 630.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-09 09:15:00 | 638.30 | 628.95 | 630.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 627.20 | 628.60 | 629.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-09 13:15:00 | 622.35 | 626.06 | 628.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-09 14:15:00 | 622.05 | 625.34 | 627.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-12 10:15:00 | 637.65 | 629.37 | 628.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2024-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-12 10:15:00 | 637.65 | 629.37 | 628.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-13 09:15:00 | 655.45 | 638.06 | 633.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-14 11:15:00 | 654.10 | 654.18 | 647.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-14 12:00:00 | 654.10 | 654.18 | 647.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 09:15:00 | 672.20 | 659.20 | 654.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-16 11:15:00 | 673.00 | 661.92 | 656.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 09:15:00 | 681.00 | 667.59 | 661.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 12:15:00 | 674.00 | 672.16 | 669.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 14:00:00 | 673.00 | 672.65 | 669.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 15:15:00 | 674.00 | 672.82 | 670.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 09:15:00 | 675.20 | 672.82 | 670.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-21 09:15:00 | 669.30 | 672.11 | 670.31 | SL hit (close<static) qty=1.00 sl=670.00 alert=retest2 |

### Cycle 72 — SELL (started 2024-02-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 09:15:00 | 667.25 | 669.27 | 669.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 13:15:00 | 664.25 | 667.43 | 668.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-23 14:15:00 | 665.00 | 664.66 | 666.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-23 14:15:00 | 665.00 | 664.66 | 666.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 14:15:00 | 665.00 | 664.66 | 666.12 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2024-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-26 11:15:00 | 668.90 | 667.14 | 666.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-26 12:15:00 | 671.50 | 668.01 | 667.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 15:15:00 | 667.00 | 668.47 | 667.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 15:15:00 | 667.00 | 668.47 | 667.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 15:15:00 | 667.00 | 668.47 | 667.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 09:15:00 | 669.50 | 668.47 | 667.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 665.00 | 667.78 | 667.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 10:00:00 | 665.00 | 667.78 | 667.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2024-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 10:15:00 | 664.75 | 667.17 | 667.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 13:15:00 | 659.50 | 664.39 | 665.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 14:15:00 | 665.00 | 664.51 | 665.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-27 15:00:00 | 665.00 | 664.51 | 665.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 661.35 | 663.16 | 664.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 10:30:00 | 653.50 | 659.83 | 663.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-01 09:15:00 | 694.50 | 647.23 | 648.40 | SL hit (close>static) qty=1.00 sl=667.60 alert=retest2 |

### Cycle 75 — BUY (started 2024-03-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 10:15:00 | 677.85 | 653.36 | 651.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 15:15:00 | 706.00 | 676.30 | 664.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 11:15:00 | 700.00 | 700.59 | 684.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-04 12:00:00 | 700.00 | 700.59 | 684.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 15:15:00 | 694.00 | 698.62 | 688.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 09:15:00 | 686.40 | 698.62 | 688.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 09:15:00 | 672.95 | 693.49 | 687.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 10:00:00 | 672.95 | 693.49 | 687.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 10:15:00 | 669.00 | 688.59 | 685.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 10:30:00 | 670.80 | 688.59 | 685.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2024-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 12:15:00 | 667.30 | 681.06 | 682.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 14:15:00 | 655.45 | 673.51 | 678.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 12:15:00 | 669.50 | 665.91 | 672.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 12:15:00 | 669.50 | 665.91 | 672.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 12:15:00 | 669.50 | 665.91 | 672.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 13:00:00 | 669.50 | 665.91 | 672.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 13:15:00 | 668.55 | 666.43 | 671.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 14:00:00 | 668.55 | 666.43 | 671.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 14:15:00 | 672.80 | 667.71 | 671.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 15:00:00 | 672.80 | 667.71 | 671.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 15:15:00 | 677.30 | 669.63 | 672.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 09:15:00 | 669.95 | 669.63 | 672.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 10:00:00 | 671.10 | 669.92 | 672.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 11:00:00 | 670.75 | 670.09 | 672.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 12:15:00 | 670.05 | 670.47 | 672.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 12:15:00 | 671.10 | 670.60 | 672.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 14:00:00 | 666.00 | 669.68 | 671.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 09:15:00 | 636.45 | 649.66 | 654.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 09:15:00 | 637.54 | 649.66 | 654.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 09:15:00 | 637.21 | 649.66 | 654.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 09:15:00 | 636.55 | 649.66 | 654.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 09:15:00 | 632.70 | 649.66 | 654.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-14 11:15:00 | 639.00 | 636.82 | 642.97 | SL hit (close>ema200) qty=0.50 sl=636.82 alert=retest2 |

### Cycle 77 — BUY (started 2024-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 09:15:00 | 630.40 | 619.01 | 618.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 10:15:00 | 633.80 | 621.97 | 620.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 15:15:00 | 638.25 | 643.37 | 638.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 15:15:00 | 638.25 | 643.37 | 638.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 15:15:00 | 638.25 | 643.37 | 638.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 09:15:00 | 649.20 | 641.30 | 639.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 14:15:00 | 648.10 | 647.44 | 644.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-04 09:15:00 | 655.45 | 661.12 | 661.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2024-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 09:15:00 | 655.45 | 661.12 | 661.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 11:15:00 | 644.40 | 656.37 | 659.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-05 10:15:00 | 652.05 | 650.62 | 654.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 10:15:00 | 652.05 | 650.62 | 654.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 10:15:00 | 652.05 | 650.62 | 654.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-05 10:30:00 | 656.55 | 650.62 | 654.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 09:15:00 | 645.10 | 647.86 | 651.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-08 10:45:00 | 642.00 | 646.50 | 650.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-12 09:15:00 | 609.90 | 619.32 | 626.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-12 10:15:00 | 626.90 | 620.84 | 626.82 | SL hit (close>ema200) qty=0.50 sl=620.84 alert=retest2 |

### Cycle 79 — BUY (started 2024-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 10:15:00 | 636.00 | 627.28 | 627.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 10:15:00 | 640.20 | 634.01 | 631.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 09:15:00 | 637.75 | 642.74 | 637.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 09:15:00 | 637.75 | 642.74 | 637.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 637.75 | 642.74 | 637.70 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2024-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-22 15:15:00 | 633.55 | 639.09 | 639.16 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2024-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 12:15:00 | 644.00 | 639.53 | 639.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 15:15:00 | 646.10 | 641.41 | 640.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 14:15:00 | 640.90 | 643.31 | 641.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-24 14:15:00 | 640.90 | 643.31 | 641.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 14:15:00 | 640.90 | 643.31 | 641.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-24 15:00:00 | 640.90 | 643.31 | 641.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 15:15:00 | 642.00 | 643.04 | 641.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 11:00:00 | 642.95 | 642.23 | 641.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 10:00:00 | 642.50 | 654.04 | 650.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 10:30:00 | 643.30 | 652.44 | 650.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-29 15:15:00 | 646.55 | 649.37 | 649.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2024-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 15:15:00 | 646.55 | 649.37 | 649.40 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 14:15:00 | 660.35 | 650.50 | 649.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-02 14:15:00 | 671.00 | 661.53 | 656.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-03 09:15:00 | 662.20 | 663.02 | 658.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-03 10:00:00 | 662.20 | 663.02 | 658.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 13:15:00 | 660.50 | 662.70 | 659.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 14:00:00 | 660.50 | 662.70 | 659.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 14:15:00 | 673.70 | 664.90 | 660.88 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2024-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 15:15:00 | 657.00 | 663.13 | 663.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-08 15:15:00 | 654.70 | 661.74 | 662.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-09 11:15:00 | 661.35 | 661.04 | 662.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-09 11:15:00 | 661.35 | 661.04 | 662.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 11:15:00 | 661.35 | 661.04 | 662.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-09 11:45:00 | 660.10 | 661.04 | 662.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 12:15:00 | 657.15 | 660.26 | 661.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 13:15:00 | 651.00 | 660.26 | 661.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 14:15:00 | 663.65 | 650.06 | 651.47 | SL hit (close>static) qty=1.00 sl=662.90 alert=retest2 |

### Cycle 85 — BUY (started 2024-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 15:15:00 | 663.00 | 652.65 | 652.52 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 14:15:00 | 649.40 | 654.35 | 654.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 09:15:00 | 638.90 | 650.25 | 652.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-17 14:15:00 | 644.15 | 642.05 | 645.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-17 14:15:00 | 644.15 | 642.05 | 645.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 14:15:00 | 644.15 | 642.05 | 645.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 15:00:00 | 644.15 | 642.05 | 645.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 15:15:00 | 644.00 | 642.44 | 645.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-18 09:15:00 | 646.95 | 642.44 | 645.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 648.85 | 643.72 | 645.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-18 11:30:00 | 650.70 | 644.48 | 645.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 12:15:00 | 645.00 | 644.58 | 645.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-21 09:30:00 | 636.70 | 642.04 | 644.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-22 13:15:00 | 604.87 | 620.98 | 630.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-22 15:15:00 | 623.00 | 621.29 | 628.79 | SL hit (close>ema200) qty=0.50 sl=621.29 alert=retest2 |

### Cycle 87 — BUY (started 2024-05-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 13:15:00 | 632.95 | 630.48 | 630.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 10:15:00 | 634.00 | 631.78 | 630.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 15:15:00 | 643.20 | 647.16 | 643.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 09:15:00 | 642.10 | 646.15 | 642.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 642.10 | 646.15 | 642.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 10:00:00 | 642.10 | 646.15 | 642.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 642.65 | 645.45 | 642.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 11:15:00 | 640.25 | 645.45 | 642.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 11:15:00 | 641.40 | 644.64 | 642.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 12:00:00 | 641.40 | 644.64 | 642.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 12:15:00 | 643.75 | 644.46 | 642.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 12:45:00 | 643.70 | 644.46 | 642.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 13:15:00 | 636.35 | 642.84 | 642.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 14:00:00 | 636.35 | 642.84 | 642.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2024-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 14:15:00 | 633.25 | 640.92 | 641.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 09:15:00 | 618.05 | 634.68 | 638.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 12:15:00 | 615.20 | 613.02 | 621.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-03 13:00:00 | 615.20 | 613.02 | 621.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 15:15:00 | 610.65 | 611.91 | 619.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:15:00 | 614.40 | 611.91 | 619.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 620.05 | 613.54 | 619.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:45:00 | 621.65 | 613.54 | 619.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 609.60 | 612.75 | 618.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 12:15:00 | 604.05 | 612.21 | 617.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-05 09:15:00 | 633.65 | 619.54 | 619.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2024-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 09:15:00 | 633.65 | 619.54 | 619.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 645.15 | 636.67 | 631.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 09:15:00 | 672.95 | 678.59 | 671.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-12 09:15:00 | 672.95 | 678.59 | 671.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 672.95 | 678.59 | 671.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 10:00:00 | 672.95 | 678.59 | 671.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 10:15:00 | 666.60 | 676.19 | 670.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 10:30:00 | 667.90 | 676.19 | 670.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 11:15:00 | 672.30 | 675.41 | 670.81 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 11:15:00 | 660.75 | 670.07 | 670.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-13 13:15:00 | 659.55 | 666.50 | 668.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-13 14:15:00 | 666.85 | 666.57 | 668.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-13 15:00:00 | 666.85 | 666.57 | 668.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 15:15:00 | 673.00 | 667.86 | 668.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:15:00 | 698.10 | 667.86 | 668.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2024-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 09:15:00 | 691.70 | 672.63 | 670.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 14:15:00 | 715.95 | 694.18 | 683.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 13:15:00 | 740.20 | 741.31 | 726.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 11:15:00 | 726.90 | 734.45 | 728.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 726.90 | 734.45 | 728.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 12:00:00 | 726.90 | 734.45 | 728.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 12:15:00 | 712.60 | 730.08 | 726.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 13:00:00 | 712.60 | 730.08 | 726.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 13:15:00 | 713.30 | 726.72 | 725.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 14:15:00 | 722.00 | 726.72 | 725.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 09:15:00 | 713.75 | 723.14 | 724.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 09:15:00 | 713.75 | 723.14 | 724.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 10:15:00 | 710.10 | 720.53 | 722.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 13:15:00 | 719.15 | 719.07 | 721.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-21 14:00:00 | 719.15 | 719.07 | 721.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 713.00 | 717.86 | 720.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 15:00:00 | 713.00 | 717.86 | 720.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 710.80 | 714.76 | 718.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 12:00:00 | 700.15 | 710.98 | 716.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 14:15:00 | 720.30 | 711.18 | 711.51 | SL hit (close>static) qty=1.00 sl=719.00 alert=retest2 |

### Cycle 93 — BUY (started 2024-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 15:15:00 | 720.70 | 713.09 | 712.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 09:15:00 | 731.90 | 716.85 | 714.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 13:15:00 | 830.95 | 831.18 | 818.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-03 13:45:00 | 829.95 | 831.18 | 818.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 813.05 | 825.53 | 819.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:45:00 | 811.20 | 825.53 | 819.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 812.00 | 822.82 | 818.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:45:00 | 812.30 | 822.82 | 818.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 817.85 | 821.83 | 818.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 12:45:00 | 818.70 | 821.32 | 818.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 13:45:00 | 819.25 | 820.51 | 818.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 09:15:00 | 807.30 | 816.45 | 816.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 09:15:00 | 807.30 | 816.45 | 816.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 09:15:00 | 798.25 | 810.69 | 813.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 12:15:00 | 808.85 | 799.77 | 803.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 12:15:00 | 808.85 | 799.77 | 803.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 12:15:00 | 808.85 | 799.77 | 803.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 13:00:00 | 808.85 | 799.77 | 803.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 817.45 | 803.31 | 805.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 13:30:00 | 823.95 | 803.31 | 805.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 09:15:00 | 810.55 | 806.82 | 806.39 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 797.25 | 804.90 | 805.56 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 11:15:00 | 819.30 | 807.78 | 806.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 09:15:00 | 823.70 | 816.24 | 811.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 10:15:00 | 841.05 | 842.96 | 831.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-12 10:45:00 | 839.00 | 842.96 | 831.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 842.15 | 844.08 | 838.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 12:30:00 | 845.50 | 844.50 | 839.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 09:15:00 | 850.00 | 843.24 | 839.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 10:00:00 | 854.60 | 865.49 | 859.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 12:15:00 | 834.05 | 852.84 | 854.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2024-07-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 12:15:00 | 834.05 | 852.84 | 854.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 13:15:00 | 830.25 | 848.32 | 852.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 11:15:00 | 841.80 | 837.30 | 844.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 11:15:00 | 841.80 | 837.30 | 844.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 841.80 | 837.30 | 844.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:45:00 | 844.90 | 837.30 | 844.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 838.55 | 837.55 | 843.50 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 09:15:00 | 861.95 | 846.21 | 845.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 12:15:00 | 883.95 | 868.39 | 859.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 14:15:00 | 921.10 | 922.84 | 905.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 15:00:00 | 921.10 | 922.84 | 905.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 10:15:00 | 903.55 | 920.71 | 908.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 11:00:00 | 903.55 | 920.71 | 908.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 11:15:00 | 890.50 | 914.67 | 907.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 12:00:00 | 890.50 | 914.67 | 907.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 984.45 | 987.57 | 970.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:45:00 | 975.85 | 987.57 | 970.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 977.90 | 985.64 | 970.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:30:00 | 976.70 | 985.64 | 970.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 988.00 | 984.37 | 972.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:30:00 | 975.60 | 984.37 | 972.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 12:15:00 | 1005.75 | 991.44 | 981.02 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2024-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 13:15:00 | 967.85 | 982.61 | 984.08 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 11:15:00 | 995.25 | 985.31 | 984.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 12:15:00 | 1000.65 | 988.38 | 985.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-07 15:15:00 | 990.50 | 992.29 | 988.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 15:15:00 | 990.50 | 992.29 | 988.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 990.50 | 992.29 | 988.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 1004.35 | 992.29 | 988.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-08 10:00:00 | 1000.50 | 993.93 | 989.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-08 15:15:00 | 1001.10 | 997.13 | 993.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 13:15:00 | 980.40 | 991.07 | 992.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2024-08-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 13:15:00 | 980.40 | 991.07 | 992.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 14:15:00 | 974.80 | 987.82 | 990.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 09:15:00 | 1013.95 | 990.79 | 991.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 09:15:00 | 1013.95 | 990.79 | 991.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 1013.95 | 990.79 | 991.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:00:00 | 1013.95 | 990.79 | 991.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 10:15:00 | 1019.60 | 996.55 | 993.92 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 10:15:00 | 991.15 | 1005.52 | 1005.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 12:15:00 | 977.70 | 998.01 | 1002.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 15:15:00 | 1011.00 | 995.62 | 999.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 15:15:00 | 1011.00 | 995.62 | 999.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 15:15:00 | 1011.00 | 995.62 | 999.58 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 13:15:00 | 1002.55 | 1001.07 | 1001.01 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 14:15:00 | 1000.60 | 1000.97 | 1000.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-16 15:15:00 | 1000.00 | 1000.78 | 1000.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 14:15:00 | 977.10 | 974.62 | 980.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-20 15:00:00 | 977.10 | 974.62 | 980.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 1015.30 | 983.13 | 983.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 10:00:00 | 1015.30 | 983.13 | 983.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2024-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 10:15:00 | 996.90 | 985.89 | 984.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 11:15:00 | 1023.00 | 993.31 | 988.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-26 10:15:00 | 1060.95 | 1064.39 | 1049.70 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 11:30:00 | 1067.85 | 1064.66 | 1051.16 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 13:00:00 | 1066.65 | 1065.06 | 1052.57 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 13:15:00 | 1065.00 | 1070.20 | 1063.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 13:30:00 | 1065.00 | 1070.20 | 1063.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 14:15:00 | 1076.00 | 1071.36 | 1064.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 14:45:00 | 1065.00 | 1071.36 | 1064.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 1065.00 | 1070.09 | 1064.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:15:00 | 1056.20 | 1070.09 | 1064.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 1057.80 | 1067.63 | 1063.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-28 09:15:00 | 1057.80 | 1067.63 | 1063.70 | SL hit (close<ema400) qty=1.00 sl=1063.70 alert=retest1 |

### Cycle 108 — SELL (started 2024-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 15:15:00 | 1060.50 | 1062.84 | 1063.00 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 1071.35 | 1064.54 | 1063.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 13:15:00 | 1090.90 | 1072.31 | 1067.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 14:15:00 | 1070.75 | 1072.00 | 1068.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-30 15:00:00 | 1070.75 | 1072.00 | 1068.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 1090.00 | 1075.60 | 1070.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:15:00 | 1091.00 | 1075.60 | 1070.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 1149.75 | 1090.43 | 1077.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 09:15:00 | 1181.75 | 1143.54 | 1127.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 09:15:00 | 1180.70 | 1192.46 | 1192.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 09:15:00 | 1180.70 | 1192.46 | 1192.52 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 10:15:00 | 1204.20 | 1192.23 | 1191.38 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2024-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 12:15:00 | 1183.95 | 1192.99 | 1193.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 15:15:00 | 1174.75 | 1185.90 | 1189.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 1155.20 | 1150.36 | 1159.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 1155.20 | 1150.36 | 1159.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 1155.20 | 1150.36 | 1159.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:00:00 | 1155.20 | 1150.36 | 1159.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 1215.00 | 1163.29 | 1164.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:00:00 | 1215.00 | 1163.29 | 1164.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 11:15:00 | 1188.80 | 1168.39 | 1166.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 09:15:00 | 1249.60 | 1212.88 | 1196.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 09:15:00 | 1234.05 | 1236.34 | 1218.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 10:00:00 | 1234.05 | 1236.34 | 1218.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 1217.00 | 1231.45 | 1219.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 12:00:00 | 1217.00 | 1231.45 | 1219.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 1216.40 | 1228.44 | 1219.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 13:00:00 | 1216.40 | 1228.44 | 1219.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2024-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 15:15:00 | 1195.00 | 1213.60 | 1214.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 11:15:00 | 1173.75 | 1198.66 | 1206.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 14:15:00 | 1198.85 | 1192.20 | 1201.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 14:15:00 | 1198.85 | 1192.20 | 1201.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 1198.85 | 1192.20 | 1201.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 15:00:00 | 1198.85 | 1192.20 | 1201.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 1192.60 | 1192.28 | 1200.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 09:45:00 | 1179.10 | 1189.72 | 1198.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 11:45:00 | 1177.00 | 1186.99 | 1195.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 12:30:00 | 1177.70 | 1184.68 | 1193.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 09:45:00 | 1178.40 | 1182.17 | 1189.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 1181.85 | 1172.68 | 1180.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:45:00 | 1180.75 | 1172.68 | 1180.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 1183.55 | 1174.85 | 1180.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:30:00 | 1199.90 | 1174.85 | 1180.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 1173.55 | 1174.59 | 1179.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 13:30:00 | 1171.15 | 1174.30 | 1178.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 09:15:00 | 1162.15 | 1174.88 | 1178.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 11:15:00 | 1186.40 | 1176.86 | 1178.21 | SL hit (close>static) qty=1.00 sl=1185.00 alert=retest2 |

### Cycle 115 — BUY (started 2024-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 12:15:00 | 1201.55 | 1181.80 | 1180.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 13:15:00 | 1212.35 | 1187.91 | 1183.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 13:15:00 | 1186.95 | 1192.97 | 1189.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 13:15:00 | 1186.95 | 1192.97 | 1189.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 1186.95 | 1192.97 | 1189.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 14:00:00 | 1186.95 | 1192.97 | 1189.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 1192.50 | 1192.87 | 1189.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 10:45:00 | 1200.00 | 1193.25 | 1190.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-07 09:15:00 | 1180.95 | 1192.25 | 1191.53 | SL hit (close<static) qty=1.00 sl=1185.70 alert=retest2 |

### Cycle 116 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 1152.25 | 1184.25 | 1187.96 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2024-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 12:15:00 | 1193.05 | 1180.88 | 1180.62 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 11:15:00 | 1175.20 | 1180.35 | 1180.97 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2024-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 12:15:00 | 1194.80 | 1183.24 | 1182.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 14:15:00 | 1206.55 | 1189.93 | 1185.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 11:15:00 | 1196.30 | 1196.99 | 1190.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 11:30:00 | 1197.55 | 1196.99 | 1190.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 1198.25 | 1197.24 | 1191.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 13:15:00 | 1202.00 | 1197.24 | 1191.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 13:45:00 | 1201.70 | 1195.25 | 1192.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 11:15:00 | 1185.65 | 1192.60 | 1192.46 | SL hit (close<static) qty=1.00 sl=1186.80 alert=retest2 |

### Cycle 120 — SELL (started 2024-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 12:15:00 | 1256.55 | 1257.09 | 1257.16 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2024-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 13:15:00 | 1258.05 | 1257.28 | 1257.24 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 1239.30 | 1255.78 | 1256.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 10:15:00 | 1236.65 | 1251.96 | 1254.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-25 14:15:00 | 1251.85 | 1249.89 | 1252.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 14:15:00 | 1251.85 | 1249.89 | 1252.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 14:15:00 | 1251.85 | 1249.89 | 1252.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-25 15:00:00 | 1251.85 | 1249.89 | 1252.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 15:15:00 | 1248.70 | 1249.65 | 1252.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 09:15:00 | 1258.10 | 1249.65 | 1252.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 1240.85 | 1247.89 | 1251.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 11:15:00 | 1237.25 | 1246.39 | 1250.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 15:15:00 | 1288.00 | 1259.08 | 1255.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2024-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 15:15:00 | 1288.00 | 1259.08 | 1255.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 14:15:00 | 1317.95 | 1284.92 | 1274.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 12:15:00 | 1321.30 | 1324.24 | 1311.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-05 13:00:00 | 1321.30 | 1324.24 | 1311.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 1329.05 | 1327.28 | 1317.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:30:00 | 1315.20 | 1327.28 | 1317.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 1322.15 | 1332.97 | 1324.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 15:00:00 | 1322.15 | 1332.97 | 1324.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 15:15:00 | 1322.60 | 1330.90 | 1324.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:15:00 | 1318.35 | 1330.90 | 1324.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 1310.05 | 1326.73 | 1322.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 1310.05 | 1326.73 | 1322.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 1295.50 | 1320.48 | 1320.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:45:00 | 1295.10 | 1320.48 | 1320.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2024-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 11:15:00 | 1285.00 | 1313.39 | 1317.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 13:15:00 | 1284.20 | 1303.38 | 1311.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 12:15:00 | 1290.55 | 1289.93 | 1299.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-08 13:00:00 | 1290.55 | 1289.93 | 1299.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 1268.30 | 1261.83 | 1273.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:00:00 | 1268.30 | 1261.83 | 1273.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 1227.90 | 1232.87 | 1252.06 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2024-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 10:15:00 | 1280.00 | 1258.70 | 1256.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-14 11:15:00 | 1285.00 | 1263.96 | 1259.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-14 15:15:00 | 1269.80 | 1272.18 | 1265.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 15:15:00 | 1269.80 | 1272.18 | 1265.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 15:15:00 | 1269.80 | 1272.18 | 1265.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-18 09:45:00 | 1261.10 | 1270.16 | 1265.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 10:15:00 | 1266.10 | 1269.34 | 1265.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-18 12:30:00 | 1280.60 | 1274.46 | 1268.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-19 13:15:00 | 1274.55 | 1279.90 | 1274.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-19 14:15:00 | 1274.40 | 1278.10 | 1274.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 09:45:00 | 1275.00 | 1278.65 | 1275.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 1279.05 | 1278.73 | 1276.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 09:45:00 | 1287.65 | 1276.71 | 1275.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 11:00:00 | 1284.05 | 1278.18 | 1276.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 11:30:00 | 1284.45 | 1281.63 | 1278.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 09:45:00 | 1286.45 | 1287.30 | 1283.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 10:15:00 | 1300.45 | 1289.93 | 1284.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 14:15:00 | 1315.55 | 1293.77 | 1287.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 15:15:00 | 1275.00 | 1290.21 | 1287.27 | SL hit (close<static) qty=1.00 sl=1275.30 alert=retest2 |

### Cycle 126 — SELL (started 2024-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 15:15:00 | 1285.00 | 1286.31 | 1286.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 09:15:00 | 1279.75 | 1285.00 | 1285.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 11:15:00 | 1271.95 | 1265.15 | 1271.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 11:15:00 | 1271.95 | 1265.15 | 1271.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 1271.95 | 1265.15 | 1271.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 12:00:00 | 1271.95 | 1265.15 | 1271.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 1260.20 | 1264.16 | 1270.74 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2024-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 09:15:00 | 1310.15 | 1277.25 | 1275.12 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2024-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 12:15:00 | 1288.05 | 1298.27 | 1298.56 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 09:15:00 | 1301.30 | 1297.85 | 1297.53 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 10:15:00 | 1294.75 | 1297.23 | 1297.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 12:15:00 | 1286.95 | 1295.19 | 1296.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 14:15:00 | 1297.70 | 1295.64 | 1296.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 14:15:00 | 1297.70 | 1295.64 | 1296.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 1297.70 | 1295.64 | 1296.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:45:00 | 1298.95 | 1295.64 | 1296.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 15:15:00 | 1293.35 | 1295.18 | 1296.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:15:00 | 1293.05 | 1295.18 | 1296.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 1287.90 | 1293.72 | 1295.33 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2024-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 12:15:00 | 1298.00 | 1294.77 | 1294.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 14:15:00 | 1308.15 | 1298.03 | 1296.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 09:15:00 | 1294.20 | 1297.27 | 1296.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 09:15:00 | 1294.20 | 1297.27 | 1296.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 1294.20 | 1297.27 | 1296.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:30:00 | 1296.80 | 1297.27 | 1296.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 1289.60 | 1295.74 | 1295.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 11:00:00 | 1289.60 | 1295.74 | 1295.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 11:15:00 | 1284.80 | 1293.55 | 1294.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 12:15:00 | 1275.00 | 1289.84 | 1292.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 14:15:00 | 1294.70 | 1290.14 | 1292.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 14:15:00 | 1294.70 | 1290.14 | 1292.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 1294.70 | 1290.14 | 1292.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 15:00:00 | 1294.70 | 1290.14 | 1292.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 1296.60 | 1291.43 | 1292.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:15:00 | 1305.90 | 1291.43 | 1292.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 1288.40 | 1290.83 | 1292.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:30:00 | 1295.80 | 1290.83 | 1292.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 1291.20 | 1290.90 | 1292.22 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2024-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 13:15:00 | 1298.25 | 1293.24 | 1293.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 14:15:00 | 1298.45 | 1294.28 | 1293.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 15:15:00 | 1293.50 | 1294.13 | 1293.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 15:15:00 | 1293.50 | 1294.13 | 1293.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 15:15:00 | 1293.50 | 1294.13 | 1293.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:15:00 | 1297.60 | 1294.13 | 1293.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 1292.90 | 1293.88 | 1293.46 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 10:15:00 | 1285.00 | 1292.10 | 1292.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 12:15:00 | 1281.55 | 1289.18 | 1291.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 1273.90 | 1269.73 | 1277.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 14:15:00 | 1273.90 | 1269.73 | 1277.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 1273.90 | 1269.73 | 1277.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 1273.90 | 1269.73 | 1277.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1273.00 | 1269.62 | 1275.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:15:00 | 1272.30 | 1269.62 | 1275.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 1270.00 | 1269.70 | 1275.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 11:30:00 | 1260.10 | 1268.37 | 1274.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:30:00 | 1269.90 | 1267.04 | 1271.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 14:30:00 | 1267.80 | 1260.13 | 1263.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 09:15:00 | 1206.40 | 1226.64 | 1240.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 09:15:00 | 1204.41 | 1226.64 | 1240.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 14:15:00 | 1197.09 | 1204.85 | 1223.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-23 09:15:00 | 1134.09 | 1188.40 | 1212.74 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 135 — BUY (started 2024-12-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 12:15:00 | 1161.20 | 1147.33 | 1146.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 14:15:00 | 1167.35 | 1153.51 | 1149.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 10:15:00 | 1143.25 | 1154.28 | 1151.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 10:15:00 | 1143.25 | 1154.28 | 1151.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 1143.25 | 1154.28 | 1151.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 11:00:00 | 1143.25 | 1154.28 | 1151.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 11:15:00 | 1123.20 | 1148.06 | 1148.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 1078.65 | 1129.27 | 1139.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 15:15:00 | 1199.95 | 1143.16 | 1144.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 15:15:00 | 1199.95 | 1143.16 | 1144.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 1199.95 | 1143.16 | 1144.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:15:00 | 1183.95 | 1143.16 | 1144.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 1142.45 | 1143.02 | 1144.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 12:15:00 | 1136.05 | 1143.43 | 1144.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 14:00:00 | 1132.95 | 1139.08 | 1141.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 09:15:00 | 1098.15 | 1140.65 | 1142.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 11:15:00 | 1079.25 | 1113.67 | 1128.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 11:15:00 | 1076.30 | 1113.67 | 1128.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-01 13:15:00 | 1120.35 | 1111.69 | 1124.66 | SL hit (close>ema200) qty=0.50 sl=1111.69 alert=retest2 |

### Cycle 137 — BUY (started 2025-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 09:15:00 | 1116.55 | 1100.17 | 1099.71 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 1072.75 | 1096.46 | 1098.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 11:15:00 | 1066.50 | 1086.39 | 1093.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 1060.45 | 1056.58 | 1067.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 1060.45 | 1056.58 | 1067.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 1060.45 | 1056.58 | 1067.30 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 1077.55 | 1063.75 | 1062.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 1079.95 | 1071.03 | 1067.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 15:15:00 | 1075.00 | 1075.76 | 1071.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-20 09:15:00 | 1059.60 | 1075.76 | 1071.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 1060.30 | 1072.67 | 1070.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:30:00 | 1060.05 | 1072.67 | 1070.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 1070.10 | 1072.15 | 1070.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 15:15:00 | 1080.00 | 1071.41 | 1070.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 10:15:00 | 1050.80 | 1069.29 | 1070.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 1050.80 | 1069.29 | 1070.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 15:15:00 | 1015.70 | 1032.73 | 1046.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 14:15:00 | 1039.05 | 1031.68 | 1039.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 14:15:00 | 1039.05 | 1031.68 | 1039.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 1039.05 | 1031.68 | 1039.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 15:00:00 | 1039.05 | 1031.68 | 1039.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 1034.30 | 1032.21 | 1039.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:15:00 | 1024.60 | 1032.21 | 1039.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 973.37 | 1000.02 | 1017.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-27 14:15:00 | 997.30 | 983.49 | 1000.69 | SL hit (close>ema200) qty=0.50 sl=983.49 alert=retest2 |

### Cycle 141 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 1005.15 | 986.36 | 985.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 1014.15 | 994.53 | 989.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 15:15:00 | 1027.50 | 1028.15 | 1013.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 09:15:00 | 1031.50 | 1028.15 | 1013.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 1115.05 | 1103.67 | 1083.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 11:45:00 | 1119.45 | 1108.63 | 1089.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 12:45:00 | 1120.05 | 1110.68 | 1092.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 13:45:00 | 1123.40 | 1114.07 | 1095.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 10:00:00 | 1123.05 | 1122.34 | 1104.37 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 1155.50 | 1133.21 | 1118.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 12:30:00 | 1161.95 | 1145.91 | 1128.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 15:00:00 | 1160.00 | 1152.97 | 1143.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 10:15:00 | 1113.60 | 1140.77 | 1140.08 | SL hit (close<static) qty=1.00 sl=1114.00 alert=retest2 |

### Cycle 142 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 11:15:00 | 1113.25 | 1135.27 | 1137.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 1089.95 | 1116.85 | 1126.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 1145.00 | 1077.28 | 1086.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 1145.00 | 1077.28 | 1086.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 1145.00 | 1077.28 | 1086.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 1145.00 | 1077.28 | 1086.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 1146.00 | 1091.03 | 1092.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:45:00 | 1147.00 | 1091.03 | 1092.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 11:15:00 | 1127.15 | 1098.25 | 1095.44 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-02-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 11:15:00 | 1085.35 | 1097.55 | 1098.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 13:15:00 | 1078.90 | 1091.48 | 1095.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-14 15:15:00 | 1092.50 | 1090.19 | 1094.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-14 15:15:00 | 1092.50 | 1090.19 | 1094.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 15:15:00 | 1092.50 | 1090.19 | 1094.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 09:15:00 | 1129.20 | 1090.19 | 1094.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — BUY (started 2025-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 09:15:00 | 1128.70 | 1097.89 | 1097.19 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 12:15:00 | 1083.40 | 1103.88 | 1104.50 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 1128.20 | 1104.36 | 1103.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 10:15:00 | 1145.20 | 1112.52 | 1107.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 1184.95 | 1191.59 | 1166.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:15:00 | 1182.05 | 1191.59 | 1166.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 1210.85 | 1213.33 | 1202.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-27 10:30:00 | 1223.10 | 1206.70 | 1204.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 12:15:00 | 1191.55 | 1202.37 | 1202.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2025-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 12:15:00 | 1191.55 | 1202.37 | 1202.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 13:15:00 | 1161.00 | 1194.10 | 1199.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 14:15:00 | 1212.15 | 1169.01 | 1177.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 14:15:00 | 1212.15 | 1169.01 | 1177.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 1212.15 | 1169.01 | 1177.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 1212.15 | 1169.01 | 1177.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 1224.00 | 1180.01 | 1181.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:15:00 | 1144.00 | 1180.01 | 1181.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 10:15:00 | 1170.60 | 1162.57 | 1161.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 1170.60 | 1162.57 | 1161.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 13:15:00 | 1182.85 | 1169.16 | 1165.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 11:15:00 | 1171.15 | 1173.09 | 1168.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-06 11:15:00 | 1171.15 | 1173.09 | 1168.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 11:15:00 | 1171.15 | 1173.09 | 1168.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 11:45:00 | 1168.95 | 1173.09 | 1168.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 14:15:00 | 1169.20 | 1172.38 | 1169.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 15:00:00 | 1169.20 | 1172.38 | 1169.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 1150.80 | 1168.06 | 1167.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:15:00 | 1151.35 | 1168.06 | 1167.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — SELL (started 2025-03-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 09:15:00 | 1152.50 | 1164.95 | 1166.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 09:15:00 | 1145.25 | 1153.58 | 1157.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 13:15:00 | 1152.00 | 1149.90 | 1153.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 14:00:00 | 1152.00 | 1149.90 | 1153.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 1153.65 | 1150.65 | 1153.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 1153.65 | 1150.65 | 1153.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 1163.00 | 1153.12 | 1154.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 09:15:00 | 1151.90 | 1153.12 | 1154.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 11:30:00 | 1153.75 | 1146.93 | 1148.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 15:15:00 | 1167.75 | 1140.99 | 1138.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2025-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 15:15:00 | 1167.75 | 1140.99 | 1138.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 10:15:00 | 1194.85 | 1156.43 | 1146.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 09:15:00 | 1174.50 | 1181.50 | 1166.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 10:00:00 | 1174.50 | 1181.50 | 1166.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 1175.80 | 1193.21 | 1181.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:45:00 | 1173.10 | 1193.21 | 1181.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 10:15:00 | 1174.70 | 1189.51 | 1180.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 15:00:00 | 1178.25 | 1181.73 | 1179.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 09:15:00 | 1142.05 | 1172.24 | 1175.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2025-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 09:15:00 | 1142.05 | 1172.24 | 1175.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 09:15:00 | 1101.30 | 1140.85 | 1155.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 15:15:00 | 1160.00 | 1122.88 | 1137.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 15:15:00 | 1160.00 | 1122.88 | 1137.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 15:15:00 | 1160.00 | 1122.88 | 1137.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 11:45:00 | 1083.25 | 1107.00 | 1125.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 11:15:00 | 1152.90 | 1125.29 | 1125.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 1152.90 | 1125.29 | 1125.01 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 11:15:00 | 1093.00 | 1124.38 | 1127.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 14:15:00 | 1071.20 | 1104.64 | 1116.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 10:15:00 | 1109.75 | 1100.12 | 1110.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 10:15:00 | 1109.75 | 1100.12 | 1110.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 1109.75 | 1100.12 | 1110.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:00:00 | 1109.75 | 1100.12 | 1110.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 1100.05 | 1100.11 | 1109.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 13:30:00 | 1095.40 | 1098.29 | 1107.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 09:15:00 | 1124.50 | 1102.11 | 1106.68 | SL hit (close>static) qty=1.00 sl=1110.75 alert=retest2 |

### Cycle 155 — BUY (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 11:15:00 | 1127.95 | 1111.31 | 1110.32 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 1084.60 | 1111.54 | 1111.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 1042.20 | 1085.61 | 1097.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 1072.45 | 1062.67 | 1076.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 1072.45 | 1062.67 | 1076.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1072.45 | 1062.67 | 1076.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 12:30:00 | 1064.30 | 1071.03 | 1075.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 15:00:00 | 1046.20 | 1063.99 | 1071.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 10:15:00 | 1121.20 | 1080.47 | 1077.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 1121.20 | 1080.47 | 1077.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 1132.60 | 1090.89 | 1082.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 10:15:00 | 1169.00 | 1174.34 | 1149.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 10:30:00 | 1170.60 | 1174.34 | 1149.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 14:15:00 | 1166.30 | 1179.83 | 1171.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 15:00:00 | 1166.30 | 1179.83 | 1171.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 1166.70 | 1177.20 | 1170.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:15:00 | 1178.50 | 1177.20 | 1170.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 1200.00 | 1216.90 | 1204.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 1200.00 | 1216.90 | 1204.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 1204.00 | 1214.32 | 1204.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 1200.00 | 1214.32 | 1204.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 1200.60 | 1211.58 | 1203.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:00:00 | 1200.60 | 1211.58 | 1203.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 1206.10 | 1210.48 | 1204.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 09:30:00 | 1216.70 | 1212.34 | 1206.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 14:00:00 | 1209.50 | 1217.37 | 1211.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 1172.50 | 1204.42 | 1206.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 1172.50 | 1204.42 | 1206.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 1166.70 | 1196.87 | 1203.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 15:15:00 | 1154.00 | 1153.26 | 1167.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-29 09:15:00 | 1150.00 | 1153.26 | 1167.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 1135.30 | 1149.67 | 1164.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 10:15:00 | 1131.20 | 1149.67 | 1164.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 15:15:00 | 1122.00 | 1141.75 | 1154.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 15:15:00 | 1133.30 | 1142.37 | 1148.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 14:15:00 | 1133.80 | 1140.29 | 1144.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 1142.10 | 1140.04 | 1143.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:45:00 | 1150.70 | 1140.04 | 1143.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1141.00 | 1140.23 | 1143.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:30:00 | 1142.00 | 1140.23 | 1143.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 1143.20 | 1140.83 | 1143.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:00:00 | 1143.20 | 1140.83 | 1143.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 1136.80 | 1140.02 | 1142.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:45:00 | 1140.40 | 1140.02 | 1142.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 1150.70 | 1142.28 | 1143.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 15:00:00 | 1150.70 | 1142.28 | 1143.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 1136.50 | 1141.13 | 1142.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:15:00 | 1134.10 | 1141.13 | 1142.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 1130.30 | 1138.96 | 1141.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 13:00:00 | 1126.40 | 1137.42 | 1140.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 1076.63 | 1115.52 | 1128.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 1077.11 | 1115.52 | 1128.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 14:15:00 | 1074.64 | 1100.50 | 1115.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-08 09:15:00 | 1119.40 | 1102.41 | 1113.45 | SL hit (close>ema200) qty=0.50 sl=1102.41 alert=retest2 |

### Cycle 159 — BUY (started 2025-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 15:15:00 | 1082.00 | 1077.26 | 1076.86 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 11:15:00 | 1071.40 | 1076.82 | 1076.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-14 13:15:00 | 1067.00 | 1073.68 | 1075.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-14 14:15:00 | 1081.10 | 1075.17 | 1075.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 14:15:00 | 1081.10 | 1075.17 | 1075.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 1081.10 | 1075.17 | 1075.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-14 15:00:00 | 1081.10 | 1075.17 | 1075.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 15:15:00 | 1070.00 | 1074.13 | 1075.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:15:00 | 1079.00 | 1074.13 | 1075.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 1070.70 | 1073.45 | 1074.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-15 11:00:00 | 1065.30 | 1071.82 | 1074.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 12:30:00 | 1067.40 | 1062.02 | 1066.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 14:15:00 | 1085.10 | 1069.72 | 1069.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2025-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 14:15:00 | 1085.10 | 1069.72 | 1069.03 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 09:15:00 | 1044.00 | 1065.90 | 1067.50 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 1073.00 | 1065.68 | 1064.79 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 1052.40 | 1062.17 | 1063.36 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 13:15:00 | 1075.60 | 1064.67 | 1063.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 14:15:00 | 1081.10 | 1070.03 | 1067.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 13:15:00 | 1075.50 | 1078.66 | 1073.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-23 14:00:00 | 1075.50 | 1078.66 | 1073.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 1088.80 | 1080.69 | 1075.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 15:15:00 | 1092.60 | 1080.69 | 1075.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:30:00 | 1090.50 | 1085.16 | 1078.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 13:00:00 | 1091.70 | 1088.96 | 1081.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 09:30:00 | 1092.60 | 1092.23 | 1085.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1089.30 | 1091.64 | 1086.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:45:00 | 1089.90 | 1091.64 | 1086.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 1092.70 | 1091.95 | 1088.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:45:00 | 1092.00 | 1091.95 | 1088.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 1089.00 | 1091.36 | 1088.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 1097.40 | 1091.36 | 1088.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 10:00:00 | 1095.00 | 1092.09 | 1088.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 09:15:00 | 1067.20 | 1096.91 | 1094.53 | SL hit (close<static) qty=1.00 sl=1074.00 alert=retest2 |

### Cycle 166 — SELL (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 10:15:00 | 1062.40 | 1090.01 | 1091.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 10:15:00 | 1056.50 | 1069.57 | 1078.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 15:15:00 | 1029.00 | 1024.50 | 1036.84 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 09:30:00 | 1009.00 | 1022.08 | 1034.62 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 1040.80 | 1026.73 | 1031.07 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-05 09:15:00 | 1040.80 | 1026.73 | 1031.07 | SL hit (close>ema400) qty=1.00 sl=1031.07 alert=retest1 |

### Cycle 167 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 1048.70 | 1036.39 | 1034.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 14:15:00 | 1064.00 | 1043.64 | 1038.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 11:15:00 | 1052.50 | 1052.73 | 1045.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 11:30:00 | 1052.00 | 1052.73 | 1045.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 1056.00 | 1053.84 | 1047.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 1056.00 | 1053.84 | 1047.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 1035.30 | 1050.00 | 1047.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:00:00 | 1035.30 | 1050.00 | 1047.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 1027.20 | 1045.44 | 1045.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 11:00:00 | 1027.20 | 1045.44 | 1045.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — SELL (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 11:15:00 | 1026.10 | 1041.57 | 1043.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 09:15:00 | 1019.20 | 1030.24 | 1036.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 1032.00 | 1010.04 | 1015.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 1032.00 | 1010.04 | 1015.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1032.00 | 1010.04 | 1015.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 1032.20 | 1010.04 | 1015.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 1022.80 | 1012.60 | 1016.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 11:45:00 | 1015.90 | 1013.82 | 1016.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 11:15:00 | 1010.60 | 998.74 | 997.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — BUY (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 11:15:00 | 1010.60 | 998.74 | 997.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 12:15:00 | 1015.30 | 1002.05 | 999.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 10:15:00 | 1005.70 | 1007.68 | 1003.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 10:15:00 | 1005.70 | 1007.68 | 1003.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 1005.70 | 1007.68 | 1003.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:00:00 | 1005.70 | 1007.68 | 1003.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 1003.10 | 1006.76 | 1003.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 1003.10 | 1006.76 | 1003.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 1001.20 | 1005.65 | 1003.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:45:00 | 1000.60 | 1005.65 | 1003.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 996.60 | 1003.84 | 1002.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:45:00 | 992.50 | 1003.84 | 1002.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — SELL (started 2025-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 14:15:00 | 992.00 | 1001.47 | 1001.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 982.40 | 997.66 | 999.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 09:15:00 | 997.80 | 988.42 | 992.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 997.80 | 988.42 | 992.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 997.80 | 988.42 | 992.54 | EMA400 retest candle locked (from downside) |

### Cycle 171 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 10:15:00 | 1005.00 | 991.78 | 990.49 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 14:15:00 | 981.20 | 988.63 | 989.56 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 11:15:00 | 995.70 | 989.60 | 989.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 12:15:00 | 999.70 | 991.62 | 990.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 09:15:00 | 996.20 | 996.67 | 993.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 09:15:00 | 996.20 | 996.67 | 993.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 996.20 | 996.67 | 993.69 | EMA400 retest candle locked (from upside) |

### Cycle 174 — SELL (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 11:15:00 | 977.00 | 990.63 | 991.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 12:15:00 | 970.30 | 986.56 | 989.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 12:15:00 | 966.20 | 962.76 | 972.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 12:15:00 | 966.20 | 962.76 | 972.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 966.20 | 962.76 | 972.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 948.45 | 965.44 | 971.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 10:00:00 | 954.35 | 960.27 | 965.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 11:30:00 | 958.00 | 958.80 | 963.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 990.95 | 968.95 | 966.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 990.95 | 968.95 | 966.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 10:15:00 | 997.00 | 974.56 | 969.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 10:15:00 | 1020.15 | 1021.41 | 1006.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-07 11:00:00 | 1020.15 | 1021.41 | 1006.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 1005.85 | 1018.30 | 1006.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:00:00 | 1005.85 | 1018.30 | 1006.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 1014.35 | 1017.51 | 1007.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 14:45:00 | 1021.75 | 1018.26 | 1009.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 998.10 | 1012.36 | 1008.83 | SL hit (close<static) qty=1.00 sl=1005.00 alert=retest2 |

### Cycle 176 — SELL (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 12:15:00 | 993.45 | 1005.58 | 1006.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 14:15:00 | 986.65 | 995.50 | 999.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 09:15:00 | 996.50 | 995.30 | 998.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 996.50 | 995.30 | 998.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 996.50 | 995.30 | 998.80 | EMA400 retest candle locked (from downside) |

### Cycle 177 — BUY (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 09:15:00 | 1019.95 | 1001.16 | 1000.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 14:15:00 | 1030.95 | 1016.84 | 1011.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 13:15:00 | 1086.15 | 1087.42 | 1075.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 14:00:00 | 1086.15 | 1087.42 | 1075.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 1074.10 | 1084.45 | 1076.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 1067.20 | 1084.45 | 1076.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1075.15 | 1082.59 | 1076.28 | EMA400 retest candle locked (from upside) |

### Cycle 178 — SELL (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 15:15:00 | 1070.00 | 1073.57 | 1073.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 09:15:00 | 1067.60 | 1072.37 | 1073.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 14:15:00 | 1008.40 | 998.27 | 1008.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 14:15:00 | 1008.40 | 998.27 | 1008.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 1008.40 | 998.27 | 1008.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 1008.40 | 998.27 | 1008.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 1002.00 | 999.02 | 1007.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 1010.60 | 999.02 | 1007.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1008.00 | 1000.82 | 1007.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:30:00 | 1004.75 | 1001.26 | 1007.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 11:45:00 | 1002.50 | 1001.65 | 1006.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 12:15:00 | 954.51 | 973.96 | 985.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 14:15:00 | 981.00 | 974.22 | 983.63 | SL hit (close>ema200) qty=0.50 sl=974.22 alert=retest2 |

### Cycle 179 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 10:15:00 | 949.40 | 935.93 | 934.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 11:15:00 | 950.80 | 938.90 | 936.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 941.20 | 969.82 | 960.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 941.20 | 969.82 | 960.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 941.20 | 969.82 | 960.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:45:00 | 943.30 | 969.82 | 960.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 919.30 | 959.71 | 956.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 919.30 | 959.71 | 956.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 913.85 | 950.54 | 952.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 13:15:00 | 909.75 | 937.02 | 945.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 15:15:00 | 907.00 | 904.71 | 919.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 09:15:00 | 901.30 | 904.71 | 919.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 916.50 | 907.72 | 917.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:00:00 | 916.50 | 907.72 | 917.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 914.60 | 909.10 | 916.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:15:00 | 917.45 | 909.10 | 916.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 917.45 | 910.77 | 916.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:00:00 | 917.45 | 910.77 | 916.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 920.30 | 912.67 | 917.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 920.30 | 912.67 | 917.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 921.05 | 914.35 | 917.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 908.50 | 914.35 | 917.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 10:00:00 | 909.70 | 893.26 | 897.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 10:15:00 | 863.07 | 880.07 | 887.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 10:15:00 | 864.22 | 880.07 | 887.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 10:15:00 | 876.05 | 873.20 | 879.70 | SL hit (close>ema200) qty=0.50 sl=873.20 alert=retest2 |

### Cycle 181 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 895.55 | 882.55 | 881.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 15:15:00 | 904.90 | 892.23 | 887.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 907.50 | 910.10 | 902.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:00:00 | 907.50 | 910.10 | 902.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 904.15 | 908.51 | 903.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 15:00:00 | 904.15 | 908.51 | 903.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 902.90 | 907.39 | 903.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 897.05 | 907.39 | 903.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 905.75 | 907.06 | 903.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:45:00 | 915.15 | 904.58 | 903.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:45:00 | 918.00 | 908.10 | 905.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-08 12:15:00 | 1006.67 | 949.36 | 928.33 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 182 — SELL (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 15:15:00 | 976.00 | 986.15 | 986.85 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 12:15:00 | 991.80 | 985.26 | 984.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 13:15:00 | 992.85 | 986.77 | 985.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 15:15:00 | 985.50 | 987.19 | 985.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 15:15:00 | 985.50 | 987.19 | 985.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 985.50 | 987.19 | 985.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 990.65 | 987.19 | 985.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 992.00 | 988.15 | 986.47 | EMA400 retest candle locked (from upside) |

### Cycle 184 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 972.55 | 984.49 | 985.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 14:15:00 | 970.80 | 981.75 | 983.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 13:15:00 | 917.20 | 916.27 | 934.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-19 13:30:00 | 917.50 | 916.27 | 934.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 900.00 | 894.47 | 903.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 887.60 | 894.47 | 903.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:45:00 | 892.00 | 893.44 | 901.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 15:15:00 | 906.00 | 893.04 | 897.50 | SL hit (close>static) qty=1.00 sl=905.00 alert=retest2 |

### Cycle 185 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 875.25 | 875.06 | 875.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 15:15:00 | 879.00 | 875.85 | 875.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 873.50 | 875.38 | 875.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 873.50 | 875.38 | 875.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 873.50 | 875.38 | 875.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:30:00 | 872.15 | 875.38 | 875.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — SELL (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 10:15:00 | 873.20 | 874.94 | 875.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 11:15:00 | 868.90 | 873.73 | 874.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 10:15:00 | 874.15 | 870.91 | 872.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 10:15:00 | 874.15 | 870.91 | 872.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 874.15 | 870.91 | 872.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 874.15 | 870.91 | 872.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 874.45 | 871.62 | 872.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:45:00 | 875.05 | 871.62 | 872.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 874.65 | 872.23 | 872.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:30:00 | 875.30 | 872.23 | 872.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — BUY (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 14:15:00 | 875.20 | 873.31 | 873.19 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 15:15:00 | 871.95 | 873.04 | 873.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 866.20 | 871.67 | 872.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 870.55 | 865.13 | 867.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 870.55 | 865.13 | 867.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 870.55 | 865.13 | 867.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 866.65 | 865.13 | 867.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 877.00 | 867.51 | 868.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 877.00 | 867.51 | 868.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — BUY (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 11:15:00 | 882.20 | 870.44 | 869.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 11:15:00 | 885.65 | 879.23 | 875.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 890.50 | 892.92 | 884.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 09:45:00 | 890.15 | 892.92 | 884.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 889.55 | 892.25 | 885.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:30:00 | 889.85 | 892.25 | 885.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 890.00 | 891.80 | 885.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:45:00 | 887.55 | 891.80 | 885.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 886.65 | 889.31 | 886.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 882.40 | 889.31 | 886.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 874.00 | 886.25 | 885.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:00:00 | 874.00 | 886.25 | 885.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 873.10 | 883.62 | 884.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 13:15:00 | 866.80 | 878.97 | 881.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 12:15:00 | 871.30 | 870.49 | 875.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 13:00:00 | 871.30 | 870.49 | 875.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 866.60 | 870.20 | 874.45 | EMA400 retest candle locked (from downside) |

### Cycle 191 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 892.30 | 876.69 | 876.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 11:15:00 | 896.00 | 880.56 | 878.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 889.00 | 889.93 | 884.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 12:15:00 | 881.90 | 887.72 | 884.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 881.90 | 887.72 | 884.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 881.90 | 887.72 | 884.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 888.20 | 887.81 | 885.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 14:30:00 | 889.95 | 888.38 | 885.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 15:00:00 | 890.65 | 888.38 | 885.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 880.75 | 887.26 | 885.64 | SL hit (close<static) qty=1.00 sl=881.10 alert=retest2 |

### Cycle 192 — SELL (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 11:15:00 | 878.45 | 884.27 | 884.48 | EMA200 below EMA400 |

### Cycle 193 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 903.00 | 886.41 | 885.21 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 882.45 | 885.84 | 886.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 11:15:00 | 880.90 | 884.85 | 885.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 10:15:00 | 860.35 | 859.27 | 867.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 11:00:00 | 860.35 | 859.27 | 867.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 859.90 | 860.90 | 866.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:45:00 | 859.30 | 860.23 | 865.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-29 09:15:00 | 773.37 | 848.90 | 859.61 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 195 — BUY (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 13:15:00 | 575.15 | 570.20 | 570.03 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 564.75 | 569.88 | 570.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 10:15:00 | 563.45 | 568.59 | 569.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 566.10 | 565.86 | 567.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 566.10 | 565.86 | 567.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 566.10 | 565.86 | 567.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 10:45:00 | 562.50 | 565.36 | 567.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:30:00 | 561.80 | 564.71 | 566.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 534.38 | 546.57 | 548.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 533.71 | 546.57 | 548.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 532.10 | 530.78 | 535.27 | SL hit (close>ema200) qty=0.50 sl=530.78 alert=retest2 |

### Cycle 197 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 535.00 | 532.37 | 532.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 536.80 | 533.36 | 532.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 12:15:00 | 532.90 | 534.44 | 533.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 12:15:00 | 532.90 | 534.44 | 533.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 532.90 | 534.44 | 533.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 13:00:00 | 532.90 | 534.44 | 533.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 13:15:00 | 534.05 | 534.36 | 533.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 535.00 | 533.31 | 533.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 526.60 | 535.02 | 534.95 | SL hit (close<static) qty=1.00 sl=532.65 alert=retest2 |

### Cycle 198 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 522.95 | 532.61 | 533.86 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 535.70 | 530.59 | 529.94 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 12:15:00 | 525.95 | 529.32 | 529.46 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 538.50 | 531.32 | 530.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 11:15:00 | 540.80 | 534.98 | 532.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 537.50 | 537.90 | 535.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 09:15:00 | 537.50 | 537.90 | 535.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 537.50 | 537.90 | 535.23 | EMA400 retest candle locked (from upside) |

### Cycle 202 — SELL (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 11:15:00 | 535.10 | 535.95 | 535.98 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 14:15:00 | 537.00 | 536.04 | 536.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 15:15:00 | 538.00 | 536.43 | 536.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 09:15:00 | 535.00 | 536.14 | 536.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 535.00 | 536.14 | 536.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 535.00 | 536.14 | 536.08 | EMA400 retest candle locked (from upside) |

### Cycle 204 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 528.70 | 534.66 | 535.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 524.95 | 531.16 | 533.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 528.75 | 527.94 | 529.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 528.75 | 527.94 | 529.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 528.75 | 527.94 | 529.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 15:15:00 | 527.50 | 528.64 | 529.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:30:00 | 527.10 | 528.03 | 529.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 15:15:00 | 501.12 | 508.25 | 513.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 15:15:00 | 500.75 | 508.25 | 513.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 512.35 | 509.07 | 513.35 | SL hit (close>ema200) qty=0.50 sl=509.07 alert=retest2 |

### Cycle 205 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 383.85 | 380.28 | 379.83 | EMA200 above EMA400 |

### Cycle 206 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 371.85 | 378.33 | 379.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 366.40 | 372.85 | 375.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 371.40 | 367.58 | 371.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 371.40 | 367.58 | 371.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 371.40 | 367.58 | 371.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 371.40 | 367.58 | 371.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 373.45 | 368.76 | 371.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 384.80 | 368.76 | 371.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 384.50 | 371.91 | 372.88 | EMA400 retest candle locked (from downside) |

### Cycle 207 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 386.25 | 374.77 | 374.10 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2026-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 12:15:00 | 367.80 | 374.82 | 375.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 14:15:00 | 366.35 | 372.32 | 374.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 353.65 | 345.39 | 352.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 353.65 | 345.39 | 352.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 353.65 | 345.39 | 352.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 353.65 | 345.39 | 352.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 357.85 | 347.88 | 352.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 357.85 | 347.88 | 352.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 350.75 | 348.31 | 352.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:45:00 | 353.20 | 348.31 | 352.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 352.55 | 349.60 | 351.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 09:30:00 | 348.05 | 351.44 | 351.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 14:15:00 | 356.10 | 352.49 | 352.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — BUY (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 14:15:00 | 356.10 | 352.49 | 352.10 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2026-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 12:15:00 | 346.40 | 351.04 | 351.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 324.30 | 345.75 | 349.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 311.85 | 311.50 | 321.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 09:45:00 | 313.80 | 311.50 | 321.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 325.70 | 312.55 | 316.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:30:00 | 328.95 | 312.55 | 316.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 326.65 | 315.37 | 317.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:45:00 | 336.55 | 315.37 | 317.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — BUY (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 12:15:00 | 324.00 | 319.28 | 319.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 13:15:00 | 329.35 | 321.29 | 320.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 322.55 | 322.57 | 321.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 322.55 | 322.57 | 321.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 322.55 | 322.57 | 321.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:45:00 | 321.85 | 322.57 | 321.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 319.00 | 321.86 | 320.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:45:00 | 319.80 | 321.86 | 320.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 317.20 | 320.92 | 320.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 317.20 | 320.92 | 320.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 212 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 314.65 | 319.67 | 320.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 09:15:00 | 311.90 | 315.88 | 317.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 315.70 | 310.30 | 313.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 315.70 | 310.30 | 313.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 315.70 | 310.30 | 313.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 322.15 | 310.30 | 313.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 313.20 | 310.88 | 313.31 | EMA400 retest candle locked (from downside) |

### Cycle 213 — BUY (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 13:15:00 | 317.50 | 315.07 | 314.87 | EMA200 above EMA400 |

### Cycle 214 — SELL (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 14:15:00 | 313.00 | 314.65 | 314.70 | EMA200 below EMA400 |

### Cycle 215 — BUY (started 2026-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 15:15:00 | 315.10 | 314.74 | 314.74 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 310.10 | 313.81 | 314.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 12:15:00 | 304.50 | 310.22 | 312.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 310.00 | 309.49 | 311.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-24 15:00:00 | 310.00 | 309.49 | 311.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 307.85 | 309.27 | 311.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:30:00 | 306.40 | 310.14 | 310.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 09:15:00 | 304.90 | 309.14 | 309.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 12:15:00 | 305.70 | 307.47 | 308.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 10:30:00 | 304.75 | 307.42 | 308.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 289.40 | 297.81 | 301.53 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 09:15:00 | 291.08 | 297.81 | 301.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 09:15:00 | 289.65 | 297.81 | 301.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 09:15:00 | 290.41 | 297.81 | 301.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 09:15:00 | 289.51 | 297.81 | 301.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 285.60 | 292.53 | 296.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-09 09:15:00 | 275.76 | 280.58 | 287.13 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 217 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 293.25 | 285.84 | 285.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 295.20 | 287.71 | 286.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 291.80 | 297.17 | 293.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 291.80 | 297.17 | 293.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 291.80 | 297.17 | 293.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 291.80 | 297.17 | 293.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 291.50 | 296.03 | 292.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 286.45 | 296.03 | 292.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 286.65 | 294.16 | 292.33 | EMA400 retest candle locked (from upside) |

### Cycle 218 — SELL (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 12:15:00 | 288.15 | 291.24 | 291.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 284.80 | 289.29 | 290.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 11:15:00 | 289.25 | 288.65 | 289.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 11:15:00 | 289.25 | 288.65 | 289.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 289.25 | 288.65 | 289.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:30:00 | 289.10 | 288.65 | 289.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 12:15:00 | 290.85 | 289.09 | 289.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 13:00:00 | 290.85 | 289.09 | 289.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 219 — BUY (started 2026-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 13:15:00 | 296.60 | 290.59 | 290.51 | EMA200 above EMA400 |

### Cycle 220 — SELL (started 2026-03-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 13:15:00 | 289.00 | 291.06 | 291.19 | EMA200 below EMA400 |

### Cycle 221 — BUY (started 2026-03-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 14:15:00 | 292.95 | 291.43 | 291.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 09:15:00 | 300.05 | 293.56 | 292.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 12:15:00 | 301.85 | 302.27 | 298.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-18 13:00:00 | 301.85 | 302.27 | 298.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 301.75 | 303.75 | 300.81 | EMA400 retest candle locked (from upside) |

### Cycle 222 — SELL (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 10:15:00 | 297.35 | 299.98 | 300.03 | EMA200 below EMA400 |

### Cycle 223 — BUY (started 2026-03-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-23 14:15:00 | 300.00 | 299.57 | 299.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-23 15:15:00 | 301.00 | 299.86 | 299.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-24 09:15:00 | 298.95 | 299.68 | 299.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 298.95 | 299.68 | 299.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 298.95 | 299.68 | 299.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:45:00 | 299.00 | 299.68 | 299.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 224 — SELL (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 10:15:00 | 295.50 | 298.84 | 299.25 | EMA200 below EMA400 |

### Cycle 225 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 306.90 | 299.88 | 299.53 | EMA200 above EMA400 |

### Cycle 226 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 300.50 | 301.43 | 301.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 299.55 | 300.87 | 301.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 11:15:00 | 302.25 | 300.64 | 300.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 11:15:00 | 302.25 | 300.64 | 300.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 11:15:00 | 302.25 | 300.64 | 300.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 12:00:00 | 302.25 | 300.64 | 300.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 12:15:00 | 302.55 | 301.02 | 301.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 12:30:00 | 303.25 | 301.02 | 301.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 227 — BUY (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 13:15:00 | 303.70 | 301.56 | 301.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 09:15:00 | 312.60 | 304.02 | 302.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 302.50 | 308.23 | 306.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 302.50 | 308.23 | 306.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 302.50 | 308.23 | 306.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 302.50 | 308.23 | 306.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 303.35 | 307.26 | 305.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:30:00 | 304.85 | 307.03 | 305.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 09:15:00 | 293.40 | 304.56 | 305.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 228 — SELL (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 09:15:00 | 293.40 | 304.56 | 305.24 | EMA200 below EMA400 |

### Cycle 229 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 307.90 | 303.41 | 303.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 09:15:00 | 310.35 | 307.61 | 305.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 309.45 | 309.87 | 307.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 13:15:00 | 309.45 | 309.87 | 307.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 309.45 | 309.87 | 307.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 15:00:00 | 314.00 | 310.69 | 308.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-10 09:15:00 | 345.40 | 321.48 | 313.61 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 230 — SELL (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 10:15:00 | 361.40 | 368.23 | 368.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 13:15:00 | 360.30 | 364.67 | 366.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 09:15:00 | 368.00 | 364.05 | 365.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 368.00 | 364.05 | 365.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 368.00 | 364.05 | 365.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:00:00 | 368.00 | 364.05 | 365.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 367.45 | 364.73 | 365.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 11:15:00 | 364.90 | 364.73 | 365.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 11:15:00 | 374.10 | 367.11 | 366.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 231 — BUY (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 11:15:00 | 374.10 | 367.11 | 366.31 | EMA200 above EMA400 |

### Cycle 232 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 364.95 | 365.94 | 365.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 360.00 | 364.03 | 365.04 | Break + close below crossover candle low |

### Cycle 233 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 432.10 | 375.13 | 369.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 506.10 | 434.95 | 406.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 10:15:00 | 472.30 | 475.04 | 448.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 10:45:00 | 472.00 | 475.04 | 448.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 447.25 | 466.86 | 455.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 12:15:00 | 470.50 | 463.08 | 455.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 11:15:00 | 460.80 | 466.84 | 466.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 234 — SELL (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 11:15:00 | 460.80 | 466.84 | 466.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 15:15:00 | 454.00 | 460.12 | 463.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 483.75 | 464.85 | 465.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 483.75 | 464.85 | 465.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 483.75 | 464.85 | 465.23 | EMA400 retest candle locked (from downside) |

### Cycle 235 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 476.50 | 467.18 | 466.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 12:15:00 | 484.10 | 472.87 | 469.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 14:15:00 | 487.45 | 488.44 | 483.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 15:00:00 | 487.45 | 488.44 | 483.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 487.90 | 488.33 | 484.08 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-18 11:00:00 | 477.00 | 2023-05-19 09:15:00 | 473.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2023-06-01 10:15:00 | 478.45 | 2023-06-01 13:15:00 | 474.10 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2023-06-14 15:00:00 | 477.50 | 2023-06-30 10:15:00 | 484.15 | STOP_HIT | 1.00 | 1.39% |
| BUY | retest2 | 2023-06-15 14:00:00 | 477.10 | 2023-06-30 10:15:00 | 484.15 | STOP_HIT | 1.00 | 1.48% |
| BUY | retest2 | 2023-06-15 15:00:00 | 477.65 | 2023-06-30 10:15:00 | 484.15 | STOP_HIT | 1.00 | 1.36% |
| BUY | retest2 | 2023-06-16 09:45:00 | 477.50 | 2023-06-30 10:15:00 | 484.15 | STOP_HIT | 1.00 | 1.39% |
| BUY | retest2 | 2023-06-26 15:15:00 | 489.40 | 2023-06-30 10:15:00 | 484.15 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2023-06-27 09:30:00 | 488.50 | 2023-06-30 10:15:00 | 484.15 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2023-06-27 13:00:00 | 490.45 | 2023-06-30 10:15:00 | 484.15 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2023-06-27 15:30:00 | 488.40 | 2023-06-30 10:15:00 | 484.15 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2023-07-04 14:45:00 | 484.95 | 2023-07-05 11:15:00 | 485.90 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2023-07-18 09:15:00 | 489.40 | 2023-07-18 11:15:00 | 485.50 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2023-07-19 15:15:00 | 484.75 | 2023-07-25 09:15:00 | 485.70 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2023-07-20 09:30:00 | 485.05 | 2023-07-25 09:15:00 | 485.70 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2023-07-20 10:00:00 | 484.75 | 2023-07-25 09:15:00 | 485.70 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2023-07-20 11:15:00 | 485.00 | 2023-07-25 09:15:00 | 485.70 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2023-07-24 09:15:00 | 480.35 | 2023-07-25 09:15:00 | 485.70 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2023-08-04 10:15:00 | 488.50 | 2023-08-07 09:15:00 | 494.50 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2023-08-04 12:30:00 | 486.95 | 2023-08-07 09:15:00 | 494.50 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2023-08-04 13:00:00 | 488.00 | 2023-08-07 09:15:00 | 494.50 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2023-08-14 09:15:00 | 505.00 | 2023-08-23 10:15:00 | 520.00 | STOP_HIT | 1.00 | 2.97% |
| SELL | retest2 | 2023-08-28 09:45:00 | 513.85 | 2023-09-01 09:15:00 | 514.50 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2023-08-28 10:30:00 | 514.50 | 2023-09-01 09:15:00 | 514.50 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2023-08-28 13:30:00 | 514.90 | 2023-09-01 09:15:00 | 514.50 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2023-08-29 09:45:00 | 514.35 | 2023-09-01 09:15:00 | 514.50 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2023-09-11 15:00:00 | 518.55 | 2023-09-12 13:15:00 | 510.65 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2023-09-12 10:15:00 | 518.50 | 2023-09-12 13:15:00 | 510.65 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2023-09-25 15:15:00 | 526.00 | 2023-09-26 11:15:00 | 537.25 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2023-09-26 09:30:00 | 526.90 | 2023-09-26 11:15:00 | 537.25 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2023-09-28 12:15:00 | 540.55 | 2023-10-10 11:15:00 | 562.00 | STOP_HIT | 1.00 | 3.97% |
| BUY | retest2 | 2023-09-28 14:15:00 | 540.95 | 2023-10-10 11:15:00 | 562.00 | STOP_HIT | 1.00 | 3.89% |
| BUY | retest2 | 2023-09-28 15:00:00 | 541.40 | 2023-10-10 11:15:00 | 562.00 | STOP_HIT | 1.00 | 3.80% |
| SELL | retest2 | 2023-10-19 09:15:00 | 570.25 | 2023-10-20 09:15:00 | 599.45 | STOP_HIT | 1.00 | -5.12% |
| SELL | retest2 | 2023-10-25 12:30:00 | 574.80 | 2023-10-26 09:15:00 | 546.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-25 12:30:00 | 574.80 | 2023-10-26 13:15:00 | 569.15 | STOP_HIT | 0.50 | 0.98% |
| BUY | retest1 | 2023-11-02 09:15:00 | 575.95 | 2023-11-03 10:15:00 | 571.65 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest1 | 2023-11-03 10:15:00 | 575.50 | 2023-11-03 10:15:00 | 571.65 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2023-11-08 11:15:00 | 574.45 | 2023-11-08 15:15:00 | 565.90 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2023-11-16 14:00:00 | 570.30 | 2023-11-17 11:15:00 | 576.45 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest1 | 2023-11-22 09:15:00 | 615.00 | 2023-11-28 10:15:00 | 645.75 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-11-22 09:15:00 | 615.00 | 2023-12-01 09:15:00 | 676.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-12-07 09:15:00 | 660.25 | 2023-12-07 13:15:00 | 670.00 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2023-12-28 09:15:00 | 702.40 | 2023-12-28 10:15:00 | 694.45 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2023-12-28 14:15:00 | 699.10 | 2024-01-08 10:15:00 | 729.40 | STOP_HIT | 1.00 | 4.33% |
| BUY | retest2 | 2023-12-28 15:15:00 | 700.00 | 2024-01-08 10:15:00 | 729.40 | STOP_HIT | 1.00 | 4.20% |
| BUY | retest2 | 2023-12-29 10:15:00 | 703.00 | 2024-01-08 10:15:00 | 729.40 | STOP_HIT | 1.00 | 3.76% |
| BUY | retest2 | 2023-12-29 11:15:00 | 704.45 | 2024-01-08 10:15:00 | 729.40 | STOP_HIT | 1.00 | 3.54% |
| BUY | retest2 | 2023-12-29 12:30:00 | 704.40 | 2024-01-08 10:15:00 | 729.40 | STOP_HIT | 1.00 | 3.55% |
| SELL | retest2 | 2024-01-10 12:15:00 | 723.40 | 2024-01-16 10:15:00 | 687.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-11 14:00:00 | 726.00 | 2024-01-16 10:15:00 | 689.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-11 14:30:00 | 721.80 | 2024-01-16 10:15:00 | 685.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-11 15:00:00 | 725.25 | 2024-01-16 10:15:00 | 688.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-12 09:15:00 | 720.10 | 2024-01-16 12:15:00 | 684.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-12 10:30:00 | 716.75 | 2024-01-16 12:15:00 | 680.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-10 12:15:00 | 723.40 | 2024-01-17 15:15:00 | 653.40 | TARGET_HIT | 0.50 | 9.68% |
| SELL | retest2 | 2024-01-11 14:00:00 | 726.00 | 2024-01-17 15:15:00 | 652.73 | TARGET_HIT | 0.50 | 10.09% |
| SELL | retest2 | 2024-01-11 14:30:00 | 721.80 | 2024-01-18 11:15:00 | 678.55 | STOP_HIT | 0.50 | 5.99% |
| SELL | retest2 | 2024-01-11 15:00:00 | 725.25 | 2024-01-18 11:15:00 | 678.55 | STOP_HIT | 0.50 | 6.44% |
| SELL | retest2 | 2024-01-12 09:15:00 | 720.10 | 2024-01-18 11:15:00 | 678.55 | STOP_HIT | 0.50 | 5.77% |
| SELL | retest2 | 2024-01-12 10:30:00 | 716.75 | 2024-01-18 11:15:00 | 678.55 | STOP_HIT | 0.50 | 5.33% |
| SELL | retest2 | 2024-01-25 12:30:00 | 677.15 | 2024-01-30 11:15:00 | 692.65 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2024-01-25 15:15:00 | 675.00 | 2024-01-30 11:15:00 | 692.65 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2024-01-29 12:45:00 | 679.85 | 2024-01-30 11:15:00 | 692.65 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-01-29 13:15:00 | 679.35 | 2024-01-30 11:15:00 | 692.65 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-01-29 15:15:00 | 677.45 | 2024-01-30 11:15:00 | 692.65 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2024-02-09 13:15:00 | 622.35 | 2024-02-12 10:15:00 | 637.65 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2024-02-09 14:15:00 | 622.05 | 2024-02-12 10:15:00 | 637.65 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2024-02-16 11:15:00 | 673.00 | 2024-02-21 09:15:00 | 669.30 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-02-19 09:15:00 | 681.00 | 2024-02-22 09:15:00 | 667.25 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2024-02-20 12:15:00 | 674.00 | 2024-02-22 09:15:00 | 667.25 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-02-20 14:00:00 | 673.00 | 2024-02-22 09:15:00 | 667.25 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-02-21 09:15:00 | 675.20 | 2024-02-22 09:15:00 | 667.25 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-02-28 10:30:00 | 653.50 | 2024-03-01 09:15:00 | 694.50 | STOP_HIT | 1.00 | -6.27% |
| SELL | retest2 | 2024-03-07 09:15:00 | 669.95 | 2024-03-13 09:15:00 | 636.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-07 10:00:00 | 671.10 | 2024-03-13 09:15:00 | 637.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-07 11:00:00 | 670.75 | 2024-03-13 09:15:00 | 637.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-07 12:15:00 | 670.05 | 2024-03-13 09:15:00 | 636.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-07 14:00:00 | 666.00 | 2024-03-13 09:15:00 | 632.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-07 09:15:00 | 669.95 | 2024-03-14 11:15:00 | 639.00 | STOP_HIT | 0.50 | 4.62% |
| SELL | retest2 | 2024-03-07 10:00:00 | 671.10 | 2024-03-14 11:15:00 | 639.00 | STOP_HIT | 0.50 | 4.78% |
| SELL | retest2 | 2024-03-07 11:00:00 | 670.75 | 2024-03-14 11:15:00 | 639.00 | STOP_HIT | 0.50 | 4.73% |
| SELL | retest2 | 2024-03-07 12:15:00 | 670.05 | 2024-03-14 11:15:00 | 639.00 | STOP_HIT | 0.50 | 4.63% |
| SELL | retest2 | 2024-03-07 14:00:00 | 666.00 | 2024-03-14 11:15:00 | 639.00 | STOP_HIT | 0.50 | 4.05% |
| BUY | retest2 | 2024-03-28 09:15:00 | 649.20 | 2024-04-04 09:15:00 | 655.45 | STOP_HIT | 1.00 | 0.96% |
| BUY | retest2 | 2024-03-28 14:15:00 | 648.10 | 2024-04-04 09:15:00 | 655.45 | STOP_HIT | 1.00 | 1.13% |
| SELL | retest2 | 2024-04-08 10:45:00 | 642.00 | 2024-04-12 09:15:00 | 609.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-08 10:45:00 | 642.00 | 2024-04-12 10:15:00 | 626.90 | STOP_HIT | 0.50 | 2.35% |
| BUY | retest2 | 2024-04-25 11:00:00 | 642.95 | 2024-04-29 15:15:00 | 646.55 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2024-04-29 10:00:00 | 642.50 | 2024-04-29 15:15:00 | 646.55 | STOP_HIT | 1.00 | 0.63% |
| BUY | retest2 | 2024-04-29 10:30:00 | 643.30 | 2024-04-29 15:15:00 | 646.55 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest2 | 2024-05-09 13:15:00 | 651.00 | 2024-05-13 14:15:00 | 663.65 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2024-05-21 09:30:00 | 636.70 | 2024-05-22 13:15:00 | 604.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-21 09:30:00 | 636.70 | 2024-05-22 15:15:00 | 623.00 | STOP_HIT | 0.50 | 2.15% |
| SELL | retest2 | 2024-06-04 12:15:00 | 604.05 | 2024-06-05 09:15:00 | 633.65 | STOP_HIT | 1.00 | -4.90% |
| BUY | retest2 | 2024-06-20 14:15:00 | 722.00 | 2024-06-21 09:15:00 | 713.75 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-06-24 12:00:00 | 700.15 | 2024-06-25 14:15:00 | 720.30 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2024-07-04 12:45:00 | 818.70 | 2024-07-05 09:15:00 | 807.30 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-07-04 13:45:00 | 819.25 | 2024-07-05 09:15:00 | 807.30 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-07-15 12:30:00 | 845.50 | 2024-07-19 12:15:00 | 834.05 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-07-16 09:15:00 | 850.00 | 2024-07-19 12:15:00 | 834.05 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2024-07-19 10:00:00 | 854.60 | 2024-07-19 12:15:00 | 834.05 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2024-08-08 09:15:00 | 1004.35 | 2024-08-09 13:15:00 | 980.40 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2024-08-08 10:00:00 | 1000.50 | 2024-08-09 13:15:00 | 980.40 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2024-08-08 15:15:00 | 1001.10 | 2024-08-09 13:15:00 | 980.40 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest1 | 2024-08-26 11:30:00 | 1067.85 | 2024-08-28 09:15:00 | 1057.80 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest1 | 2024-08-26 13:00:00 | 1066.65 | 2024-08-28 09:15:00 | 1057.80 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-08-28 13:00:00 | 1062.55 | 2024-08-29 15:15:00 | 1060.50 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2024-08-29 09:15:00 | 1072.70 | 2024-08-29 15:15:00 | 1060.50 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-08-29 13:30:00 | 1065.00 | 2024-08-29 15:15:00 | 1060.50 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2024-08-29 14:00:00 | 1063.75 | 2024-08-29 15:15:00 | 1060.50 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2024-09-06 09:15:00 | 1181.75 | 2024-09-12 09:15:00 | 1180.70 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2024-09-26 09:45:00 | 1179.10 | 2024-10-01 11:15:00 | 1186.40 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-09-26 11:45:00 | 1177.00 | 2024-10-01 11:15:00 | 1186.40 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-09-26 12:30:00 | 1177.70 | 2024-10-01 12:15:00 | 1201.55 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-09-27 09:45:00 | 1178.40 | 2024-10-01 12:15:00 | 1201.55 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-09-30 13:30:00 | 1171.15 | 2024-10-01 12:15:00 | 1201.55 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2024-10-01 09:15:00 | 1162.15 | 2024-10-01 12:15:00 | 1201.55 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2024-10-04 10:45:00 | 1200.00 | 2024-10-07 09:15:00 | 1180.95 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-10-10 13:15:00 | 1202.00 | 2024-10-14 11:15:00 | 1185.65 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-10-11 13:45:00 | 1201.70 | 2024-10-14 11:15:00 | 1185.65 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-10-14 12:45:00 | 1225.40 | 2024-10-24 12:15:00 | 1256.55 | STOP_HIT | 1.00 | 2.54% |
| BUY | retest2 | 2024-10-14 14:15:00 | 1205.00 | 2024-10-24 12:15:00 | 1256.55 | STOP_HIT | 1.00 | 4.28% |
| BUY | retest2 | 2024-10-16 12:15:00 | 1216.75 | 2024-10-24 12:15:00 | 1256.55 | STOP_HIT | 1.00 | 3.27% |
| BUY | retest2 | 2024-10-16 12:45:00 | 1217.55 | 2024-10-24 12:15:00 | 1256.55 | STOP_HIT | 1.00 | 3.20% |
| SELL | retest2 | 2024-10-28 11:15:00 | 1237.25 | 2024-10-28 15:15:00 | 1288.00 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2024-11-18 12:30:00 | 1280.60 | 2024-11-25 15:15:00 | 1275.00 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2024-11-19 13:15:00 | 1274.55 | 2024-11-25 15:15:00 | 1275.00 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2024-11-19 14:15:00 | 1274.40 | 2024-11-25 15:15:00 | 1275.00 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2024-11-21 09:45:00 | 1275.00 | 2024-11-25 15:15:00 | 1275.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2024-11-22 09:45:00 | 1287.65 | 2024-11-25 15:15:00 | 1275.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-11-22 11:00:00 | 1284.05 | 2024-11-26 15:15:00 | 1285.00 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2024-11-22 11:30:00 | 1284.45 | 2024-11-26 15:15:00 | 1285.00 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2024-11-25 09:45:00 | 1286.45 | 2024-11-26 15:15:00 | 1285.00 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2024-11-25 14:15:00 | 1315.55 | 2024-11-26 15:15:00 | 1285.00 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2024-12-16 11:30:00 | 1260.10 | 2024-12-20 09:15:00 | 1206.40 | PARTIAL | 0.50 | 4.26% |
| SELL | retest2 | 2024-12-17 09:30:00 | 1269.90 | 2024-12-20 09:15:00 | 1204.41 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2024-12-18 14:30:00 | 1267.80 | 2024-12-20 14:15:00 | 1197.09 | PARTIAL | 0.50 | 5.58% |
| SELL | retest2 | 2024-12-16 11:30:00 | 1260.10 | 2024-12-23 09:15:00 | 1134.09 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-17 09:30:00 | 1269.90 | 2024-12-23 09:15:00 | 1142.91 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-18 14:30:00 | 1267.80 | 2024-12-23 09:15:00 | 1141.02 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-31 12:15:00 | 1136.05 | 2025-01-01 11:15:00 | 1079.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-31 14:00:00 | 1132.95 | 2025-01-01 11:15:00 | 1076.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-31 12:15:00 | 1136.05 | 2025-01-01 13:15:00 | 1120.35 | STOP_HIT | 0.50 | 1.38% |
| SELL | retest2 | 2024-12-31 14:00:00 | 1132.95 | 2025-01-01 13:15:00 | 1120.35 | STOP_HIT | 0.50 | 1.11% |
| SELL | retest2 | 2025-01-01 09:15:00 | 1098.15 | 2025-01-09 09:15:00 | 1116.55 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-01-20 15:15:00 | 1080.00 | 2025-01-21 10:15:00 | 1050.80 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-01-24 09:15:00 | 1024.60 | 2025-01-27 09:15:00 | 973.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:15:00 | 1024.60 | 2025-01-27 14:15:00 | 997.30 | STOP_HIT | 0.50 | 2.66% |
| BUY | retest2 | 2025-02-04 11:45:00 | 1119.45 | 2025-02-10 10:15:00 | 1113.60 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-02-04 12:45:00 | 1120.05 | 2025-02-10 10:15:00 | 1113.60 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-02-04 13:45:00 | 1123.40 | 2025-02-10 11:15:00 | 1113.25 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-02-05 10:00:00 | 1123.05 | 2025-02-10 11:15:00 | 1113.25 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-02-06 12:30:00 | 1161.95 | 2025-02-10 11:15:00 | 1113.25 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest2 | 2025-02-07 15:00:00 | 1160.00 | 2025-02-10 11:15:00 | 1113.25 | STOP_HIT | 1.00 | -4.03% |
| BUY | retest2 | 2025-02-27 10:30:00 | 1223.10 | 2025-02-27 12:15:00 | 1191.55 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2025-03-03 09:15:00 | 1144.00 | 2025-03-05 10:15:00 | 1170.60 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-03-13 09:15:00 | 1151.90 | 2025-03-19 15:15:00 | 1167.75 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-03-17 11:30:00 | 1153.75 | 2025-03-19 15:15:00 | 1167.75 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-03-24 15:00:00 | 1178.25 | 2025-03-25 09:15:00 | 1142.05 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2025-03-27 11:45:00 | 1083.25 | 2025-03-28 11:15:00 | 1152.90 | STOP_HIT | 1.00 | -6.43% |
| SELL | retest2 | 2025-04-02 13:30:00 | 1095.40 | 2025-04-03 09:15:00 | 1124.50 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-04-09 12:30:00 | 1064.30 | 2025-04-11 10:15:00 | 1121.20 | STOP_HIT | 1.00 | -5.35% |
| SELL | retest2 | 2025-04-09 15:00:00 | 1046.20 | 2025-04-11 10:15:00 | 1121.20 | STOP_HIT | 1.00 | -7.17% |
| BUY | retest2 | 2025-04-24 09:30:00 | 1216.70 | 2025-04-25 09:15:00 | 1172.50 | STOP_HIT | 1.00 | -3.63% |
| BUY | retest2 | 2025-04-24 14:00:00 | 1209.50 | 2025-04-25 09:15:00 | 1172.50 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2025-04-29 10:15:00 | 1131.20 | 2025-05-07 09:15:00 | 1076.63 | PARTIAL | 0.50 | 4.82% |
| SELL | retest2 | 2025-04-29 15:15:00 | 1122.00 | 2025-05-07 09:15:00 | 1077.11 | PARTIAL | 0.50 | 4.00% |
| SELL | retest2 | 2025-04-30 15:15:00 | 1133.30 | 2025-05-07 14:15:00 | 1074.64 | PARTIAL | 0.50 | 5.18% |
| SELL | retest2 | 2025-04-29 10:15:00 | 1131.20 | 2025-05-08 09:15:00 | 1119.40 | STOP_HIT | 0.50 | 1.04% |
| SELL | retest2 | 2025-04-29 15:15:00 | 1122.00 | 2025-05-08 09:15:00 | 1119.40 | STOP_HIT | 0.50 | 0.23% |
| SELL | retest2 | 2025-04-30 15:15:00 | 1133.30 | 2025-05-08 09:15:00 | 1119.40 | STOP_HIT | 0.50 | 1.23% |
| SELL | retest2 | 2025-05-02 14:15:00 | 1133.80 | 2025-05-08 14:15:00 | 1065.90 | PARTIAL | 0.50 | 5.99% |
| SELL | retest2 | 2025-05-06 13:00:00 | 1126.40 | 2025-05-08 14:15:00 | 1070.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-02 14:15:00 | 1133.80 | 2025-05-09 13:15:00 | 1070.40 | STOP_HIT | 0.50 | 5.59% |
| SELL | retest2 | 2025-05-06 13:00:00 | 1126.40 | 2025-05-09 13:15:00 | 1070.40 | STOP_HIT | 0.50 | 4.97% |
| SELL | retest2 | 2025-05-15 11:00:00 | 1065.30 | 2025-05-16 14:15:00 | 1085.10 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-05-16 12:30:00 | 1067.40 | 2025-05-16 14:15:00 | 1085.10 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-05-23 15:15:00 | 1092.60 | 2025-05-29 09:15:00 | 1067.20 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-05-26 09:30:00 | 1090.50 | 2025-05-29 09:15:00 | 1067.20 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-05-26 13:00:00 | 1091.70 | 2025-05-29 09:15:00 | 1067.20 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-05-27 09:30:00 | 1092.60 | 2025-05-29 09:15:00 | 1067.20 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-05-28 09:15:00 | 1097.40 | 2025-05-29 09:15:00 | 1067.20 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-05-28 10:00:00 | 1095.00 | 2025-05-29 09:15:00 | 1067.20 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest1 | 2025-06-04 09:30:00 | 1009.00 | 2025-06-05 09:15:00 | 1040.80 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2025-06-12 11:45:00 | 1015.90 | 2025-06-18 11:15:00 | 1010.60 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2025-07-01 09:15:00 | 948.45 | 2025-07-03 09:15:00 | 990.95 | STOP_HIT | 1.00 | -4.48% |
| SELL | retest2 | 2025-07-02 10:00:00 | 954.35 | 2025-07-03 09:15:00 | 990.95 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2025-07-02 11:30:00 | 958.00 | 2025-07-03 09:15:00 | 990.95 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest2 | 2025-07-07 14:45:00 | 1021.75 | 2025-07-08 10:15:00 | 998.10 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-07-30 10:30:00 | 1004.75 | 2025-08-01 12:15:00 | 954.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 10:30:00 | 1004.75 | 2025-08-01 14:15:00 | 981.00 | STOP_HIT | 0.50 | 2.36% |
| SELL | retest2 | 2025-07-30 11:45:00 | 1002.50 | 2025-08-05 11:15:00 | 952.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 11:45:00 | 1002.50 | 2025-08-05 14:15:00 | 961.55 | STOP_HIT | 0.50 | 4.08% |
| SELL | retest2 | 2025-08-20 09:15:00 | 908.50 | 2025-08-28 10:15:00 | 863.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 10:00:00 | 909.70 | 2025-08-28 10:15:00 | 864.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-20 09:15:00 | 908.50 | 2025-08-29 10:15:00 | 876.05 | STOP_HIT | 0.50 | 3.57% |
| SELL | retest2 | 2025-08-25 10:00:00 | 909.70 | 2025-08-29 10:15:00 | 876.05 | STOP_HIT | 0.50 | 3.70% |
| BUY | retest2 | 2025-09-05 09:45:00 | 915.15 | 2025-09-08 12:15:00 | 1006.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-05 13:45:00 | 918.00 | 2025-09-08 14:15:00 | 1009.80 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-24 09:15:00 | 887.60 | 2025-09-24 15:15:00 | 906.00 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-09-24 10:45:00 | 892.00 | 2025-09-24 15:15:00 | 906.00 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-09-25 09:15:00 | 892.30 | 2025-10-03 14:15:00 | 875.25 | STOP_HIT | 1.00 | 1.91% |
| SELL | retest2 | 2025-09-25 13:15:00 | 895.15 | 2025-10-03 14:15:00 | 875.25 | STOP_HIT | 1.00 | 2.22% |
| SELL | retest2 | 2025-09-26 09:15:00 | 880.85 | 2025-10-03 14:15:00 | 875.25 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2025-10-17 14:30:00 | 889.95 | 2025-10-20 09:15:00 | 880.75 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-10-17 15:00:00 | 890.65 | 2025-10-20 09:15:00 | 880.75 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-10-28 14:45:00 | 859.30 | 2025-10-29 09:15:00 | 773.37 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-12-01 10:45:00 | 562.50 | 2025-12-05 09:15:00 | 534.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 11:30:00 | 561.80 | 2025-12-05 09:15:00 | 533.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 10:45:00 | 562.50 | 2025-12-09 12:15:00 | 532.10 | STOP_HIT | 0.50 | 5.40% |
| SELL | retest2 | 2025-12-01 11:30:00 | 561.80 | 2025-12-09 12:15:00 | 532.10 | STOP_HIT | 0.50 | 5.29% |
| BUY | retest2 | 2025-12-16 09:15:00 | 535.00 | 2025-12-17 09:15:00 | 526.60 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-12-31 15:15:00 | 527.50 | 2026-01-06 15:15:00 | 501.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 09:30:00 | 527.10 | 2026-01-06 15:15:00 | 500.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-31 15:15:00 | 527.50 | 2026-01-07 09:15:00 | 512.35 | STOP_HIT | 0.50 | 2.87% |
| SELL | retest2 | 2026-01-01 09:30:00 | 527.10 | 2026-01-07 09:15:00 | 512.35 | STOP_HIT | 0.50 | 2.80% |
| SELL | retest2 | 2026-02-11 09:30:00 | 348.05 | 2026-02-11 14:15:00 | 356.10 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2026-02-26 12:30:00 | 306.40 | 2026-03-05 09:15:00 | 291.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 09:15:00 | 304.90 | 2026-03-05 09:15:00 | 289.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 12:15:00 | 305.70 | 2026-03-05 09:15:00 | 290.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 10:30:00 | 304.75 | 2026-03-05 09:15:00 | 289.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 12:30:00 | 306.40 | 2026-03-09 09:15:00 | 275.76 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-27 09:15:00 | 304.90 | 2026-03-09 09:15:00 | 274.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-27 12:15:00 | 305.70 | 2026-03-09 09:15:00 | 275.13 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-02 10:30:00 | 304.75 | 2026-03-09 09:15:00 | 274.28 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-06 09:15:00 | 285.60 | 2026-03-09 09:15:00 | 271.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 09:15:00 | 285.60 | 2026-03-09 14:15:00 | 282.75 | STOP_HIT | 0.50 | 1.00% |
| BUY | retest2 | 2026-04-02 11:30:00 | 304.85 | 2026-04-06 09:15:00 | 293.40 | STOP_HIT | 1.00 | -3.76% |
| BUY | retest2 | 2026-04-09 15:00:00 | 314.00 | 2026-04-10 09:15:00 | 345.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-22 11:15:00 | 364.90 | 2026-04-23 11:15:00 | 374.10 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2026-04-30 12:15:00 | 470.50 | 2026-05-05 11:15:00 | 460.80 | STOP_HIT | 1.00 | -2.06% |
