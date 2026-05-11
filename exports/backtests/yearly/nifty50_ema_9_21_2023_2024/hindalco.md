# HINDALCO (HINDALCO)

## Backtest Summary

- **Window:** 2023-03-13 09:15:00 → 2026-05-08 15:15:00 (5443 bars)
- **Last close:** 1044.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 242 |
| ALERT1 | 154 |
| ALERT2 | 153 |
| ALERT2_SKIP | 83 |
| ALERT3 | 431 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 201 |
| PARTIAL | 8 |
| TARGET_HIT | 7 |
| STOP_HIT | 198 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 213 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 65 / 148
- **Target hits / Stop hits / Partials:** 7 / 198 / 8
- **Avg / median % per leg:** -0.05% / -0.85%
- **Sum % (uncompounded):** -9.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 124 | 43 | 34.7% | 6 | 118 | 0 | 0.09% | 11.6% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 3 | 0 | -0.92% | -2.8% |
| BUY @ 3rd Alert (retest2) | 121 | 42 | 34.7% | 6 | 115 | 0 | 0.12% | 14.4% |
| SELL (all) | 89 | 22 | 24.7% | 1 | 80 | 8 | -0.24% | -21.3% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.35% | -2.4% |
| SELL @ 3rd Alert (retest2) | 88 | 22 | 25.0% | 1 | 79 | 8 | -0.21% | -18.9% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.28% | -5.1% |
| retest2 (combined) | 209 | 64 | 30.6% | 7 | 194 | 8 | -0.02% | -4.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 09:15:00 | 410.70 | 407.19 | 407.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 10:15:00 | 412.15 | 408.18 | 407.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 09:15:00 | 406.90 | 409.71 | 408.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-24 09:15:00 | 406.90 | 409.71 | 408.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 09:15:00 | 406.90 | 409.71 | 408.92 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 11:15:00 | 405.90 | 408.34 | 408.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 12:15:00 | 404.85 | 407.64 | 408.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-24 13:15:00 | 409.50 | 408.01 | 408.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-24 13:15:00 | 409.50 | 408.01 | 408.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 13:15:00 | 409.50 | 408.01 | 408.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-24 14:00:00 | 409.50 | 408.01 | 408.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 14:15:00 | 406.90 | 407.79 | 408.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-25 09:15:00 | 403.00 | 407.61 | 407.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-26 10:15:00 | 413.25 | 406.27 | 406.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2023-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 10:15:00 | 413.25 | 406.27 | 406.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-29 09:15:00 | 422.85 | 413.25 | 410.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 09:15:00 | 418.35 | 418.98 | 415.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-30 09:15:00 | 418.35 | 418.98 | 415.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 09:15:00 | 418.35 | 418.98 | 415.37 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 09:15:00 | 409.40 | 414.01 | 414.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 14:15:00 | 405.35 | 409.80 | 411.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-02 09:15:00 | 414.25 | 408.95 | 409.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-02 09:15:00 | 414.25 | 408.95 | 409.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 09:15:00 | 414.25 | 408.95 | 409.89 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 11:15:00 | 415.90 | 411.31 | 410.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-02 13:15:00 | 416.75 | 413.21 | 411.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-06 09:15:00 | 418.85 | 418.88 | 416.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-06 09:45:00 | 418.10 | 418.88 | 416.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 10:15:00 | 416.00 | 418.30 | 416.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-06 11:00:00 | 416.00 | 418.30 | 416.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 11:15:00 | 415.00 | 417.64 | 416.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-06 11:45:00 | 415.30 | 417.64 | 416.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 12:15:00 | 415.70 | 417.25 | 416.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-07 13:15:00 | 417.90 | 416.22 | 416.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-07 13:45:00 | 417.75 | 416.70 | 416.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-09 09:45:00 | 417.45 | 419.35 | 418.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-09 12:00:00 | 417.40 | 418.45 | 418.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-09 12:15:00 | 415.35 | 417.83 | 418.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2023-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 12:15:00 | 415.35 | 417.83 | 418.15 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 12:15:00 | 418.40 | 417.85 | 417.82 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-12 14:15:00 | 416.55 | 417.57 | 417.70 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 11:15:00 | 420.35 | 418.22 | 417.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 12:15:00 | 423.55 | 419.28 | 418.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 09:15:00 | 422.00 | 423.78 | 422.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-15 09:15:00 | 422.00 | 423.78 | 422.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 09:15:00 | 422.00 | 423.78 | 422.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 09:30:00 | 420.95 | 423.78 | 422.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 10:15:00 | 423.25 | 423.68 | 422.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-15 12:45:00 | 424.80 | 423.65 | 422.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-15 13:30:00 | 424.75 | 423.57 | 422.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-16 09:15:00 | 427.90 | 423.38 | 422.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-19 14:30:00 | 425.25 | 426.63 | 426.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 15:15:00 | 426.50 | 426.60 | 426.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 09:15:00 | 424.10 | 426.60 | 426.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 09:15:00 | 428.50 | 426.98 | 426.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 09:30:00 | 425.35 | 426.98 | 426.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 424.80 | 428.24 | 427.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-06-21 10:15:00 | 420.15 | 426.62 | 426.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2023-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 10:15:00 | 420.15 | 426.62 | 426.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-21 11:15:00 | 418.30 | 424.96 | 426.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-22 09:15:00 | 423.90 | 422.98 | 424.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 09:15:00 | 423.90 | 422.98 | 424.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 09:15:00 | 423.90 | 422.98 | 424.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-22 11:45:00 | 421.35 | 422.49 | 424.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-27 13:15:00 | 418.40 | 415.65 | 415.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2023-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 13:15:00 | 418.40 | 415.65 | 415.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 09:15:00 | 419.05 | 416.86 | 416.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-30 13:15:00 | 420.20 | 420.24 | 418.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-30 14:00:00 | 420.20 | 420.24 | 418.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 424.80 | 425.40 | 424.08 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-05 13:15:00 | 420.80 | 423.28 | 423.39 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-07-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 14:15:00 | 426.30 | 423.47 | 423.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 15:15:00 | 427.30 | 424.24 | 423.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 09:15:00 | 422.75 | 423.94 | 423.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 09:15:00 | 422.75 | 423.94 | 423.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 422.75 | 423.94 | 423.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 09:45:00 | 423.90 | 423.94 | 423.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 10:15:00 | 420.25 | 423.20 | 423.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 11:00:00 | 420.25 | 423.20 | 423.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2023-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 11:15:00 | 419.55 | 422.47 | 422.87 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 09:15:00 | 430.70 | 424.07 | 423.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-10 10:15:00 | 432.25 | 425.70 | 424.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-11 10:15:00 | 426.65 | 427.43 | 426.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-11 11:00:00 | 426.65 | 427.43 | 426.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 11:15:00 | 426.55 | 427.25 | 426.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-11 11:30:00 | 425.95 | 427.25 | 426.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 12:15:00 | 425.50 | 426.90 | 426.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-11 12:45:00 | 425.55 | 426.90 | 426.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 13:15:00 | 425.35 | 426.59 | 425.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-11 14:15:00 | 424.75 | 426.59 | 425.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 14:15:00 | 424.65 | 426.20 | 425.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-11 14:30:00 | 424.20 | 426.20 | 425.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2023-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 09:15:00 | 423.50 | 425.34 | 425.51 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-07-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 09:15:00 | 436.55 | 426.87 | 425.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-13 10:15:00 | 439.05 | 429.30 | 427.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-17 14:15:00 | 447.50 | 447.79 | 442.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-17 15:00:00 | 447.50 | 447.79 | 442.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 09:15:00 | 446.85 | 447.38 | 443.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 09:45:00 | 447.15 | 447.38 | 443.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 11:15:00 | 443.95 | 446.82 | 443.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 12:00:00 | 443.95 | 446.82 | 443.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 12:15:00 | 444.40 | 446.34 | 443.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 14:00:00 | 445.70 | 446.21 | 444.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-19 09:15:00 | 440.85 | 444.71 | 443.91 | SL hit (close<static) qty=1.00 sl=443.35 alert=retest2 |

### Cycle 18 — SELL (started 2023-07-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-19 12:15:00 | 440.30 | 442.85 | 443.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-19 13:15:00 | 439.70 | 442.22 | 442.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-20 13:15:00 | 441.95 | 441.16 | 441.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 13:15:00 | 441.95 | 441.16 | 441.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 13:15:00 | 441.95 | 441.16 | 441.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 14:00:00 | 441.95 | 441.16 | 441.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 14:15:00 | 441.15 | 441.16 | 441.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 15:15:00 | 441.80 | 441.16 | 441.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 15:15:00 | 441.80 | 441.29 | 441.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-21 09:15:00 | 438.90 | 441.29 | 441.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-25 10:15:00 | 440.60 | 436.17 | 437.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-25 11:15:00 | 443.60 | 438.35 | 438.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2023-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 11:15:00 | 443.60 | 438.35 | 438.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 12:15:00 | 444.35 | 439.55 | 438.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-27 14:15:00 | 451.55 | 451.70 | 448.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 14:15:00 | 451.55 | 451.70 | 448.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 14:15:00 | 451.55 | 451.70 | 448.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 14:45:00 | 448.30 | 451.70 | 448.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 09:15:00 | 444.15 | 450.35 | 448.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-28 10:00:00 | 444.15 | 450.35 | 448.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 10:15:00 | 447.75 | 449.83 | 448.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-28 12:30:00 | 449.05 | 449.06 | 448.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-28 13:45:00 | 449.75 | 449.25 | 448.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-02 13:15:00 | 450.95 | 457.92 | 458.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2023-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 13:15:00 | 450.95 | 457.92 | 458.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 09:15:00 | 450.15 | 455.24 | 456.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 14:15:00 | 453.20 | 451.77 | 454.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-03 15:00:00 | 453.20 | 451.77 | 454.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 15:15:00 | 458.00 | 453.02 | 454.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 09:15:00 | 465.20 | 453.02 | 454.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 457.75 | 453.96 | 454.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 10:30:00 | 456.45 | 454.16 | 454.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-04 11:15:00 | 461.55 | 455.64 | 455.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2023-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 11:15:00 | 461.55 | 455.64 | 455.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 10:15:00 | 462.15 | 458.35 | 457.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 10:15:00 | 458.10 | 461.70 | 459.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 10:15:00 | 458.10 | 461.70 | 459.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 10:15:00 | 458.10 | 461.70 | 459.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 11:00:00 | 458.10 | 461.70 | 459.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 11:15:00 | 458.25 | 461.01 | 459.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 11:30:00 | 458.20 | 461.01 | 459.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2023-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 13:15:00 | 454.30 | 458.96 | 459.05 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-08-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 14:15:00 | 468.70 | 459.56 | 458.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-10 12:15:00 | 470.25 | 465.31 | 462.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-11 09:15:00 | 461.80 | 465.84 | 463.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 09:15:00 | 461.80 | 465.84 | 463.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 09:15:00 | 461.80 | 465.84 | 463.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 10:00:00 | 461.80 | 465.84 | 463.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 10:15:00 | 462.40 | 465.15 | 463.47 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 09:15:00 | 453.70 | 461.35 | 462.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-16 09:15:00 | 441.30 | 451.07 | 455.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-17 13:15:00 | 444.00 | 443.71 | 447.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-17 13:30:00 | 443.90 | 443.71 | 447.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 14:15:00 | 447.80 | 444.53 | 447.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-17 15:00:00 | 447.80 | 444.53 | 447.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 15:15:00 | 446.10 | 444.84 | 447.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 09:15:00 | 444.35 | 444.84 | 447.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-21 13:15:00 | 447.85 | 445.50 | 445.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2023-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 13:15:00 | 447.85 | 445.50 | 445.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 14:15:00 | 450.00 | 446.40 | 445.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 10:15:00 | 456.95 | 459.14 | 455.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-24 11:00:00 | 456.95 | 459.14 | 455.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 11:15:00 | 456.00 | 458.51 | 455.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 12:00:00 | 456.00 | 458.51 | 455.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 12:15:00 | 456.85 | 458.18 | 455.88 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2023-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 10:15:00 | 450.20 | 455.24 | 455.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-28 11:15:00 | 447.75 | 450.71 | 452.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-29 09:15:00 | 450.30 | 448.57 | 450.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-29 09:15:00 | 450.30 | 448.57 | 450.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 450.30 | 448.57 | 450.53 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 12:15:00 | 457.70 | 452.28 | 451.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 09:15:00 | 462.35 | 456.14 | 453.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 13:15:00 | 457.75 | 458.59 | 456.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-30 13:30:00 | 457.00 | 458.59 | 456.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 15:15:00 | 456.65 | 457.85 | 456.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-31 09:15:00 | 463.50 | 457.85 | 456.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-06 15:15:00 | 478.10 | 478.58 | 478.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2023-09-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 15:15:00 | 478.10 | 478.58 | 478.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-07 09:15:00 | 477.15 | 478.30 | 478.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-08 11:15:00 | 474.85 | 474.15 | 475.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-08 12:00:00 | 474.85 | 474.15 | 475.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 12:15:00 | 476.00 | 474.52 | 475.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-08 12:30:00 | 475.80 | 474.52 | 475.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 13:15:00 | 478.55 | 475.32 | 476.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-08 14:00:00 | 478.55 | 475.32 | 476.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 14:15:00 | 476.75 | 475.61 | 476.18 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 09:15:00 | 479.80 | 476.59 | 476.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-11 10:15:00 | 483.35 | 477.94 | 477.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 09:15:00 | 479.95 | 481.90 | 479.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-12 09:15:00 | 479.95 | 481.90 | 479.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 479.95 | 481.90 | 479.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 482.75 | 481.90 | 479.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 480.15 | 481.55 | 479.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 11:00:00 | 480.15 | 481.55 | 479.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 11:15:00 | 480.00 | 481.24 | 479.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 12:00:00 | 480.00 | 481.24 | 479.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 12:15:00 | 478.35 | 480.66 | 479.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 13:00:00 | 478.35 | 480.66 | 479.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 13:15:00 | 478.60 | 480.25 | 479.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 13:30:00 | 478.00 | 480.25 | 479.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2023-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 15:15:00 | 477.50 | 479.36 | 479.39 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-09-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 11:15:00 | 481.20 | 479.75 | 479.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-13 12:15:00 | 483.20 | 480.44 | 479.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 12:15:00 | 496.85 | 498.16 | 493.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-15 12:30:00 | 496.80 | 498.16 | 493.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 491.80 | 496.09 | 493.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 09:30:00 | 487.40 | 496.09 | 493.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 10:15:00 | 490.85 | 495.04 | 493.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 10:30:00 | 491.05 | 495.04 | 493.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2023-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 12:15:00 | 487.45 | 492.49 | 492.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 13:15:00 | 483.75 | 490.74 | 491.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 09:15:00 | 485.60 | 482.11 | 485.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 09:15:00 | 485.60 | 482.11 | 485.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 09:15:00 | 485.60 | 482.11 | 485.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 09:30:00 | 486.20 | 482.11 | 485.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 10:15:00 | 482.20 | 482.13 | 484.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-21 11:30:00 | 480.25 | 481.87 | 484.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-21 12:00:00 | 480.85 | 481.87 | 484.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-28 09:15:00 | 478.25 | 474.43 | 474.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2023-09-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-28 09:15:00 | 478.25 | 474.43 | 474.10 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-09-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 12:15:00 | 469.30 | 473.22 | 473.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 13:15:00 | 466.70 | 471.92 | 473.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 09:15:00 | 478.45 | 471.74 | 472.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-29 09:15:00 | 478.45 | 471.74 | 472.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 09:15:00 | 478.45 | 471.74 | 472.50 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2023-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 10:15:00 | 480.25 | 473.44 | 473.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 11:15:00 | 483.80 | 475.51 | 474.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 09:15:00 | 478.30 | 483.96 | 479.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 09:15:00 | 478.30 | 483.96 | 479.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 478.30 | 483.96 | 479.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 10:00:00 | 478.30 | 483.96 | 479.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 10:15:00 | 479.90 | 483.14 | 479.74 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2023-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 10:15:00 | 476.00 | 478.93 | 478.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 11:15:00 | 472.70 | 477.68 | 478.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 13:15:00 | 472.90 | 471.77 | 473.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 13:15:00 | 472.90 | 471.77 | 473.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 13:15:00 | 472.90 | 471.77 | 473.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 13:30:00 | 473.00 | 471.77 | 473.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 14:15:00 | 472.85 | 471.98 | 473.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 14:30:00 | 473.50 | 471.98 | 473.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 15:15:00 | 472.40 | 472.07 | 473.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-09 09:15:00 | 468.55 | 472.07 | 473.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-09 12:00:00 | 470.50 | 471.38 | 472.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-09 15:00:00 | 469.50 | 470.34 | 471.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-10 09:30:00 | 471.00 | 470.26 | 471.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 10:15:00 | 470.20 | 470.25 | 471.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-10 12:15:00 | 468.00 | 469.97 | 471.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-10 13:15:00 | 483.55 | 472.56 | 472.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2023-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 13:15:00 | 483.55 | 472.56 | 472.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 09:15:00 | 489.50 | 483.40 | 479.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-12 12:15:00 | 485.15 | 485.33 | 481.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-12 13:00:00 | 485.15 | 485.33 | 481.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 481.75 | 484.20 | 482.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 11:00:00 | 484.70 | 484.30 | 482.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-16 09:15:00 | 480.10 | 481.50 | 481.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2023-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 09:15:00 | 480.10 | 481.50 | 481.59 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 12:15:00 | 484.75 | 481.60 | 481.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-18 09:15:00 | 491.45 | 485.09 | 483.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 11:15:00 | 485.50 | 486.03 | 484.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-18 12:00:00 | 485.50 | 486.03 | 484.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 12:15:00 | 485.35 | 485.89 | 484.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 12:30:00 | 485.80 | 485.89 | 484.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 13:15:00 | 486.55 | 486.03 | 484.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 13:45:00 | 486.95 | 486.03 | 484.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 14:15:00 | 485.00 | 485.82 | 484.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 14:30:00 | 485.20 | 485.82 | 484.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 15:15:00 | 484.30 | 485.52 | 484.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-19 09:15:00 | 475.00 | 485.52 | 484.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2023-10-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 09:15:00 | 476.85 | 483.78 | 483.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 09:15:00 | 467.50 | 473.13 | 476.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 09:15:00 | 471.45 | 465.07 | 469.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 09:15:00 | 471.45 | 465.07 | 469.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 09:15:00 | 471.45 | 465.07 | 469.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-25 10:00:00 | 471.45 | 465.07 | 469.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 10:15:00 | 467.40 | 465.54 | 469.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-25 11:15:00 | 466.00 | 465.54 | 469.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-31 15:15:00 | 460.05 | 458.31 | 458.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2023-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 15:15:00 | 460.05 | 458.31 | 458.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-01 09:15:00 | 460.80 | 458.81 | 458.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 14:15:00 | 475.40 | 475.59 | 470.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-03 14:45:00 | 475.65 | 475.59 | 470.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 09:15:00 | 483.85 | 484.32 | 481.26 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2023-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 15:15:00 | 481.00 | 483.70 | 483.98 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2023-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 09:15:00 | 490.95 | 484.95 | 484.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 09:15:00 | 508.20 | 490.96 | 487.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 09:15:00 | 499.25 | 501.15 | 495.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-16 09:30:00 | 501.55 | 501.15 | 495.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 12:15:00 | 499.95 | 501.60 | 499.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-17 12:45:00 | 499.50 | 501.60 | 499.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 14:15:00 | 497.25 | 500.54 | 499.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-17 15:00:00 | 497.25 | 500.54 | 499.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 15:15:00 | 498.30 | 500.10 | 499.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-20 09:15:00 | 502.05 | 500.10 | 499.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-20 14:00:00 | 498.75 | 500.93 | 500.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-20 14:15:00 | 496.95 | 500.13 | 499.92 | SL hit (close<static) qty=1.00 sl=497.05 alert=retest2 |

### Cycle 44 — SELL (started 2023-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 15:15:00 | 497.65 | 499.64 | 499.71 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 09:15:00 | 509.00 | 501.51 | 500.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-21 10:15:00 | 509.80 | 503.17 | 501.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-22 09:15:00 | 501.35 | 504.92 | 503.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 09:15:00 | 501.35 | 504.92 | 503.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 501.35 | 504.92 | 503.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 10:00:00 | 501.35 | 504.92 | 503.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 10:15:00 | 503.80 | 504.70 | 503.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 10:30:00 | 503.05 | 504.70 | 503.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 11:15:00 | 502.60 | 504.28 | 503.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 12:00:00 | 502.60 | 504.28 | 503.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 12:15:00 | 501.55 | 503.73 | 503.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 12:30:00 | 501.70 | 503.73 | 503.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2023-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 13:15:00 | 499.00 | 502.79 | 502.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-23 09:15:00 | 493.45 | 500.30 | 501.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 13:15:00 | 501.10 | 498.93 | 500.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 13:15:00 | 501.10 | 498.93 | 500.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 13:15:00 | 501.10 | 498.93 | 500.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 14:00:00 | 501.10 | 498.93 | 500.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 14:15:00 | 501.65 | 499.48 | 500.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 15:15:00 | 501.60 | 499.48 | 500.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 15:15:00 | 501.60 | 499.90 | 500.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 09:15:00 | 504.80 | 499.90 | 500.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2023-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 09:15:00 | 505.95 | 501.11 | 501.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-24 10:15:00 | 507.80 | 502.45 | 501.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-29 12:15:00 | 515.15 | 515.79 | 512.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-29 13:00:00 | 515.15 | 515.79 | 512.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 516.00 | 516.63 | 513.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 09:30:00 | 513.30 | 516.63 | 513.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 12:15:00 | 515.70 | 516.14 | 514.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-30 13:45:00 | 516.95 | 515.96 | 514.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-30 14:45:00 | 516.75 | 515.87 | 514.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-01 09:15:00 | 518.75 | 515.90 | 514.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-01 12:45:00 | 516.75 | 516.62 | 515.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 15:15:00 | 517.00 | 517.57 | 516.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 09:15:00 | 522.25 | 517.57 | 516.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 14:15:00 | 518.60 | 519.45 | 517.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 15:00:00 | 518.70 | 519.30 | 518.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 09:30:00 | 518.70 | 519.60 | 518.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 12:15:00 | 519.95 | 520.54 | 519.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 12:45:00 | 520.25 | 520.54 | 519.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 13:15:00 | 519.20 | 520.27 | 519.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 14:00:00 | 519.20 | 520.27 | 519.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 14:15:00 | 521.45 | 520.51 | 519.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 14:30:00 | 518.50 | 520.51 | 519.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 09:15:00 | 518.10 | 520.11 | 519.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-06 13:45:00 | 525.50 | 520.40 | 519.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-07 11:15:00 | 516.90 | 519.74 | 519.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2023-12-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-07 11:15:00 | 516.90 | 519.74 | 519.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-07 12:15:00 | 516.40 | 519.07 | 519.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-08 09:15:00 | 521.90 | 518.48 | 518.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-08 09:15:00 | 521.90 | 518.48 | 518.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 09:15:00 | 521.90 | 518.48 | 518.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-08 09:30:00 | 524.25 | 518.48 | 518.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 10:15:00 | 521.10 | 519.01 | 519.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-08 10:45:00 | 522.70 | 519.01 | 519.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2023-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 11:15:00 | 521.00 | 519.41 | 519.29 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2023-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 12:15:00 | 518.30 | 519.18 | 519.20 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2023-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 14:15:00 | 520.50 | 519.27 | 519.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 13:15:00 | 521.45 | 520.34 | 519.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 14:15:00 | 526.65 | 528.16 | 524.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-12 14:45:00 | 526.75 | 528.16 | 524.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 11:15:00 | 525.65 | 527.64 | 525.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-13 11:30:00 | 525.85 | 527.64 | 525.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 12:15:00 | 523.70 | 526.85 | 525.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-13 12:30:00 | 523.90 | 526.85 | 525.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 13:15:00 | 531.40 | 527.76 | 526.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-13 15:00:00 | 532.60 | 528.73 | 526.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 15:15:00 | 548.05 | 557.84 | 558.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2023-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 15:15:00 | 548.05 | 557.84 | 558.25 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2023-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 09:15:00 | 567.50 | 557.49 | 557.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 14:15:00 | 571.60 | 564.73 | 561.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 09:15:00 | 608.10 | 609.43 | 599.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-29 10:00:00 | 608.10 | 609.43 | 599.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 15:15:00 | 607.50 | 612.02 | 609.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 09:15:00 | 617.70 | 612.02 | 609.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 10:45:00 | 613.65 | 612.74 | 609.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-03 09:15:00 | 600.30 | 611.98 | 611.02 | SL hit (close<static) qty=1.00 sl=606.95 alert=retest2 |

### Cycle 54 — SELL (started 2024-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 10:15:00 | 598.50 | 609.28 | 609.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-03 14:15:00 | 593.35 | 601.43 | 605.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-05 09:15:00 | 595.40 | 593.28 | 597.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-05 09:45:00 | 595.10 | 593.28 | 597.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 10:15:00 | 591.50 | 592.92 | 597.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-05 13:15:00 | 590.80 | 592.35 | 596.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-12 11:15:00 | 581.80 | 580.22 | 580.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2024-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 11:15:00 | 581.80 | 580.22 | 580.01 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 09:15:00 | 574.15 | 579.87 | 580.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 12:15:00 | 571.20 | 575.11 | 576.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-16 13:15:00 | 581.90 | 576.47 | 577.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 13:15:00 | 581.90 | 576.47 | 577.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 13:15:00 | 581.90 | 576.47 | 577.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 14:00:00 | 581.90 | 576.47 | 577.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 14:15:00 | 579.65 | 577.10 | 577.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-17 09:15:00 | 561.70 | 577.58 | 577.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-24 13:15:00 | 565.00 | 558.31 | 557.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2024-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 13:15:00 | 565.00 | 558.31 | 557.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-24 15:15:00 | 566.75 | 560.99 | 558.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-25 10:15:00 | 559.30 | 560.79 | 559.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-25 10:15:00 | 559.30 | 560.79 | 559.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 10:15:00 | 559.30 | 560.79 | 559.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 11:00:00 | 559.30 | 560.79 | 559.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 11:15:00 | 560.80 | 560.79 | 559.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-25 15:00:00 | 567.40 | 561.85 | 560.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-29 10:45:00 | 567.25 | 564.65 | 561.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-29 12:30:00 | 567.55 | 565.34 | 562.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-29 13:30:00 | 566.75 | 565.87 | 563.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 14:15:00 | 569.10 | 574.58 | 570.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 15:00:00 | 569.10 | 574.58 | 570.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 15:15:00 | 569.50 | 573.56 | 570.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 09:15:00 | 575.30 | 573.56 | 570.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 09:15:00 | 573.05 | 573.46 | 570.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 11:30:00 | 577.95 | 574.56 | 571.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 14:15:00 | 576.85 | 575.44 | 572.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 12:00:00 | 576.65 | 577.02 | 574.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 10:30:00 | 579.20 | 575.22 | 574.37 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 12:15:00 | 575.90 | 575.98 | 574.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 12:30:00 | 576.05 | 575.98 | 574.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 13:15:00 | 583.85 | 577.56 | 575.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 14:15:00 | 585.45 | 577.56 | 575.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-05 09:15:00 | 585.40 | 579.77 | 577.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-05 09:45:00 | 587.80 | 581.52 | 578.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-05 13:00:00 | 586.55 | 584.02 | 580.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 14:15:00 | 580.25 | 583.68 | 580.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 14:45:00 | 576.05 | 583.68 | 580.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 15:15:00 | 579.90 | 582.92 | 580.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-06 09:15:00 | 571.70 | 582.92 | 580.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-02-06 09:15:00 | 573.30 | 581.00 | 580.03 | SL hit (close<static) qty=1.00 sl=574.80 alert=retest2 |

### Cycle 58 — SELL (started 2024-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-06 11:15:00 | 575.35 | 578.58 | 579.02 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 14:15:00 | 586.45 | 579.80 | 579.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 09:15:00 | 592.50 | 583.29 | 581.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-07 15:15:00 | 587.40 | 588.86 | 585.48 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-08 09:15:00 | 598.75 | 588.86 | 585.48 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 587.10 | 597.12 | 593.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-02-09 09:15:00 | 587.10 | 597.12 | 593.29 | SL hit (close<ema400) qty=1.00 sl=593.29 alert=retest1 |

### Cycle 60 — SELL (started 2024-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 12:15:00 | 589.00 | 591.00 | 591.03 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-02-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 15:15:00 | 592.00 | 591.02 | 591.01 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 09:15:00 | 587.40 | 590.30 | 590.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 11:15:00 | 585.05 | 589.08 | 590.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-15 09:15:00 | 511.75 | 510.99 | 527.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-15 09:45:00 | 510.50 | 510.99 | 527.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 14:15:00 | 515.80 | 514.84 | 518.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-16 14:30:00 | 516.85 | 514.84 | 518.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 514.50 | 514.79 | 518.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-20 09:45:00 | 511.40 | 513.44 | 515.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-20 13:45:00 | 511.00 | 511.58 | 514.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-21 09:15:00 | 521.20 | 513.58 | 514.36 | SL hit (close>static) qty=1.00 sl=518.90 alert=retest2 |

### Cycle 63 — BUY (started 2024-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-21 10:15:00 | 522.10 | 515.29 | 515.06 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 15:15:00 | 511.30 | 514.74 | 515.05 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-02-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 09:15:00 | 519.30 | 515.65 | 515.43 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 09:15:00 | 509.60 | 516.85 | 517.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 10:15:00 | 507.00 | 514.88 | 516.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 14:15:00 | 509.05 | 507.59 | 509.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-27 15:00:00 | 509.05 | 507.59 | 509.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 15:15:00 | 510.00 | 508.07 | 509.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-28 09:15:00 | 512.95 | 508.07 | 509.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 510.70 | 508.60 | 510.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 10:45:00 | 507.50 | 508.53 | 509.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-01 09:15:00 | 514.85 | 505.80 | 506.15 | SL hit (close>static) qty=1.00 sl=514.45 alert=retest2 |

### Cycle 67 — BUY (started 2024-03-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 10:15:00 | 517.00 | 508.04 | 507.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 11:15:00 | 519.85 | 510.41 | 508.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 12:15:00 | 525.20 | 525.36 | 521.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-05 12:30:00 | 524.15 | 525.36 | 521.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 13:15:00 | 522.15 | 524.72 | 521.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 14:00:00 | 522.15 | 524.72 | 521.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 14:15:00 | 521.50 | 524.08 | 521.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 15:00:00 | 521.50 | 524.08 | 521.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 15:15:00 | 524.00 | 524.06 | 522.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 09:15:00 | 520.20 | 524.06 | 522.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 517.30 | 522.71 | 521.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:00:00 | 517.30 | 522.71 | 521.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 10:15:00 | 517.00 | 521.57 | 521.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-06 12:00:00 | 519.90 | 521.23 | 521.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-12 13:15:00 | 530.50 | 531.36 | 531.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 13:15:00 | 530.50 | 531.36 | 531.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 09:15:00 | 521.05 | 528.58 | 530.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 520.20 | 515.10 | 520.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 10:15:00 | 520.20 | 515.10 | 520.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 10:15:00 | 520.20 | 515.10 | 520.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 11:00:00 | 520.20 | 515.10 | 520.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 11:15:00 | 523.60 | 516.80 | 520.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 12:00:00 | 523.60 | 516.80 | 520.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 12:15:00 | 523.00 | 518.04 | 520.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 13:15:00 | 522.75 | 518.04 | 520.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 13:15:00 | 522.10 | 518.85 | 521.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 10:15:00 | 518.00 | 521.10 | 521.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 11:00:00 | 519.20 | 520.72 | 521.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-15 12:15:00 | 526.75 | 521.94 | 521.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2024-03-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 12:15:00 | 526.75 | 521.94 | 521.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-15 14:15:00 | 532.65 | 524.87 | 523.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 14:15:00 | 533.10 | 533.14 | 530.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-19 15:00:00 | 533.10 | 533.14 | 530.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 518.60 | 530.11 | 529.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 10:00:00 | 518.60 | 530.11 | 529.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2024-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 10:15:00 | 522.45 | 528.58 | 528.77 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 09:15:00 | 541.80 | 530.02 | 529.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 10:15:00 | 545.00 | 539.75 | 535.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 14:15:00 | 557.75 | 558.56 | 553.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-27 15:00:00 | 557.75 | 558.56 | 553.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 10:15:00 | 565.15 | 565.85 | 562.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-02 10:30:00 | 562.80 | 565.85 | 562.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 11:15:00 | 572.75 | 575.96 | 572.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 12:00:00 | 572.75 | 575.96 | 572.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 12:15:00 | 575.20 | 575.80 | 572.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-04 13:15:00 | 577.60 | 575.80 | 572.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-05 12:15:00 | 570.25 | 572.71 | 572.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2024-04-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 12:15:00 | 570.25 | 572.71 | 572.76 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2024-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 10:15:00 | 576.95 | 573.11 | 572.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 12:15:00 | 578.35 | 574.70 | 573.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 12:15:00 | 597.35 | 598.00 | 593.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-12 12:30:00 | 596.85 | 598.00 | 593.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 14:15:00 | 608.90 | 612.35 | 608.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-16 14:45:00 | 608.45 | 612.35 | 608.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 15:15:00 | 610.50 | 611.98 | 608.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-18 09:15:00 | 614.60 | 611.98 | 608.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-19 10:15:00 | 600.70 | 612.35 | 611.91 | SL hit (close<static) qty=1.00 sl=607.50 alert=retest2 |

### Cycle 74 — SELL (started 2024-04-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 11:15:00 | 608.65 | 611.61 | 611.62 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-04-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-19 12:15:00 | 612.30 | 611.75 | 611.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-19 13:15:00 | 614.20 | 612.24 | 611.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 09:15:00 | 614.00 | 616.75 | 615.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 09:15:00 | 614.00 | 616.75 | 615.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 09:15:00 | 614.00 | 616.75 | 615.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 10:00:00 | 614.00 | 616.75 | 615.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 10:15:00 | 614.50 | 616.30 | 615.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 12:30:00 | 615.80 | 615.70 | 615.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 14:00:00 | 616.00 | 615.76 | 615.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-23 14:15:00 | 611.95 | 615.00 | 614.93 | SL hit (close<static) qty=1.00 sl=612.55 alert=retest2 |

### Cycle 76 — SELL (started 2024-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 15:15:00 | 612.00 | 614.40 | 614.66 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 09:15:00 | 620.05 | 615.53 | 615.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 10:15:00 | 626.50 | 617.72 | 616.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-29 09:15:00 | 642.05 | 646.12 | 639.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-29 10:00:00 | 642.05 | 646.12 | 639.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 11:15:00 | 643.80 | 646.07 | 643.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 12:00:00 | 643.80 | 646.07 | 643.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 12:15:00 | 644.90 | 645.84 | 643.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 13:15:00 | 648.55 | 645.84 | 643.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 14:00:00 | 647.20 | 646.11 | 644.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-02 09:15:00 | 637.35 | 643.92 | 643.51 | SL hit (close<static) qty=1.00 sl=643.60 alert=retest2 |

### Cycle 78 — SELL (started 2024-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 10:15:00 | 639.45 | 643.02 | 643.14 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-05-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 09:15:00 | 647.25 | 643.39 | 643.09 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 09:15:00 | 634.00 | 643.05 | 643.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 10:15:00 | 625.50 | 635.91 | 639.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 624.45 | 624.21 | 630.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 09:45:00 | 625.05 | 624.21 | 630.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 627.30 | 624.85 | 629.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 12:45:00 | 628.05 | 624.85 | 629.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 13:15:00 | 630.00 | 625.88 | 629.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 14:00:00 | 630.00 | 625.88 | 629.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 14:15:00 | 633.10 | 627.32 | 629.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 15:00:00 | 633.10 | 627.32 | 629.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 15:15:00 | 633.90 | 628.64 | 630.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-09 09:15:00 | 631.30 | 628.64 | 630.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 10:15:00 | 628.15 | 628.44 | 629.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-09 10:45:00 | 630.15 | 628.44 | 629.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 11:15:00 | 630.35 | 628.82 | 629.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-09 11:45:00 | 629.80 | 628.82 | 629.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 12:15:00 | 623.25 | 627.71 | 629.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 13:15:00 | 621.65 | 627.71 | 629.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 14:30:00 | 619.95 | 625.69 | 627.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 09:30:00 | 622.30 | 624.27 | 626.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 13:00:00 | 622.80 | 624.45 | 626.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 13:15:00 | 625.15 | 624.59 | 626.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 13:45:00 | 626.85 | 624.59 | 626.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 14:15:00 | 626.15 | 624.90 | 626.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 15:00:00 | 626.15 | 624.90 | 626.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 15:15:00 | 625.00 | 624.92 | 626.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 09:15:00 | 618.90 | 624.92 | 626.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 13:15:00 | 624.35 | 623.81 | 625.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 13:15:00 | 633.20 | 625.69 | 625.80 | SL hit (close>static) qty=1.00 sl=630.55 alert=retest2 |

### Cycle 81 — BUY (started 2024-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 14:15:00 | 633.25 | 627.20 | 626.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 09:15:00 | 646.35 | 632.19 | 628.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 10:15:00 | 649.75 | 651.64 | 646.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 11:00:00 | 649.75 | 651.64 | 646.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 644.40 | 649.96 | 646.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:45:00 | 643.30 | 649.96 | 646.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 652.90 | 650.55 | 647.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 656.80 | 650.90 | 647.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 12:15:00 | 655.65 | 652.04 | 649.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 13:00:00 | 656.45 | 652.92 | 649.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 15:15:00 | 686.00 | 693.80 | 694.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2024-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 15:15:00 | 686.00 | 693.80 | 694.28 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 704.25 | 695.89 | 695.19 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 09:15:00 | 681.70 | 694.11 | 694.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 636.90 | 682.67 | 689.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 668.55 | 656.46 | 666.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 12:15:00 | 668.55 | 656.46 | 666.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 668.55 | 656.46 | 666.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:00:00 | 668.55 | 656.46 | 666.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 680.50 | 661.26 | 667.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:00:00 | 680.50 | 661.26 | 667.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 697.05 | 668.42 | 670.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 15:00:00 | 697.05 | 668.42 | 670.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2024-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 15:15:00 | 691.00 | 672.94 | 672.45 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-11 09:15:00 | 676.00 | 677.75 | 677.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-11 14:15:00 | 672.60 | 676.06 | 677.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-12 09:15:00 | 679.70 | 676.64 | 677.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-12 09:15:00 | 679.70 | 676.64 | 677.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 679.70 | 676.64 | 677.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-12 09:30:00 | 678.65 | 676.64 | 677.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 10:15:00 | 676.90 | 676.69 | 677.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-12 11:00:00 | 676.90 | 676.69 | 677.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 11:15:00 | 675.30 | 676.41 | 676.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-12 11:45:00 | 677.65 | 676.41 | 676.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 12:15:00 | 675.00 | 676.13 | 676.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-12 13:45:00 | 673.60 | 675.70 | 676.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-12 15:00:00 | 674.20 | 675.40 | 676.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-13 09:15:00 | 683.00 | 676.66 | 676.70 | SL hit (close>static) qty=1.00 sl=677.50 alert=retest2 |

### Cycle 87 — BUY (started 2024-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 10:15:00 | 679.20 | 677.17 | 676.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 09:15:00 | 684.25 | 680.06 | 678.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-14 15:15:00 | 682.10 | 682.92 | 681.08 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 09:15:00 | 687.45 | 682.92 | 681.08 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 680.80 | 682.50 | 681.05 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-18 09:15:00 | 680.80 | 682.50 | 681.05 | SL hit (close<ema400) qty=1.00 sl=681.05 alert=retest1 |

### Cycle 88 — SELL (started 2024-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 15:15:00 | 678.00 | 680.91 | 680.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 09:15:00 | 670.00 | 678.73 | 679.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 09:15:00 | 668.60 | 668.32 | 672.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 10:15:00 | 672.35 | 669.13 | 672.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 672.35 | 669.13 | 672.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 11:00:00 | 672.35 | 669.13 | 672.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 681.00 | 671.50 | 673.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 12:00:00 | 681.00 | 671.50 | 673.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 12:15:00 | 681.00 | 673.40 | 674.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 13:00:00 | 681.00 | 673.40 | 674.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2024-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 14:15:00 | 676.20 | 675.03 | 674.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 09:15:00 | 687.05 | 677.90 | 676.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 12:15:00 | 682.80 | 686.24 | 684.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 12:15:00 | 682.80 | 686.24 | 684.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 12:15:00 | 682.80 | 686.24 | 684.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 13:00:00 | 682.80 | 686.24 | 684.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 13:15:00 | 685.30 | 686.05 | 684.35 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2024-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 09:15:00 | 673.55 | 683.24 | 683.46 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 15:15:00 | 684.00 | 680.12 | 679.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 09:15:00 | 692.50 | 682.60 | 681.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 13:15:00 | 691.70 | 692.12 | 688.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-01 13:30:00 | 692.30 | 692.12 | 688.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 14:15:00 | 690.50 | 691.80 | 688.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 14:45:00 | 689.00 | 691.80 | 688.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 694.10 | 692.14 | 689.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 10:45:00 | 701.40 | 694.16 | 690.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 13:15:00 | 698.60 | 695.18 | 691.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 14:30:00 | 696.45 | 695.13 | 692.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 15:15:00 | 696.00 | 695.13 | 692.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 692.75 | 694.80 | 692.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 10:00:00 | 692.75 | 694.80 | 692.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 686.15 | 693.07 | 692.13 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-03 10:15:00 | 686.15 | 693.07 | 692.13 | SL hit (close<static) qty=1.00 sl=688.40 alert=retest2 |

### Cycle 92 — SELL (started 2024-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 12:15:00 | 689.15 | 691.44 | 691.50 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2024-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 14:15:00 | 693.20 | 691.80 | 691.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 701.90 | 693.85 | 692.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 14:15:00 | 691.45 | 695.43 | 694.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 14:15:00 | 691.45 | 695.43 | 694.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 691.45 | 695.43 | 694.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 15:00:00 | 691.45 | 695.43 | 694.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 693.50 | 695.04 | 694.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 09:15:00 | 699.10 | 695.04 | 694.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 12:30:00 | 693.90 | 697.02 | 696.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 13:30:00 | 694.30 | 696.62 | 696.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 11:15:00 | 692.85 | 698.38 | 698.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2024-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 11:15:00 | 692.85 | 698.38 | 698.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 12:15:00 | 689.20 | 696.55 | 697.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 09:15:00 | 694.70 | 694.39 | 696.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 09:15:00 | 694.70 | 694.39 | 696.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 694.70 | 694.39 | 696.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 10:45:00 | 694.20 | 694.49 | 695.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 10:15:00 | 694.05 | 695.93 | 696.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 10:45:00 | 694.35 | 695.41 | 695.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 11:15:00 | 694.00 | 693.91 | 694.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-15 11:15:00 | 700.85 | 695.30 | 695.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2024-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 11:15:00 | 700.85 | 695.30 | 695.08 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 13:15:00 | 690.25 | 695.86 | 696.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 668.55 | 687.32 | 691.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 10:15:00 | 669.70 | 669.57 | 677.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 11:00:00 | 669.70 | 669.57 | 677.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 659.50 | 648.83 | 652.32 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2024-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 12:15:00 | 661.70 | 655.59 | 654.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 13:15:00 | 666.45 | 657.76 | 655.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 13:15:00 | 667.10 | 667.26 | 662.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 14:00:00 | 667.10 | 667.26 | 662.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 656.85 | 664.99 | 662.85 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2024-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 12:15:00 | 659.05 | 661.44 | 661.54 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 13:15:00 | 663.30 | 661.81 | 661.70 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 14:15:00 | 659.65 | 661.38 | 661.52 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2024-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 09:15:00 | 670.20 | 663.07 | 662.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 15:15:00 | 672.00 | 667.42 | 665.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 12:15:00 | 671.40 | 672.15 | 668.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-01 12:45:00 | 672.15 | 672.15 | 668.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 669.20 | 671.56 | 668.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 13:45:00 | 669.75 | 671.56 | 668.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 674.55 | 672.16 | 669.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:30:00 | 671.00 | 672.16 | 669.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 658.35 | 669.41 | 668.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:45:00 | 655.75 | 669.41 | 668.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 10:15:00 | 655.60 | 666.65 | 667.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 12:15:00 | 654.55 | 662.50 | 665.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 625.65 | 618.44 | 627.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 09:15:00 | 625.65 | 618.44 | 627.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 625.65 | 618.44 | 627.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:00:00 | 625.65 | 618.44 | 627.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 623.90 | 619.53 | 627.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:45:00 | 627.25 | 619.53 | 627.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 628.70 | 622.10 | 625.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 09:15:00 | 623.50 | 622.10 | 625.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 619.95 | 621.67 | 625.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 11:00:00 | 616.10 | 620.56 | 624.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:15:00 | 615.05 | 619.38 | 623.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:45:00 | 615.55 | 618.54 | 622.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 13:00:00 | 616.05 | 617.68 | 620.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 625.10 | 619.16 | 620.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 14:00:00 | 625.10 | 619.16 | 620.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 623.05 | 619.94 | 620.80 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-12 09:15:00 | 626.25 | 621.65 | 621.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 09:15:00 | 626.25 | 621.65 | 621.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 10:15:00 | 629.30 | 623.18 | 622.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 10:15:00 | 627.05 | 627.70 | 625.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 10:15:00 | 627.05 | 627.70 | 625.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 627.05 | 627.70 | 625.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 11:00:00 | 627.05 | 627.70 | 625.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 629.25 | 628.01 | 625.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 11:30:00 | 626.70 | 628.01 | 625.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 626.40 | 627.69 | 625.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:45:00 | 626.70 | 627.69 | 625.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 628.10 | 627.77 | 626.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 628.10 | 627.77 | 626.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 620.55 | 626.33 | 625.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:00:00 | 620.55 | 626.33 | 625.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 620.90 | 625.24 | 625.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 09:15:00 | 633.60 | 625.24 | 625.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 10:15:00 | 620.10 | 624.58 | 624.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 10:15:00 | 620.10 | 624.58 | 624.91 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 09:15:00 | 631.20 | 625.05 | 624.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 653.80 | 636.09 | 631.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 11:15:00 | 681.35 | 682.35 | 673.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 12:00:00 | 681.35 | 682.35 | 673.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 680.20 | 685.52 | 680.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:00:00 | 680.20 | 685.52 | 680.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 684.20 | 685.26 | 680.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 15:00:00 | 685.10 | 685.23 | 680.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 691.40 | 684.98 | 681.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 11:15:00 | 694.80 | 700.47 | 700.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 11:15:00 | 694.80 | 700.47 | 700.67 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2024-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 13:15:00 | 701.95 | 700.34 | 700.20 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 09:15:00 | 691.85 | 698.87 | 699.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 10:15:00 | 690.00 | 697.10 | 698.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 669.75 | 669.38 | 676.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 668.25 | 669.45 | 673.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 668.25 | 669.45 | 673.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 09:30:00 | 672.50 | 669.45 | 673.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 669.15 | 669.41 | 672.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 12:30:00 | 672.45 | 669.41 | 672.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 13:15:00 | 670.15 | 669.56 | 671.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:00:00 | 670.15 | 669.56 | 671.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 654.25 | 657.83 | 662.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 09:45:00 | 651.00 | 657.57 | 660.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 09:45:00 | 653.00 | 651.90 | 655.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 10:30:00 | 651.85 | 652.96 | 655.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 13:15:00 | 668.30 | 657.11 | 656.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2024-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 13:15:00 | 668.30 | 657.11 | 656.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 14:15:00 | 676.95 | 661.08 | 658.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 13:15:00 | 672.15 | 672.58 | 666.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 13:45:00 | 671.50 | 672.58 | 666.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 678.15 | 682.58 | 679.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 15:00:00 | 686.60 | 681.82 | 680.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 09:15:00 | 685.65 | 682.11 | 680.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 15:15:00 | 687.30 | 681.18 | 680.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-27 09:15:00 | 755.26 | 734.09 | 722.84 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 13:15:00 | 749.80 | 751.08 | 751.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 741.75 | 748.33 | 749.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 10:15:00 | 734.00 | 725.91 | 731.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 10:15:00 | 734.00 | 725.91 | 731.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 734.00 | 725.91 | 731.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 11:00:00 | 734.00 | 725.91 | 731.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 11:15:00 | 730.80 | 726.89 | 731.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 12:30:00 | 729.55 | 727.51 | 731.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 13:15:00 | 729.10 | 727.51 | 731.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 14:00:00 | 729.85 | 727.98 | 730.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 12:15:00 | 727.80 | 729.60 | 730.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 730.95 | 729.87 | 730.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:00:00 | 730.95 | 729.87 | 730.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 730.30 | 729.95 | 730.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 14:15:00 | 728.80 | 729.95 | 730.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-11 09:15:00 | 744.90 | 733.06 | 731.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2024-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 09:15:00 | 744.90 | 733.06 | 731.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 10:15:00 | 747.15 | 735.88 | 733.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 10:15:00 | 743.50 | 743.81 | 739.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-14 10:45:00 | 743.40 | 743.81 | 739.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 739.95 | 742.84 | 740.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 13:45:00 | 741.40 | 742.84 | 740.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 14:15:00 | 742.85 | 742.84 | 740.37 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2024-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 11:15:00 | 731.95 | 738.50 | 738.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-15 12:15:00 | 724.60 | 735.72 | 737.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 09:15:00 | 737.45 | 732.38 | 735.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 09:15:00 | 737.45 | 732.38 | 735.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 737.45 | 732.38 | 735.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:30:00 | 736.80 | 732.38 | 735.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 735.35 | 732.98 | 735.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 11:15:00 | 733.65 | 732.98 | 735.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 739.15 | 733.77 | 734.42 | SL hit (close>static) qty=1.00 sl=738.90 alert=retest2 |

### Cycle 113 — BUY (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-17 10:15:00 | 741.00 | 735.22 | 735.02 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2024-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 15:15:00 | 732.55 | 735.07 | 735.13 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 10:15:00 | 736.80 | 735.47 | 735.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 11:15:00 | 750.50 | 738.48 | 736.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 10:15:00 | 744.65 | 746.85 | 742.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-21 11:00:00 | 744.65 | 746.85 | 742.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 741.30 | 745.74 | 742.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 12:00:00 | 741.30 | 745.74 | 742.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 12:15:00 | 740.20 | 744.63 | 742.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 13:00:00 | 740.20 | 744.63 | 742.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 740.00 | 743.12 | 742.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 14:45:00 | 740.25 | 743.12 | 742.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 15:15:00 | 737.70 | 742.04 | 741.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:15:00 | 744.50 | 742.04 | 741.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2024-10-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 09:15:00 | 732.90 | 740.21 | 740.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 12:15:00 | 727.90 | 735.70 | 738.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 730.00 | 729.07 | 733.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 10:00:00 | 730.00 | 729.07 | 733.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 694.65 | 684.80 | 692.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 10:30:00 | 695.85 | 684.80 | 692.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 694.80 | 686.80 | 692.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:45:00 | 693.50 | 686.80 | 692.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 13:15:00 | 693.30 | 689.40 | 693.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 15:15:00 | 691.50 | 690.27 | 693.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 14:45:00 | 691.35 | 686.91 | 689.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 09:15:00 | 691.25 | 688.49 | 689.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 10:15:00 | 696.45 | 691.26 | 691.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 696.45 | 691.26 | 691.03 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 14:15:00 | 687.60 | 690.93 | 691.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 09:15:00 | 682.70 | 689.04 | 690.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 11:15:00 | 688.05 | 687.58 | 689.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-31 12:00:00 | 688.05 | 687.58 | 689.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 685.30 | 687.12 | 688.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 13:45:00 | 683.30 | 685.75 | 688.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-01 17:15:00 | 690.40 | 686.52 | 687.85 | SL hit (close>static) qty=1.00 sl=689.55 alert=retest2 |

### Cycle 119 — BUY (started 2024-11-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 13:15:00 | 696.35 | 684.37 | 683.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 14:15:00 | 698.40 | 687.17 | 684.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 661.80 | 695.14 | 693.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 661.80 | 695.14 | 693.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 661.80 | 695.14 | 693.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:45:00 | 658.80 | 695.14 | 693.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 10:15:00 | 652.40 | 686.59 | 689.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 11:15:00 | 648.65 | 679.00 | 685.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 12:15:00 | 659.10 | 659.01 | 668.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-08 13:00:00 | 659.10 | 659.01 | 668.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 660.40 | 654.55 | 658.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:45:00 | 657.10 | 655.89 | 658.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 14:15:00 | 624.25 | 634.69 | 644.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 11:15:00 | 636.10 | 633.43 | 640.48 | SL hit (close>ema200) qty=0.50 sl=633.43 alert=retest2 |

### Cycle 121 — BUY (started 2024-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 11:15:00 | 656.65 | 641.70 | 640.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-18 12:15:00 | 662.00 | 645.76 | 642.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 10:15:00 | 647.55 | 649.07 | 646.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 10:15:00 | 647.55 | 649.07 | 646.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 647.55 | 649.07 | 646.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:45:00 | 646.60 | 649.07 | 646.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 647.75 | 648.81 | 646.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 12:00:00 | 647.75 | 648.81 | 646.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 639.30 | 647.55 | 646.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 15:00:00 | 639.30 | 647.55 | 646.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 638.00 | 645.64 | 645.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:15:00 | 644.50 | 645.64 | 645.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 643.15 | 645.14 | 645.36 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2024-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 10:15:00 | 648.90 | 645.90 | 645.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 09:15:00 | 652.15 | 648.13 | 647.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-22 15:15:00 | 652.00 | 652.10 | 649.87 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 09:15:00 | 659.75 | 652.10 | 649.87 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 660.75 | 663.68 | 661.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-27 14:15:00 | 660.75 | 663.68 | 661.51 | SL hit (close<ema400) qty=1.00 sl=661.51 alert=retest1 |

### Cycle 124 — SELL (started 2024-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 11:15:00 | 656.30 | 660.17 | 660.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 12:15:00 | 654.10 | 658.95 | 659.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 13:15:00 | 653.70 | 653.16 | 655.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 13:15:00 | 653.70 | 653.16 | 655.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 653.70 | 653.16 | 655.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 13:30:00 | 657.15 | 653.16 | 655.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 655.95 | 653.72 | 655.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 14:30:00 | 657.00 | 653.72 | 655.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 657.00 | 654.37 | 655.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:15:00 | 655.75 | 654.37 | 655.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 660.30 | 655.56 | 656.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:00:00 | 660.30 | 655.56 | 656.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 657.40 | 655.93 | 656.29 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 11:15:00 | 659.50 | 656.64 | 656.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 12:15:00 | 661.60 | 657.63 | 657.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 09:15:00 | 658.80 | 659.74 | 658.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 09:15:00 | 658.80 | 659.74 | 658.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 658.80 | 659.74 | 658.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:00:00 | 658.80 | 659.74 | 658.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 660.40 | 659.87 | 658.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:45:00 | 658.50 | 659.87 | 658.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 11:15:00 | 660.85 | 660.07 | 658.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 11:45:00 | 660.95 | 660.07 | 658.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 662.95 | 663.97 | 661.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:00:00 | 662.95 | 663.97 | 661.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 663.10 | 663.80 | 661.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:30:00 | 661.50 | 663.80 | 661.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 13:15:00 | 664.80 | 664.00 | 662.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 14:15:00 | 661.80 | 664.00 | 662.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 663.30 | 663.86 | 662.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 14:30:00 | 662.60 | 663.86 | 662.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 662.40 | 663.57 | 662.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:45:00 | 661.30 | 663.00 | 662.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 661.45 | 662.69 | 662.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:45:00 | 659.85 | 662.69 | 662.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 667.60 | 663.67 | 662.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:30:00 | 661.35 | 663.67 | 662.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 672.25 | 667.36 | 664.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:45:00 | 665.25 | 667.36 | 664.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 660.55 | 668.57 | 667.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:00:00 | 660.55 | 668.57 | 667.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2024-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 10:15:00 | 655.55 | 665.96 | 666.39 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2024-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 15:15:00 | 670.30 | 666.72 | 666.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 10:15:00 | 672.50 | 668.61 | 667.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 12:15:00 | 666.00 | 668.20 | 667.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 12:15:00 | 666.00 | 668.20 | 667.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 12:15:00 | 666.00 | 668.20 | 667.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 13:00:00 | 666.00 | 668.20 | 667.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 13:15:00 | 667.20 | 668.00 | 667.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 14:30:00 | 667.75 | 668.24 | 667.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 15:00:00 | 669.20 | 668.24 | 667.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 11:15:00 | 663.45 | 668.81 | 669.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 11:15:00 | 663.45 | 668.81 | 669.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 651.55 | 664.76 | 667.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 12:15:00 | 662.70 | 662.09 | 665.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 13:00:00 | 662.70 | 662.09 | 665.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 664.80 | 662.63 | 665.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:00:00 | 664.80 | 662.63 | 665.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 661.80 | 662.46 | 664.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 09:15:00 | 659.00 | 662.97 | 664.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 626.05 | 632.58 | 640.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-19 15:15:00 | 630.00 | 629.66 | 635.19 | SL hit (close>ema200) qty=0.50 sl=629.66 alert=retest2 |

### Cycle 129 — BUY (started 2025-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 09:15:00 | 591.95 | 585.57 | 585.12 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 575.05 | 585.34 | 585.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 569.25 | 576.16 | 580.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 579.35 | 571.01 | 574.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 579.35 | 571.01 | 574.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 579.35 | 571.01 | 574.87 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2025-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 12:15:00 | 586.75 | 578.20 | 577.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-14 14:15:00 | 590.65 | 582.09 | 579.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-15 13:15:00 | 588.25 | 588.64 | 584.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-15 14:00:00 | 588.25 | 588.64 | 584.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 614.60 | 618.87 | 615.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 14:45:00 | 615.60 | 618.87 | 615.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 613.65 | 617.82 | 614.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 609.35 | 617.82 | 614.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 607.90 | 615.84 | 614.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 10:00:00 | 607.90 | 615.84 | 614.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 611.00 | 614.87 | 613.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:15:00 | 606.50 | 614.87 | 613.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2025-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 12:15:00 | 611.15 | 613.30 | 613.34 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 613.90 | 613.26 | 613.25 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 14:15:00 | 608.05 | 612.38 | 612.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 09:15:00 | 601.50 | 609.51 | 611.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-24 11:15:00 | 613.30 | 609.63 | 611.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 11:15:00 | 613.30 | 609.63 | 611.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 11:15:00 | 613.30 | 609.63 | 611.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 12:00:00 | 613.30 | 609.63 | 611.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 12:15:00 | 612.50 | 610.20 | 611.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 13:30:00 | 609.65 | 610.02 | 611.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 579.17 | 589.94 | 598.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-29 13:15:00 | 581.55 | 580.22 | 585.94 | SL hit (close>ema200) qty=0.50 sl=580.22 alert=retest2 |

### Cycle 135 — BUY (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 11:15:00 | 589.75 | 588.28 | 588.17 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 12:15:00 | 585.75 | 587.77 | 587.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 13:15:00 | 584.20 | 587.06 | 587.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 14:15:00 | 587.65 | 587.18 | 587.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 14:15:00 | 587.65 | 587.18 | 587.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 587.65 | 587.18 | 587.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 15:00:00 | 587.65 | 587.18 | 587.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 590.60 | 587.86 | 587.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:15:00 | 587.25 | 587.86 | 587.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 585.65 | 587.42 | 587.68 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2025-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 10:15:00 | 592.95 | 588.53 | 588.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 12:15:00 | 593.60 | 590.02 | 588.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 585.65 | 590.85 | 590.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 585.65 | 590.85 | 590.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 585.65 | 590.85 | 590.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 585.65 | 590.85 | 590.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 589.90 | 590.66 | 590.16 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 15:15:00 | 586.10 | 589.14 | 589.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 570.35 | 585.38 | 587.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 593.15 | 580.10 | 582.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 593.15 | 580.10 | 582.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 593.15 | 580.10 | 582.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:45:00 | 593.40 | 580.10 | 582.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 589.00 | 581.88 | 583.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 12:15:00 | 586.85 | 583.23 | 583.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 12:15:00 | 589.95 | 584.57 | 584.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2025-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 12:15:00 | 589.95 | 584.57 | 584.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 596.20 | 587.18 | 585.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 10:15:00 | 593.70 | 596.20 | 592.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 11:00:00 | 593.70 | 596.20 | 592.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 594.15 | 595.43 | 592.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:45:00 | 593.05 | 595.43 | 592.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 593.25 | 594.99 | 592.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 593.90 | 594.99 | 592.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 605.10 | 597.85 | 594.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 608.15 | 597.85 | 594.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 14:45:00 | 606.00 | 603.48 | 598.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 592.50 | 602.20 | 599.10 | SL hit (close<static) qty=1.00 sl=594.70 alert=retest2 |

### Cycle 140 — SELL (started 2025-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 15:15:00 | 595.35 | 597.96 | 598.00 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-11 09:15:00 | 599.60 | 598.29 | 598.15 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 12:15:00 | 593.15 | 597.76 | 598.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-12 09:15:00 | 587.85 | 595.03 | 596.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 597.15 | 595.46 | 596.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 10:15:00 | 597.15 | 595.46 | 596.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 597.15 | 595.46 | 596.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:00:00 | 597.15 | 595.46 | 596.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 602.15 | 596.79 | 597.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 602.15 | 596.79 | 597.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2025-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 12:15:00 | 602.40 | 597.92 | 597.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 11:15:00 | 605.70 | 601.12 | 599.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 09:15:00 | 599.25 | 602.09 | 600.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-14 09:15:00 | 599.25 | 602.09 | 600.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 599.25 | 602.09 | 600.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 10:00:00 | 599.25 | 602.09 | 600.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 601.55 | 601.99 | 600.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-14 14:30:00 | 608.00 | 601.23 | 600.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-17 09:15:00 | 590.65 | 599.56 | 599.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 09:15:00 | 590.65 | 599.56 | 599.99 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 11:15:00 | 604.05 | 600.89 | 600.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-17 13:15:00 | 605.75 | 602.37 | 601.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-18 11:15:00 | 604.70 | 605.73 | 603.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-18 11:15:00 | 604.70 | 605.73 | 603.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 11:15:00 | 604.70 | 605.73 | 603.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 11:45:00 | 602.10 | 605.73 | 603.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 12:15:00 | 605.00 | 605.59 | 603.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 12:45:00 | 603.65 | 605.59 | 603.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 13:15:00 | 610.00 | 606.47 | 604.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-18 14:15:00 | 612.05 | 606.47 | 604.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-19 09:15:00 | 613.90 | 607.76 | 605.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 10:15:00 | 620.95 | 634.78 | 636.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2025-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 10:15:00 | 620.95 | 634.78 | 636.66 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2025-02-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-28 15:15:00 | 634.00 | 630.06 | 629.53 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-03-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 09:15:00 | 624.65 | 628.98 | 629.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 11:15:00 | 622.40 | 626.76 | 628.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 633.70 | 628.15 | 628.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 12:15:00 | 633.70 | 628.15 | 628.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 633.70 | 628.15 | 628.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:00:00 | 633.70 | 628.15 | 628.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2025-03-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 13:15:00 | 636.15 | 629.75 | 629.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-03 14:15:00 | 637.50 | 631.30 | 629.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 13:15:00 | 692.60 | 693.70 | 683.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 14:00:00 | 692.60 | 693.70 | 683.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 689.60 | 691.76 | 684.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 10:15:00 | 691.95 | 691.76 | 684.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 14:00:00 | 692.65 | 691.08 | 686.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 14:15:00 | 692.15 | 691.00 | 689.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 10:15:00 | 677.70 | 687.28 | 687.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 10:15:00 | 677.70 | 687.28 | 687.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 13:15:00 | 677.00 | 682.53 | 685.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 681.60 | 680.91 | 683.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 681.60 | 680.91 | 683.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 681.60 | 680.91 | 683.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 682.85 | 680.91 | 683.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 696.10 | 683.83 | 683.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 699.35 | 688.66 | 685.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 11:15:00 | 694.35 | 695.27 | 691.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-19 12:00:00 | 694.35 | 695.27 | 691.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 10:15:00 | 700.00 | 703.22 | 699.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 11:00:00 | 700.00 | 703.22 | 699.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 11:15:00 | 701.30 | 702.84 | 699.97 | EMA400 retest candle locked (from upside) |

### Cycle 152 — SELL (started 2025-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-24 09:15:00 | 696.60 | 698.85 | 698.85 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 13:15:00 | 704.80 | 699.99 | 699.36 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 09:15:00 | 692.10 | 699.01 | 699.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 12:15:00 | 687.25 | 694.00 | 696.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-25 13:15:00 | 695.00 | 694.20 | 696.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 13:15:00 | 695.00 | 694.20 | 696.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 695.00 | 694.20 | 696.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:45:00 | 695.30 | 694.20 | 696.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 698.50 | 694.69 | 696.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 13:30:00 | 691.70 | 695.12 | 695.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 14:30:00 | 692.45 | 694.33 | 695.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 10:45:00 | 692.45 | 694.03 | 695.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 11:15:00 | 691.90 | 694.03 | 695.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 693.00 | 693.82 | 694.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:45:00 | 693.40 | 693.82 | 694.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 695.15 | 694.09 | 694.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:00:00 | 695.15 | 694.09 | 694.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 692.60 | 693.79 | 694.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 09:30:00 | 683.45 | 691.56 | 693.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 09:15:00 | 657.12 | 666.45 | 675.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 09:15:00 | 657.83 | 666.45 | 675.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 09:15:00 | 657.83 | 666.45 | 675.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 09:15:00 | 657.30 | 666.45 | 675.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-02 14:15:00 | 661.30 | 660.99 | 668.88 | SL hit (close>ema200) qty=0.50 sl=660.99 alert=retest2 |

### Cycle 155 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 599.90 | 578.28 | 577.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 12:15:00 | 606.50 | 583.93 | 580.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 09:15:00 | 610.00 | 610.30 | 600.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 09:45:00 | 610.00 | 610.30 | 600.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 13:15:00 | 612.75 | 609.27 | 602.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 13:30:00 | 603.10 | 609.27 | 602.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 604.25 | 608.43 | 604.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:00:00 | 604.25 | 608.43 | 604.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 609.40 | 608.62 | 604.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:15:00 | 612.60 | 608.62 | 604.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:15:00 | 612.25 | 609.19 | 605.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 13:45:00 | 612.20 | 609.77 | 606.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:45:00 | 613.30 | 610.41 | 607.38 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 624.50 | 627.28 | 625.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 624.50 | 627.28 | 625.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 616.70 | 625.16 | 624.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 616.70 | 625.16 | 624.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-25 11:15:00 | 618.85 | 623.90 | 624.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 618.85 | 623.90 | 624.05 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 630.55 | 624.32 | 624.08 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 14:15:00 | 622.95 | 625.16 | 625.40 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 09:15:00 | 629.25 | 625.55 | 625.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 10:15:00 | 630.95 | 626.63 | 626.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 13:15:00 | 622.40 | 625.92 | 625.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 13:15:00 | 622.40 | 625.92 | 625.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 13:15:00 | 622.40 | 625.92 | 625.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 14:00:00 | 622.40 | 625.92 | 625.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 14:15:00 | 625.45 | 625.82 | 625.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 15:15:00 | 620.55 | 625.82 | 625.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — SELL (started 2025-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 15:15:00 | 620.55 | 624.77 | 625.33 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 09:15:00 | 647.65 | 629.34 | 627.36 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 630.00 | 631.86 | 632.08 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 10:15:00 | 635.20 | 632.53 | 632.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 11:15:00 | 636.60 | 633.34 | 632.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 09:15:00 | 629.55 | 633.77 | 633.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 09:15:00 | 629.55 | 633.77 | 633.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 629.55 | 633.77 | 633.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:45:00 | 627.90 | 633.77 | 633.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — SELL (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 10:15:00 | 629.50 | 632.92 | 633.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 11:15:00 | 626.85 | 631.71 | 632.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 12:15:00 | 626.30 | 623.21 | 626.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 12:15:00 | 626.30 | 623.21 | 626.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 12:15:00 | 626.30 | 623.21 | 626.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 13:00:00 | 626.30 | 623.21 | 626.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 624.80 | 623.53 | 626.29 | EMA400 retest candle locked (from downside) |

### Cycle 165 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 644.00 | 628.52 | 627.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 651.60 | 647.82 | 643.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 10:15:00 | 655.40 | 656.66 | 651.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 11:00:00 | 655.40 | 656.66 | 651.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 657.30 | 656.89 | 653.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 654.75 | 656.89 | 653.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 663.50 | 662.06 | 659.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:30:00 | 660.50 | 662.06 | 659.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 662.35 | 662.78 | 660.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:30:00 | 656.10 | 662.78 | 660.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 662.40 | 662.70 | 660.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 662.40 | 662.70 | 660.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 657.70 | 661.70 | 660.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 657.70 | 661.70 | 660.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 659.85 | 661.33 | 660.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:15:00 | 660.50 | 661.33 | 660.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 654.00 | 660.32 | 660.10 | SL hit (close<static) qty=1.00 sl=655.70 alert=retest2 |

### Cycle 166 — SELL (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 10:15:00 | 655.00 | 659.26 | 659.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 649.45 | 656.78 | 658.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 11:15:00 | 652.70 | 651.89 | 654.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 12:00:00 | 652.70 | 651.89 | 654.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 654.40 | 652.39 | 654.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:45:00 | 654.45 | 652.39 | 654.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 654.50 | 652.81 | 654.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:30:00 | 653.45 | 652.81 | 654.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 649.55 | 652.16 | 654.14 | EMA400 retest candle locked (from downside) |

### Cycle 167 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 660.05 | 655.90 | 655.46 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 10:15:00 | 652.55 | 658.09 | 658.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 11:15:00 | 652.10 | 656.89 | 657.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 12:15:00 | 651.00 | 650.88 | 653.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-29 13:00:00 | 651.00 | 650.88 | 653.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 634.85 | 632.45 | 637.04 | EMA400 retest candle locked (from downside) |

### Cycle 169 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 640.15 | 636.69 | 636.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 642.95 | 638.55 | 637.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 10:15:00 | 656.25 | 656.54 | 652.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 11:00:00 | 656.25 | 656.54 | 652.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 654.80 | 656.91 | 654.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:00:00 | 654.80 | 656.91 | 654.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 658.20 | 656.92 | 654.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 10:15:00 | 658.75 | 656.92 | 654.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 11:15:00 | 659.05 | 656.99 | 655.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 15:15:00 | 651.00 | 653.79 | 654.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 651.00 | 653.79 | 654.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 645.10 | 652.05 | 653.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 644.65 | 643.87 | 647.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 10:45:00 | 645.30 | 643.87 | 647.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 646.75 | 644.91 | 647.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:30:00 | 648.00 | 644.91 | 647.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 647.80 | 645.49 | 647.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 647.80 | 645.49 | 647.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 650.00 | 646.39 | 647.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 645.75 | 646.85 | 647.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:45:00 | 646.85 | 644.41 | 645.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 10:15:00 | 653.85 | 644.85 | 644.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 10:15:00 | 653.85 | 644.85 | 644.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 657.40 | 648.11 | 646.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 10:15:00 | 692.35 | 693.41 | 686.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 11:15:00 | 692.30 | 693.41 | 686.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 685.60 | 691.76 | 688.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:00:00 | 685.60 | 691.76 | 688.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 687.10 | 690.83 | 688.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:45:00 | 686.85 | 690.83 | 688.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 695.20 | 691.70 | 689.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 12:30:00 | 695.55 | 692.27 | 689.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 699.10 | 693.36 | 690.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 10:15:00 | 696.65 | 693.77 | 691.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:15:00 | 698.45 | 694.03 | 691.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 695.80 | 696.90 | 693.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:45:00 | 695.70 | 696.90 | 693.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 696.75 | 699.34 | 697.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:00:00 | 696.75 | 699.34 | 697.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 692.35 | 697.94 | 696.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 15:00:00 | 692.35 | 697.94 | 696.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 692.35 | 696.82 | 696.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 692.95 | 696.82 | 696.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-04 11:15:00 | 693.90 | 695.75 | 695.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — SELL (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 11:15:00 | 693.90 | 695.75 | 695.80 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 14:15:00 | 699.50 | 696.10 | 695.92 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 691.35 | 695.15 | 695.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 11:15:00 | 689.00 | 693.35 | 694.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 13:15:00 | 676.45 | 675.56 | 679.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 14:00:00 | 676.45 | 675.56 | 679.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 675.05 | 675.46 | 679.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:45:00 | 676.20 | 675.46 | 679.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 679.25 | 674.15 | 676.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:00:00 | 679.25 | 674.15 | 676.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 672.20 | 673.76 | 675.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:15:00 | 666.95 | 673.76 | 675.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 13:30:00 | 670.30 | 670.01 | 671.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 667.00 | 670.40 | 671.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 10:30:00 | 669.00 | 668.80 | 669.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 673.80 | 669.80 | 669.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:45:00 | 673.45 | 669.80 | 669.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-17 12:15:00 | 674.75 | 670.79 | 670.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — BUY (started 2025-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 12:15:00 | 674.75 | 670.79 | 670.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 09:15:00 | 681.05 | 676.09 | 674.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 693.35 | 695.58 | 692.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 10:00:00 | 693.35 | 695.58 | 692.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 692.55 | 694.98 | 692.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:00:00 | 692.55 | 694.98 | 692.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 690.90 | 694.16 | 692.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:45:00 | 691.30 | 694.16 | 692.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 691.70 | 693.67 | 692.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 15:00:00 | 693.90 | 693.38 | 692.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 685.25 | 691.87 | 691.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — SELL (started 2025-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 09:15:00 | 685.25 | 691.87 | 691.89 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 693.05 | 690.80 | 690.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 15:15:00 | 695.00 | 691.64 | 690.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 09:15:00 | 691.45 | 691.60 | 690.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 09:15:00 | 691.45 | 691.60 | 690.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 691.45 | 691.60 | 690.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:15:00 | 687.90 | 691.60 | 690.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 690.60 | 691.40 | 690.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:30:00 | 686.90 | 691.40 | 690.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 687.85 | 690.69 | 690.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 12:00:00 | 687.85 | 690.69 | 690.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — SELL (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 12:15:00 | 685.20 | 689.59 | 690.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 678.65 | 686.97 | 688.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 11:15:00 | 687.20 | 686.35 | 688.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 11:15:00 | 687.20 | 686.35 | 688.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 687.20 | 686.35 | 688.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:30:00 | 690.40 | 686.35 | 688.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 689.40 | 686.96 | 688.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:00:00 | 689.40 | 686.96 | 688.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 690.10 | 687.59 | 688.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 14:00:00 | 690.10 | 687.59 | 688.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 685.80 | 679.03 | 681.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:00:00 | 685.80 | 679.03 | 681.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 686.30 | 680.48 | 682.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:00:00 | 686.30 | 680.48 | 682.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — BUY (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 13:15:00 | 687.10 | 683.25 | 683.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 15:15:00 | 688.50 | 684.95 | 684.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 12:15:00 | 682.80 | 685.41 | 684.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 12:15:00 | 682.80 | 685.41 | 684.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 682.80 | 685.41 | 684.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:00:00 | 682.80 | 685.41 | 684.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 683.30 | 684.99 | 684.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 14:30:00 | 685.30 | 685.41 | 684.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 15:00:00 | 687.10 | 685.41 | 684.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 09:15:00 | 679.50 | 685.19 | 685.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 679.50 | 685.19 | 685.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 676.15 | 681.91 | 683.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 686.50 | 682.67 | 683.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 686.50 | 682.67 | 683.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 686.50 | 682.67 | 683.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 686.50 | 682.67 | 683.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 689.65 | 684.07 | 684.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 682.70 | 684.07 | 684.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 697.10 | 674.17 | 674.22 | SL hit (close>static) qty=1.00 sl=689.90 alert=retest2 |

### Cycle 181 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 10:15:00 | 702.55 | 679.84 | 676.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 712.00 | 697.19 | 691.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 10:15:00 | 707.20 | 709.02 | 702.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 11:00:00 | 707.20 | 709.02 | 702.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 700.35 | 706.07 | 703.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 698.95 | 706.07 | 703.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 702.25 | 705.31 | 703.50 | EMA400 retest candle locked (from upside) |

### Cycle 182 — SELL (started 2025-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 15:15:00 | 701.00 | 702.39 | 702.54 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 706.15 | 703.27 | 702.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 11:15:00 | 706.95 | 704.00 | 703.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 09:15:00 | 705.65 | 705.80 | 704.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 09:15:00 | 705.65 | 705.80 | 704.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 705.65 | 705.80 | 704.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 704.85 | 705.80 | 704.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 705.65 | 705.77 | 704.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 705.65 | 705.77 | 704.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 704.75 | 706.16 | 705.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 15:00:00 | 704.75 | 706.16 | 705.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 702.65 | 705.46 | 705.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:15:00 | 715.45 | 705.46 | 705.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 13:00:00 | 705.40 | 709.57 | 709.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 13:15:00 | 704.80 | 708.62 | 708.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 704.80 | 708.62 | 708.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 15:15:00 | 704.00 | 707.19 | 707.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 11:15:00 | 706.05 | 705.67 | 706.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 11:15:00 | 706.05 | 705.67 | 706.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 706.05 | 705.67 | 706.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:30:00 | 706.05 | 705.67 | 706.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 705.60 | 702.93 | 704.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 705.60 | 702.93 | 704.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 706.00 | 703.55 | 704.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:45:00 | 706.00 | 703.55 | 704.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 705.80 | 704.00 | 705.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:45:00 | 706.65 | 704.00 | 705.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 702.80 | 703.77 | 704.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 14:45:00 | 706.00 | 703.77 | 704.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 709.00 | 704.85 | 705.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 709.00 | 704.85 | 705.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 710.10 | 705.90 | 705.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 713.25 | 707.37 | 706.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 738.70 | 738.89 | 732.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 15:15:00 | 738.00 | 738.89 | 732.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 741.95 | 743.54 | 740.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:00:00 | 741.95 | 743.54 | 740.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 738.70 | 742.58 | 740.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 15:00:00 | 738.70 | 742.58 | 740.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 738.50 | 741.76 | 739.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 09:30:00 | 742.75 | 741.99 | 740.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 12:15:00 | 742.00 | 741.76 | 740.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 740.05 | 742.26 | 742.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — SELL (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 12:15:00 | 740.05 | 742.26 | 742.38 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 750.00 | 743.81 | 743.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 13:15:00 | 758.95 | 748.68 | 745.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 13:15:00 | 753.50 | 755.00 | 751.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 14:00:00 | 753.50 | 755.00 | 751.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 752.75 | 754.15 | 751.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:45:00 | 753.95 | 754.15 | 751.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 753.50 | 754.02 | 751.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:45:00 | 750.40 | 754.02 | 751.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 753.70 | 753.95 | 752.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 13:30:00 | 756.00 | 754.43 | 752.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 14:15:00 | 756.00 | 754.43 | 752.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 15:00:00 | 756.20 | 754.78 | 752.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 10:15:00 | 751.15 | 753.98 | 753.03 | SL hit (close<static) qty=1.00 sl=751.75 alert=retest2 |

### Cycle 188 — SELL (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 14:15:00 | 750.05 | 752.19 | 752.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 09:15:00 | 746.75 | 750.67 | 751.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 14:15:00 | 750.50 | 748.53 | 749.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 14:15:00 | 750.50 | 748.53 | 749.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 750.50 | 748.53 | 749.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:00:00 | 750.50 | 748.53 | 749.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 750.00 | 748.82 | 749.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 746.00 | 748.82 | 749.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 743.50 | 747.76 | 749.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:45:00 | 741.90 | 746.17 | 748.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 13:15:00 | 750.00 | 747.78 | 747.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — BUY (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 13:15:00 | 750.00 | 747.78 | 747.71 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 745.55 | 747.33 | 747.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 741.55 | 746.22 | 746.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 750.80 | 745.98 | 746.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 12:15:00 | 750.80 | 745.98 | 746.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 750.80 | 745.98 | 746.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:00:00 | 750.80 | 745.98 | 746.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 749.70 | 746.73 | 746.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 15:00:00 | 745.80 | 746.54 | 746.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 752.95 | 745.86 | 745.99 | SL hit (close>static) qty=1.00 sl=752.25 alert=retest2 |

### Cycle 191 — BUY (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 10:15:00 | 751.45 | 746.98 | 746.48 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 09:15:00 | 740.15 | 745.95 | 746.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 10:15:00 | 735.80 | 743.92 | 745.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 14:15:00 | 744.10 | 741.72 | 743.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 14:15:00 | 744.10 | 741.72 | 743.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 744.10 | 741.72 | 743.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:30:00 | 747.00 | 741.72 | 743.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 742.10 | 741.79 | 743.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:15:00 | 750.70 | 741.79 | 743.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 751.85 | 743.81 | 744.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 751.70 | 743.81 | 744.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — BUY (started 2025-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 10:15:00 | 754.05 | 745.85 | 745.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 12:15:00 | 756.35 | 749.13 | 746.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 10:15:00 | 757.00 | 760.83 | 756.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 10:15:00 | 757.00 | 760.83 | 756.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 757.00 | 760.83 | 756.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:45:00 | 756.20 | 760.83 | 756.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 761.35 | 760.94 | 757.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:30:00 | 756.15 | 760.94 | 757.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 778.15 | 777.25 | 774.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:45:00 | 771.80 | 777.25 | 774.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 771.00 | 776.00 | 773.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:45:00 | 771.55 | 776.00 | 773.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 769.95 | 774.79 | 773.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:15:00 | 768.35 | 774.79 | 773.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — SELL (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 14:15:00 | 767.50 | 771.95 | 772.46 | EMA200 below EMA400 |

### Cycle 195 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 775.75 | 772.12 | 772.10 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 09:15:00 | 765.95 | 772.40 | 772.60 | EMA200 below EMA400 |

### Cycle 197 — BUY (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 09:15:00 | 772.25 | 772.17 | 772.17 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 769.70 | 771.68 | 771.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 764.80 | 769.93 | 770.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 764.05 | 762.39 | 765.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 764.05 | 762.39 | 765.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 764.05 | 762.39 | 765.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 766.95 | 762.39 | 765.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 765.10 | 762.93 | 765.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 765.10 | 762.93 | 765.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 765.85 | 763.52 | 765.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:00:00 | 765.85 | 763.52 | 765.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 764.35 | 763.68 | 765.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 14:45:00 | 763.15 | 763.71 | 765.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 768.95 | 764.96 | 765.58 | SL hit (close>static) qty=1.00 sl=766.80 alert=retest2 |

### Cycle 199 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 775.50 | 767.07 | 766.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 13:15:00 | 778.05 | 771.98 | 769.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 10:15:00 | 775.85 | 775.85 | 772.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 11:00:00 | 775.85 | 775.85 | 772.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 767.25 | 774.00 | 771.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 767.25 | 774.00 | 771.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 769.95 | 773.19 | 771.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 14:15:00 | 772.50 | 773.19 | 771.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-28 09:15:00 | 849.75 | 838.86 | 824.44 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 200 — SELL (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 15:15:00 | 845.50 | 850.18 | 850.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 842.95 | 848.73 | 849.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 795.30 | 793.21 | 808.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:45:00 | 796.75 | 793.21 | 808.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 793.80 | 787.93 | 792.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 15:00:00 | 793.80 | 787.93 | 792.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 792.65 | 788.87 | 792.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 796.80 | 788.87 | 792.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 803.85 | 791.87 | 793.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:00:00 | 803.85 | 791.87 | 793.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 803.90 | 795.90 | 795.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 804.70 | 797.88 | 796.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 807.05 | 807.66 | 803.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 10:15:00 | 808.55 | 807.84 | 803.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 808.55 | 807.84 | 803.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:30:00 | 803.75 | 807.84 | 803.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 804.60 | 807.19 | 803.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:45:00 | 803.70 | 807.19 | 803.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 799.70 | 805.69 | 803.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:00:00 | 799.70 | 805.69 | 803.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 800.15 | 804.58 | 803.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:45:00 | 799.70 | 804.58 | 803.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 803.00 | 804.17 | 803.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 804.90 | 804.17 | 803.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 803.10 | 803.96 | 803.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 13:30:00 | 807.35 | 804.57 | 803.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 15:00:00 | 806.90 | 805.03 | 803.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 789.50 | 802.40 | 802.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 789.50 | 802.40 | 802.92 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 14:15:00 | 799.50 | 797.42 | 797.33 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 784.70 | 795.30 | 796.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 14:15:00 | 775.95 | 780.50 | 785.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 785.75 | 780.48 | 784.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 785.75 | 780.48 | 784.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 785.75 | 780.48 | 784.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:45:00 | 784.75 | 780.48 | 784.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 784.80 | 781.34 | 784.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 11:30:00 | 783.85 | 781.84 | 784.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 12:45:00 | 782.25 | 782.16 | 784.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 794.00 | 786.64 | 785.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 794.00 | 786.64 | 785.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 10:15:00 | 799.80 | 789.27 | 787.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 14:15:00 | 807.65 | 808.09 | 803.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 15:00:00 | 807.65 | 808.09 | 803.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 806.50 | 810.14 | 807.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 806.50 | 810.14 | 807.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 805.95 | 809.30 | 807.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:00:00 | 805.95 | 809.30 | 807.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 805.80 | 808.60 | 807.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:00:00 | 805.80 | 808.60 | 807.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 803.75 | 807.63 | 806.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:45:00 | 802.75 | 807.63 | 806.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 806.95 | 807.50 | 806.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 15:00:00 | 808.05 | 807.61 | 807.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 10:30:00 | 808.80 | 807.08 | 806.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 11:30:00 | 808.10 | 808.22 | 807.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 10:00:00 | 811.05 | 819.66 | 819.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 817.50 | 819.23 | 819.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:30:00 | 812.00 | 819.23 | 819.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 815.15 | 818.42 | 818.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — SELL (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 11:15:00 | 815.15 | 818.42 | 818.75 | EMA200 below EMA400 |

### Cycle 207 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 825.20 | 818.78 | 818.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 12:15:00 | 829.50 | 822.00 | 820.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 14:15:00 | 822.15 | 822.18 | 820.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 14:15:00 | 822.15 | 822.18 | 820.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 822.15 | 822.18 | 820.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:45:00 | 820.70 | 822.18 | 820.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 821.70 | 822.08 | 820.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 827.60 | 822.08 | 820.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 15:00:00 | 824.50 | 823.94 | 822.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 13:15:00 | 832.60 | 838.89 | 839.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 832.60 | 838.89 | 839.21 | EMA200 below EMA400 |

### Cycle 209 — BUY (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 09:15:00 | 845.50 | 839.57 | 839.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 09:15:00 | 853.30 | 847.10 | 843.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 10:15:00 | 849.65 | 854.04 | 850.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 10:15:00 | 849.65 | 854.04 | 850.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 849.65 | 854.04 | 850.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 849.65 | 854.04 | 850.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 851.40 | 853.51 | 850.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 15:15:00 | 853.75 | 852.02 | 850.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 15:15:00 | 865.35 | 867.65 | 867.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 210 — SELL (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 15:15:00 | 865.35 | 867.65 | 867.95 | EMA200 below EMA400 |

### Cycle 211 — BUY (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 09:15:00 | 875.35 | 869.19 | 868.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 11:15:00 | 880.55 | 872.52 | 870.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 13:15:00 | 942.80 | 944.84 | 930.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 14:00:00 | 942.80 | 944.84 | 930.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 939.50 | 942.09 | 936.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:45:00 | 939.90 | 942.09 | 936.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 918.75 | 936.74 | 935.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 918.75 | 936.74 | 935.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 212 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 905.90 | 930.57 | 932.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 903.10 | 925.08 | 929.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 15:15:00 | 904.00 | 902.79 | 911.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 09:15:00 | 902.00 | 902.79 | 911.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 902.45 | 902.72 | 910.28 | EMA400 retest candle locked (from downside) |

### Cycle 213 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 921.35 | 912.76 | 912.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 09:15:00 | 927.50 | 915.71 | 913.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 10:15:00 | 946.70 | 948.39 | 939.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 11:00:00 | 946.70 | 948.39 | 939.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 939.65 | 946.20 | 939.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:00:00 | 939.65 | 946.20 | 939.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 932.05 | 943.37 | 939.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 932.05 | 943.37 | 939.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 934.50 | 941.59 | 938.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 932.15 | 941.59 | 938.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 941.45 | 939.15 | 938.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 12:00:00 | 945.60 | 940.44 | 938.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 13:15:00 | 933.80 | 938.49 | 938.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 214 — SELL (started 2026-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 13:15:00 | 933.80 | 938.49 | 938.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 927.10 | 936.21 | 937.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 10:15:00 | 934.60 | 934.37 | 936.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 11:00:00 | 934.60 | 934.37 | 936.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 11:15:00 | 936.30 | 934.76 | 936.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 12:00:00 | 936.30 | 934.76 | 936.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 940.40 | 935.89 | 936.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 940.40 | 935.89 | 936.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 935.80 | 935.87 | 936.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:30:00 | 938.85 | 935.87 | 936.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 938.45 | 936.39 | 936.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 14:45:00 | 939.55 | 936.39 | 936.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 215 — BUY (started 2026-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 15:15:00 | 940.00 | 937.11 | 937.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 09:15:00 | 943.30 | 938.35 | 937.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-22 10:15:00 | 936.30 | 937.94 | 937.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 10:15:00 | 936.30 | 937.94 | 937.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 936.30 | 937.94 | 937.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 936.30 | 937.94 | 937.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 939.60 | 938.27 | 937.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 12:15:00 | 942.80 | 938.27 | 937.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 13:45:00 | 943.95 | 940.05 | 938.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 12:15:00 | 957.40 | 986.16 | 988.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 216 — SELL (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 12:15:00 | 957.40 | 986.16 | 988.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 09:15:00 | 921.40 | 963.65 | 976.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 920.00 | 916.92 | 935.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 14:00:00 | 920.00 | 916.92 | 935.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 936.10 | 922.86 | 934.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 952.15 | 922.86 | 934.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 953.65 | 929.02 | 936.54 | EMA400 retest candle locked (from downside) |

### Cycle 217 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 968.80 | 944.71 | 942.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 15:15:00 | 970.00 | 961.17 | 954.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 932.00 | 955.33 | 952.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 932.00 | 955.33 | 952.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 932.00 | 955.33 | 952.59 | EMA400 retest candle locked (from upside) |

### Cycle 218 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 931.75 | 950.62 | 950.70 | EMA200 below EMA400 |

### Cycle 219 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 967.45 | 946.71 | 944.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 11:15:00 | 970.40 | 962.32 | 955.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 953.90 | 964.20 | 959.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 953.90 | 964.20 | 959.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 953.90 | 964.20 | 959.48 | EMA400 retest candle locked (from upside) |

### Cycle 220 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 907.85 | 952.01 | 956.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 09:15:00 | 886.00 | 903.94 | 917.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 09:15:00 | 897.30 | 894.06 | 904.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 897.30 | 894.06 | 904.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 897.30 | 894.06 | 904.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:30:00 | 903.25 | 894.06 | 904.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 901.85 | 897.28 | 901.22 | EMA400 retest candle locked (from downside) |

### Cycle 221 — BUY (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 13:15:00 | 914.05 | 905.17 | 904.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 10:15:00 | 923.95 | 910.42 | 906.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 10:15:00 | 924.80 | 927.15 | 918.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-23 11:00:00 | 924.80 | 927.15 | 918.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 914.50 | 924.62 | 918.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:45:00 | 911.70 | 924.62 | 918.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 908.50 | 921.40 | 917.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 13:00:00 | 908.50 | 921.40 | 917.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 910.65 | 919.25 | 917.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 15:00:00 | 916.55 | 918.71 | 916.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 918.40 | 917.87 | 916.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 12:00:00 | 915.75 | 917.79 | 917.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 13:00:00 | 916.80 | 917.60 | 916.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 917.45 | 917.57 | 917.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:30:00 | 914.00 | 917.57 | 917.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 923.15 | 918.68 | 917.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 15:15:00 | 925.30 | 918.68 | 917.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 15:15:00 | 925.40 | 931.92 | 931.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 15:15:00 | 925.40 | 930.61 | 931.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 222 — SELL (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 15:15:00 | 925.40 | 930.61 | 931.21 | EMA200 below EMA400 |

### Cycle 223 — BUY (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 09:15:00 | 936.95 | 931.88 | 931.73 | EMA200 above EMA400 |

### Cycle 224 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 925.70 | 930.64 | 931.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 923.10 | 929.14 | 930.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 13:15:00 | 934.40 | 929.69 | 930.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 13:15:00 | 934.40 | 929.69 | 930.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 934.40 | 929.69 | 930.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 14:00:00 | 934.40 | 929.69 | 930.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 225 — BUY (started 2026-03-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 14:15:00 | 941.15 | 931.98 | 931.41 | EMA200 above EMA400 |

### Cycle 226 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 924.10 | 930.13 | 930.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 915.65 | 927.23 | 929.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 13:15:00 | 931.05 | 927.43 | 929.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 13:15:00 | 931.05 | 927.43 | 929.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 13:15:00 | 931.05 | 927.43 | 929.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:00:00 | 931.05 | 927.43 | 929.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 920.95 | 926.14 | 928.34 | EMA400 retest candle locked (from downside) |

### Cycle 227 — BUY (started 2026-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 09:15:00 | 970.00 | 934.33 | 931.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 10:15:00 | 981.90 | 943.84 | 936.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-05 14:15:00 | 954.35 | 958.06 | 946.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 14:15:00 | 954.35 | 958.06 | 946.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 954.35 | 958.06 | 946.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 954.35 | 958.06 | 946.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 963.10 | 958.85 | 948.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 10:15:00 | 966.40 | 958.85 | 948.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 12:45:00 | 964.95 | 960.97 | 952.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 14:00:00 | 964.70 | 961.71 | 953.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 940.60 | 957.29 | 953.50 | SL hit (close<static) qty=1.00 sl=943.15 alert=retest2 |

### Cycle 228 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 941.00 | 949.74 | 950.46 | EMA200 below EMA400 |

### Cycle 229 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 955.55 | 948.15 | 947.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 965.15 | 953.20 | 950.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 954.60 | 956.40 | 952.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 12:15:00 | 954.60 | 956.40 | 952.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 954.60 | 956.40 | 952.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 954.60 | 956.40 | 952.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 936.35 | 953.42 | 952.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:00:00 | 936.35 | 953.42 | 952.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 230 — SELL (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 10:15:00 | 945.00 | 951.74 | 951.94 | EMA200 below EMA400 |

### Cycle 231 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 953.60 | 952.09 | 952.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 13:15:00 | 966.05 | 954.88 | 953.34 | Break + close above crossover candle high |

### Cycle 232 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 915.90 | 951.30 | 952.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 13:15:00 | 913.50 | 930.33 | 940.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 09:15:00 | 926.45 | 923.84 | 934.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 926.45 | 923.84 | 934.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 926.45 | 923.84 | 934.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:45:00 | 930.65 | 923.84 | 934.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 937.95 | 927.46 | 934.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 11:30:00 | 940.45 | 927.46 | 934.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 926.05 | 927.18 | 933.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 13:15:00 | 923.50 | 927.18 | 933.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 15:00:00 | 920.00 | 925.07 | 931.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 15:15:00 | 935.00 | 933.11 | 933.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 233 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 935.00 | 933.11 | 933.00 | EMA200 above EMA400 |

### Cycle 234 — SELL (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 09:15:00 | 925.10 | 931.50 | 932.28 | EMA200 below EMA400 |

### Cycle 235 — BUY (started 2026-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 14:15:00 | 934.20 | 932.70 | 932.53 | EMA200 above EMA400 |

### Cycle 236 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 922.80 | 930.58 | 931.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 906.65 | 921.58 | 926.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 861.00 | 852.29 | 866.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 861.80 | 852.29 | 866.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 869.15 | 856.48 | 863.97 | EMA400 retest candle locked (from downside) |

### Cycle 237 — BUY (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 11:15:00 | 867.00 | 866.49 | 866.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-30 09:15:00 | 907.50 | 875.27 | 870.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 889.75 | 901.03 | 893.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 889.75 | 901.03 | 893.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 889.75 | 901.03 | 893.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 889.75 | 901.03 | 893.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 890.15 | 898.85 | 893.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:45:00 | 887.80 | 898.85 | 893.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 12:15:00 | 910.40 | 900.31 | 894.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 13:15:00 | 916.90 | 900.31 | 894.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 14:45:00 | 916.60 | 906.40 | 898.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 1008.59 | 985.17 | 978.80 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 238 — SELL (started 2026-04-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 14:15:00 | 1015.55 | 1022.17 | 1022.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 15:15:00 | 1013.10 | 1020.35 | 1021.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 10:15:00 | 1022.75 | 1020.51 | 1021.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 10:15:00 | 1022.75 | 1020.51 | 1021.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 1022.75 | 1020.51 | 1021.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:45:00 | 1021.30 | 1020.51 | 1021.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 1022.20 | 1020.85 | 1021.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:15:00 | 1022.10 | 1020.85 | 1021.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 1021.65 | 1021.01 | 1021.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 13:30:00 | 1019.75 | 1021.06 | 1021.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 14:30:00 | 1019.65 | 1021.26 | 1021.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 09:15:00 | 1031.60 | 1023.33 | 1022.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 239 — BUY (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 09:15:00 | 1031.60 | 1023.33 | 1022.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 12:15:00 | 1038.00 | 1028.78 | 1025.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 09:15:00 | 1034.15 | 1038.75 | 1034.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 09:15:00 | 1034.15 | 1038.75 | 1034.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 1034.15 | 1038.75 | 1034.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 10:15:00 | 1029.00 | 1038.75 | 1034.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 1032.75 | 1037.55 | 1034.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 10:30:00 | 1028.00 | 1037.55 | 1034.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 1068.00 | 1068.81 | 1061.27 | EMA400 retest candle locked (from upside) |

### Cycle 240 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 1045.05 | 1060.91 | 1061.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 14:15:00 | 1038.60 | 1050.57 | 1056.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 15:15:00 | 1043.90 | 1043.86 | 1048.60 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 09:15:00 | 1033.40 | 1043.86 | 1048.60 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 1041.40 | 1043.37 | 1047.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:00:00 | 1041.40 | 1043.37 | 1047.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 1047.20 | 1044.13 | 1047.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:00:00 | 1047.20 | 1044.13 | 1047.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 1045.00 | 1044.31 | 1047.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:15:00 | 1049.40 | 1044.31 | 1047.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 1057.70 | 1046.99 | 1048.53 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 1057.70 | 1046.99 | 1048.53 | SL hit (close>ema400) qty=1.00 sl=1048.53 alert=retest1 |

### Cycle 241 — BUY (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 14:15:00 | 1055.30 | 1049.79 | 1049.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 15:15:00 | 1057.00 | 1051.23 | 1050.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 14:15:00 | 1044.00 | 1053.84 | 1052.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 14:15:00 | 1044.00 | 1053.84 | 1052.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 1044.00 | 1053.84 | 1052.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 15:00:00 | 1044.00 | 1053.84 | 1052.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 1044.20 | 1051.91 | 1051.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 1052.30 | 1051.91 | 1051.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 1053.60 | 1052.18 | 1051.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 12:00:00 | 1059.20 | 1053.59 | 1052.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 15:15:00 | 1057.00 | 1054.55 | 1053.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 09:15:00 | 1043.00 | 1052.63 | 1052.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 242 — SELL (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 09:15:00 | 1043.00 | 1052.63 | 1052.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 12:15:00 | 1039.60 | 1047.61 | 1050.16 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-17 10:00:00 | 411.30 | 2023-05-23 09:15:00 | 410.70 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2023-05-18 10:15:00 | 410.65 | 2023-05-23 09:15:00 | 410.70 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2023-05-25 09:15:00 | 403.00 | 2023-05-26 10:15:00 | 413.25 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2023-06-07 13:15:00 | 417.90 | 2023-06-09 12:15:00 | 415.35 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2023-06-07 13:45:00 | 417.75 | 2023-06-09 12:15:00 | 415.35 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2023-06-09 09:45:00 | 417.45 | 2023-06-09 12:15:00 | 415.35 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2023-06-09 12:00:00 | 417.40 | 2023-06-09 12:15:00 | 415.35 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2023-06-15 12:45:00 | 424.80 | 2023-06-21 10:15:00 | 420.15 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2023-06-15 13:30:00 | 424.75 | 2023-06-21 10:15:00 | 420.15 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2023-06-16 09:15:00 | 427.90 | 2023-06-21 10:15:00 | 420.15 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2023-06-19 14:30:00 | 425.25 | 2023-06-21 10:15:00 | 420.15 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2023-06-22 11:45:00 | 421.35 | 2023-06-27 13:15:00 | 418.40 | STOP_HIT | 1.00 | 0.70% |
| BUY | retest2 | 2023-07-18 14:00:00 | 445.70 | 2023-07-19 09:15:00 | 440.85 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2023-07-21 09:15:00 | 438.90 | 2023-07-25 11:15:00 | 443.60 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2023-07-25 10:15:00 | 440.60 | 2023-07-25 11:15:00 | 443.60 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2023-07-28 12:30:00 | 449.05 | 2023-08-02 13:15:00 | 450.95 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2023-07-28 13:45:00 | 449.75 | 2023-08-02 13:15:00 | 450.95 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2023-08-04 10:30:00 | 456.45 | 2023-08-04 11:15:00 | 461.55 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2023-08-18 09:15:00 | 444.35 | 2023-08-21 13:15:00 | 447.85 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2023-08-31 09:15:00 | 463.50 | 2023-09-06 15:15:00 | 478.10 | STOP_HIT | 1.00 | 3.15% |
| SELL | retest2 | 2023-09-21 11:30:00 | 480.25 | 2023-09-28 09:15:00 | 478.25 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2023-09-21 12:00:00 | 480.85 | 2023-09-28 09:15:00 | 478.25 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2023-10-09 09:15:00 | 468.55 | 2023-10-10 13:15:00 | 483.55 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2023-10-09 12:00:00 | 470.50 | 2023-10-10 13:15:00 | 483.55 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2023-10-09 15:00:00 | 469.50 | 2023-10-10 13:15:00 | 483.55 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2023-10-10 09:30:00 | 471.00 | 2023-10-10 13:15:00 | 483.55 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2023-10-10 12:15:00 | 468.00 | 2023-10-10 13:15:00 | 483.55 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2023-10-13 11:00:00 | 484.70 | 2023-10-16 09:15:00 | 480.10 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2023-10-25 11:15:00 | 466.00 | 2023-10-31 15:15:00 | 460.05 | STOP_HIT | 1.00 | 1.28% |
| BUY | retest2 | 2023-11-20 09:15:00 | 502.05 | 2023-11-20 14:15:00 | 496.95 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2023-11-20 14:00:00 | 498.75 | 2023-11-20 14:15:00 | 496.95 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2023-11-30 13:45:00 | 516.95 | 2023-12-07 11:15:00 | 516.90 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2023-11-30 14:45:00 | 516.75 | 2023-12-07 11:15:00 | 516.90 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2023-12-01 09:15:00 | 518.75 | 2023-12-07 11:15:00 | 516.90 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2023-12-01 12:45:00 | 516.75 | 2023-12-07 11:15:00 | 516.90 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2023-12-04 09:15:00 | 522.25 | 2023-12-07 11:15:00 | 516.90 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2023-12-04 14:15:00 | 518.60 | 2023-12-07 11:15:00 | 516.90 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2023-12-04 15:00:00 | 518.70 | 2023-12-07 11:15:00 | 516.90 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2023-12-05 09:30:00 | 518.70 | 2023-12-07 11:15:00 | 516.90 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2023-12-06 13:45:00 | 525.50 | 2023-12-07 11:15:00 | 516.90 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2023-12-13 15:00:00 | 532.60 | 2023-12-20 15:15:00 | 548.05 | STOP_HIT | 1.00 | 2.90% |
| BUY | retest2 | 2024-01-02 09:15:00 | 617.70 | 2024-01-03 09:15:00 | 600.30 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2024-01-02 10:45:00 | 613.65 | 2024-01-03 09:15:00 | 600.30 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2024-01-05 13:15:00 | 590.80 | 2024-01-12 11:15:00 | 581.80 | STOP_HIT | 1.00 | 1.52% |
| SELL | retest2 | 2024-01-17 09:15:00 | 561.70 | 2024-01-24 13:15:00 | 565.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2024-01-25 15:00:00 | 567.40 | 2024-02-06 09:15:00 | 573.30 | STOP_HIT | 1.00 | 1.04% |
| BUY | retest2 | 2024-01-29 10:45:00 | 567.25 | 2024-02-06 09:15:00 | 573.30 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2024-01-29 12:30:00 | 567.55 | 2024-02-06 09:15:00 | 573.30 | STOP_HIT | 1.00 | 1.01% |
| BUY | retest2 | 2024-01-29 13:30:00 | 566.75 | 2024-02-06 09:15:00 | 573.30 | STOP_HIT | 1.00 | 1.16% |
| BUY | retest2 | 2024-01-31 11:30:00 | 577.95 | 2024-02-06 11:15:00 | 575.35 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-01-31 14:15:00 | 576.85 | 2024-02-06 11:15:00 | 575.35 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2024-02-01 12:00:00 | 576.65 | 2024-02-06 11:15:00 | 575.35 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2024-02-02 10:30:00 | 579.20 | 2024-02-06 11:15:00 | 575.35 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-02-02 14:15:00 | 585.45 | 2024-02-06 11:15:00 | 575.35 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-02-05 09:15:00 | 585.40 | 2024-02-06 11:15:00 | 575.35 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-02-05 09:45:00 | 587.80 | 2024-02-06 11:15:00 | 575.35 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-02-05 13:00:00 | 586.55 | 2024-02-06 11:15:00 | 575.35 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest1 | 2024-02-08 09:15:00 | 598.75 | 2024-02-09 09:15:00 | 587.10 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2024-02-20 09:45:00 | 511.40 | 2024-02-21 09:15:00 | 521.20 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-02-20 13:45:00 | 511.00 | 2024-02-21 09:15:00 | 521.20 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-02-28 10:45:00 | 507.50 | 2024-03-01 09:15:00 | 514.85 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-03-06 12:00:00 | 519.90 | 2024-03-12 13:15:00 | 530.50 | STOP_HIT | 1.00 | 2.04% |
| SELL | retest2 | 2024-03-15 10:15:00 | 518.00 | 2024-03-15 12:15:00 | 526.75 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-03-15 11:00:00 | 519.20 | 2024-03-15 12:15:00 | 526.75 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-04-04 13:15:00 | 577.60 | 2024-04-05 12:15:00 | 570.25 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-04-18 09:15:00 | 614.60 | 2024-04-19 10:15:00 | 600.70 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2024-04-23 12:30:00 | 615.80 | 2024-04-23 14:15:00 | 611.95 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-04-23 14:00:00 | 616.00 | 2024-04-23 14:15:00 | 611.95 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-04-30 13:15:00 | 648.55 | 2024-05-02 09:15:00 | 637.35 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-04-30 14:00:00 | 647.20 | 2024-05-02 09:15:00 | 637.35 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-05-09 13:15:00 | 621.65 | 2024-05-13 13:15:00 | 633.20 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-05-09 14:30:00 | 619.95 | 2024-05-13 13:15:00 | 633.20 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-05-10 09:30:00 | 622.30 | 2024-05-13 13:15:00 | 633.20 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-05-10 13:00:00 | 622.80 | 2024-05-13 13:15:00 | 633.20 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2024-05-13 09:15:00 | 618.90 | 2024-05-13 13:15:00 | 633.20 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2024-05-13 13:15:00 | 624.35 | 2024-05-13 13:15:00 | 633.20 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-05-17 09:15:00 | 656.80 | 2024-05-31 15:15:00 | 686.00 | STOP_HIT | 1.00 | 4.45% |
| BUY | retest2 | 2024-05-17 12:15:00 | 655.65 | 2024-05-31 15:15:00 | 686.00 | STOP_HIT | 1.00 | 4.63% |
| BUY | retest2 | 2024-05-17 13:00:00 | 656.45 | 2024-05-31 15:15:00 | 686.00 | STOP_HIT | 1.00 | 4.50% |
| SELL | retest2 | 2024-06-12 13:45:00 | 673.60 | 2024-06-13 09:15:00 | 683.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-06-12 15:00:00 | 674.20 | 2024-06-13 09:15:00 | 683.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest1 | 2024-06-18 09:15:00 | 687.45 | 2024-06-18 09:15:00 | 680.80 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-07-02 10:45:00 | 701.40 | 2024-07-03 10:15:00 | 686.15 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2024-07-02 13:15:00 | 698.60 | 2024-07-03 10:15:00 | 686.15 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2024-07-02 14:30:00 | 696.45 | 2024-07-03 10:15:00 | 686.15 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2024-07-02 15:15:00 | 696.00 | 2024-07-03 10:15:00 | 686.15 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-07-05 09:15:00 | 699.10 | 2024-07-10 11:15:00 | 692.85 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-07-08 12:30:00 | 693.90 | 2024-07-10 11:15:00 | 692.85 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2024-07-08 13:30:00 | 694.30 | 2024-07-10 11:15:00 | 692.85 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-07-11 10:45:00 | 694.20 | 2024-07-15 11:15:00 | 700.85 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-07-12 10:15:00 | 694.05 | 2024-07-15 11:15:00 | 700.85 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-07-12 10:45:00 | 694.35 | 2024-07-15 11:15:00 | 700.85 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-07-15 11:15:00 | 694.00 | 2024-07-15 11:15:00 | 700.85 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-08-08 11:00:00 | 616.10 | 2024-08-12 09:15:00 | 626.25 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-08-08 13:15:00 | 615.05 | 2024-08-12 09:15:00 | 626.25 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-08-08 13:45:00 | 615.55 | 2024-08-12 09:15:00 | 626.25 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-08-09 13:00:00 | 616.05 | 2024-08-12 09:15:00 | 626.25 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-08-14 09:15:00 | 633.60 | 2024-08-14 10:15:00 | 620.10 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-08-23 15:00:00 | 685.10 | 2024-08-29 11:15:00 | 694.80 | STOP_HIT | 1.00 | 1.42% |
| BUY | retest2 | 2024-08-26 09:15:00 | 691.40 | 2024-08-29 11:15:00 | 694.80 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2024-09-11 09:45:00 | 651.00 | 2024-09-12 13:15:00 | 668.30 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2024-09-12 09:45:00 | 653.00 | 2024-09-12 13:15:00 | 668.30 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2024-09-12 10:30:00 | 651.85 | 2024-09-12 13:15:00 | 668.30 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2024-09-18 15:00:00 | 686.60 | 2024-09-27 09:15:00 | 755.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-19 09:15:00 | 685.65 | 2024-09-27 09:15:00 | 754.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-19 15:15:00 | 687.30 | 2024-09-27 09:15:00 | 756.03 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-09 12:30:00 | 729.55 | 2024-10-11 09:15:00 | 744.90 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2024-10-09 13:15:00 | 729.10 | 2024-10-11 09:15:00 | 744.90 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2024-10-09 14:00:00 | 729.85 | 2024-10-11 09:15:00 | 744.90 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2024-10-10 12:15:00 | 727.80 | 2024-10-11 09:15:00 | 744.90 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2024-10-10 14:15:00 | 728.80 | 2024-10-11 09:15:00 | 744.90 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2024-10-16 11:15:00 | 733.65 | 2024-10-17 09:15:00 | 739.15 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-10-28 15:15:00 | 691.50 | 2024-10-30 10:15:00 | 696.45 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-10-29 14:45:00 | 691.35 | 2024-10-30 10:15:00 | 696.45 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-10-30 09:15:00 | 691.25 | 2024-10-30 10:15:00 | 696.45 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-10-31 13:45:00 | 683.30 | 2024-11-01 17:15:00 | 690.40 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-11-04 09:15:00 | 679.30 | 2024-11-05 13:15:00 | 696.35 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2024-11-05 09:30:00 | 681.85 | 2024-11-05 13:15:00 | 696.35 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-11-05 10:30:00 | 682.55 | 2024-11-05 13:15:00 | 696.35 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-11-12 12:45:00 | 657.10 | 2024-11-13 14:15:00 | 624.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 12:45:00 | 657.10 | 2024-11-14 11:15:00 | 636.10 | STOP_HIT | 0.50 | 3.20% |
| BUY | retest1 | 2024-11-25 09:15:00 | 659.75 | 2024-11-27 14:15:00 | 660.75 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2024-11-28 10:15:00 | 662.55 | 2024-11-28 11:15:00 | 656.30 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-12-10 14:30:00 | 667.75 | 2024-12-12 11:15:00 | 663.45 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-12-10 15:00:00 | 669.20 | 2024-12-12 11:15:00 | 663.45 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-12-16 09:15:00 | 659.00 | 2024-12-19 09:15:00 | 626.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 09:15:00 | 659.00 | 2024-12-19 15:15:00 | 630.00 | STOP_HIT | 0.50 | 4.40% |
| SELL | retest2 | 2025-01-24 13:30:00 | 609.65 | 2025-01-28 09:15:00 | 579.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 13:30:00 | 609.65 | 2025-01-29 13:15:00 | 581.55 | STOP_HIT | 0.50 | 4.61% |
| SELL | retest2 | 2025-02-04 12:15:00 | 586.85 | 2025-02-04 12:15:00 | 589.95 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-02-07 11:15:00 | 608.15 | 2025-02-10 09:15:00 | 592.50 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-02-07 14:45:00 | 606.00 | 2025-02-10 09:15:00 | 592.50 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-02-14 14:30:00 | 608.00 | 2025-02-17 09:15:00 | 590.65 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2025-02-18 14:15:00 | 612.05 | 2025-02-25 10:15:00 | 620.95 | STOP_HIT | 1.00 | 1.45% |
| BUY | retest2 | 2025-02-19 09:15:00 | 613.90 | 2025-02-25 10:15:00 | 620.95 | STOP_HIT | 1.00 | 1.15% |
| BUY | retest2 | 2025-03-11 10:15:00 | 691.95 | 2025-03-13 10:15:00 | 677.70 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-03-11 14:00:00 | 692.65 | 2025-03-13 10:15:00 | 677.70 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-03-12 14:15:00 | 692.15 | 2025-03-13 10:15:00 | 677.70 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-03-26 13:30:00 | 691.70 | 2025-04-02 09:15:00 | 657.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-26 14:30:00 | 692.45 | 2025-04-02 09:15:00 | 657.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-27 10:45:00 | 692.45 | 2025-04-02 09:15:00 | 657.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-27 11:15:00 | 691.90 | 2025-04-02 09:15:00 | 657.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-26 13:30:00 | 691.70 | 2025-04-02 14:15:00 | 661.30 | STOP_HIT | 0.50 | 4.39% |
| SELL | retest2 | 2025-03-26 14:30:00 | 692.45 | 2025-04-02 14:15:00 | 661.30 | STOP_HIT | 0.50 | 4.50% |
| SELL | retest2 | 2025-03-27 10:45:00 | 692.45 | 2025-04-02 14:15:00 | 661.30 | STOP_HIT | 0.50 | 4.50% |
| SELL | retest2 | 2025-03-27 11:15:00 | 691.90 | 2025-04-02 14:15:00 | 661.30 | STOP_HIT | 0.50 | 4.42% |
| SELL | retest2 | 2025-03-28 09:30:00 | 683.45 | 2025-04-03 09:15:00 | 649.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 09:30:00 | 683.45 | 2025-04-04 09:15:00 | 615.11 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-04-17 11:15:00 | 612.60 | 2025-04-25 11:15:00 | 618.85 | STOP_HIT | 1.00 | 1.02% |
| BUY | retest2 | 2025-04-17 12:15:00 | 612.25 | 2025-04-25 11:15:00 | 618.85 | STOP_HIT | 1.00 | 1.08% |
| BUY | retest2 | 2025-04-17 13:45:00 | 612.20 | 2025-04-25 11:15:00 | 618.85 | STOP_HIT | 1.00 | 1.09% |
| BUY | retest2 | 2025-04-21 09:45:00 | 613.30 | 2025-04-25 11:15:00 | 618.85 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest2 | 2025-05-21 13:15:00 | 660.50 | 2025-05-22 09:15:00 | 654.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-06-12 10:15:00 | 658.75 | 2025-06-12 15:15:00 | 651.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-06-12 11:15:00 | 659.05 | 2025-06-12 15:15:00 | 651.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-06-17 11:45:00 | 645.75 | 2025-06-20 10:15:00 | 653.85 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-06-18 10:45:00 | 646.85 | 2025-06-20 10:15:00 | 653.85 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-07-01 12:30:00 | 695.55 | 2025-07-04 11:15:00 | 693.90 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-07-02 09:15:00 | 699.10 | 2025-07-04 11:15:00 | 693.90 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-07-02 10:15:00 | 696.65 | 2025-07-04 11:15:00 | 693.90 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-07-02 11:15:00 | 698.45 | 2025-07-04 11:15:00 | 693.90 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-07-14 11:15:00 | 666.95 | 2025-07-17 12:15:00 | 674.75 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-07-15 13:30:00 | 670.30 | 2025-07-17 12:15:00 | 674.75 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-07-16 09:15:00 | 667.00 | 2025-07-17 12:15:00 | 674.75 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-07-17 10:30:00 | 669.00 | 2025-07-17 12:15:00 | 674.75 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-07-25 15:00:00 | 693.90 | 2025-07-28 09:15:00 | 685.25 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-08-05 14:30:00 | 685.30 | 2025-08-07 09:15:00 | 679.50 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-08-05 15:00:00 | 687.10 | 2025-08-07 09:15:00 | 679.50 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-08-08 09:15:00 | 682.70 | 2025-08-13 09:15:00 | 697.10 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-08-25 09:15:00 | 715.45 | 2025-08-26 13:15:00 | 704.80 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-08-26 13:00:00 | 705.40 | 2025-08-26 13:15:00 | 704.80 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-09-09 09:30:00 | 742.75 | 2025-09-11 12:15:00 | 740.05 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-09-09 12:15:00 | 742.00 | 2025-09-11 12:15:00 | 740.05 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-09-16 13:30:00 | 756.00 | 2025-09-17 10:15:00 | 751.15 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-09-16 14:15:00 | 756.00 | 2025-09-17 10:15:00 | 751.15 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-09-16 15:00:00 | 756.20 | 2025-09-17 10:15:00 | 751.15 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-09-17 13:15:00 | 756.25 | 2025-09-17 13:15:00 | 748.15 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-09-19 10:45:00 | 741.90 | 2025-09-22 13:15:00 | 750.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-09-23 15:00:00 | 745.80 | 2025-09-25 09:15:00 | 752.95 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-10-15 14:45:00 | 763.15 | 2025-10-16 09:15:00 | 768.95 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-10-17 14:15:00 | 772.50 | 2025-10-28 09:15:00 | 849.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-17 13:30:00 | 807.35 | 2025-11-18 09:15:00 | 789.50 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-11-17 15:00:00 | 806.90 | 2025-11-18 09:15:00 | 789.50 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-11-25 11:30:00 | 783.85 | 2025-11-26 09:15:00 | 794.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-11-25 12:45:00 | 782.25 | 2025-11-26 09:15:00 | 794.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-12-02 15:00:00 | 808.05 | 2025-12-09 11:15:00 | 815.15 | STOP_HIT | 1.00 | 0.88% |
| BUY | retest2 | 2025-12-03 10:30:00 | 808.80 | 2025-12-09 11:15:00 | 815.15 | STOP_HIT | 1.00 | 0.79% |
| BUY | retest2 | 2025-12-03 11:30:00 | 808.10 | 2025-12-09 11:15:00 | 815.15 | STOP_HIT | 1.00 | 0.87% |
| BUY | retest2 | 2025-12-09 10:00:00 | 811.05 | 2025-12-09 11:15:00 | 815.15 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest2 | 2025-12-11 09:15:00 | 827.60 | 2025-12-16 13:15:00 | 832.60 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest2 | 2025-12-11 15:00:00 | 824.50 | 2025-12-16 13:15:00 | 832.60 | STOP_HIT | 1.00 | 0.98% |
| BUY | retest2 | 2025-12-19 15:15:00 | 853.75 | 2025-12-29 15:15:00 | 865.35 | STOP_HIT | 1.00 | 1.36% |
| BUY | retest2 | 2026-01-19 12:00:00 | 945.60 | 2026-01-20 13:15:00 | 933.80 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2026-01-22 12:15:00 | 942.80 | 2026-01-30 12:15:00 | 957.40 | STOP_HIT | 1.00 | 1.55% |
| BUY | retest2 | 2026-01-22 13:45:00 | 943.95 | 2026-01-30 12:15:00 | 957.40 | STOP_HIT | 1.00 | 1.42% |
| BUY | retest2 | 2026-02-23 15:00:00 | 916.55 | 2026-02-27 15:15:00 | 925.40 | STOP_HIT | 1.00 | 0.97% |
| BUY | retest2 | 2026-02-24 09:15:00 | 918.40 | 2026-02-27 15:15:00 | 925.40 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2026-02-24 12:00:00 | 915.75 | 2026-02-27 15:15:00 | 925.40 | STOP_HIT | 1.00 | 1.05% |
| BUY | retest2 | 2026-02-24 13:00:00 | 916.80 | 2026-02-27 15:15:00 | 925.40 | STOP_HIT | 1.00 | 0.94% |
| BUY | retest2 | 2026-02-24 15:15:00 | 925.30 | 2026-02-27 15:15:00 | 925.40 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2026-02-27 15:15:00 | 925.40 | 2026-02-27 15:15:00 | 925.40 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2026-03-06 10:15:00 | 966.40 | 2026-03-09 09:15:00 | 940.60 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2026-03-06 12:45:00 | 964.95 | 2026-03-09 09:15:00 | 940.60 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2026-03-06 14:00:00 | 964.70 | 2026-03-09 09:15:00 | 940.60 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2026-03-16 13:15:00 | 923.50 | 2026-03-17 15:15:00 | 935.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2026-03-16 15:00:00 | 920.00 | 2026-03-17 15:15:00 | 935.00 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2026-04-02 13:15:00 | 916.90 | 2026-04-15 09:15:00 | 1008.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 14:45:00 | 916.60 | 2026-04-15 09:15:00 | 1008.26 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-21 13:30:00 | 1019.75 | 2026-04-22 09:15:00 | 1031.60 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-04-21 14:30:00 | 1019.65 | 2026-04-22 09:15:00 | 1031.60 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest1 | 2026-05-05 09:15:00 | 1033.40 | 2026-05-05 12:15:00 | 1057.70 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2026-05-07 12:00:00 | 1059.20 | 2026-05-08 09:15:00 | 1043.00 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2026-05-07 15:15:00 | 1057.00 | 2026-05-08 09:15:00 | 1043.00 | STOP_HIT | 1.00 | -1.32% |
