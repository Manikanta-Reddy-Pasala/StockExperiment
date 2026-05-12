# ITC (ITC)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 307.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 90 |
| ALERT1 | 56 |
| ALERT2 | 54 |
| ALERT2_SKIP | 29 |
| ALERT3 | 134 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 50 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 51 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 53 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 40
- **Target hits / Stop hits / Partials:** 1 / 50 / 2
- **Avg / median % per leg:** 0.03% / -0.40%
- **Sum % (uncompounded):** 1.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 6 | 27.3% | 0 | 22 | 0 | -0.27% | -5.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 22 | 6 | 27.3% | 0 | 22 | 0 | -0.27% | -5.9% |
| SELL (all) | 31 | 7 | 22.6% | 1 | 28 | 2 | 0.24% | 7.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.18% | -1.2% |
| SELL @ 3rd Alert (retest2) | 30 | 7 | 23.3% | 1 | 27 | 2 | 0.29% | 8.7% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.18% | -1.2% |
| retest2 (combined) | 52 | 13 | 25.0% | 1 | 49 | 2 | 0.06% | 2.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 432.20 | 429.41 | 429.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 434.55 | 431.09 | 430.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 431.35 | 432.43 | 431.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 431.35 | 432.43 | 431.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 431.35 | 432.43 | 431.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 09:45:00 | 430.70 | 432.43 | 431.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 10:15:00 | 431.60 | 432.27 | 431.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 10:30:00 | 430.30 | 432.27 | 431.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 11:15:00 | 430.60 | 431.93 | 431.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:00:00 | 430.60 | 431.93 | 431.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 426.80 | 430.91 | 430.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:45:00 | 425.80 | 430.91 | 430.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 13:15:00 | 427.35 | 430.20 | 430.36 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 434.60 | 430.28 | 429.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 12:15:00 | 435.85 | 433.39 | 431.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 11:15:00 | 434.05 | 434.86 | 433.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 12:00:00 | 434.05 | 434.86 | 433.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 433.90 | 434.67 | 433.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 12:30:00 | 433.85 | 434.67 | 433.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 433.05 | 434.35 | 433.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 433.05 | 434.35 | 433.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 434.75 | 434.43 | 433.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 15:15:00 | 436.10 | 434.43 | 433.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:30:00 | 435.95 | 435.92 | 435.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 11:45:00 | 435.85 | 435.77 | 435.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 14:15:00 | 433.20 | 434.85 | 434.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 14:15:00 | 433.20 | 434.85 | 434.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 15:15:00 | 432.40 | 434.36 | 434.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 434.90 | 428.80 | 430.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 434.90 | 428.80 | 430.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 434.90 | 428.80 | 430.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 434.90 | 428.80 | 430.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 438.00 | 430.64 | 431.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 438.00 | 430.64 | 431.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 437.15 | 431.94 | 431.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 439.30 | 435.53 | 433.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 437.85 | 439.53 | 437.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 437.85 | 439.53 | 437.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 437.85 | 439.53 | 437.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 437.85 | 439.53 | 437.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 440.60 | 439.74 | 437.41 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 15:15:00 | 434.70 | 436.37 | 436.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 421.65 | 433.43 | 435.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 15:15:00 | 418.75 | 418.15 | 420.75 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-02 09:15:00 | 415.30 | 418.15 | 420.75 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 419.70 | 418.29 | 420.14 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-02 12:15:00 | 420.20 | 418.68 | 420.14 | SL hit (close>ema400) qty=1.00 sl=420.14 alert=retest1 |

### Cycle 7 — BUY (started 2025-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 14:15:00 | 419.15 | 418.31 | 418.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 15:15:00 | 419.45 | 418.54 | 418.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 12:15:00 | 426.00 | 426.40 | 424.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 13:00:00 | 426.00 | 426.40 | 424.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 423.75 | 425.73 | 424.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:00:00 | 423.75 | 425.73 | 424.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 421.15 | 424.82 | 424.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 421.15 | 424.82 | 424.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 420.95 | 424.04 | 424.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 417.15 | 421.38 | 422.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 417.85 | 416.98 | 419.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:00:00 | 417.85 | 416.98 | 419.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 418.45 | 417.41 | 418.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:30:00 | 418.60 | 417.41 | 418.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 418.00 | 417.82 | 418.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:30:00 | 416.10 | 417.32 | 418.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:00:00 | 416.25 | 417.12 | 417.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:15:00 | 415.80 | 417.06 | 417.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 14:30:00 | 415.85 | 416.64 | 416.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 418.35 | 416.82 | 417.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:00:00 | 418.35 | 416.82 | 417.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 418.10 | 417.08 | 417.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:30:00 | 418.40 | 417.08 | 417.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-20 11:15:00 | 418.90 | 417.44 | 417.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 11:15:00 | 418.90 | 417.44 | 417.26 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 09:15:00 | 414.50 | 417.12 | 417.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-23 15:15:00 | 413.40 | 415.70 | 416.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 417.90 | 416.14 | 416.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 417.90 | 416.14 | 416.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 417.90 | 416.14 | 416.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:30:00 | 417.90 | 416.14 | 416.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 418.45 | 416.60 | 416.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:00:00 | 418.45 | 416.60 | 416.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 420.10 | 417.30 | 417.02 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 13:15:00 | 414.50 | 416.68 | 416.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 14:15:00 | 414.15 | 416.17 | 416.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 10:15:00 | 417.95 | 416.22 | 416.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 10:15:00 | 417.95 | 416.22 | 416.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 417.95 | 416.22 | 416.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 11:00:00 | 417.95 | 416.22 | 416.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 416.75 | 416.33 | 416.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 12:30:00 | 416.45 | 416.50 | 416.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 13:15:00 | 417.25 | 416.65 | 416.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 13:15:00 | 417.25 | 416.65 | 416.60 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 15:15:00 | 416.20 | 416.55 | 416.56 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 09:15:00 | 418.65 | 416.97 | 416.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 13:15:00 | 419.15 | 417.66 | 417.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 13:15:00 | 419.05 | 419.14 | 418.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 13:30:00 | 418.95 | 419.14 | 418.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 418.65 | 419.04 | 418.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 418.65 | 419.04 | 418.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 418.00 | 418.93 | 418.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:45:00 | 418.00 | 418.93 | 418.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 418.35 | 418.82 | 418.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 12:00:00 | 418.35 | 418.82 | 418.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 13:15:00 | 416.10 | 417.86 | 418.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 14:15:00 | 415.00 | 416.39 | 417.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 12:15:00 | 414.10 | 413.87 | 414.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 12:30:00 | 414.05 | 413.87 | 414.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 414.15 | 413.92 | 414.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:30:00 | 414.90 | 413.92 | 414.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 413.60 | 413.77 | 414.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 13:15:00 | 412.35 | 413.48 | 414.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 09:15:00 | 415.35 | 413.44 | 413.85 | SL hit (close>static) qty=1.00 sl=415.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 11:15:00 | 416.55 | 414.47 | 414.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 15:15:00 | 418.00 | 416.53 | 415.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 418.00 | 418.54 | 417.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 10:00:00 | 418.00 | 418.54 | 417.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 417.20 | 418.27 | 417.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:45:00 | 417.20 | 418.27 | 417.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 416.70 | 417.96 | 417.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 12:45:00 | 417.65 | 417.89 | 417.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 417.55 | 417.48 | 417.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 416.00 | 417.15 | 417.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 416.00 | 417.15 | 417.19 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 09:15:00 | 418.40 | 417.24 | 417.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 10:15:00 | 419.00 | 417.59 | 417.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 423.20 | 423.40 | 421.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 10:00:00 | 423.20 | 423.40 | 421.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 423.15 | 424.07 | 423.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:30:00 | 422.55 | 423.75 | 422.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 422.20 | 423.44 | 422.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 421.50 | 423.44 | 422.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 421.15 | 422.55 | 422.56 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 14:15:00 | 422.95 | 422.61 | 422.58 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 422.05 | 422.50 | 422.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 11:15:00 | 421.25 | 422.14 | 422.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 410.45 | 409.94 | 411.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 09:30:00 | 410.25 | 409.94 | 411.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 410.15 | 409.80 | 410.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:30:00 | 411.00 | 409.80 | 410.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 409.50 | 408.29 | 408.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:00:00 | 409.50 | 408.29 | 408.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 410.85 | 408.80 | 409.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:45:00 | 411.75 | 408.80 | 409.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 11:15:00 | 411.00 | 409.24 | 409.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 12:15:00 | 412.50 | 409.89 | 409.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 13:15:00 | 417.00 | 417.18 | 415.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-04 14:00:00 | 417.00 | 417.18 | 415.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 414.05 | 416.50 | 415.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:45:00 | 414.15 | 416.50 | 415.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 413.35 | 415.87 | 415.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:45:00 | 413.60 | 415.87 | 415.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 14:15:00 | 414.15 | 414.86 | 414.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 412.30 | 414.18 | 414.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 15:15:00 | 412.40 | 412.37 | 413.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 09:15:00 | 413.45 | 412.37 | 413.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 414.00 | 412.70 | 413.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:15:00 | 414.05 | 412.70 | 413.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 412.55 | 412.67 | 413.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 11:45:00 | 412.35 | 412.66 | 413.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 10:15:00 | 414.20 | 413.52 | 413.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 10:15:00 | 414.20 | 413.52 | 413.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 09:15:00 | 416.15 | 414.42 | 413.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 13:15:00 | 416.95 | 417.00 | 416.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 14:15:00 | 416.40 | 416.88 | 416.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 416.40 | 416.88 | 416.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 416.40 | 416.88 | 416.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 416.10 | 416.72 | 416.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:45:00 | 415.80 | 416.59 | 416.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 414.85 | 416.24 | 415.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:45:00 | 414.95 | 416.24 | 415.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 414.00 | 415.79 | 415.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 414.00 | 415.79 | 415.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 12:15:00 | 415.05 | 415.64 | 415.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 10:15:00 | 413.00 | 414.43 | 415.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 12:15:00 | 407.80 | 407.53 | 409.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 13:00:00 | 407.80 | 407.53 | 409.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 409.40 | 408.57 | 409.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 12:15:00 | 408.40 | 408.62 | 409.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 11:15:00 | 402.00 | 401.48 | 401.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 11:15:00 | 402.00 | 401.48 | 401.45 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 15:15:00 | 400.80 | 401.38 | 401.43 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 09:15:00 | 404.90 | 402.08 | 401.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 12:15:00 | 408.20 | 404.62 | 403.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-01 10:15:00 | 407.40 | 407.43 | 405.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-01 11:00:00 | 407.40 | 407.43 | 405.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 405.55 | 407.05 | 405.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:00:00 | 405.55 | 407.05 | 405.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 405.15 | 406.67 | 405.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:45:00 | 405.20 | 406.67 | 405.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 406.00 | 406.54 | 405.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 09:30:00 | 406.45 | 406.37 | 405.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 10:00:00 | 406.80 | 406.37 | 405.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 10:45:00 | 406.35 | 406.46 | 405.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 12:30:00 | 406.30 | 406.45 | 405.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 405.00 | 406.16 | 405.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 405.00 | 406.16 | 405.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 406.80 | 406.29 | 405.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 409.10 | 406.43 | 405.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:45:00 | 407.90 | 413.34 | 411.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 406.70 | 410.89 | 410.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 406.70 | 410.89 | 410.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 405.45 | 409.80 | 410.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 13:15:00 | 408.65 | 408.34 | 409.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 14:00:00 | 408.65 | 408.34 | 409.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 407.70 | 408.21 | 408.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:30:00 | 408.70 | 408.21 | 408.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 409.45 | 408.48 | 408.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:00:00 | 409.45 | 408.48 | 408.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 409.85 | 408.76 | 408.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:30:00 | 409.75 | 408.76 | 408.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 13:15:00 | 410.70 | 409.33 | 409.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 11:15:00 | 411.05 | 409.95 | 409.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 09:15:00 | 413.20 | 414.47 | 413.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 10:00:00 | 413.20 | 414.47 | 413.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 416.00 | 414.78 | 413.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 11:15:00 | 416.65 | 414.78 | 413.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 11:15:00 | 412.50 | 413.60 | 413.46 | SL hit (close<static) qty=1.00 sl=412.85 alert=retest2 |

### Cycle 32 — SELL (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 14:15:00 | 412.60 | 413.22 | 413.30 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 10:15:00 | 413.70 | 413.39 | 413.36 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 11:15:00 | 413.10 | 413.33 | 413.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 09:15:00 | 410.85 | 412.81 | 413.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 09:15:00 | 411.40 | 410.75 | 411.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 411.40 | 410.75 | 411.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 411.40 | 410.75 | 411.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:30:00 | 411.55 | 410.75 | 411.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 412.85 | 411.17 | 411.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:00:00 | 412.85 | 411.17 | 411.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 411.50 | 411.24 | 411.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 13:30:00 | 411.15 | 411.19 | 411.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 410.85 | 411.65 | 411.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 14:15:00 | 405.15 | 403.11 | 402.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-26 14:15:00 | 405.15 | 403.11 | 402.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 09:15:00 | 406.45 | 404.08 | 403.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 15:15:00 | 404.40 | 405.21 | 404.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 15:15:00 | 404.40 | 405.21 | 404.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 404.40 | 405.21 | 404.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:30:00 | 404.05 | 404.70 | 404.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 403.15 | 404.39 | 404.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:30:00 | 402.65 | 404.39 | 404.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 12:15:00 | 402.25 | 403.70 | 403.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 13:15:00 | 400.90 | 403.14 | 403.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 11:15:00 | 403.65 | 402.15 | 402.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 11:15:00 | 403.65 | 402.15 | 402.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 403.65 | 402.15 | 402.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 403.65 | 402.15 | 402.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 405.50 | 402.82 | 403.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:00:00 | 405.50 | 402.82 | 403.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 405.50 | 403.36 | 403.27 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 09:15:00 | 400.95 | 403.62 | 403.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 12:15:00 | 400.20 | 402.17 | 402.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 401.40 | 400.76 | 401.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 401.40 | 400.76 | 401.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 401.40 | 400.76 | 401.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 401.40 | 400.76 | 401.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 399.50 | 400.50 | 401.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 11:15:00 | 398.75 | 400.50 | 401.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 14:00:00 | 399.05 | 399.81 | 400.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 12:00:00 | 399.25 | 399.79 | 400.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 402.30 | 400.22 | 400.35 | SL hit (close>static) qty=1.00 sl=401.35 alert=retest2 |

### Cycle 39 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 402.25 | 400.63 | 400.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 11:15:00 | 403.20 | 401.14 | 400.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 400.25 | 401.50 | 401.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 400.25 | 401.50 | 401.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 400.25 | 401.50 | 401.14 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 399.10 | 400.76 | 400.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 397.90 | 399.53 | 400.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 399.60 | 398.15 | 398.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 399.60 | 398.15 | 398.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 399.60 | 398.15 | 398.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 399.15 | 398.15 | 398.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 399.75 | 398.47 | 399.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 399.75 | 398.47 | 399.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 399.70 | 398.78 | 399.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:30:00 | 400.00 | 398.78 | 399.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 399.35 | 398.90 | 399.11 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 400.30 | 399.34 | 399.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 401.20 | 399.71 | 399.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 13:15:00 | 415.00 | 415.60 | 412.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 14:00:00 | 415.00 | 415.60 | 412.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 414.50 | 415.47 | 413.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 12:15:00 | 416.40 | 415.50 | 413.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 13:30:00 | 416.30 | 415.77 | 414.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 15:00:00 | 416.85 | 415.99 | 414.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 10:15:00 | 417.00 | 419.59 | 419.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 417.00 | 419.59 | 419.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 11:15:00 | 413.95 | 418.46 | 419.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 09:15:00 | 406.35 | 405.77 | 408.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 09:45:00 | 406.40 | 405.77 | 408.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 407.15 | 406.10 | 407.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 407.15 | 406.10 | 407.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 405.20 | 405.92 | 406.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 404.50 | 406.44 | 406.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 10:00:00 | 403.95 | 405.94 | 406.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 14:15:00 | 408.65 | 406.34 | 406.38 | SL hit (close>static) qty=1.00 sl=407.20 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 15:15:00 | 408.25 | 406.72 | 406.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 409.40 | 407.26 | 406.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 10:15:00 | 407.10 | 407.23 | 406.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-17 10:45:00 | 407.30 | 407.23 | 406.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 407.45 | 407.27 | 406.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:30:00 | 407.20 | 407.27 | 406.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 406.70 | 407.16 | 406.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:45:00 | 406.70 | 407.16 | 406.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 407.50 | 407.22 | 406.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:30:00 | 407.15 | 407.22 | 406.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 407.00 | 407.18 | 406.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:45:00 | 406.60 | 407.18 | 406.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 407.50 | 407.24 | 406.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:15:00 | 405.95 | 407.24 | 406.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 405.10 | 406.82 | 406.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 403.85 | 405.59 | 406.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 405.75 | 404.49 | 405.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 405.75 | 404.49 | 405.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 405.75 | 404.49 | 405.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:45:00 | 405.30 | 404.49 | 405.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 406.50 | 404.89 | 405.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:45:00 | 406.95 | 404.89 | 405.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 405.85 | 405.34 | 405.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 13:45:00 | 406.45 | 405.34 | 405.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 14:15:00 | 405.90 | 405.45 | 405.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 10:15:00 | 406.55 | 405.68 | 405.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 10:15:00 | 406.50 | 407.08 | 406.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 10:15:00 | 406.50 | 407.08 | 406.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 406.50 | 407.08 | 406.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:00:00 | 406.50 | 407.08 | 406.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 405.30 | 406.73 | 406.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:45:00 | 404.50 | 406.73 | 406.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 404.75 | 406.33 | 406.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 12:45:00 | 405.10 | 406.33 | 406.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 13:15:00 | 405.00 | 406.07 | 406.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 14:15:00 | 403.05 | 405.46 | 405.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 403.15 | 402.86 | 404.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 402.60 | 402.68 | 403.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 402.60 | 402.68 | 403.36 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 12:15:00 | 404.00 | 403.49 | 403.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 14:15:00 | 404.30 | 403.60 | 403.48 | Break + close above crossover candle high |

### Cycle 48 — SELL (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 09:15:00 | 401.60 | 403.31 | 403.38 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 11:15:00 | 404.25 | 403.55 | 403.48 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 401.55 | 403.26 | 403.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 400.40 | 401.66 | 402.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 11:15:00 | 401.55 | 401.47 | 402.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 12:00:00 | 401.55 | 401.47 | 402.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 401.45 | 400.97 | 401.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:45:00 | 401.55 | 400.97 | 401.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 403.00 | 401.37 | 401.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 403.00 | 401.37 | 401.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 402.65 | 401.63 | 401.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:30:00 | 402.85 | 401.63 | 401.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 401.55 | 401.62 | 401.78 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 14:15:00 | 403.10 | 401.91 | 401.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 11:15:00 | 404.20 | 402.88 | 402.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 403.40 | 403.91 | 403.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 403.40 | 403.91 | 403.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 403.40 | 403.91 | 403.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:30:00 | 403.50 | 403.91 | 403.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 402.50 | 403.86 | 403.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 402.50 | 403.86 | 403.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 401.00 | 403.29 | 403.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:00:00 | 401.00 | 403.29 | 403.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 15:15:00 | 402.00 | 402.94 | 403.01 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 10:15:00 | 404.45 | 403.04 | 403.03 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 11:15:00 | 402.95 | 403.02 | 403.02 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 12:15:00 | 403.85 | 403.19 | 403.10 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 14:15:00 | 400.35 | 402.54 | 402.81 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 403.30 | 402.91 | 402.90 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 09:15:00 | 402.00 | 402.85 | 402.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 10:15:00 | 400.80 | 402.13 | 402.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 10:15:00 | 402.30 | 401.18 | 401.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 10:15:00 | 402.30 | 401.18 | 401.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 402.30 | 401.18 | 401.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 402.30 | 401.18 | 401.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 401.75 | 401.30 | 401.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 12:45:00 | 401.50 | 401.46 | 401.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 403.10 | 402.10 | 401.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 09:15:00 | 403.10 | 402.10 | 401.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 10:15:00 | 405.10 | 402.70 | 402.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 12:15:00 | 402.00 | 402.64 | 402.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 12:15:00 | 402.00 | 402.64 | 402.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 402.00 | 402.64 | 402.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 402.00 | 402.64 | 402.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 403.00 | 402.71 | 402.37 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 11:15:00 | 399.90 | 401.78 | 402.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 14:15:00 | 399.65 | 401.04 | 401.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 09:15:00 | 401.00 | 400.91 | 401.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 09:15:00 | 401.00 | 400.91 | 401.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 401.00 | 400.91 | 401.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:30:00 | 401.15 | 400.91 | 401.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 400.90 | 400.91 | 401.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:00:00 | 400.90 | 400.91 | 401.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 401.05 | 400.99 | 401.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:45:00 | 401.35 | 400.99 | 401.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 401.20 | 400.82 | 401.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 401.20 | 400.82 | 401.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 401.00 | 400.86 | 401.12 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 402.00 | 401.40 | 401.32 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 15:15:00 | 400.95 | 401.25 | 401.27 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 402.50 | 401.50 | 401.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 403.85 | 402.80 | 402.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 14:15:00 | 405.85 | 406.48 | 405.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 14:45:00 | 405.90 | 406.48 | 405.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 405.00 | 406.25 | 405.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:45:00 | 405.15 | 406.25 | 405.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 404.70 | 405.94 | 405.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 11:45:00 | 405.30 | 405.74 | 405.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 12:15:00 | 404.05 | 405.40 | 405.16 | SL hit (close<static) qty=1.00 sl=404.35 alert=retest2 |

### Cycle 64 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 404.35 | 404.94 | 404.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 402.55 | 404.46 | 404.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 402.40 | 401.77 | 402.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 402.40 | 401.77 | 402.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 402.40 | 401.77 | 402.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 402.40 | 401.77 | 402.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 402.80 | 401.97 | 402.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:30:00 | 402.65 | 401.97 | 402.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 403.15 | 402.21 | 402.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:00:00 | 403.15 | 402.21 | 402.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 403.20 | 402.41 | 402.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:30:00 | 403.00 | 402.41 | 402.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 379.35 | 398.08 | 400.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 10:45:00 | 376.35 | 392.60 | 397.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 09:15:00 | 357.53 | 369.01 | 382.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-06 13:15:00 | 338.72 | 346.26 | 353.77 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 65 — BUY (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 14:15:00 | 322.10 | 321.17 | 321.07 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 315.20 | 320.12 | 320.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 311.20 | 317.21 | 319.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 312.10 | 310.80 | 314.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:00:00 | 312.10 | 310.80 | 314.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 313.90 | 311.42 | 314.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:30:00 | 314.25 | 311.42 | 314.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 314.90 | 312.12 | 314.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 314.90 | 312.12 | 314.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 314.75 | 312.64 | 314.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 313.55 | 312.64 | 314.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 313.20 | 312.76 | 314.50 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 316.50 | 315.28 | 315.24 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 10:15:00 | 313.10 | 314.96 | 315.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 312.15 | 313.65 | 314.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 10:15:00 | 320.00 | 312.66 | 312.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 10:15:00 | 320.00 | 312.66 | 312.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 320.00 | 312.66 | 312.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:00:00 | 320.00 | 312.66 | 312.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 11:15:00 | 325.15 | 315.16 | 314.09 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 11:15:00 | 318.00 | 320.18 | 320.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 12:15:00 | 317.15 | 319.58 | 320.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 12:15:00 | 318.95 | 318.45 | 319.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 12:15:00 | 318.95 | 318.45 | 319.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 318.95 | 318.45 | 319.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:45:00 | 319.15 | 318.45 | 319.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 319.60 | 318.68 | 319.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 13:30:00 | 319.80 | 318.68 | 319.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 317.30 | 318.40 | 318.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 316.65 | 318.20 | 318.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 12:00:00 | 316.75 | 316.04 | 316.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 326.05 | 318.84 | 317.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 326.05 | 318.84 | 317.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 330.00 | 324.69 | 321.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 328.30 | 329.70 | 326.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 12:00:00 | 328.30 | 329.70 | 326.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 328.00 | 329.25 | 327.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:00:00 | 328.00 | 329.25 | 327.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 325.80 | 328.56 | 327.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 325.80 | 328.56 | 327.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 325.80 | 328.01 | 326.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 326.55 | 328.01 | 326.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 10:15:00 | 326.40 | 326.74 | 326.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 10:15:00 | 326.40 | 326.74 | 326.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 11:15:00 | 324.65 | 326.32 | 326.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 13:15:00 | 324.40 | 324.33 | 325.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-24 14:00:00 | 324.40 | 324.33 | 325.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 322.95 | 323.92 | 324.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 10:45:00 | 321.70 | 323.44 | 324.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 305.61 | 309.20 | 310.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 305.95 | 305.84 | 307.78 | SL hit (close>ema200) qty=0.50 sl=305.84 alert=retest2 |

### Cycle 73 — BUY (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 11:15:00 | 308.90 | 308.44 | 308.38 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 307.90 | 308.33 | 308.34 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 13:15:00 | 309.25 | 308.52 | 308.42 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 305.65 | 308.05 | 308.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 14:15:00 | 304.50 | 306.58 | 307.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 09:15:00 | 304.70 | 303.29 | 304.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 304.70 | 303.29 | 304.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 304.70 | 303.29 | 304.74 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2026-03-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 15:15:00 | 307.45 | 305.27 | 305.11 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 09:15:00 | 304.50 | 305.36 | 305.37 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 306.40 | 305.57 | 305.47 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 13:15:00 | 305.30 | 305.41 | 305.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-18 14:15:00 | 304.10 | 305.15 | 305.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 304.65 | 301.34 | 302.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 304.65 | 301.34 | 302.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 304.65 | 301.34 | 302.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 304.65 | 301.34 | 302.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 307.00 | 302.47 | 303.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 307.00 | 302.47 | 303.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 11:15:00 | 307.35 | 303.45 | 303.42 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 14:15:00 | 300.00 | 303.53 | 303.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 15:15:00 | 299.80 | 302.78 | 303.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 13:15:00 | 292.30 | 291.95 | 295.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:45:00 | 292.05 | 291.95 | 295.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 296.10 | 292.73 | 294.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 296.10 | 292.73 | 294.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 298.25 | 293.84 | 294.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 11:00:00 | 298.25 | 293.84 | 294.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2026-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 15:15:00 | 295.75 | 295.51 | 295.49 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 293.00 | 295.00 | 295.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 292.55 | 294.40 | 294.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 292.70 | 291.02 | 292.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 292.70 | 291.02 | 292.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 292.70 | 291.02 | 292.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 292.75 | 291.02 | 292.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 292.45 | 291.31 | 292.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 289.55 | 292.07 | 292.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 13:45:00 | 291.25 | 291.26 | 291.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 293.45 | 291.70 | 291.97 | SL hit (close>static) qty=1.00 sl=293.00 alert=retest2 |

### Cycle 85 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 293.45 | 292.17 | 292.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 295.15 | 293.02 | 292.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 299.15 | 303.08 | 302.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 299.15 | 303.08 | 302.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 299.15 | 303.08 | 302.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:45:00 | 299.55 | 303.08 | 302.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2026-04-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 12:15:00 | 299.25 | 301.45 | 301.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 14:15:00 | 298.65 | 300.50 | 301.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 302.65 | 300.67 | 301.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 302.65 | 300.67 | 301.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 302.65 | 300.67 | 301.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:00:00 | 302.65 | 300.67 | 301.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 301.90 | 300.91 | 301.22 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2026-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 12:15:00 | 303.50 | 301.77 | 301.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 308.00 | 304.14 | 303.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 13:15:00 | 305.75 | 306.28 | 305.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 14:00:00 | 305.75 | 306.28 | 305.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 305.20 | 306.06 | 305.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 15:00:00 | 305.20 | 306.06 | 305.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 304.90 | 305.83 | 305.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 305.65 | 305.83 | 305.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:00:00 | 305.40 | 305.74 | 305.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 12:00:00 | 305.55 | 305.63 | 305.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 13:45:00 | 305.45 | 305.91 | 305.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 306.05 | 306.81 | 306.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:30:00 | 306.50 | 306.81 | 306.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 306.95 | 306.84 | 306.25 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 304.65 | 305.77 | 305.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 304.65 | 305.77 | 305.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 303.70 | 305.02 | 305.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 305.00 | 303.36 | 304.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 305.00 | 303.36 | 304.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 305.00 | 303.36 | 304.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 305.05 | 303.36 | 304.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 304.10 | 303.51 | 304.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:30:00 | 303.55 | 303.48 | 304.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 12:00:00 | 303.35 | 303.48 | 304.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 11:15:00 | 304.40 | 304.32 | 304.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 11:15:00 | 304.40 | 304.32 | 304.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 12:15:00 | 305.30 | 304.51 | 304.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 15:15:00 | 304.10 | 304.46 | 304.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 15:15:00 | 304.10 | 304.46 | 304.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 304.10 | 304.46 | 304.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 311.50 | 304.46 | 304.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 10:15:00 | 310.25 | 311.85 | 312.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 310.25 | 311.85 | 312.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-07 13:15:00 | 308.90 | 309.94 | 310.55 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-19 15:15:00 | 436.10 | 2025-05-21 14:15:00 | 433.20 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-05-21 09:30:00 | 435.95 | 2025-05-21 14:15:00 | 433.20 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-05-21 11:45:00 | 435.85 | 2025-05-21 14:15:00 | 433.20 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest1 | 2025-06-02 09:15:00 | 415.30 | 2025-06-02 12:15:00 | 420.20 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-06-03 09:45:00 | 418.30 | 2025-06-05 11:15:00 | 420.50 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-06-03 11:00:00 | 417.75 | 2025-06-05 11:15:00 | 420.50 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-06-03 11:30:00 | 418.05 | 2025-06-05 11:15:00 | 420.50 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-06-03 12:45:00 | 416.50 | 2025-06-05 11:15:00 | 420.50 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-06-17 12:30:00 | 416.10 | 2025-06-20 11:15:00 | 418.90 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-06-18 11:00:00 | 416.25 | 2025-06-20 11:15:00 | 418.90 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-06-18 12:15:00 | 415.80 | 2025-06-20 11:15:00 | 418.90 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-06-19 14:30:00 | 415.85 | 2025-06-20 11:15:00 | 418.90 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-06-25 12:30:00 | 416.45 | 2025-06-25 13:15:00 | 417.25 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-07-04 13:15:00 | 412.35 | 2025-07-07 09:15:00 | 415.35 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-07-10 12:45:00 | 417.65 | 2025-07-11 10:15:00 | 416.00 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-07-11 09:15:00 | 417.55 | 2025-07-11 10:15:00 | 416.00 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-08-07 11:45:00 | 412.35 | 2025-08-08 10:15:00 | 414.20 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-08-20 12:15:00 | 408.40 | 2025-08-28 11:15:00 | 402.00 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2025-09-02 09:30:00 | 406.45 | 2025-09-05 11:15:00 | 406.70 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-09-02 10:00:00 | 406.80 | 2025-09-05 11:15:00 | 406.70 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2025-09-02 10:45:00 | 406.35 | 2025-09-05 11:15:00 | 406.70 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-09-02 12:30:00 | 406.30 | 2025-09-05 11:15:00 | 406.70 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2025-09-03 09:15:00 | 409.10 | 2025-09-05 11:15:00 | 406.70 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-09-05 09:45:00 | 407.90 | 2025-09-05 11:15:00 | 406.70 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-09-12 11:15:00 | 416.65 | 2025-09-15 11:15:00 | 412.50 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-18 13:30:00 | 411.15 | 2025-09-26 14:15:00 | 405.15 | STOP_HIT | 1.00 | 1.46% |
| SELL | retest2 | 2025-09-19 09:15:00 | 410.85 | 2025-09-26 14:15:00 | 405.15 | STOP_HIT | 1.00 | 1.39% |
| SELL | retest2 | 2025-10-08 11:15:00 | 398.75 | 2025-10-10 09:15:00 | 402.30 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-10-08 14:00:00 | 399.05 | 2025-10-10 09:15:00 | 402.30 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-10-09 12:00:00 | 399.25 | 2025-10-10 09:15:00 | 402.30 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-10-24 12:15:00 | 416.40 | 2025-11-03 10:15:00 | 417.00 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2025-10-24 13:30:00 | 416.30 | 2025-11-03 10:15:00 | 417.00 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2025-10-24 15:00:00 | 416.85 | 2025-11-03 10:15:00 | 417.00 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-11-14 09:15:00 | 404.50 | 2025-11-14 14:15:00 | 408.65 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-11-14 10:00:00 | 403.95 | 2025-11-14 14:15:00 | 408.65 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-12-15 12:45:00 | 401.50 | 2025-12-16 09:15:00 | 403.10 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-12-26 11:45:00 | 405.30 | 2025-12-26 12:15:00 | 404.05 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2026-01-01 10:45:00 | 376.35 | 2026-01-02 09:15:00 | 357.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 10:45:00 | 376.35 | 2026-01-06 13:15:00 | 338.72 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-13 09:15:00 | 316.65 | 2026-02-17 09:15:00 | 326.05 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2026-02-16 12:00:00 | 316.75 | 2026-02-17 09:15:00 | 326.05 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2026-02-20 09:15:00 | 326.55 | 2026-02-23 10:15:00 | 326.40 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2026-02-25 10:45:00 | 321.70 | 2026-03-09 09:15:00 | 305.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 10:45:00 | 321.70 | 2026-03-10 10:15:00 | 305.95 | STOP_HIT | 0.50 | 4.90% |
| SELL | retest2 | 2026-04-02 09:15:00 | 289.55 | 2026-04-02 14:15:00 | 293.45 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2026-04-02 13:45:00 | 291.25 | 2026-04-02 14:15:00 | 293.45 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2026-04-21 09:15:00 | 305.65 | 2026-04-23 10:15:00 | 304.65 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2026-04-21 10:00:00 | 305.40 | 2026-04-23 10:15:00 | 304.65 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2026-04-21 12:00:00 | 305.55 | 2026-04-23 10:15:00 | 304.65 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2026-04-21 13:45:00 | 305.45 | 2026-04-23 10:15:00 | 304.65 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2026-04-27 11:30:00 | 303.55 | 2026-04-28 11:15:00 | 304.40 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2026-04-27 12:00:00 | 303.35 | 2026-04-28 11:15:00 | 304.40 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2026-04-29 09:15:00 | 311.50 | 2026-05-05 10:15:00 | 310.25 | STOP_HIT | 1.00 | -0.40% |
