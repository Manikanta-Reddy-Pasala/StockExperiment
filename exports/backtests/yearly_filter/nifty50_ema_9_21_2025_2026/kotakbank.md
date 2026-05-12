# KOTAKBANK (KOTAKBANK)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 381.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 70 |
| ALERT1 | 49 |
| ALERT2 | 48 |
| ALERT2_SKIP | 16 |
| ALERT3 | 145 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 66 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 63 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 5 / 58
- **Target hits / Stop hits / Partials:** 0 / 63 / 0
- **Avg / median % per leg:** -0.88% / -0.95%
- **Sum % (uncompounded):** -55.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 4 | 19.0% | 0 | 21 | 0 | -0.34% | -7.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 4 | 19.0% | 0 | 21 | 0 | -0.34% | -7.1% |
| SELL (all) | 42 | 1 | 2.4% | 0 | 42 | 0 | -1.15% | -48.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 42 | 1 | 2.4% | 0 | 42 | 0 | -1.15% | -48.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 63 | 5 | 7.9% | 0 | 63 | 0 | -0.88% | -55.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 13:15:00 | 421.26 | 423.78 | 423.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-14 11:15:00 | 419.90 | 422.57 | 423.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 12:15:00 | 418.14 | 417.59 | 419.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-15 13:00:00 | 418.14 | 417.59 | 419.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 421.26 | 418.33 | 419.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:45:00 | 420.60 | 418.33 | 419.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 421.50 | 418.96 | 420.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 419.30 | 419.23 | 420.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 15:15:00 | 421.96 | 420.39 | 420.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 15:15:00 | 421.96 | 420.39 | 420.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 12:15:00 | 423.88 | 421.69 | 420.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 419.60 | 421.48 | 421.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 419.60 | 421.48 | 421.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 419.60 | 421.48 | 421.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:00:00 | 419.60 | 421.48 | 421.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 420.22 | 421.22 | 421.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:30:00 | 419.60 | 421.22 | 421.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 418.94 | 420.68 | 420.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 418.00 | 420.15 | 420.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 14:15:00 | 413.00 | 412.87 | 415.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 14:45:00 | 413.38 | 412.87 | 415.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 414.60 | 413.22 | 415.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:15:00 | 415.40 | 413.22 | 415.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 417.10 | 413.99 | 415.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 417.10 | 413.99 | 415.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 417.08 | 414.61 | 415.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:45:00 | 417.46 | 414.61 | 415.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 419.84 | 416.15 | 415.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 13:15:00 | 421.08 | 417.14 | 416.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 13:15:00 | 417.94 | 418.71 | 417.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 13:15:00 | 417.94 | 418.71 | 417.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 417.94 | 418.71 | 417.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:00:00 | 417.94 | 418.71 | 417.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 417.72 | 418.52 | 417.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:30:00 | 418.08 | 418.52 | 417.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 418.34 | 418.48 | 417.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 413.68 | 418.48 | 417.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 413.44 | 417.47 | 417.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:30:00 | 414.60 | 417.47 | 417.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 10:15:00 | 415.40 | 417.06 | 417.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 11:15:00 | 413.00 | 414.65 | 415.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 13:15:00 | 414.66 | 414.62 | 415.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 14:15:00 | 414.72 | 414.64 | 415.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 414.72 | 414.64 | 415.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:30:00 | 415.04 | 414.64 | 415.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 414.18 | 414.63 | 415.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 10:30:00 | 413.00 | 414.31 | 415.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 14:00:00 | 413.66 | 413.63 | 414.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 15:15:00 | 419.10 | 415.33 | 415.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 15:15:00 | 419.10 | 415.33 | 415.21 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 11:15:00 | 413.44 | 414.99 | 415.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 412.56 | 414.37 | 414.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 410.12 | 409.52 | 411.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 10:00:00 | 410.12 | 409.52 | 411.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 410.50 | 409.72 | 411.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:30:00 | 411.42 | 409.72 | 411.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 411.10 | 410.05 | 410.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:00:00 | 411.10 | 410.05 | 410.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 408.98 | 409.84 | 410.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 408.80 | 409.87 | 410.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 11:00:00 | 408.60 | 409.64 | 410.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 12:30:00 | 408.70 | 409.47 | 410.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 14:30:00 | 408.72 | 409.15 | 409.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 408.42 | 408.78 | 409.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:00:00 | 408.42 | 408.78 | 409.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-06 10:15:00 | 415.80 | 410.18 | 410.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 415.80 | 410.18 | 410.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 426.52 | 415.64 | 413.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 15:15:00 | 429.40 | 429.75 | 425.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:15:00 | 427.52 | 429.75 | 425.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 427.34 | 428.90 | 427.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:00:00 | 427.34 | 428.90 | 427.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 427.06 | 428.53 | 427.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:45:00 | 427.10 | 428.53 | 427.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 426.78 | 428.18 | 427.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:30:00 | 426.88 | 428.18 | 427.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 426.04 | 427.75 | 427.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:00:00 | 426.04 | 427.75 | 427.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 425.18 | 427.24 | 426.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 425.18 | 427.24 | 426.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 424.44 | 426.33 | 426.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 417.66 | 424.60 | 425.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 424.02 | 422.45 | 423.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 424.02 | 422.45 | 423.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 424.02 | 422.45 | 423.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:00:00 | 424.02 | 422.45 | 423.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 426.24 | 423.21 | 423.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 426.24 | 423.21 | 423.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 427.40 | 424.05 | 424.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 427.40 | 424.05 | 424.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2025-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 12:15:00 | 427.70 | 424.78 | 424.59 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 424.34 | 425.71 | 425.84 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 15:15:00 | 426.98 | 425.91 | 425.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-19 09:15:00 | 429.60 | 426.65 | 426.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 15:15:00 | 428.62 | 428.67 | 427.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-20 09:15:00 | 427.52 | 428.67 | 427.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 429.14 | 428.77 | 427.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:30:00 | 427.58 | 428.77 | 427.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 439.78 | 441.09 | 439.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:30:00 | 440.06 | 441.09 | 439.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 439.00 | 440.67 | 439.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:15:00 | 438.84 | 440.67 | 439.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 438.22 | 440.18 | 439.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 15:00:00 | 441.02 | 440.05 | 439.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 14:15:00 | 441.40 | 439.66 | 439.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 09:15:00 | 435.10 | 439.07 | 439.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 09:15:00 | 435.10 | 439.07 | 439.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 12:15:00 | 433.88 | 437.25 | 438.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 09:15:00 | 436.22 | 435.08 | 436.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-01 10:00:00 | 436.22 | 435.08 | 436.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 435.60 | 435.18 | 436.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:30:00 | 435.80 | 435.18 | 436.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 436.38 | 435.39 | 436.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 14:00:00 | 436.38 | 435.39 | 436.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 436.60 | 435.63 | 436.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 15:00:00 | 436.60 | 435.63 | 436.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 436.40 | 435.79 | 436.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 438.52 | 435.79 | 436.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 439.18 | 436.47 | 436.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:45:00 | 438.64 | 436.47 | 436.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 437.16 | 436.60 | 436.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 11:15:00 | 436.04 | 436.60 | 436.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 446.70 | 431.79 | 429.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 09:15:00 | 446.70 | 431.79 | 429.87 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 13:15:00 | 440.00 | 442.21 | 442.48 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 445.02 | 442.91 | 442.69 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 12:15:00 | 439.14 | 442.23 | 442.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-15 13:15:00 | 437.80 | 441.34 | 442.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 430.46 | 429.60 | 432.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 430.46 | 429.60 | 432.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 430.46 | 429.60 | 432.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:45:00 | 431.56 | 429.60 | 432.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 431.70 | 430.02 | 432.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:00:00 | 431.70 | 430.02 | 432.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 432.18 | 430.45 | 432.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:30:00 | 432.00 | 430.45 | 432.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 431.60 | 430.68 | 432.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:45:00 | 431.36 | 430.68 | 432.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 433.34 | 431.36 | 432.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 433.34 | 431.36 | 432.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 432.90 | 431.67 | 432.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 432.12 | 431.67 | 432.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 431.66 | 431.67 | 432.21 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 10:15:00 | 433.58 | 432.09 | 432.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 11:15:00 | 434.58 | 432.58 | 432.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 15:15:00 | 433.08 | 433.65 | 432.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 09:15:00 | 430.18 | 433.65 | 432.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 429.80 | 432.88 | 432.69 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 429.72 | 432.25 | 432.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 428.40 | 430.62 | 431.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 11:15:00 | 392.86 | 391.88 | 396.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-31 12:00:00 | 392.86 | 391.88 | 396.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 398.62 | 393.23 | 396.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:00:00 | 398.62 | 393.23 | 396.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 397.72 | 394.13 | 396.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 14:45:00 | 396.82 | 394.47 | 396.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 10:15:00 | 400.36 | 396.58 | 396.89 | SL hit (close>static) qty=1.00 sl=399.72 alert=retest2 |

### Cycle 20 — BUY (started 2025-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 12:15:00 | 399.60 | 397.52 | 397.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 11:15:00 | 400.14 | 398.56 | 397.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 13:15:00 | 399.20 | 399.85 | 399.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 13:15:00 | 399.20 | 399.85 | 399.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 399.20 | 399.85 | 399.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:30:00 | 399.24 | 399.85 | 399.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 400.86 | 400.05 | 399.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 403.34 | 400.20 | 399.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 09:15:00 | 396.34 | 399.83 | 399.83 | SL hit (close<static) qty=1.00 sl=398.96 alert=retest2 |

### Cycle 21 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 396.62 | 399.19 | 399.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 09:15:00 | 395.56 | 397.92 | 398.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 393.60 | 393.47 | 395.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 09:30:00 | 392.40 | 393.47 | 395.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 392.94 | 393.46 | 395.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:30:00 | 394.28 | 393.46 | 395.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 392.62 | 393.29 | 394.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 14:30:00 | 391.92 | 393.26 | 394.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 11:15:00 | 396.54 | 394.79 | 394.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 11:15:00 | 396.54 | 394.79 | 394.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 401.88 | 397.34 | 396.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 14:15:00 | 399.94 | 400.05 | 398.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 15:00:00 | 399.94 | 400.05 | 398.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 401.84 | 400.40 | 398.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 10:30:00 | 403.94 | 401.11 | 399.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 13:30:00 | 403.04 | 403.66 | 402.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 14:15:00 | 403.38 | 403.66 | 402.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:30:00 | 403.32 | 403.42 | 403.12 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 398.86 | 402.51 | 402.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 398.86 | 402.51 | 402.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 11:15:00 | 398.44 | 401.69 | 402.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 09:15:00 | 394.12 | 391.49 | 393.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 09:15:00 | 394.12 | 391.49 | 393.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 394.12 | 391.49 | 393.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:45:00 | 389.74 | 392.15 | 392.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 10:30:00 | 389.98 | 390.54 | 391.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 12:00:00 | 389.58 | 390.35 | 391.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 12:15:00 | 389.50 | 391.20 | 391.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 389.98 | 390.96 | 391.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:00:00 | 389.98 | 390.96 | 391.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 389.50 | 390.16 | 390.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 10:45:00 | 389.00 | 389.96 | 390.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 15:00:00 | 388.96 | 389.48 | 390.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:30:00 | 388.70 | 388.87 | 389.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:00:00 | 388.44 | 389.20 | 389.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 389.40 | 389.24 | 389.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:30:00 | 389.80 | 389.24 | 389.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 389.38 | 389.27 | 389.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-09 13:15:00 | 391.10 | 389.64 | 389.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 13:15:00 | 391.10 | 389.64 | 389.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 14:15:00 | 391.94 | 390.10 | 389.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 393.02 | 393.68 | 392.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 09:30:00 | 392.94 | 393.68 | 392.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 393.76 | 394.96 | 394.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:00:00 | 393.76 | 394.96 | 394.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 394.50 | 394.87 | 394.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 395.48 | 394.88 | 394.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 12:30:00 | 395.22 | 394.66 | 394.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 397.98 | 394.45 | 394.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 12:15:00 | 406.56 | 407.01 | 407.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2025-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 12:15:00 | 406.56 | 407.01 | 407.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 405.64 | 406.73 | 406.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 407.12 | 405.20 | 405.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 12:15:00 | 407.12 | 405.20 | 405.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 407.12 | 405.20 | 405.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:00:00 | 407.12 | 405.20 | 405.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 408.54 | 405.86 | 406.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:30:00 | 409.20 | 405.86 | 406.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 14:15:00 | 410.04 | 406.70 | 406.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 15:15:00 | 410.40 | 407.44 | 406.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 12:15:00 | 407.48 | 407.69 | 407.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 12:15:00 | 407.48 | 407.69 | 407.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 407.48 | 407.69 | 407.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:30:00 | 406.94 | 407.69 | 407.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 406.00 | 407.35 | 407.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 14:00:00 | 406.00 | 407.35 | 407.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 406.18 | 407.12 | 406.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 15:15:00 | 406.40 | 407.12 | 406.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 406.40 | 406.86 | 406.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 10:15:00 | 404.00 | 406.29 | 406.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 14:15:00 | 399.24 | 398.15 | 400.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 15:00:00 | 399.24 | 398.15 | 400.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 399.50 | 398.13 | 399.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:30:00 | 398.68 | 398.42 | 399.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:15:00 | 398.54 | 398.42 | 399.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 14:45:00 | 398.22 | 398.83 | 399.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 406.00 | 400.32 | 400.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 406.00 | 400.32 | 400.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 11:15:00 | 410.00 | 403.44 | 401.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 13:15:00 | 425.42 | 426.98 | 422.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 14:00:00 | 425.42 | 426.98 | 422.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 423.08 | 425.41 | 423.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 423.08 | 425.41 | 423.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 423.62 | 425.05 | 423.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:15:00 | 423.16 | 425.05 | 423.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 424.76 | 425.00 | 423.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:30:00 | 423.78 | 425.00 | 423.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 422.62 | 424.34 | 423.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 422.62 | 424.34 | 423.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 422.56 | 423.99 | 423.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 421.72 | 423.99 | 423.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 422.04 | 423.60 | 423.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:45:00 | 420.72 | 423.60 | 423.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 426.64 | 424.21 | 423.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 11:30:00 | 427.10 | 425.04 | 424.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 10:15:00 | 435.26 | 441.45 | 441.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 435.26 | 441.45 | 441.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 09:15:00 | 431.22 | 436.99 | 438.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 14:15:00 | 431.86 | 431.33 | 433.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 15:00:00 | 431.86 | 431.33 | 433.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 432.76 | 431.62 | 433.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 433.96 | 431.62 | 433.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 432.18 | 431.73 | 433.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 10:45:00 | 431.28 | 431.54 | 433.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 421.68 | 416.67 | 416.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 421.68 | 416.67 | 416.36 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 15:15:00 | 417.30 | 419.35 | 419.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 09:15:00 | 415.40 | 418.06 | 418.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 416.66 | 415.74 | 416.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 10:00:00 | 416.66 | 415.74 | 416.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 416.46 | 415.85 | 416.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:45:00 | 416.64 | 415.85 | 416.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 418.20 | 416.32 | 416.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:00:00 | 418.20 | 416.32 | 416.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 418.98 | 416.85 | 417.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 14:15:00 | 419.22 | 416.85 | 417.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 420.26 | 417.53 | 417.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 10:15:00 | 423.68 | 419.30 | 418.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 13:15:00 | 428.86 | 429.74 | 427.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 14:00:00 | 428.86 | 429.74 | 427.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 429.00 | 429.59 | 428.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:45:00 | 428.10 | 429.59 | 428.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 428.72 | 429.42 | 428.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 425.98 | 429.42 | 428.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 424.50 | 428.43 | 427.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 424.50 | 428.43 | 427.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 425.58 | 427.86 | 427.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:30:00 | 425.18 | 427.86 | 427.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 12:15:00 | 425.80 | 427.16 | 427.26 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 14:15:00 | 429.52 | 427.42 | 427.35 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 11:15:00 | 425.86 | 427.38 | 427.42 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 09:15:00 | 429.86 | 427.40 | 427.31 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 15:15:00 | 425.96 | 428.17 | 428.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 422.74 | 427.09 | 427.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 15:15:00 | 426.40 | 425.99 | 426.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-10 09:15:00 | 428.34 | 425.99 | 426.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 429.48 | 426.68 | 427.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 430.28 | 426.68 | 427.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 428.98 | 427.14 | 427.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:30:00 | 430.12 | 427.14 | 427.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 427.18 | 427.14 | 427.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 13:15:00 | 427.90 | 427.14 | 427.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 425.42 | 426.80 | 427.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 13:30:00 | 428.00 | 426.80 | 427.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 430.84 | 427.30 | 427.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 10:15:00 | 436.86 | 429.21 | 428.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 14:15:00 | 435.66 | 435.93 | 433.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-12 15:00:00 | 435.66 | 435.93 | 433.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 432.10 | 435.09 | 433.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:30:00 | 432.36 | 435.09 | 433.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 432.68 | 434.61 | 433.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 12:00:00 | 434.32 | 434.55 | 433.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 12:15:00 | 433.10 | 435.09 | 435.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 433.10 | 435.09 | 435.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 12:15:00 | 432.36 | 433.52 | 434.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 433.22 | 432.89 | 433.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 433.22 | 432.89 | 433.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 433.22 | 432.89 | 433.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 12:15:00 | 431.94 | 432.80 | 433.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 10:15:00 | 434.40 | 432.96 | 432.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 10:15:00 | 434.40 | 432.96 | 432.79 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 431.76 | 432.66 | 432.75 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 13:15:00 | 433.84 | 432.70 | 432.70 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 432.34 | 432.65 | 432.68 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 09:15:00 | 433.18 | 432.76 | 432.72 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 431.78 | 432.56 | 432.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 14:15:00 | 431.52 | 432.12 | 432.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 431.22 | 430.74 | 431.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 13:15:00 | 431.22 | 430.74 | 431.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 431.22 | 430.74 | 431.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:45:00 | 431.40 | 430.74 | 431.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 430.50 | 430.69 | 431.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:15:00 | 431.04 | 430.69 | 431.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 431.04 | 430.76 | 431.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 433.16 | 430.76 | 431.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 432.46 | 431.10 | 431.43 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 434.60 | 431.94 | 431.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 12:15:00 | 437.72 | 433.10 | 432.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 13:15:00 | 441.70 | 442.71 | 440.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 14:00:00 | 441.70 | 442.71 | 440.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 439.96 | 442.16 | 440.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 15:00:00 | 439.96 | 442.16 | 440.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 437.98 | 441.32 | 439.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 09:45:00 | 440.96 | 441.31 | 440.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 14:15:00 | 438.00 | 439.48 | 439.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2026-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 14:15:00 | 438.00 | 439.48 | 439.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 435.18 | 438.25 | 438.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 428.70 | 427.70 | 431.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 15:00:00 | 428.70 | 427.70 | 431.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 426.18 | 424.90 | 426.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:30:00 | 426.48 | 424.90 | 426.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 425.96 | 425.11 | 426.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:30:00 | 426.14 | 425.11 | 426.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 426.98 | 425.49 | 426.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:00:00 | 426.98 | 425.49 | 426.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 426.80 | 425.75 | 426.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 426.40 | 425.75 | 426.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 426.60 | 426.10 | 426.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:15:00 | 427.32 | 426.10 | 426.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 425.80 | 426.04 | 426.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:30:00 | 428.02 | 426.04 | 426.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 424.50 | 425.73 | 426.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 423.50 | 425.86 | 426.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:15:00 | 423.80 | 425.08 | 425.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:00:00 | 423.90 | 423.71 | 424.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 10:30:00 | 423.20 | 421.45 | 422.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 427.20 | 422.60 | 423.09 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-19 11:15:00 | 427.20 | 422.60 | 423.09 | SL hit (close>static) qty=1.00 sl=426.50 alert=retest2 |

### Cycle 48 — BUY (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 12:15:00 | 428.20 | 423.72 | 423.56 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 418.80 | 424.04 | 424.70 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 427.70 | 424.41 | 424.02 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 422.50 | 423.88 | 424.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 09:15:00 | 405.90 | 420.38 | 422.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 11:15:00 | 412.30 | 411.70 | 415.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 11:30:00 | 412.90 | 411.70 | 415.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 413.70 | 411.47 | 413.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 13:00:00 | 413.70 | 411.47 | 413.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 413.20 | 411.82 | 413.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 408.10 | 412.00 | 412.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 10:30:00 | 410.40 | 409.73 | 410.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 409.90 | 410.52 | 410.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 418.50 | 410.11 | 409.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 418.50 | 410.11 | 409.69 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 410.40 | 412.61 | 412.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 10:15:00 | 406.80 | 411.45 | 412.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 14:15:00 | 409.15 | 408.81 | 410.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-05 15:00:00 | 409.15 | 408.81 | 410.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 411.65 | 409.38 | 410.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 412.35 | 409.38 | 410.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 411.70 | 409.84 | 410.68 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 10:15:00 | 417.05 | 411.29 | 411.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 11:15:00 | 422.70 | 413.57 | 412.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 14:15:00 | 429.35 | 429.88 | 425.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 15:00:00 | 429.35 | 429.88 | 425.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 428.65 | 429.24 | 427.70 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 423.90 | 426.86 | 427.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 12:15:00 | 422.65 | 425.15 | 426.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 13:15:00 | 422.20 | 421.87 | 423.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 13:30:00 | 422.00 | 421.87 | 423.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 425.80 | 422.66 | 423.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 425.80 | 422.66 | 423.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 426.15 | 423.36 | 423.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 422.40 | 423.36 | 423.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 15:15:00 | 424.30 | 423.45 | 423.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:45:00 | 422.90 | 423.41 | 423.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 14:15:00 | 426.30 | 423.84 | 423.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 14:15:00 | 426.30 | 423.84 | 423.69 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 419.70 | 423.14 | 423.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 416.15 | 421.18 | 422.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 420.70 | 420.24 | 421.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 10:30:00 | 421.35 | 420.24 | 421.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 421.40 | 420.47 | 421.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:45:00 | 421.90 | 420.47 | 421.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 420.95 | 420.57 | 421.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 422.20 | 420.57 | 421.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 422.15 | 420.88 | 421.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:00:00 | 422.15 | 420.88 | 421.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 421.55 | 421.02 | 421.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:45:00 | 422.95 | 421.02 | 421.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 420.15 | 420.84 | 421.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 426.20 | 420.84 | 421.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 427.30 | 422.13 | 422.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 10:15:00 | 428.60 | 423.43 | 422.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 12:15:00 | 428.95 | 429.18 | 426.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-24 13:00:00 | 428.95 | 429.18 | 426.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 428.00 | 428.75 | 427.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:15:00 | 428.00 | 428.75 | 427.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 428.00 | 428.60 | 427.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 429.80 | 428.60 | 427.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 427.50 | 428.38 | 427.24 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 12:15:00 | 422.80 | 426.09 | 426.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 419.90 | 423.41 | 424.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 413.70 | 413.38 | 417.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 413.70 | 413.38 | 417.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 406.65 | 407.03 | 409.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 409.70 | 407.03 | 409.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 370.25 | 367.91 | 371.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 370.50 | 367.91 | 371.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 372.30 | 369.02 | 371.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 372.30 | 369.02 | 371.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 372.10 | 369.64 | 371.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 370.80 | 369.64 | 371.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 14:15:00 | 374.30 | 371.71 | 372.19 | SL hit (close>static) qty=1.00 sl=373.95 alert=retest2 |

### Cycle 60 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 373.15 | 372.55 | 372.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 375.45 | 373.13 | 372.75 | Break + close above crossover candle high |

### Cycle 61 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 364.40 | 372.00 | 372.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 359.05 | 366.61 | 368.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 364.60 | 361.33 | 364.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 364.60 | 361.33 | 364.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 364.60 | 361.33 | 364.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:00:00 | 364.60 | 361.33 | 364.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 364.40 | 361.94 | 364.31 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 368.60 | 365.77 | 365.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 376.30 | 367.88 | 366.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 12:15:00 | 364.35 | 368.75 | 367.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 12:15:00 | 364.35 | 368.75 | 367.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 12:15:00 | 364.35 | 368.75 | 367.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 13:00:00 | 364.35 | 368.75 | 367.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 13:15:00 | 370.25 | 369.05 | 367.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 13:30:00 | 368.95 | 369.05 | 367.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 363.25 | 368.37 | 367.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 363.25 | 368.37 | 367.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 364.00 | 367.49 | 367.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 364.00 | 367.49 | 367.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 364.70 | 366.93 | 367.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 354.20 | 363.59 | 365.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 360.35 | 357.68 | 360.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 360.35 | 357.68 | 360.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 360.35 | 357.68 | 360.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 358.30 | 357.68 | 360.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:45:00 | 357.95 | 357.99 | 360.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 12:15:00 | 358.75 | 357.99 | 360.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:15:00 | 358.50 | 358.40 | 360.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 356.90 | 358.10 | 360.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 347.10 | 357.23 | 359.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 14:30:00 | 354.95 | 354.41 | 356.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 350.40 | 355.09 | 356.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 14:15:00 | 360.65 | 356.89 | 356.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 360.65 | 356.89 | 356.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 14:15:00 | 362.60 | 359.86 | 358.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 12:15:00 | 372.60 | 374.82 | 370.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 12:45:00 | 372.75 | 374.82 | 370.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 370.85 | 374.03 | 370.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:00:00 | 370.85 | 374.03 | 370.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 372.40 | 373.70 | 370.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 371.65 | 373.70 | 370.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 374.60 | 375.12 | 373.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:45:00 | 373.30 | 375.12 | 373.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 365.95 | 373.26 | 372.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 10:00:00 | 365.95 | 373.26 | 372.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 10:15:00 | 367.00 | 372.01 | 372.05 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 380.40 | 373.21 | 372.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 385.20 | 380.20 | 376.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 12:15:00 | 379.20 | 380.67 | 377.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 12:45:00 | 379.55 | 380.67 | 377.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 15:15:00 | 378.85 | 380.05 | 378.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 09:15:00 | 380.90 | 380.05 | 378.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 382.80 | 380.60 | 378.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 11:00:00 | 383.45 | 381.17 | 379.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 11:45:00 | 383.25 | 381.51 | 379.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 12:15:00 | 383.20 | 381.51 | 379.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 10:30:00 | 383.60 | 382.39 | 381.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 11:15:00 | 381.30 | 382.17 | 381.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 12:15:00 | 381.30 | 382.17 | 381.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 12:15:00 | 379.40 | 381.62 | 380.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 13:00:00 | 379.40 | 381.62 | 380.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 379.45 | 381.18 | 380.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 14:15:00 | 380.00 | 381.18 | 380.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 15:15:00 | 380.00 | 380.85 | 380.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 10:15:00 | 379.55 | 380.80 | 380.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 379.55 | 380.80 | 380.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 13:15:00 | 377.55 | 379.47 | 380.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 15:15:00 | 371.00 | 370.77 | 372.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:15:00 | 379.40 | 370.77 | 372.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 379.30 | 372.48 | 373.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 379.30 | 372.48 | 373.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 378.20 | 373.62 | 373.88 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 376.70 | 374.24 | 374.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 381.95 | 377.17 | 375.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 378.15 | 378.60 | 377.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 14:00:00 | 378.15 | 378.60 | 377.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 377.25 | 378.27 | 377.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 380.20 | 378.27 | 377.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 09:30:00 | 379.65 | 381.17 | 379.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 374.95 | 380.59 | 380.22 | SL hit (close<static) qty=1.00 sl=376.55 alert=retest2 |

### Cycle 69 — SELL (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 10:15:00 | 375.10 | 379.50 | 379.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 11:15:00 | 373.15 | 378.23 | 379.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 10:15:00 | 374.55 | 373.66 | 376.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-05 10:45:00 | 374.65 | 373.66 | 376.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 373.25 | 372.77 | 374.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 11:30:00 | 370.95 | 372.28 | 373.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 14:15:00 | 376.05 | 371.70 | 373.13 | SL hit (close>static) qty=1.00 sl=375.10 alert=retest2 |

### Cycle 70 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 379.80 | 374.19 | 374.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 11:15:00 | 380.25 | 376.24 | 375.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 376.50 | 377.97 | 376.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 376.50 | 377.97 | 376.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 376.50 | 377.97 | 376.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 11:45:00 | 379.10 | 378.07 | 376.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 12:15:00 | 379.90 | 378.07 | 376.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 14:30:00 | 379.90 | 378.41 | 377.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-16 09:15:00 | 419.30 | 2025-05-16 15:15:00 | 421.96 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-05-29 10:30:00 | 413.00 | 2025-05-29 15:15:00 | 419.10 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-05-29 14:00:00 | 413.66 | 2025-05-29 15:15:00 | 419.10 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-06-05 09:15:00 | 408.80 | 2025-06-06 10:15:00 | 415.80 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-06-05 11:00:00 | 408.60 | 2025-06-06 10:15:00 | 415.80 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-06-05 12:30:00 | 408.70 | 2025-06-06 10:15:00 | 415.80 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-06-05 14:30:00 | 408.72 | 2025-06-06 10:15:00 | 415.80 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-06-26 15:00:00 | 441.02 | 2025-06-30 09:15:00 | 435.10 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-06-27 14:15:00 | 441.40 | 2025-06-30 09:15:00 | 435.10 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-07-02 11:15:00 | 436.04 | 2025-07-08 09:15:00 | 446.70 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-07-31 14:45:00 | 396.82 | 2025-08-01 10:15:00 | 400.36 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-08-06 09:15:00 | 403.34 | 2025-08-07 09:15:00 | 396.34 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-08-12 14:30:00 | 391.92 | 2025-08-13 11:15:00 | 396.54 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-08-19 10:30:00 | 403.94 | 2025-08-22 10:15:00 | 398.86 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-08-20 13:30:00 | 403.04 | 2025-08-22 10:15:00 | 398.86 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-08-20 14:15:00 | 403.38 | 2025-08-22 10:15:00 | 398.86 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-08-22 09:30:00 | 403.32 | 2025-08-22 10:15:00 | 398.86 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-09-02 13:45:00 | 389.74 | 2025-09-09 13:15:00 | 391.10 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-09-03 10:30:00 | 389.98 | 2025-09-09 13:15:00 | 391.10 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-09-03 12:00:00 | 389.58 | 2025-09-09 13:15:00 | 391.10 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-09-04 12:15:00 | 389.50 | 2025-09-09 13:15:00 | 391.10 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-09-05 10:45:00 | 389.00 | 2025-09-09 13:15:00 | 391.10 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-09-05 15:00:00 | 388.96 | 2025-09-09 13:15:00 | 391.10 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-09-08 09:30:00 | 388.70 | 2025-09-09 13:15:00 | 391.10 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-09-09 11:00:00 | 388.44 | 2025-09-09 13:15:00 | 391.10 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-09-15 09:15:00 | 395.48 | 2025-09-22 12:15:00 | 406.56 | STOP_HIT | 1.00 | 2.80% |
| BUY | retest2 | 2025-09-15 12:30:00 | 395.22 | 2025-09-22 12:15:00 | 406.56 | STOP_HIT | 1.00 | 2.87% |
| BUY | retest2 | 2025-09-16 09:15:00 | 397.98 | 2025-09-22 12:15:00 | 406.56 | STOP_HIT | 1.00 | 2.16% |
| SELL | retest2 | 2025-09-30 10:30:00 | 398.68 | 2025-10-01 09:15:00 | 406.00 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-09-30 11:15:00 | 398.54 | 2025-10-01 09:15:00 | 406.00 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-09-30 14:45:00 | 398.22 | 2025-10-01 09:15:00 | 406.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-10-09 11:30:00 | 427.10 | 2025-10-24 10:15:00 | 435.26 | STOP_HIT | 1.00 | 1.91% |
| SELL | retest2 | 2025-10-29 10:45:00 | 431.28 | 2025-11-17 09:15:00 | 421.68 | STOP_HIT | 1.00 | 2.23% |
| BUY | retest2 | 2025-12-15 12:00:00 | 434.32 | 2025-12-17 12:15:00 | 433.10 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-12-22 12:15:00 | 431.94 | 2025-12-24 10:15:00 | 434.40 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2026-01-05 09:45:00 | 440.96 | 2026-01-05 14:15:00 | 438.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2026-01-14 09:15:00 | 423.50 | 2026-01-19 11:15:00 | 427.20 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-01-14 14:15:00 | 423.80 | 2026-01-19 11:15:00 | 427.20 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-01-16 11:00:00 | 423.90 | 2026-01-19 11:15:00 | 427.20 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2026-01-19 10:30:00 | 423.20 | 2026-01-19 11:15:00 | 427.20 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-01-30 09:15:00 | 408.10 | 2026-02-03 09:15:00 | 418.50 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2026-02-01 10:30:00 | 410.40 | 2026-02-03 09:15:00 | 418.50 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-02-01 12:15:00 | 409.90 | 2026-02-03 09:15:00 | 418.50 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2026-02-17 09:15:00 | 422.40 | 2026-02-18 14:15:00 | 426.30 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-02-17 15:15:00 | 424.30 | 2026-02-18 14:15:00 | 426.30 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2026-02-18 09:45:00 | 422.90 | 2026-02-18 14:15:00 | 426.30 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-03-17 11:15:00 | 370.80 | 2026-03-17 14:15:00 | 374.30 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2026-04-01 10:15:00 | 358.30 | 2026-04-06 14:15:00 | 360.65 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2026-04-01 11:45:00 | 357.95 | 2026-04-06 14:15:00 | 360.65 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2026-04-01 12:15:00 | 358.75 | 2026-04-06 14:15:00 | 360.65 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2026-04-01 13:15:00 | 358.50 | 2026-04-06 14:15:00 | 360.65 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2026-04-02 09:15:00 | 347.10 | 2026-04-06 14:15:00 | 360.65 | STOP_HIT | 1.00 | -3.90% |
| SELL | retest2 | 2026-04-02 14:30:00 | 354.95 | 2026-04-06 14:15:00 | 360.65 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-04-06 09:15:00 | 350.40 | 2026-04-06 14:15:00 | 360.65 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2026-04-17 11:00:00 | 383.45 | 2026-04-22 10:15:00 | 379.55 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2026-04-17 11:45:00 | 383.25 | 2026-04-22 10:15:00 | 379.55 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-04-17 12:15:00 | 383.20 | 2026-04-22 10:15:00 | 379.55 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2026-04-20 10:30:00 | 383.60 | 2026-04-22 10:15:00 | 379.55 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2026-04-20 14:15:00 | 380.00 | 2026-04-22 10:15:00 | 379.55 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2026-04-20 15:15:00 | 380.00 | 2026-04-22 10:15:00 | 379.55 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2026-04-29 09:15:00 | 380.20 | 2026-05-04 09:15:00 | 374.95 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-04-30 09:30:00 | 379.65 | 2026-05-04 09:15:00 | 374.95 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-05-06 11:30:00 | 370.95 | 2026-05-06 14:15:00 | 376.05 | STOP_HIT | 1.00 | -1.37% |
