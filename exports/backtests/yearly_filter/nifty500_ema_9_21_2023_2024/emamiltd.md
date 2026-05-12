# Emami Ltd. (EMAMILTD)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 456.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 232 |
| ALERT1 | 147 |
| ALERT2 | 143 |
| ALERT2_SKIP | 82 |
| ALERT3 | 333 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 184 |
| PARTIAL | 19 |
| TARGET_HIT | 7 |
| STOP_HIT | 183 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 208 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 63 / 145
- **Target hits / Stop hits / Partials:** 7 / 182 / 19
- **Avg / median % per leg:** 0.30% / -0.82%
- **Sum % (uncompounded):** 62.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 75 | 20 | 26.7% | 7 | 68 | 0 | 0.21% | 16.0% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 4 | 0 | 0.89% | 3.5% |
| BUY @ 3rd Alert (retest2) | 71 | 16 | 22.5% | 7 | 64 | 0 | 0.17% | 12.4% |
| SELL (all) | 133 | 43 | 32.3% | 0 | 114 | 19 | 0.35% | 46.2% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.98% | -2.0% |
| SELL @ 3rd Alert (retest2) | 132 | 43 | 32.6% | 0 | 113 | 19 | 0.36% | 48.2% |
| retest1 (combined) | 5 | 4 | 80.0% | 0 | 5 | 0 | 0.31% | 1.6% |
| retest2 (combined) | 203 | 59 | 29.1% | 7 | 177 | 19 | 0.30% | 60.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 09:15:00 | 386.85 | 388.48 | 388.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 10:15:00 | 385.75 | 387.94 | 388.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-17 11:15:00 | 388.65 | 388.08 | 388.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-17 11:15:00 | 388.65 | 388.08 | 388.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 11:15:00 | 388.65 | 388.08 | 388.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-17 12:00:00 | 388.65 | 388.08 | 388.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 12:15:00 | 387.15 | 387.89 | 388.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-17 13:15:00 | 386.20 | 387.89 | 388.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-17 14:15:00 | 386.00 | 387.56 | 388.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-18 11:15:00 | 390.85 | 388.68 | 388.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 11:15:00 | 390.85 | 388.68 | 388.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-19 11:15:00 | 393.65 | 390.72 | 389.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-19 15:15:00 | 390.60 | 391.50 | 390.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-19 15:15:00 | 390.60 | 391.50 | 390.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 15:15:00 | 390.60 | 391.50 | 390.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-22 09:15:00 | 391.80 | 391.50 | 390.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 391.95 | 391.59 | 390.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-23 09:45:00 | 394.40 | 392.54 | 391.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-23 10:45:00 | 393.95 | 392.79 | 391.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-24 10:15:00 | 389.75 | 392.13 | 391.97 | SL hit (close<static) qty=1.00 sl=389.80 alert=retest2 |

### Cycle 3 — SELL (started 2023-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 11:15:00 | 389.70 | 391.65 | 391.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 12:15:00 | 388.65 | 391.05 | 391.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-26 09:15:00 | 389.80 | 386.84 | 388.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-26 09:15:00 | 389.80 | 386.84 | 388.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 389.80 | 386.84 | 388.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 09:45:00 | 389.90 | 386.84 | 388.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 10:15:00 | 390.10 | 387.49 | 388.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 11:00:00 | 390.10 | 387.49 | 388.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2023-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 13:15:00 | 396.70 | 390.08 | 389.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 14:15:00 | 400.30 | 392.12 | 390.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 14:15:00 | 399.00 | 400.61 | 396.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-29 14:15:00 | 399.00 | 400.61 | 396.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 14:15:00 | 399.00 | 400.61 | 396.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-29 14:45:00 | 397.00 | 400.61 | 396.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 15:15:00 | 400.55 | 400.59 | 397.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 09:15:00 | 400.70 | 400.59 | 397.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 10:00:00 | 401.40 | 400.76 | 397.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-31 13:15:00 | 396.00 | 398.47 | 398.40 | SL hit (close<static) qty=1.00 sl=397.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 14:15:00 | 391.60 | 397.10 | 397.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 15:15:00 | 391.00 | 395.88 | 397.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-02 09:15:00 | 394.85 | 392.29 | 394.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-02 09:15:00 | 394.85 | 392.29 | 394.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 09:15:00 | 394.85 | 392.29 | 394.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-02 09:45:00 | 395.50 | 392.29 | 394.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 10:15:00 | 393.30 | 392.49 | 394.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-02 11:45:00 | 391.85 | 392.37 | 394.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-02 14:15:00 | 396.90 | 393.35 | 394.09 | SL hit (close>static) qty=1.00 sl=394.85 alert=retest2 |

### Cycle 6 — BUY (started 2023-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 11:15:00 | 395.85 | 394.64 | 394.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 12:15:00 | 397.35 | 395.18 | 394.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-05 13:15:00 | 394.80 | 395.10 | 394.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-05 13:15:00 | 394.80 | 395.10 | 394.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 13:15:00 | 394.80 | 395.10 | 394.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-05 14:00:00 | 394.80 | 395.10 | 394.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 14:15:00 | 395.00 | 395.08 | 394.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-05 14:45:00 | 393.60 | 395.08 | 394.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 15:15:00 | 394.30 | 394.93 | 394.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-06 09:15:00 | 394.95 | 394.93 | 394.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 09:15:00 | 395.00 | 394.94 | 394.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 11:00:00 | 397.20 | 395.39 | 395.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 11:30:00 | 396.70 | 395.55 | 395.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-07 09:15:00 | 394.75 | 394.93 | 394.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2023-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-07 09:15:00 | 394.75 | 394.93 | 394.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-08 11:15:00 | 392.00 | 393.84 | 394.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-13 14:15:00 | 382.80 | 376.16 | 379.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 14:15:00 | 382.80 | 376.16 | 379.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 14:15:00 | 382.80 | 376.16 | 379.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 15:00:00 | 382.80 | 376.16 | 379.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 15:15:00 | 381.95 | 377.32 | 379.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-14 09:15:00 | 376.85 | 377.32 | 379.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-15 12:30:00 | 379.30 | 379.11 | 379.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-15 14:00:00 | 380.45 | 379.38 | 379.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-15 14:15:00 | 381.70 | 379.84 | 379.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2023-06-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-15 14:15:00 | 381.70 | 379.84 | 379.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 09:15:00 | 396.50 | 383.52 | 381.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-20 11:15:00 | 412.80 | 413.15 | 405.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-20 12:00:00 | 412.80 | 413.15 | 405.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 15:15:00 | 414.00 | 416.36 | 412.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-22 09:15:00 | 423.05 | 416.36 | 412.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-23 09:15:00 | 409.55 | 416.35 | 415.03 | SL hit (close<static) qty=1.00 sl=410.00 alert=retest2 |

### Cycle 9 — SELL (started 2023-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 12:15:00 | 408.90 | 413.74 | 414.07 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 10:15:00 | 416.10 | 414.33 | 414.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-26 11:15:00 | 420.25 | 415.52 | 414.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-27 11:15:00 | 419.00 | 419.60 | 417.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-27 12:00:00 | 419.00 | 419.60 | 417.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 13:15:00 | 418.70 | 419.32 | 417.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 13:30:00 | 416.60 | 419.32 | 417.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 14:15:00 | 418.40 | 419.14 | 417.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 14:30:00 | 416.85 | 419.14 | 417.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 15:15:00 | 417.50 | 418.81 | 417.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 09:45:00 | 414.10 | 417.68 | 417.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2023-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-28 10:15:00 | 413.00 | 416.74 | 416.96 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 10:15:00 | 420.00 | 416.31 | 416.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 12:15:00 | 420.90 | 417.84 | 417.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 13:15:00 | 423.80 | 427.80 | 425.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-04 13:15:00 | 423.80 | 427.80 | 425.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 13:15:00 | 423.80 | 427.80 | 425.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 14:00:00 | 423.80 | 427.80 | 425.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 14:15:00 | 425.50 | 427.34 | 425.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 15:00:00 | 425.50 | 427.34 | 425.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 15:15:00 | 424.00 | 426.67 | 425.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 09:15:00 | 422.25 | 426.67 | 425.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 419.00 | 425.14 | 424.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 09:45:00 | 419.40 | 425.14 | 424.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2023-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-05 10:15:00 | 419.85 | 424.08 | 424.24 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-07-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 11:15:00 | 428.10 | 423.79 | 423.59 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 12:15:00 | 418.30 | 423.82 | 424.16 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 11:15:00 | 427.95 | 424.47 | 424.14 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-11 09:15:00 | 420.00 | 424.05 | 424.19 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 13:15:00 | 425.60 | 423.04 | 422.93 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-07-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 11:15:00 | 419.75 | 422.74 | 422.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 12:15:00 | 418.35 | 421.87 | 422.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 09:15:00 | 419.60 | 418.91 | 420.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-14 10:00:00 | 419.60 | 418.91 | 420.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 422.90 | 417.25 | 418.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 10:00:00 | 422.90 | 417.25 | 418.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 10:15:00 | 422.95 | 418.39 | 419.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 10:30:00 | 423.20 | 418.39 | 419.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — BUY (started 2023-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 12:15:00 | 423.20 | 420.12 | 419.75 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 09:15:00 | 419.30 | 419.63 | 419.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-18 10:15:00 | 416.35 | 418.97 | 419.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-20 09:15:00 | 416.70 | 414.81 | 415.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 09:15:00 | 416.70 | 414.81 | 415.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 09:15:00 | 416.70 | 414.81 | 415.80 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-07-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 11:15:00 | 419.50 | 416.62 | 416.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-21 12:15:00 | 421.90 | 419.04 | 417.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-24 10:15:00 | 420.05 | 420.61 | 419.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-24 11:00:00 | 420.05 | 420.61 | 419.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 09:15:00 | 419.85 | 420.81 | 419.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 10:00:00 | 419.85 | 420.81 | 419.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 10:15:00 | 420.15 | 420.68 | 419.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-25 12:15:00 | 420.35 | 420.55 | 419.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-25 12:45:00 | 420.30 | 420.44 | 419.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-25 13:30:00 | 421.15 | 420.65 | 420.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2023-07-31 14:15:00 | 462.39 | 456.95 | 450.16 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2023-08-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 14:15:00 | 446.05 | 453.46 | 454.38 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 10:15:00 | 455.50 | 452.40 | 452.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 14:15:00 | 460.15 | 454.59 | 453.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 10:15:00 | 502.85 | 504.27 | 491.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-10 10:30:00 | 500.55 | 504.27 | 491.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 09:15:00 | 515.10 | 516.04 | 508.73 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-08-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 09:15:00 | 506.25 | 509.71 | 510.15 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 09:15:00 | 515.85 | 510.52 | 510.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-18 11:15:00 | 520.00 | 513.29 | 511.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 09:15:00 | 531.05 | 531.48 | 525.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-22 09:30:00 | 533.00 | 531.48 | 525.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 13:15:00 | 537.50 | 538.46 | 534.15 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-08-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 13:15:00 | 523.00 | 531.51 | 532.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-24 14:15:00 | 516.50 | 528.51 | 531.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 10:15:00 | 518.30 | 515.99 | 520.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 10:15:00 | 518.30 | 515.99 | 520.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 10:15:00 | 518.30 | 515.99 | 520.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 11:00:00 | 518.30 | 515.99 | 520.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 12:15:00 | 519.00 | 516.98 | 520.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-28 13:30:00 | 516.15 | 516.73 | 519.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-29 11:15:00 | 514.75 | 516.40 | 518.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-29 15:15:00 | 514.25 | 516.35 | 517.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-30 13:15:00 | 520.00 | 518.16 | 518.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2023-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 13:15:00 | 520.00 | 518.16 | 518.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 15:15:00 | 520.05 | 518.83 | 518.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 13:15:00 | 524.00 | 525.44 | 522.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-31 14:00:00 | 524.00 | 525.44 | 522.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 14:15:00 | 523.80 | 525.11 | 522.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 14:45:00 | 522.80 | 525.11 | 522.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 15:15:00 | 523.85 | 524.86 | 522.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-01 09:15:00 | 539.15 | 524.86 | 522.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-04 15:15:00 | 523.60 | 527.29 | 527.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2023-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 15:15:00 | 523.60 | 527.29 | 527.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-05 14:15:00 | 521.30 | 524.88 | 526.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-06 11:15:00 | 524.10 | 523.88 | 525.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-06 11:15:00 | 524.10 | 523.88 | 525.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 11:15:00 | 524.10 | 523.88 | 525.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-06 11:45:00 | 524.35 | 523.88 | 525.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 12:15:00 | 527.10 | 524.52 | 525.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-06 13:00:00 | 527.10 | 524.52 | 525.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 13:15:00 | 526.80 | 524.98 | 525.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-06 13:30:00 | 527.10 | 524.98 | 525.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 14:15:00 | 528.35 | 525.65 | 525.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-06 15:15:00 | 527.00 | 525.65 | 525.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2023-09-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 15:15:00 | 527.00 | 525.92 | 525.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-07 13:15:00 | 531.50 | 528.59 | 527.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 09:15:00 | 560.20 | 561.85 | 551.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-12 09:45:00 | 558.55 | 561.85 | 551.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 566.60 | 562.80 | 552.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 10:30:00 | 558.10 | 562.80 | 552.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 12:15:00 | 553.15 | 561.25 | 553.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 13:00:00 | 553.15 | 561.25 | 553.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 13:15:00 | 558.10 | 560.62 | 554.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 11:45:00 | 563.70 | 557.34 | 554.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-14 13:15:00 | 560.25 | 561.55 | 558.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-14 14:00:00 | 560.10 | 561.26 | 559.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-15 11:00:00 | 560.25 | 560.97 | 559.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 11:15:00 | 561.85 | 561.14 | 559.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-15 11:30:00 | 561.40 | 561.14 | 559.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 12:15:00 | 550.75 | 559.07 | 559.04 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-09-15 12:15:00 | 550.75 | 559.07 | 559.04 | SL hit (close<static) qty=1.00 sl=552.25 alert=retest2 |

### Cycle 31 — SELL (started 2023-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-15 13:15:00 | 549.50 | 557.15 | 558.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-15 14:15:00 | 547.70 | 555.26 | 557.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 09:15:00 | 523.90 | 523.72 | 530.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 09:15:00 | 523.90 | 523.72 | 530.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 09:15:00 | 523.90 | 523.72 | 530.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 09:30:00 | 534.70 | 523.72 | 530.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 519.55 | 519.45 | 524.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 09:30:00 | 522.60 | 519.45 | 524.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 523.15 | 519.76 | 522.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 09:30:00 | 522.10 | 519.76 | 522.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 10:15:00 | 515.40 | 518.89 | 521.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 13:00:00 | 514.15 | 517.29 | 520.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 14:30:00 | 514.10 | 515.99 | 519.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 15:00:00 | 513.05 | 515.99 | 519.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-28 09:30:00 | 512.45 | 514.35 | 517.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 09:15:00 | 530.20 | 516.14 | 516.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-09-29 09:15:00 | 530.20 | 516.14 | 516.49 | SL hit (close>static) qty=1.00 sl=523.55 alert=retest2 |

### Cycle 32 — BUY (started 2023-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 10:15:00 | 537.70 | 520.45 | 518.42 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-10-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 14:15:00 | 529.95 | 534.39 | 534.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-06 09:15:00 | 524.65 | 531.74 | 533.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 10:15:00 | 516.40 | 513.18 | 519.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-10 10:30:00 | 516.00 | 513.18 | 519.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 12:15:00 | 518.05 | 514.86 | 519.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 12:30:00 | 521.70 | 514.86 | 519.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 13:15:00 | 517.00 | 515.29 | 518.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-10 15:15:00 | 515.10 | 515.63 | 518.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-11 09:15:00 | 520.60 | 516.54 | 518.65 | SL hit (close>static) qty=1.00 sl=519.50 alert=retest2 |

### Cycle 34 — BUY (started 2023-10-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 09:15:00 | 524.75 | 520.28 | 519.79 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2023-10-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 15:15:00 | 514.85 | 519.16 | 519.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-13 13:15:00 | 512.45 | 516.14 | 517.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 09:15:00 | 507.95 | 507.51 | 511.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 10:15:00 | 511.75 | 508.36 | 511.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 10:15:00 | 511.75 | 508.36 | 511.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 11:00:00 | 511.75 | 508.36 | 511.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 11:15:00 | 511.40 | 508.97 | 511.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-17 13:45:00 | 509.45 | 508.50 | 510.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 09:45:00 | 508.85 | 505.74 | 505.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-20 10:15:00 | 515.75 | 507.74 | 506.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2023-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-20 10:15:00 | 515.75 | 507.74 | 506.80 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 14:15:00 | 497.10 | 508.07 | 508.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 15:15:00 | 496.00 | 505.65 | 507.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 09:15:00 | 507.90 | 506.10 | 507.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 09:15:00 | 507.90 | 506.10 | 507.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 09:15:00 | 507.90 | 506.10 | 507.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-25 09:30:00 | 508.90 | 506.10 | 507.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 10:15:00 | 507.75 | 506.43 | 507.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-25 12:30:00 | 505.65 | 505.05 | 506.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-25 14:15:00 | 510.50 | 506.77 | 507.07 | SL hit (close>static) qty=1.00 sl=509.95 alert=retest2 |

### Cycle 38 — BUY (started 2023-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 10:15:00 | 513.75 | 505.66 | 504.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-01 10:15:00 | 516.25 | 511.10 | 509.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 12:15:00 | 511.55 | 511.96 | 509.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-01 13:00:00 | 511.55 | 511.96 | 509.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 15:15:00 | 510.00 | 511.92 | 510.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 14:45:00 | 514.00 | 510.41 | 510.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-03 09:15:00 | 515.00 | 510.33 | 510.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-03 11:15:00 | 508.70 | 509.88 | 509.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2023-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 11:15:00 | 508.70 | 509.88 | 509.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 15:15:00 | 508.35 | 509.57 | 509.82 | Break + close below crossover candle low |

### Cycle 40 — BUY (started 2023-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-06 09:15:00 | 518.20 | 511.30 | 510.58 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2023-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 09:15:00 | 508.30 | 513.42 | 513.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-08 10:15:00 | 507.80 | 512.30 | 513.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 14:15:00 | 510.65 | 510.19 | 511.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-08 14:15:00 | 510.65 | 510.19 | 511.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 14:15:00 | 510.65 | 510.19 | 511.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-08 14:30:00 | 512.40 | 510.19 | 511.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 15:15:00 | 510.90 | 510.33 | 511.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-09 09:15:00 | 509.50 | 510.33 | 511.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-09 10:45:00 | 509.95 | 509.96 | 511.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-09 11:15:00 | 515.50 | 511.07 | 511.63 | SL hit (close>static) qty=1.00 sl=511.85 alert=retest2 |

### Cycle 42 — BUY (started 2023-11-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-09 12:15:00 | 518.20 | 512.50 | 512.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-09 15:15:00 | 520.00 | 515.87 | 514.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-10 11:15:00 | 514.90 | 516.23 | 514.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 11:15:00 | 514.90 | 516.23 | 514.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 11:15:00 | 514.90 | 516.23 | 514.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 12:00:00 | 514.90 | 516.23 | 514.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 12:15:00 | 514.95 | 515.97 | 514.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 12:30:00 | 515.00 | 515.97 | 514.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 13:15:00 | 514.20 | 515.62 | 514.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 13:30:00 | 512.95 | 515.62 | 514.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 14:15:00 | 511.55 | 514.80 | 514.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 14:30:00 | 511.00 | 514.80 | 514.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 15:15:00 | 512.95 | 514.43 | 514.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-12 18:15:00 | 516.95 | 514.43 | 514.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-13 09:15:00 | 511.10 | 514.02 | 514.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2023-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 09:15:00 | 511.10 | 514.02 | 514.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-15 09:15:00 | 509.15 | 511.01 | 512.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-16 09:15:00 | 506.70 | 506.43 | 508.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-16 10:00:00 | 506.70 | 506.43 | 508.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 494.70 | 493.60 | 496.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 09:30:00 | 494.20 | 493.60 | 496.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 493.00 | 493.93 | 495.29 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2023-11-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 15:15:00 | 497.95 | 494.88 | 494.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-24 09:15:00 | 505.80 | 497.06 | 495.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-28 09:15:00 | 501.95 | 502.03 | 499.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 10:15:00 | 499.50 | 501.53 | 499.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 10:15:00 | 499.50 | 501.53 | 499.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 11:00:00 | 499.50 | 501.53 | 499.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 11:15:00 | 499.85 | 501.19 | 499.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 11:30:00 | 497.20 | 501.19 | 499.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 12:15:00 | 502.80 | 501.51 | 499.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-29 13:45:00 | 503.00 | 500.91 | 500.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-29 15:15:00 | 498.30 | 500.26 | 500.09 | SL hit (close<static) qty=1.00 sl=499.30 alert=retest2 |

### Cycle 45 — SELL (started 2023-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 12:15:00 | 503.20 | 508.37 | 509.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-06 13:15:00 | 500.75 | 506.85 | 508.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-07 13:15:00 | 501.35 | 499.86 | 503.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-07 13:30:00 | 501.85 | 499.86 | 503.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 09:15:00 | 500.70 | 500.47 | 502.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-08 10:45:00 | 499.45 | 500.17 | 502.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-08 11:45:00 | 498.90 | 500.04 | 502.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-08 12:15:00 | 499.00 | 500.04 | 502.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-11 12:15:00 | 507.25 | 499.83 | 500.54 | SL hit (close>static) qty=1.00 sl=504.60 alert=retest2 |

### Cycle 46 — BUY (started 2023-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 13:15:00 | 506.65 | 501.20 | 501.10 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2023-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 12:15:00 | 497.30 | 501.14 | 501.35 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2023-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 11:15:00 | 503.10 | 500.98 | 500.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 09:15:00 | 507.00 | 503.90 | 502.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 14:15:00 | 497.80 | 504.69 | 503.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-15 14:15:00 | 497.80 | 504.69 | 503.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 14:15:00 | 497.80 | 504.69 | 503.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 15:00:00 | 497.80 | 504.69 | 503.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2023-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 15:15:00 | 497.50 | 503.25 | 503.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-18 13:15:00 | 495.00 | 498.59 | 500.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-19 10:15:00 | 497.90 | 497.45 | 499.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-19 10:45:00 | 498.00 | 497.45 | 499.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 11:15:00 | 500.60 | 498.08 | 499.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-19 12:00:00 | 500.60 | 498.08 | 499.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 12:15:00 | 500.00 | 498.46 | 499.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-19 12:45:00 | 500.70 | 498.46 | 499.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 13:15:00 | 500.60 | 498.89 | 499.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-19 14:00:00 | 500.60 | 498.89 | 499.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 14:15:00 | 500.90 | 499.29 | 499.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-19 15:00:00 | 500.90 | 499.29 | 499.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 15:15:00 | 497.70 | 498.97 | 499.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-20 09:15:00 | 500.60 | 498.97 | 499.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2023-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 09:15:00 | 504.00 | 499.98 | 499.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-20 10:15:00 | 510.30 | 502.04 | 500.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 14:15:00 | 499.80 | 503.10 | 501.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 14:15:00 | 499.80 | 503.10 | 501.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 14:15:00 | 499.80 | 503.10 | 501.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 15:00:00 | 499.80 | 503.10 | 501.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 15:15:00 | 495.45 | 501.57 | 501.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-21 09:15:00 | 491.50 | 501.57 | 501.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2023-12-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-21 09:15:00 | 494.75 | 500.21 | 500.79 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2023-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 09:15:00 | 508.20 | 497.80 | 497.54 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2023-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-27 14:15:00 | 498.35 | 498.74 | 498.78 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2023-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 15:15:00 | 499.20 | 498.83 | 498.82 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2023-12-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-28 09:15:00 | 497.50 | 498.56 | 498.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-28 10:15:00 | 497.00 | 498.25 | 498.55 | Break + close below crossover candle low |

### Cycle 56 — BUY (started 2023-12-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 12:15:00 | 512.60 | 500.89 | 499.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-28 14:15:00 | 528.00 | 508.86 | 503.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 09:15:00 | 558.05 | 565.60 | 552.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-02 10:00:00 | 558.05 | 565.60 | 552.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 10:15:00 | 558.95 | 562.70 | 557.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-03 10:45:00 | 557.30 | 562.70 | 557.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 11:15:00 | 554.75 | 561.11 | 557.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-03 12:00:00 | 554.75 | 561.11 | 557.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 12:15:00 | 549.75 | 558.84 | 556.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-03 12:45:00 | 548.90 | 558.84 | 556.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2024-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 14:15:00 | 545.00 | 554.66 | 554.96 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2024-01-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 10:15:00 | 566.50 | 557.08 | 555.92 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 12:15:00 | 556.50 | 559.57 | 559.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 14:15:00 | 554.75 | 558.04 | 558.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 14:15:00 | 552.35 | 551.89 | 554.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-09 15:00:00 | 552.35 | 551.89 | 554.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 547.05 | 550.36 | 553.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-10 12:15:00 | 542.35 | 549.08 | 552.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 10:45:00 | 543.15 | 546.03 | 549.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-17 11:15:00 | 515.23 | 525.86 | 529.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-17 11:15:00 | 515.99 | 525.86 | 529.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-19 09:15:00 | 511.40 | 510.84 | 516.20 | SL hit (close>ema200) qty=0.50 sl=510.84 alert=retest2 |

### Cycle 60 — BUY (started 2024-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 10:15:00 | 507.95 | 500.44 | 499.44 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-01-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-31 15:15:00 | 500.00 | 500.28 | 500.29 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-01 10:15:00 | 504.55 | 501.09 | 500.65 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 13:15:00 | 499.80 | 500.39 | 500.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 14:15:00 | 497.30 | 499.77 | 500.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-01 15:15:00 | 500.00 | 499.82 | 500.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 15:15:00 | 500.00 | 499.82 | 500.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 15:15:00 | 500.00 | 499.82 | 500.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 09:15:00 | 499.00 | 499.82 | 500.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 497.65 | 499.38 | 499.88 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 14:15:00 | 503.40 | 500.25 | 500.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-05 10:15:00 | 506.35 | 502.42 | 501.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-05 13:15:00 | 503.75 | 504.30 | 502.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-05 14:00:00 | 503.75 | 504.30 | 502.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 14:15:00 | 503.95 | 504.23 | 502.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 14:30:00 | 499.60 | 504.23 | 502.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 15:15:00 | 502.10 | 503.80 | 502.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-06 09:30:00 | 502.90 | 503.25 | 502.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 10:15:00 | 501.65 | 502.93 | 502.36 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2024-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 09:15:00 | 501.50 | 502.22 | 502.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-07 10:15:00 | 499.20 | 501.62 | 501.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-09 13:15:00 | 486.40 | 484.58 | 489.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 13:15:00 | 486.40 | 484.58 | 489.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 13:15:00 | 486.40 | 484.58 | 489.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-12 09:30:00 | 479.35 | 483.38 | 487.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-12 15:15:00 | 480.00 | 482.21 | 485.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-19 12:15:00 | 479.90 | 467.18 | 466.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2024-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 12:15:00 | 479.90 | 467.18 | 466.32 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 15:15:00 | 467.45 | 468.92 | 468.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 09:15:00 | 465.60 | 468.26 | 468.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-26 09:15:00 | 465.05 | 461.69 | 463.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 09:15:00 | 465.05 | 461.69 | 463.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 465.05 | 461.69 | 463.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-26 09:45:00 | 469.00 | 461.69 | 463.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 10:15:00 | 464.20 | 462.19 | 463.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-26 10:30:00 | 464.25 | 462.19 | 463.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 11:15:00 | 465.10 | 462.77 | 463.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-26 12:00:00 | 465.10 | 462.77 | 463.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 12:15:00 | 464.05 | 463.03 | 463.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-26 13:30:00 | 463.95 | 463.43 | 463.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-26 14:15:00 | 469.80 | 464.71 | 464.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2024-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-26 14:15:00 | 469.80 | 464.71 | 464.37 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 10:15:00 | 458.30 | 465.20 | 465.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-01 12:15:00 | 454.05 | 460.25 | 462.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-02 09:15:00 | 458.90 | 457.73 | 460.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-02 09:30:00 | 459.20 | 457.73 | 460.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 12:15:00 | 459.80 | 458.15 | 459.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-04 09:15:00 | 454.75 | 458.15 | 459.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 454.90 | 457.50 | 459.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-06 09:45:00 | 452.75 | 457.94 | 458.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-06 11:00:00 | 452.00 | 456.75 | 457.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 09:30:00 | 451.95 | 448.00 | 450.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 10:15:00 | 430.11 | 434.84 | 438.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 10:15:00 | 429.40 | 434.84 | 438.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 10:15:00 | 429.35 | 434.84 | 438.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-18 10:15:00 | 429.45 | 428.01 | 432.25 | SL hit (close>ema200) qty=0.50 sl=428.01 alert=retest2 |

### Cycle 70 — BUY (started 2024-03-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 15:15:00 | 430.00 | 427.30 | 427.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 10:15:00 | 437.95 | 429.94 | 428.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 14:15:00 | 421.85 | 428.35 | 428.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-22 14:15:00 | 421.85 | 428.35 | 428.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 14:15:00 | 421.85 | 428.35 | 428.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-22 14:45:00 | 425.35 | 428.35 | 428.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2024-03-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 15:15:00 | 424.50 | 427.58 | 427.87 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-26 11:15:00 | 430.65 | 428.18 | 428.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-26 14:15:00 | 435.50 | 430.23 | 429.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-28 12:15:00 | 434.25 | 435.29 | 433.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-28 13:15:00 | 433.35 | 435.29 | 433.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 13:15:00 | 433.10 | 434.85 | 433.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 14:00:00 | 433.10 | 434.85 | 433.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 14:15:00 | 429.50 | 433.78 | 433.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 15:00:00 | 429.50 | 433.78 | 433.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 15:15:00 | 429.00 | 432.82 | 432.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-01 09:15:00 | 436.55 | 432.82 | 432.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-15 09:15:00 | 445.90 | 452.65 | 453.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2024-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 09:15:00 | 445.90 | 452.65 | 453.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-18 10:15:00 | 438.40 | 445.81 | 448.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 13:15:00 | 442.20 | 439.29 | 442.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-19 13:30:00 | 441.60 | 439.29 | 442.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 14:15:00 | 442.05 | 439.84 | 442.46 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2024-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 12:15:00 | 447.10 | 444.03 | 443.72 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 14:15:00 | 439.75 | 444.18 | 444.38 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 15:15:00 | 446.95 | 444.20 | 443.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 09:15:00 | 452.15 | 445.79 | 444.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-30 10:15:00 | 483.25 | 483.43 | 476.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-30 11:15:00 | 481.95 | 483.43 | 476.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 13:15:00 | 528.75 | 530.49 | 524.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 13:30:00 | 528.95 | 530.49 | 524.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 14:15:00 | 525.25 | 529.44 | 525.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 15:00:00 | 525.25 | 529.44 | 525.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 15:15:00 | 529.95 | 529.54 | 525.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-10 09:15:00 | 530.30 | 529.54 | 525.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 532.35 | 530.10 | 526.09 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2024-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 09:15:00 | 518.40 | 525.31 | 525.61 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 11:15:00 | 527.00 | 522.11 | 521.67 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 09:15:00 | 519.05 | 522.30 | 522.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 09:15:00 | 513.10 | 516.64 | 518.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 09:15:00 | 517.50 | 513.99 | 515.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 09:15:00 | 517.50 | 513.99 | 515.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 517.50 | 513.99 | 515.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 10:00:00 | 517.50 | 513.99 | 515.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 527.80 | 516.75 | 516.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:00:00 | 527.80 | 516.75 | 516.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2024-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 11:15:00 | 535.55 | 520.51 | 518.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 11:15:00 | 540.00 | 533.61 | 527.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 09:15:00 | 530.05 | 535.88 | 531.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 530.05 | 535.88 | 531.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 530.05 | 535.88 | 531.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 11:30:00 | 540.80 | 536.98 | 532.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 12:00:00 | 542.10 | 536.98 | 532.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 14:45:00 | 543.75 | 538.07 | 534.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 09:15:00 | 526.70 | 537.44 | 534.64 | SL hit (close<static) qty=1.00 sl=529.10 alert=retest2 |

### Cycle 81 — SELL (started 2024-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 12:15:00 | 524.35 | 531.26 | 532.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 15:15:00 | 516.10 | 525.82 | 528.04 | Break + close below crossover candle low |

### Cycle 82 — BUY (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 09:15:00 | 596.40 | 539.93 | 534.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 09:15:00 | 630.75 | 596.93 | 571.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 623.05 | 629.33 | 615.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 623.05 | 629.33 | 615.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 623.05 | 629.33 | 615.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 623.05 | 629.33 | 615.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 619.95 | 627.46 | 615.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 619.95 | 627.46 | 615.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 09:15:00 | 726.00 | 726.24 | 708.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 09:45:00 | 719.50 | 726.24 | 708.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 707.90 | 734.53 | 726.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 15:00:00 | 707.90 | 734.53 | 726.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 712.00 | 730.02 | 725.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 717.70 | 730.02 | 725.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 11:15:00 | 711.45 | 720.58 | 721.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2024-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 11:15:00 | 711.45 | 720.58 | 721.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-13 09:15:00 | 706.90 | 715.41 | 718.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-13 15:15:00 | 710.00 | 707.49 | 712.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-14 09:15:00 | 713.00 | 707.49 | 712.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 711.30 | 708.25 | 712.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 10:15:00 | 716.00 | 708.25 | 712.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 10:15:00 | 716.85 | 709.97 | 712.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 10:45:00 | 715.30 | 709.97 | 712.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 11:15:00 | 713.35 | 710.65 | 712.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-14 12:15:00 | 711.85 | 710.65 | 712.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-14 14:15:00 | 711.10 | 711.54 | 712.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-18 09:15:00 | 721.00 | 712.80 | 713.03 | SL hit (close>static) qty=1.00 sl=718.40 alert=retest2 |

### Cycle 84 — BUY (started 2024-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 10:15:00 | 737.95 | 717.83 | 715.29 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2024-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 12:15:00 | 718.75 | 722.09 | 722.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 13:15:00 | 713.60 | 720.39 | 721.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 10:15:00 | 714.70 | 714.39 | 717.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-21 11:00:00 | 714.70 | 714.39 | 717.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 714.35 | 714.38 | 717.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 11:30:00 | 718.45 | 714.38 | 717.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 715.15 | 714.53 | 717.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 14:30:00 | 712.30 | 712.00 | 715.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 15:00:00 | 700.10 | 712.00 | 715.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 09:15:00 | 726.25 | 712.92 | 715.39 | SL hit (close>static) qty=1.00 sl=718.80 alert=retest2 |

### Cycle 86 — BUY (started 2024-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 12:15:00 | 727.00 | 718.85 | 717.75 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 13:15:00 | 713.45 | 717.25 | 717.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 13:15:00 | 705.20 | 712.02 | 714.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 10:15:00 | 697.00 | 695.90 | 702.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 10:15:00 | 697.00 | 695.90 | 702.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 697.00 | 695.90 | 702.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 11:30:00 | 690.35 | 694.69 | 701.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 09:15:00 | 762.95 | 706.02 | 703.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 09:15:00 | 762.95 | 706.02 | 703.49 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-07-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 12:15:00 | 727.25 | 731.32 | 731.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 14:15:00 | 724.50 | 729.19 | 730.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 09:15:00 | 733.75 | 729.51 | 730.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 09:15:00 | 733.75 | 729.51 | 730.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 733.75 | 729.51 | 730.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:45:00 | 731.45 | 729.51 | 730.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 10:15:00 | 767.95 | 737.19 | 733.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 09:15:00 | 782.65 | 752.84 | 743.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 11:15:00 | 766.45 | 768.67 | 759.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-10 12:00:00 | 766.45 | 768.67 | 759.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 765.00 | 768.63 | 763.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:45:00 | 763.85 | 768.63 | 763.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 766.10 | 768.13 | 763.74 | EMA400 retest candle locked (from upside) |

### Cycle 91 — SELL (started 2024-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 12:15:00 | 758.55 | 763.22 | 763.39 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-07-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 14:15:00 | 764.90 | 763.51 | 763.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 09:15:00 | 772.55 | 765.47 | 764.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 14:15:00 | 762.20 | 766.11 | 765.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 14:15:00 | 762.20 | 766.11 | 765.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 762.20 | 766.11 | 765.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 15:00:00 | 762.20 | 766.11 | 765.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 765.00 | 765.89 | 765.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 09:15:00 | 769.70 | 765.89 | 765.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-25 10:15:00 | 796.20 | 796.89 | 796.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 10:15:00 | 796.20 | 796.89 | 796.97 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 12:15:00 | 799.10 | 797.39 | 797.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 14:15:00 | 801.45 | 798.79 | 797.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 09:15:00 | 796.85 | 799.23 | 798.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 09:15:00 | 796.85 | 799.23 | 798.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 796.85 | 799.23 | 798.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:00:00 | 796.85 | 799.23 | 798.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 794.05 | 798.20 | 797.92 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2024-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-26 11:15:00 | 792.00 | 796.96 | 797.38 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 09:15:00 | 800.50 | 797.62 | 797.42 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 15:15:00 | 796.20 | 797.57 | 797.57 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 10:15:00 | 801.25 | 798.10 | 797.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 09:15:00 | 804.60 | 799.63 | 798.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 13:15:00 | 826.50 | 826.87 | 816.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-01 14:00:00 | 826.50 | 826.87 | 816.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 793.90 | 819.53 | 815.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 10:00:00 | 793.90 | 819.53 | 815.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — SELL (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 10:15:00 | 783.00 | 812.23 | 812.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 15:15:00 | 777.00 | 791.08 | 800.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 765.10 | 758.02 | 774.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 765.10 | 758.02 | 774.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 765.10 | 758.02 | 774.18 | EMA400 retest candle locked (from downside) |

### Cycle 100 — BUY (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 13:15:00 | 784.00 | 775.81 | 775.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 15:15:00 | 788.15 | 779.59 | 777.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 778.85 | 782.47 | 780.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 14:15:00 | 778.85 | 782.47 | 780.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 778.85 | 782.47 | 780.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 15:00:00 | 778.85 | 782.47 | 780.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 778.00 | 781.58 | 779.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 785.00 | 781.58 | 779.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 13:15:00 | 802.55 | 810.72 | 811.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2024-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 13:15:00 | 802.55 | 810.72 | 811.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 11:15:00 | 799.95 | 805.47 | 808.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 09:15:00 | 802.20 | 801.72 | 805.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 09:15:00 | 802.20 | 801.72 | 805.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 802.20 | 801.72 | 805.09 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2024-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 13:15:00 | 811.00 | 807.46 | 807.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-23 09:15:00 | 822.05 | 811.64 | 809.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 11:15:00 | 812.00 | 812.72 | 810.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 12:00:00 | 812.00 | 812.72 | 810.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 808.95 | 811.97 | 810.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:45:00 | 805.45 | 811.97 | 810.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 808.85 | 811.35 | 810.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 813.10 | 810.81 | 810.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 12:15:00 | 814.45 | 826.98 | 827.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 12:15:00 | 814.45 | 826.98 | 827.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 14:15:00 | 812.90 | 822.11 | 825.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 13:15:00 | 820.25 | 817.26 | 820.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 13:15:00 | 820.25 | 817.26 | 820.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 13:15:00 | 820.25 | 817.26 | 820.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 13:45:00 | 823.90 | 817.26 | 820.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 811.45 | 816.09 | 819.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 09:15:00 | 792.05 | 815.48 | 819.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 11:45:00 | 804.05 | 811.37 | 816.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 14:30:00 | 810.60 | 811.07 | 814.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 09:15:00 | 830.50 | 814.95 | 815.92 | SL hit (close>static) qty=1.00 sl=823.40 alert=retest2 |

### Cycle 104 — BUY (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 10:15:00 | 834.20 | 818.80 | 817.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 10:15:00 | 837.10 | 827.87 | 824.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 13:15:00 | 836.25 | 837.50 | 832.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-06 14:15:00 | 829.75 | 837.50 | 832.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 833.50 | 836.70 | 832.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:30:00 | 827.90 | 836.70 | 832.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 832.00 | 835.76 | 832.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:15:00 | 833.60 | 835.76 | 832.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 833.80 | 835.37 | 832.91 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2024-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 14:15:00 | 826.50 | 831.38 | 831.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-10 10:15:00 | 820.00 | 827.74 | 829.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 11:15:00 | 746.70 | 743.52 | 756.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-18 11:45:00 | 748.25 | 743.52 | 756.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 751.50 | 745.73 | 752.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:30:00 | 743.90 | 749.38 | 750.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 10:15:00 | 760.00 | 752.93 | 752.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 760.00 | 752.93 | 752.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 14:15:00 | 763.95 | 757.59 | 754.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 11:15:00 | 759.00 | 760.18 | 757.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 11:15:00 | 759.00 | 760.18 | 757.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 759.00 | 760.18 | 757.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 12:00:00 | 759.00 | 760.18 | 757.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 760.75 | 761.15 | 758.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 15:00:00 | 760.75 | 761.15 | 758.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 742.95 | 757.88 | 757.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 742.95 | 757.88 | 757.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2024-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 10:15:00 | 742.20 | 754.74 | 756.06 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2024-09-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 13:15:00 | 754.30 | 748.90 | 748.36 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 10:15:00 | 744.95 | 747.54 | 747.89 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 13:15:00 | 752.55 | 748.46 | 748.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 14:15:00 | 766.70 | 752.10 | 749.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 10:15:00 | 751.60 | 754.17 | 751.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 10:15:00 | 751.60 | 754.17 | 751.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 751.60 | 754.17 | 751.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 11:00:00 | 751.60 | 754.17 | 751.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 11:15:00 | 757.85 | 754.91 | 752.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:15:00 | 754.80 | 754.91 | 752.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 754.45 | 754.82 | 752.36 | EMA400 retest candle locked (from upside) |

### Cycle 111 — SELL (started 2024-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 12:15:00 | 743.65 | 751.68 | 751.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 13:15:00 | 742.10 | 749.76 | 751.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 10:15:00 | 750.00 | 748.62 | 749.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 10:15:00 | 750.00 | 748.62 | 749.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 750.00 | 748.62 | 749.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:30:00 | 748.90 | 748.62 | 749.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 744.95 | 747.89 | 749.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 12:45:00 | 741.50 | 746.88 | 748.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 14:15:00 | 741.00 | 745.92 | 748.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:15:00 | 741.50 | 744.49 | 746.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 11:15:00 | 748.90 | 742.70 | 741.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 748.90 | 742.70 | 741.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 12:15:00 | 752.00 | 744.56 | 742.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 10:15:00 | 744.90 | 747.54 | 745.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 10:15:00 | 744.90 | 747.54 | 745.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 744.90 | 747.54 | 745.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:00:00 | 744.90 | 747.54 | 745.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 744.05 | 746.84 | 745.14 | EMA400 retest candle locked (from upside) |

### Cycle 113 — SELL (started 2024-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 09:15:00 | 734.45 | 742.95 | 743.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 10:15:00 | 733.25 | 741.01 | 742.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 09:15:00 | 730.00 | 725.18 | 729.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 09:15:00 | 730.00 | 725.18 | 729.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 730.00 | 725.18 | 729.90 | EMA400 retest candle locked (from downside) |

### Cycle 114 — BUY (started 2024-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 13:15:00 | 739.60 | 733.39 | 732.80 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 09:15:00 | 722.75 | 732.27 | 732.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 11:15:00 | 720.75 | 728.63 | 730.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 14:15:00 | 742.95 | 729.42 | 730.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 14:15:00 | 742.95 | 729.42 | 730.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 742.95 | 729.42 | 730.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 15:00:00 | 742.95 | 729.42 | 730.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2024-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 15:15:00 | 738.65 | 731.27 | 731.15 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 721.40 | 729.29 | 730.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 10:15:00 | 715.60 | 726.55 | 728.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 15:15:00 | 698.00 | 696.71 | 703.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-22 09:15:00 | 690.25 | 696.71 | 703.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 684.40 | 694.25 | 701.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 10:15:00 | 682.30 | 694.25 | 701.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 10:15:00 | 648.18 | 661.02 | 672.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-25 13:15:00 | 645.45 | 641.84 | 651.96 | SL hit (close>ema200) qty=0.50 sl=641.84 alert=retest2 |

### Cycle 118 — BUY (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 10:15:00 | 656.50 | 651.74 | 651.68 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 09:15:00 | 650.80 | 652.09 | 652.14 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 652.75 | 652.22 | 652.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 11:15:00 | 665.75 | 654.93 | 653.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 674.70 | 693.89 | 681.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 674.70 | 693.89 | 681.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 674.70 | 693.89 | 681.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 674.70 | 693.89 | 681.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 671.30 | 689.38 | 680.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 671.30 | 689.38 | 680.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 14:15:00 | 675.30 | 681.94 | 679.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 15:00:00 | 675.30 | 681.94 | 679.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 15:15:00 | 682.00 | 681.95 | 679.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:45:00 | 684.05 | 683.49 | 680.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 15:15:00 | 688.45 | 686.54 | 685.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-07 10:15:00 | 677.55 | 683.82 | 684.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 10:15:00 | 677.55 | 683.82 | 684.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 11:15:00 | 672.10 | 681.47 | 683.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 15:15:00 | 670.75 | 670.31 | 674.66 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-11 09:15:00 | 662.75 | 670.31 | 674.66 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 675.85 | 671.42 | 674.77 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-11 09:15:00 | 675.85 | 671.42 | 674.77 | SL hit (close>ema400) qty=1.00 sl=674.77 alert=retest1 |

### Cycle 122 — BUY (started 2024-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 09:15:00 | 684.20 | 675.62 | 675.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-12 11:15:00 | 697.90 | 681.39 | 678.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-12 15:15:00 | 673.30 | 684.66 | 681.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 15:15:00 | 673.30 | 684.66 | 681.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 15:15:00 | 673.30 | 684.66 | 681.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 09:45:00 | 670.00 | 681.06 | 679.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2024-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 10:15:00 | 664.75 | 677.80 | 678.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 11:15:00 | 658.85 | 674.01 | 676.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 11:15:00 | 622.45 | 622.33 | 633.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 14:15:00 | 644.35 | 627.61 | 633.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 644.35 | 627.61 | 633.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 15:00:00 | 644.35 | 627.61 | 633.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 643.90 | 630.87 | 634.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 09:15:00 | 634.40 | 630.87 | 634.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 13:15:00 | 637.70 | 633.29 | 632.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 637.70 | 633.29 | 632.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 15:15:00 | 640.00 | 635.37 | 633.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 654.60 | 667.50 | 653.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 09:15:00 | 654.60 | 667.50 | 653.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 654.60 | 667.50 | 653.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 10:00:00 | 654.60 | 667.50 | 653.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 661.30 | 666.26 | 654.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 10:30:00 | 654.80 | 666.26 | 654.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 659.75 | 662.76 | 656.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 14:45:00 | 659.85 | 662.76 | 656.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 660.00 | 662.21 | 656.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:15:00 | 656.45 | 662.21 | 656.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 655.00 | 660.77 | 656.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:00:00 | 655.00 | 660.77 | 656.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 653.90 | 659.39 | 656.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 12:00:00 | 656.10 | 658.74 | 656.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 14:15:00 | 656.50 | 657.33 | 655.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 14:45:00 | 659.60 | 658.18 | 656.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 09:30:00 | 656.25 | 673.63 | 669.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 657.95 | 670.49 | 668.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:00:00 | 657.95 | 670.49 | 668.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-02 12:15:00 | 655.60 | 665.68 | 666.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2024-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 12:15:00 | 655.60 | 665.68 | 666.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 13:15:00 | 644.70 | 661.48 | 664.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 14:15:00 | 641.10 | 638.16 | 647.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-03 15:00:00 | 641.10 | 638.16 | 647.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 638.15 | 634.15 | 637.42 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2024-12-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 15:15:00 | 641.40 | 639.06 | 638.77 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 624.10 | 636.07 | 637.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 12:15:00 | 622.15 | 629.74 | 633.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 09:15:00 | 628.75 | 626.38 | 630.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 09:15:00 | 628.75 | 626.38 | 630.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 628.75 | 626.38 | 630.63 | EMA400 retest candle locked (from downside) |

### Cycle 128 — BUY (started 2024-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 13:15:00 | 638.25 | 632.53 | 632.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 14:15:00 | 642.40 | 634.51 | 633.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 09:15:00 | 636.15 | 636.42 | 634.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 09:15:00 | 636.15 | 636.42 | 634.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 636.15 | 636.42 | 634.56 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2024-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 13:15:00 | 623.45 | 631.84 | 632.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 14:15:00 | 622.60 | 629.99 | 631.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 09:15:00 | 596.15 | 596.00 | 605.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-16 09:30:00 | 599.15 | 596.00 | 605.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 597.80 | 595.98 | 601.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 14:30:00 | 599.25 | 595.98 | 601.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 586.00 | 594.30 | 599.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 11:30:00 | 584.50 | 591.08 | 597.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 12:00:00 | 583.20 | 587.55 | 592.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:15:00 | 582.10 | 587.27 | 590.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-19 13:15:00 | 596.50 | 592.54 | 592.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2024-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 13:15:00 | 596.50 | 592.54 | 592.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-20 09:15:00 | 602.45 | 596.25 | 594.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 12:15:00 | 598.60 | 599.60 | 596.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 13:00:00 | 598.60 | 599.60 | 596.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 592.90 | 598.26 | 596.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 14:00:00 | 592.90 | 598.26 | 596.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 590.45 | 596.70 | 595.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 590.45 | 596.70 | 595.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 593.00 | 595.96 | 595.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:15:00 | 578.10 | 595.96 | 595.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2024-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 09:15:00 | 572.35 | 591.24 | 593.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 13:15:00 | 567.60 | 579.14 | 586.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 579.55 | 575.82 | 582.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 10:00:00 | 579.55 | 575.82 | 582.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 11:15:00 | 578.70 | 576.62 | 581.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 11:30:00 | 580.50 | 576.62 | 581.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 13:15:00 | 592.70 | 580.47 | 582.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 14:00:00 | 592.70 | 580.47 | 582.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 14:15:00 | 597.10 | 583.80 | 584.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 15:00:00 | 597.10 | 583.80 | 584.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2024-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 15:15:00 | 600.00 | 587.04 | 585.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 13:15:00 | 602.20 | 596.13 | 593.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 11:15:00 | 598.00 | 598.61 | 596.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-01 11:45:00 | 596.80 | 598.61 | 596.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 12:15:00 | 607.95 | 600.48 | 597.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 13:15:00 | 609.30 | 600.48 | 597.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 14:30:00 | 610.10 | 603.46 | 599.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 15:15:00 | 609.90 | 603.46 | 599.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 14:45:00 | 610.40 | 607.17 | 603.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 605.20 | 607.05 | 604.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 09:30:00 | 604.95 | 607.05 | 604.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 607.95 | 607.23 | 604.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:30:00 | 606.25 | 607.23 | 604.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 596.90 | 609.21 | 607.27 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-06 09:15:00 | 596.90 | 609.21 | 607.27 | SL hit (close<static) qty=1.00 sl=597.05 alert=retest2 |

### Cycle 133 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 591.15 | 605.60 | 605.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 586.65 | 598.56 | 602.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 595.05 | 593.95 | 598.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 09:30:00 | 597.80 | 593.95 | 598.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 588.15 | 584.09 | 587.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:00:00 | 588.15 | 584.09 | 587.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 596.05 | 586.48 | 588.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 11:00:00 | 596.05 | 586.48 | 588.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — BUY (started 2025-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 11:15:00 | 608.50 | 590.89 | 590.45 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 11:15:00 | 590.95 | 591.62 | 591.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 11:15:00 | 584.00 | 588.34 | 589.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 11:15:00 | 564.70 | 563.01 | 570.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-15 11:15:00 | 564.70 | 563.01 | 570.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 564.70 | 563.01 | 570.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 12:00:00 | 564.70 | 563.01 | 570.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 566.30 | 565.17 | 568.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 09:30:00 | 570.65 | 565.17 | 568.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 15:15:00 | 565.80 | 565.50 | 567.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 09:15:00 | 565.20 | 565.50 | 567.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-17 10:15:00 | 569.50 | 566.38 | 567.43 | SL hit (close>static) qty=1.00 sl=567.40 alert=retest2 |

### Cycle 136 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 559.40 | 557.84 | 557.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 13:15:00 | 562.75 | 558.82 | 558.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 554.30 | 558.72 | 558.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 554.30 | 558.72 | 558.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 554.30 | 558.72 | 558.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 554.30 | 558.72 | 558.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 10:15:00 | 552.35 | 557.45 | 557.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 549.90 | 554.54 | 556.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 09:15:00 | 540.90 | 538.43 | 544.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 09:15:00 | 540.90 | 538.43 | 544.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 540.90 | 538.43 | 544.33 | EMA400 retest candle locked (from downside) |

### Cycle 138 — BUY (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 13:15:00 | 556.80 | 546.83 | 546.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 10:15:00 | 563.25 | 554.15 | 550.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 11:15:00 | 564.75 | 566.72 | 560.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 12:00:00 | 564.75 | 566.72 | 560.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 594.35 | 607.92 | 601.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:00:00 | 594.35 | 607.92 | 601.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 591.70 | 604.67 | 600.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 11:00:00 | 591.70 | 604.67 | 600.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2025-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 13:15:00 | 582.80 | 595.30 | 596.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 09:15:00 | 575.85 | 585.43 | 589.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 14:15:00 | 575.05 | 572.38 | 575.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 14:15:00 | 575.05 | 572.38 | 575.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 14:15:00 | 575.05 | 572.38 | 575.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 14:45:00 | 577.90 | 572.38 | 575.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 570.00 | 571.91 | 575.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 09:15:00 | 558.55 | 571.91 | 575.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:15:00 | 530.62 | 542.03 | 548.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 13:15:00 | 523.45 | 523.18 | 532.05 | SL hit (close>ema200) qty=0.50 sl=523.18 alert=retest2 |

### Cycle 140 — BUY (started 2025-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 13:15:00 | 545.55 | 533.09 | 532.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 549.85 | 540.53 | 537.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 553.10 | 553.11 | 547.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:15:00 | 552.30 | 553.11 | 547.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 548.65 | 552.67 | 549.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:30:00 | 550.35 | 552.67 | 549.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 555.00 | 553.13 | 549.56 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2025-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 13:15:00 | 548.95 | 550.20 | 550.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 14:15:00 | 547.00 | 549.56 | 550.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 547.35 | 546.02 | 547.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 14:15:00 | 547.35 | 546.02 | 547.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 547.35 | 546.02 | 547.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 15:00:00 | 547.35 | 546.02 | 547.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 15:15:00 | 543.00 | 545.41 | 547.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 532.00 | 545.41 | 547.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-06 09:15:00 | 538.55 | 528.57 | 527.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 09:15:00 | 538.55 | 528.57 | 527.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 10:15:00 | 544.00 | 535.08 | 531.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-11 09:15:00 | 550.75 | 561.00 | 553.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 09:15:00 | 550.75 | 561.00 | 553.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 550.75 | 561.00 | 553.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 14:00:00 | 566.75 | 560.04 | 556.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 15:00:00 | 567.05 | 561.44 | 557.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 11:15:00 | 553.75 | 558.42 | 558.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 11:15:00 | 553.75 | 558.42 | 558.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 12:15:00 | 549.80 | 556.70 | 557.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 552.95 | 552.12 | 554.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-18 09:45:00 | 552.70 | 552.12 | 554.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 557.90 | 553.27 | 555.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 11:00:00 | 557.90 | 553.27 | 555.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 11:15:00 | 555.65 | 553.75 | 555.20 | EMA400 retest candle locked (from downside) |

### Cycle 144 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 558.40 | 556.13 | 555.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 12:15:00 | 563.60 | 558.67 | 557.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 15:15:00 | 554.90 | 558.95 | 557.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-19 15:15:00 | 554.90 | 558.95 | 557.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 15:15:00 | 554.90 | 558.95 | 557.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 09:15:00 | 561.65 | 558.95 | 557.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 09:45:00 | 561.20 | 558.87 | 557.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 10:45:00 | 562.15 | 560.05 | 558.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 10:15:00 | 562.35 | 569.27 | 569.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2025-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 10:15:00 | 562.35 | 569.27 | 569.84 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 12:15:00 | 575.60 | 570.41 | 570.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-25 13:15:00 | 579.85 | 572.30 | 571.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 15:15:00 | 572.20 | 572.66 | 571.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-26 09:15:00 | 576.40 | 572.66 | 571.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 569.50 | 572.03 | 571.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:00:00 | 569.50 | 572.03 | 571.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 569.50 | 571.52 | 571.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:30:00 | 569.50 | 571.52 | 571.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 12:15:00 | 566.75 | 570.13 | 570.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 13:15:00 | 564.50 | 569.01 | 570.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 12:15:00 | 564.65 | 561.21 | 564.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 12:15:00 | 564.65 | 561.21 | 564.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 564.65 | 561.21 | 564.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:00:00 | 564.65 | 561.21 | 564.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 562.45 | 561.46 | 564.47 | EMA400 retest candle locked (from downside) |

### Cycle 148 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 579.15 | 567.28 | 566.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 10:15:00 | 594.00 | 578.85 | 574.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 09:15:00 | 574.50 | 582.50 | 579.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 574.50 | 582.50 | 579.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 574.50 | 582.50 | 579.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:30:00 | 575.20 | 582.50 | 579.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 574.05 | 580.81 | 578.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 11:15:00 | 572.00 | 580.81 | 578.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2025-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 13:15:00 | 573.35 | 576.89 | 577.18 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 10:15:00 | 587.00 | 578.40 | 577.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-04 15:15:00 | 594.70 | 586.57 | 582.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 580.05 | 585.26 | 582.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 580.05 | 585.26 | 582.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 580.05 | 585.26 | 582.22 | EMA400 retest candle locked (from upside) |

### Cycle 151 — SELL (started 2025-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-08 10:15:00 | 576.20 | 581.80 | 581.94 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 12:15:00 | 586.85 | 582.67 | 582.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 09:15:00 | 593.50 | 585.33 | 583.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 09:15:00 | 596.70 | 597.35 | 592.18 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-11 11:45:00 | 606.80 | 599.54 | 594.09 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-11 13:45:00 | 607.30 | 601.42 | 595.93 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-11 14:30:00 | 608.50 | 603.08 | 597.18 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 09:15:00 | 608.05 | 602.86 | 597.62 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 14:15:00 | 615.00 | 617.99 | 614.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 15:15:00 | 614.95 | 617.99 | 614.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 614.95 | 617.38 | 614.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:15:00 | 612.00 | 617.38 | 614.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 613.05 | 616.51 | 614.04 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-21 09:15:00 | 613.05 | 616.51 | 614.04 | SL hit (close<ema400) qty=1.00 sl=614.04 alert=retest1 |

### Cycle 153 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 624.90 | 635.91 | 636.71 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 12:15:00 | 634.50 | 630.24 | 629.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 15:15:00 | 636.75 | 632.91 | 631.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 625.85 | 631.50 | 630.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 625.85 | 631.50 | 630.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 625.85 | 631.50 | 630.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 11:15:00 | 633.40 | 631.29 | 630.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 13:15:00 | 627.65 | 630.40 | 630.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 13:15:00 | 627.65 | 630.40 | 630.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 14:15:00 | 624.40 | 629.20 | 629.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 631.90 | 628.27 | 629.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 631.90 | 628.27 | 629.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 631.90 | 628.27 | 629.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:15:00 | 634.15 | 628.27 | 629.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 635.25 | 629.66 | 629.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:30:00 | 626.85 | 629.20 | 629.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 15:15:00 | 628.00 | 629.01 | 629.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 633.95 | 629.84 | 629.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 633.95 | 629.84 | 629.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 11:15:00 | 637.50 | 631.82 | 630.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-05 14:15:00 | 631.90 | 632.77 | 631.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 14:15:00 | 631.90 | 632.77 | 631.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 631.90 | 632.77 | 631.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 15:00:00 | 631.90 | 632.77 | 631.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 629.50 | 632.12 | 631.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:15:00 | 627.55 | 632.12 | 631.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 630.40 | 631.77 | 631.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 10:15:00 | 642.05 | 631.77 | 631.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 12:00:00 | 635.50 | 633.25 | 632.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 14:15:00 | 627.70 | 631.35 | 631.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 14:15:00 | 627.70 | 631.35 | 631.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 15:15:00 | 622.00 | 629.48 | 630.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 633.90 | 629.32 | 630.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 10:15:00 | 633.90 | 629.32 | 630.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 633.90 | 629.32 | 630.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:00:00 | 633.90 | 629.32 | 630.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 633.95 | 630.24 | 630.55 | EMA400 retest candle locked (from downside) |

### Cycle 158 — BUY (started 2025-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 12:15:00 | 633.85 | 630.97 | 630.85 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 10:15:00 | 626.65 | 630.34 | 630.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 622.95 | 627.76 | 629.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 12:15:00 | 624.30 | 622.61 | 625.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 12:15:00 | 624.30 | 622.61 | 625.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 12:15:00 | 624.30 | 622.61 | 625.50 | EMA400 retest candle locked (from downside) |

### Cycle 160 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 634.95 | 628.06 | 627.41 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 12:15:00 | 621.60 | 630.23 | 630.24 | EMA200 below EMA400 |

### Cycle 162 — BUY (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 12:15:00 | 630.90 | 629.49 | 629.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 14:15:00 | 639.40 | 632.58 | 630.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 628.70 | 632.99 | 631.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 628.70 | 632.99 | 631.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 628.70 | 632.99 | 631.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 10:00:00 | 628.70 | 632.99 | 631.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 627.40 | 631.87 | 631.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:00:00 | 627.40 | 631.87 | 631.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 630.00 | 631.50 | 630.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:30:00 | 628.40 | 631.50 | 630.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 632.20 | 631.61 | 631.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:15:00 | 624.80 | 631.61 | 631.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 630.00 | 631.29 | 630.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:30:00 | 630.20 | 631.29 | 630.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 630.05 | 631.04 | 630.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 634.20 | 631.04 | 630.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 13:15:00 | 625.00 | 630.78 | 630.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 13:15:00 | 625.00 | 630.78 | 630.99 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 14:15:00 | 636.85 | 632.00 | 631.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 639.55 | 633.80 | 632.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 11:15:00 | 634.25 | 634.73 | 633.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 11:15:00 | 634.25 | 634.73 | 633.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 634.25 | 634.73 | 633.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 12:00:00 | 634.25 | 634.73 | 633.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 621.05 | 634.48 | 634.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 620.00 | 634.48 | 634.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — SELL (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 10:15:00 | 620.65 | 631.72 | 632.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 11:15:00 | 618.70 | 629.11 | 631.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 14:15:00 | 604.55 | 601.85 | 608.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 15:00:00 | 604.55 | 601.85 | 608.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 604.20 | 602.21 | 607.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 15:00:00 | 594.45 | 599.59 | 603.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 12:30:00 | 594.65 | 597.97 | 601.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 10:45:00 | 594.50 | 594.10 | 597.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 12:15:00 | 596.00 | 584.99 | 583.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — BUY (started 2025-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 12:15:00 | 596.00 | 584.99 | 583.50 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 579.60 | 584.19 | 584.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 15:15:00 | 578.00 | 582.12 | 583.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 10:15:00 | 581.80 | 581.75 | 583.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 14:15:00 | 582.65 | 581.96 | 582.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 582.65 | 581.96 | 582.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 582.65 | 581.96 | 582.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 584.10 | 582.39 | 582.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 582.60 | 582.39 | 582.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 580.05 | 581.92 | 582.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 14:00:00 | 578.50 | 580.49 | 581.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 09:45:00 | 579.25 | 579.10 | 580.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 14:15:00 | 583.10 | 581.20 | 581.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 14:15:00 | 583.10 | 581.20 | 581.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 15:15:00 | 583.50 | 581.66 | 581.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 588.15 | 588.39 | 585.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 10:00:00 | 588.15 | 588.39 | 585.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 584.50 | 587.61 | 585.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 584.50 | 587.61 | 585.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 583.15 | 586.72 | 585.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 583.15 | 586.72 | 585.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 584.15 | 586.21 | 585.36 | EMA400 retest candle locked (from upside) |

### Cycle 169 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 574.35 | 583.01 | 584.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 570.15 | 579.62 | 582.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 573.45 | 573.35 | 577.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 14:45:00 | 571.55 | 573.35 | 577.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 569.65 | 572.54 | 576.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 14:15:00 | 567.10 | 572.39 | 573.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:30:00 | 566.25 | 569.83 | 571.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 10:15:00 | 571.00 | 561.52 | 561.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 571.00 | 561.52 | 561.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 14:15:00 | 573.90 | 568.37 | 565.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 10:15:00 | 569.55 | 571.09 | 569.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 10:15:00 | 569.55 | 571.09 | 569.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 569.55 | 571.09 | 569.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:45:00 | 569.45 | 571.09 | 569.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 570.70 | 571.01 | 569.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 12:00:00 | 570.70 | 571.01 | 569.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 566.60 | 570.13 | 569.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 566.60 | 570.13 | 569.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 568.65 | 569.84 | 569.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 573.75 | 569.84 | 569.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 12:15:00 | 566.35 | 568.91 | 569.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — SELL (started 2025-06-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 12:15:00 | 566.35 | 568.91 | 569.05 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-06-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 15:15:00 | 571.55 | 569.33 | 569.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 09:15:00 | 576.60 | 570.78 | 569.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 11:15:00 | 560.65 | 570.22 | 569.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 11:15:00 | 560.65 | 570.22 | 569.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 560.65 | 570.22 | 569.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:00:00 | 560.65 | 570.22 | 569.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 12:15:00 | 557.70 | 567.72 | 568.73 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 09:15:00 | 570.30 | 563.39 | 562.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 10:15:00 | 578.70 | 566.46 | 564.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 11:15:00 | 571.45 | 573.29 | 570.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 11:15:00 | 571.45 | 573.29 | 570.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 571.45 | 573.29 | 570.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:30:00 | 569.55 | 573.29 | 570.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 572.65 | 573.16 | 570.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:30:00 | 571.15 | 573.16 | 570.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 580.00 | 574.93 | 572.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 10:15:00 | 584.10 | 574.93 | 572.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 581.55 | 589.19 | 589.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — SELL (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 09:15:00 | 581.55 | 589.19 | 589.98 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 11:15:00 | 591.00 | 586.33 | 585.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 12:15:00 | 592.75 | 589.11 | 587.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 09:15:00 | 583.55 | 589.13 | 588.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 583.55 | 589.13 | 588.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 583.55 | 589.13 | 588.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:45:00 | 583.85 | 589.13 | 588.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 586.55 | 588.61 | 588.30 | EMA400 retest candle locked (from upside) |

### Cycle 177 — SELL (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 10:15:00 | 585.70 | 588.24 | 588.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 582.60 | 586.80 | 587.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 12:15:00 | 580.20 | 579.75 | 582.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 12:30:00 | 580.05 | 579.75 | 582.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 575.00 | 576.24 | 578.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:30:00 | 577.55 | 576.24 | 578.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 565.05 | 562.24 | 566.21 | EMA400 retest candle locked (from downside) |

### Cycle 178 — BUY (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 09:15:00 | 580.50 | 568.49 | 567.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 10:15:00 | 581.20 | 571.03 | 568.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 14:15:00 | 611.65 | 611.88 | 603.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-04 14:30:00 | 611.70 | 611.88 | 603.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 605.70 | 609.98 | 604.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 11:00:00 | 605.70 | 609.98 | 604.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 610.25 | 610.04 | 605.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 11:30:00 | 609.65 | 610.04 | 605.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 602.15 | 609.61 | 606.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 602.15 | 609.61 | 606.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 596.55 | 607.00 | 605.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 596.55 | 607.00 | 605.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 596.45 | 604.89 | 605.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 588.40 | 598.67 | 601.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 580.50 | 578.17 | 583.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 09:30:00 | 578.05 | 578.17 | 583.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 579.40 | 578.79 | 583.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:45:00 | 582.60 | 578.79 | 583.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 582.25 | 579.83 | 582.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:45:00 | 581.95 | 579.83 | 582.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 583.80 | 580.62 | 582.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:15:00 | 582.50 | 580.62 | 582.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 580.05 | 580.51 | 582.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 11:00:00 | 574.80 | 579.37 | 581.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 15:00:00 | 576.80 | 577.92 | 580.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 15:15:00 | 582.00 | 580.56 | 580.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — BUY (started 2025-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 15:15:00 | 582.00 | 580.56 | 580.44 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 579.80 | 580.33 | 580.35 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 12:15:00 | 581.00 | 580.41 | 580.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 14:15:00 | 582.30 | 580.84 | 580.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 15:15:00 | 580.00 | 580.67 | 580.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 15:15:00 | 580.00 | 580.67 | 580.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 580.00 | 580.67 | 580.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 584.50 | 580.67 | 580.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 10:15:00 | 608.35 | 613.04 | 613.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — SELL (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 10:15:00 | 608.35 | 613.04 | 613.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 11:15:00 | 604.85 | 611.40 | 612.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 570.00 | 569.02 | 576.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 570.00 | 569.02 | 576.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 576.75 | 572.93 | 575.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 575.30 | 572.93 | 575.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 577.40 | 573.82 | 575.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:45:00 | 577.00 | 573.82 | 575.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 575.55 | 574.17 | 575.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 13:00:00 | 574.15 | 574.16 | 575.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 13:15:00 | 580.00 | 575.33 | 576.06 | SL hit (close>static) qty=1.00 sl=577.75 alert=retest2 |

### Cycle 184 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 582.00 | 576.67 | 576.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 590.15 | 580.22 | 578.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 11:15:00 | 591.25 | 592.27 | 587.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 11:45:00 | 590.30 | 592.27 | 587.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 604.15 | 606.80 | 601.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:45:00 | 599.20 | 606.80 | 601.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 602.50 | 605.94 | 601.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:45:00 | 601.75 | 605.94 | 601.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 604.00 | 605.15 | 602.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 601.30 | 605.15 | 602.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 603.65 | 604.85 | 602.47 | EMA400 retest candle locked (from upside) |

### Cycle 185 — SELL (started 2025-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 15:15:00 | 599.10 | 601.44 | 601.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 10:15:00 | 592.30 | 598.68 | 600.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 12:15:00 | 596.55 | 592.65 | 594.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 12:15:00 | 596.55 | 592.65 | 594.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 596.55 | 592.65 | 594.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:00:00 | 596.55 | 592.65 | 594.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 597.85 | 593.69 | 595.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:30:00 | 596.85 | 593.69 | 595.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 597.75 | 595.55 | 595.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:45:00 | 597.80 | 595.55 | 595.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — BUY (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 12:15:00 | 599.35 | 596.31 | 596.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 15:15:00 | 602.50 | 598.97 | 597.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 09:15:00 | 597.75 | 598.73 | 597.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 597.75 | 598.73 | 597.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 597.75 | 598.73 | 597.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 11:15:00 | 604.05 | 599.26 | 597.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 12:00:00 | 604.05 | 600.22 | 598.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:30:00 | 604.35 | 601.34 | 599.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 14:00:00 | 605.55 | 601.56 | 600.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 602.65 | 602.42 | 601.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 10:00:00 | 602.65 | 602.42 | 601.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 600.90 | 602.12 | 601.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:00:00 | 600.90 | 602.12 | 601.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 600.85 | 601.86 | 601.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:30:00 | 600.85 | 601.86 | 601.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 601.15 | 601.72 | 601.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 13:45:00 | 602.90 | 602.22 | 601.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 15:00:00 | 603.55 | 602.48 | 601.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 10:00:00 | 602.15 | 602.49 | 601.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 10:45:00 | 609.15 | 603.82 | 602.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 602.20 | 604.16 | 603.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:45:00 | 602.45 | 604.16 | 603.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 602.75 | 603.88 | 602.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:00:00 | 602.75 | 603.88 | 602.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 604.40 | 603.98 | 603.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 605.00 | 603.98 | 603.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 601.10 | 603.41 | 602.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 601.10 | 603.41 | 602.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-19 10:15:00 | 597.95 | 602.31 | 602.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — SELL (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 10:15:00 | 597.95 | 602.31 | 602.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 11:15:00 | 592.50 | 600.35 | 601.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 11:15:00 | 588.90 | 588.88 | 593.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 12:00:00 | 588.90 | 588.88 | 593.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 577.75 | 577.81 | 580.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 11:00:00 | 574.50 | 577.15 | 580.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 12:15:00 | 545.77 | 557.33 | 563.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 544.25 | 541.43 | 547.36 | SL hit (close>ema200) qty=0.50 sl=541.43 alert=retest2 |

### Cycle 188 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 550.70 | 547.80 | 547.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 11:15:00 | 553.35 | 548.91 | 548.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 10:15:00 | 552.95 | 553.46 | 551.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 10:30:00 | 553.00 | 553.46 | 551.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 550.45 | 552.86 | 551.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:45:00 | 550.40 | 552.86 | 551.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 550.30 | 552.34 | 551.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:15:00 | 552.15 | 551.75 | 550.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:45:00 | 551.65 | 551.72 | 550.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 15:15:00 | 552.30 | 551.72 | 550.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 543.70 | 550.21 | 550.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 543.70 | 550.21 | 550.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 541.00 | 548.37 | 549.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 543.60 | 543.44 | 545.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 11:00:00 | 543.60 | 543.44 | 545.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 542.10 | 541.96 | 543.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 13:30:00 | 538.80 | 541.03 | 542.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 14:45:00 | 537.75 | 540.12 | 541.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 12:15:00 | 538.05 | 536.52 | 537.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 13:15:00 | 541.10 | 538.14 | 538.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — BUY (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 13:15:00 | 541.10 | 538.14 | 538.13 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 10:15:00 | 535.10 | 537.90 | 538.09 | EMA200 below EMA400 |

### Cycle 192 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 540.00 | 538.14 | 538.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 541.65 | 539.19 | 538.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 09:15:00 | 541.10 | 545.11 | 542.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 541.10 | 545.11 | 542.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 541.10 | 545.11 | 542.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:00:00 | 541.10 | 545.11 | 542.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 543.30 | 544.75 | 542.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:00:00 | 543.30 | 544.75 | 542.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 544.40 | 544.68 | 542.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:45:00 | 544.20 | 544.68 | 542.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 541.25 | 544.09 | 543.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 15:00:00 | 541.25 | 544.09 | 543.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 541.45 | 543.56 | 542.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 545.70 | 543.79 | 543.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 10:00:00 | 544.75 | 544.03 | 543.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 11:30:00 | 542.50 | 544.93 | 544.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 13:15:00 | 541.30 | 544.10 | 544.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — SELL (started 2025-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 13:15:00 | 541.30 | 544.10 | 544.33 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 545.75 | 544.60 | 544.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 10:15:00 | 548.10 | 545.46 | 545.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 09:15:00 | 543.40 | 546.81 | 546.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 543.40 | 546.81 | 546.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 543.40 | 546.81 | 546.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 543.40 | 546.81 | 546.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 542.85 | 546.02 | 545.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 542.85 | 546.02 | 545.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 543.80 | 545.57 | 545.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 15:15:00 | 542.20 | 543.46 | 544.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 13:15:00 | 534.15 | 532.87 | 536.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 14:00:00 | 534.15 | 532.87 | 536.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 537.75 | 533.98 | 536.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:30:00 | 540.25 | 533.98 | 536.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 538.55 | 534.89 | 536.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:45:00 | 539.50 | 534.89 | 536.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 538.20 | 535.55 | 536.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:00:00 | 538.20 | 535.55 | 536.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 533.00 | 535.04 | 536.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:15:00 | 530.40 | 535.42 | 536.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 13:15:00 | 503.88 | 515.70 | 520.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-10 13:15:00 | 518.75 | 515.70 | 520.08 | SL hit (close>static) qty=0.50 sl=515.70 alert=retest2 |

### Cycle 196 — BUY (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 13:15:00 | 524.35 | 521.99 | 521.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 15:15:00 | 525.90 | 523.28 | 522.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 521.60 | 524.90 | 523.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 521.60 | 524.90 | 523.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 521.60 | 524.90 | 523.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 521.60 | 524.90 | 523.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 527.60 | 525.44 | 524.30 | EMA400 retest candle locked (from upside) |

### Cycle 197 — SELL (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 15:15:00 | 521.60 | 523.53 | 523.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 515.30 | 521.89 | 522.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 14:15:00 | 520.65 | 518.54 | 520.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 14:15:00 | 520.65 | 518.54 | 520.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 520.65 | 518.54 | 520.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 14:45:00 | 520.45 | 518.54 | 520.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 520.00 | 518.83 | 520.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 10:45:00 | 519.25 | 519.09 | 520.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 13:00:00 | 519.45 | 518.95 | 520.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 518.00 | 520.63 | 520.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 15:00:00 | 518.95 | 518.64 | 519.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 518.05 | 518.52 | 519.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 516.50 | 518.52 | 519.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 13:15:00 | 513.40 | 518.51 | 518.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 520.50 | 515.93 | 517.28 | SL hit (close>static) qty=1.00 sl=519.90 alert=retest2 |

### Cycle 198 — BUY (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 15:15:00 | 520.00 | 517.95 | 517.82 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 515.45 | 517.56 | 517.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 11:15:00 | 514.75 | 516.99 | 517.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 11:15:00 | 509.00 | 506.76 | 510.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 12:00:00 | 509.00 | 506.76 | 510.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 511.40 | 507.69 | 510.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:45:00 | 511.30 | 507.69 | 510.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 515.00 | 509.15 | 510.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:00:00 | 515.00 | 509.15 | 510.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 515.50 | 510.42 | 511.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 515.50 | 510.42 | 511.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 513.40 | 511.72 | 511.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 09:15:00 | 527.15 | 515.77 | 513.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 524.55 | 528.01 | 524.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 524.55 | 528.01 | 524.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 524.55 | 528.01 | 524.75 | EMA400 retest candle locked (from upside) |

### Cycle 201 — SELL (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 14:15:00 | 521.95 | 524.56 | 524.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 519.45 | 523.06 | 524.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 10:15:00 | 524.45 | 523.34 | 524.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 10:15:00 | 524.45 | 523.34 | 524.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 524.45 | 523.34 | 524.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:00:00 | 524.45 | 523.34 | 524.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 521.95 | 523.06 | 523.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 15:15:00 | 519.30 | 522.23 | 523.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 13:15:00 | 527.35 | 520.58 | 520.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 527.35 | 520.58 | 520.43 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 517.75 | 520.53 | 520.58 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 11:15:00 | 521.05 | 520.63 | 520.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-08 12:15:00 | 523.60 | 521.22 | 520.89 | Break + close above crossover candle high |

### Cycle 205 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 513.25 | 520.01 | 520.50 | EMA200 below EMA400 |

### Cycle 206 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 526.60 | 521.65 | 521.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 11:15:00 | 528.10 | 525.26 | 523.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 10:15:00 | 528.75 | 529.52 | 526.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-12 10:45:00 | 528.35 | 529.52 | 526.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 539.60 | 542.57 | 539.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 539.15 | 542.57 | 539.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 538.35 | 541.73 | 539.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 10:45:00 | 541.60 | 541.77 | 540.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 15:15:00 | 535.15 | 538.80 | 539.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 207 — SELL (started 2025-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 15:15:00 | 535.15 | 538.80 | 539.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 523.55 | 535.75 | 537.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 524.35 | 522.51 | 526.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-22 10:00:00 | 524.35 | 522.51 | 526.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 524.05 | 523.73 | 525.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 13:15:00 | 517.85 | 520.00 | 522.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 14:15:00 | 536.00 | 518.47 | 517.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — BUY (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 14:15:00 | 536.00 | 518.47 | 517.88 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 09:15:00 | 523.65 | 525.29 | 525.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 12:15:00 | 521.70 | 524.22 | 524.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 15:15:00 | 524.00 | 523.31 | 524.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 09:15:00 | 524.40 | 523.31 | 524.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 525.55 | 523.75 | 524.33 | EMA400 retest candle locked (from downside) |

### Cycle 210 — BUY (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 11:15:00 | 526.30 | 524.92 | 524.80 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 521.50 | 524.31 | 524.65 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 11:15:00 | 528.35 | 524.57 | 524.33 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 520.30 | 524.31 | 524.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 517.15 | 522.00 | 523.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 11:15:00 | 510.05 | 509.80 | 514.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:00:00 | 510.05 | 509.80 | 514.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 516.35 | 511.51 | 514.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 516.35 | 511.51 | 514.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 512.95 | 511.80 | 514.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:15:00 | 509.25 | 513.24 | 514.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:30:00 | 510.00 | 509.51 | 511.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:00:00 | 510.45 | 509.70 | 511.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 509.95 | 509.95 | 511.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 509.20 | 509.45 | 511.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:30:00 | 509.50 | 509.45 | 511.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 502.85 | 508.24 | 510.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:45:00 | 501.50 | 506.96 | 509.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:00:00 | 501.70 | 505.91 | 508.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 500.85 | 505.63 | 507.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 499.40 | 502.49 | 504.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 497.60 | 501.51 | 503.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 14:00:00 | 495.00 | 499.77 | 502.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 13:15:00 | 483.79 | 490.11 | 495.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 13:15:00 | 484.50 | 490.11 | 495.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 13:15:00 | 484.93 | 490.11 | 495.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 13:15:00 | 484.45 | 490.11 | 495.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 493.55 | 488.80 | 493.39 | SL hit (close>ema200) qty=0.50 sl=488.80 alert=retest2 |

### Cycle 214 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 503.00 | 495.93 | 495.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 510.45 | 500.40 | 497.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 504.75 | 505.59 | 501.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 15:00:00 | 504.75 | 505.59 | 501.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 502.60 | 504.50 | 502.07 | EMA400 retest candle locked (from upside) |

### Cycle 215 — SELL (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 15:15:00 | 498.20 | 501.47 | 501.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 09:15:00 | 490.00 | 499.17 | 500.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 485.25 | 483.64 | 488.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 485.25 | 483.64 | 488.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 485.25 | 483.64 | 488.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:30:00 | 488.55 | 483.64 | 488.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 483.45 | 482.76 | 485.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 15:00:00 | 483.45 | 482.76 | 485.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 484.00 | 483.01 | 485.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 480.65 | 483.01 | 485.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 480.50 | 482.51 | 485.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 478.50 | 482.15 | 484.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 13:00:00 | 477.30 | 481.18 | 483.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 12:45:00 | 479.25 | 479.98 | 480.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 14:15:00 | 483.30 | 480.94 | 480.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 216 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 483.30 | 480.94 | 480.65 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2026-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 10:15:00 | 477.30 | 479.97 | 480.30 | EMA200 below EMA400 |

### Cycle 218 — BUY (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 14:15:00 | 488.95 | 481.92 | 481.06 | EMA200 above EMA400 |

### Cycle 219 — SELL (started 2026-02-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 14:15:00 | 497.65 | 501.32 | 501.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 10:15:00 | 494.80 | 499.04 | 500.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 11:15:00 | 496.20 | 495.55 | 497.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 12:00:00 | 496.20 | 495.55 | 497.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 492.85 | 490.48 | 492.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 492.85 | 490.48 | 492.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 491.20 | 490.63 | 492.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 12:45:00 | 489.30 | 490.19 | 491.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:00:00 | 489.35 | 490.02 | 491.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 15:15:00 | 489.50 | 490.02 | 491.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:15:00 | 464.83 | 473.00 | 475.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:15:00 | 464.88 | 473.00 | 475.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:15:00 | 465.02 | 473.00 | 475.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 14:15:00 | 460.50 | 459.79 | 465.20 | SL hit (close>ema200) qty=0.50 sl=459.79 alert=retest2 |

### Cycle 220 — BUY (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 13:15:00 | 468.50 | 461.71 | 461.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 14:15:00 | 473.70 | 464.11 | 462.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 10:15:00 | 464.00 | 464.97 | 463.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 10:15:00 | 464.00 | 464.97 | 463.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 464.00 | 464.97 | 463.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:45:00 | 464.00 | 464.97 | 463.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 468.35 | 465.61 | 463.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:30:00 | 468.30 | 465.61 | 463.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 459.40 | 464.49 | 463.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 459.40 | 464.49 | 463.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 461.00 | 463.79 | 463.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 444.90 | 463.79 | 463.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 221 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 441.05 | 459.24 | 461.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 11:15:00 | 440.00 | 442.94 | 447.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 13:15:00 | 417.20 | 414.42 | 421.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:00:00 | 417.20 | 414.42 | 421.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 424.45 | 416.43 | 421.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 15:00:00 | 424.45 | 416.43 | 421.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 421.10 | 417.36 | 421.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 09:15:00 | 415.35 | 417.36 | 421.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 13:45:00 | 421.00 | 420.47 | 421.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 14:15:00 | 419.95 | 420.47 | 421.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 09:45:00 | 420.10 | 419.27 | 420.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 13:15:00 | 418.85 | 418.04 | 419.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:45:00 | 418.10 | 418.04 | 419.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 399.95 | 405.02 | 408.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 398.95 | 405.02 | 408.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 399.10 | 405.02 | 408.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 11:15:00 | 406.80 | 404.68 | 408.10 | SL hit (close>ema200) qty=0.50 sl=404.68 alert=retest2 |

### Cycle 222 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 415.70 | 407.47 | 406.58 | EMA200 above EMA400 |

### Cycle 223 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 399.40 | 406.64 | 407.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 395.50 | 401.35 | 404.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 397.35 | 395.64 | 398.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 397.35 | 395.64 | 398.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 397.35 | 395.64 | 398.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 395.90 | 395.64 | 398.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 392.30 | 394.77 | 395.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 11:00:00 | 394.35 | 394.24 | 395.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 398.60 | 395.48 | 395.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 224 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 398.60 | 395.48 | 395.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 401.05 | 397.12 | 396.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 416.00 | 416.31 | 410.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:45:00 | 415.85 | 416.31 | 410.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 423.30 | 423.18 | 419.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 424.35 | 423.18 | 419.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 15:00:00 | 424.60 | 425.28 | 422.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 425.55 | 424.72 | 422.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 10:15:00 | 424.95 | 426.14 | 424.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 427.50 | 426.42 | 425.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 434.40 | 425.18 | 424.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 11:15:00 | 466.79 | 457.04 | 450.29 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 225 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 455.70 | 459.79 | 460.12 | EMA200 below EMA400 |

### Cycle 226 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 462.35 | 460.43 | 460.25 | EMA200 above EMA400 |

### Cycle 227 — SELL (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 13:15:00 | 459.40 | 460.09 | 460.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-27 15:15:00 | 458.90 | 459.89 | 460.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-28 09:15:00 | 460.00 | 459.91 | 460.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 09:15:00 | 460.00 | 459.91 | 460.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 460.00 | 459.91 | 460.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:15:00 | 461.25 | 459.91 | 460.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 458.25 | 459.58 | 459.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:15:00 | 456.10 | 459.58 | 459.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 12:00:00 | 455.90 | 458.84 | 459.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 10:00:00 | 457.20 | 456.53 | 457.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 11:15:00 | 465.45 | 459.12 | 458.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 228 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 465.45 | 459.12 | 458.88 | EMA200 above EMA400 |

### Cycle 229 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 450.20 | 457.76 | 458.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 445.75 | 455.36 | 457.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 459.15 | 450.68 | 453.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 459.15 | 450.68 | 453.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 459.15 | 450.68 | 453.34 | EMA400 retest candle locked (from downside) |

### Cycle 230 — BUY (started 2026-05-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 13:15:00 | 455.85 | 455.06 | 454.95 | EMA200 above EMA400 |

### Cycle 231 — SELL (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 09:15:00 | 449.95 | 454.35 | 454.69 | EMA200 below EMA400 |

### Cycle 232 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 11:15:00 | 455.95 | 452.29 | 452.17 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-17 13:15:00 | 386.20 | 2023-05-18 11:15:00 | 390.85 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2023-05-17 14:15:00 | 386.00 | 2023-05-18 11:15:00 | 390.85 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2023-05-23 09:45:00 | 394.40 | 2023-05-24 10:15:00 | 389.75 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2023-05-23 10:45:00 | 393.95 | 2023-05-24 10:15:00 | 389.75 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2023-05-30 09:15:00 | 400.70 | 2023-05-31 13:15:00 | 396.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2023-05-30 10:00:00 | 401.40 | 2023-05-31 13:15:00 | 396.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2023-06-02 11:45:00 | 391.85 | 2023-06-02 14:15:00 | 396.90 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2023-06-06 11:00:00 | 397.20 | 2023-06-07 09:15:00 | 394.75 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2023-06-06 11:30:00 | 396.70 | 2023-06-07 09:15:00 | 394.75 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2023-06-14 09:15:00 | 376.85 | 2023-06-15 14:15:00 | 381.70 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2023-06-15 12:30:00 | 379.30 | 2023-06-15 14:15:00 | 381.70 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2023-06-15 14:00:00 | 380.45 | 2023-06-15 14:15:00 | 381.70 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2023-06-22 09:15:00 | 423.05 | 2023-06-23 09:15:00 | 409.55 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2023-07-25 12:15:00 | 420.35 | 2023-07-31 14:15:00 | 462.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-07-25 12:45:00 | 420.30 | 2023-07-31 14:15:00 | 462.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-07-25 13:30:00 | 421.15 | 2023-07-31 14:15:00 | 463.26 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-08-28 13:30:00 | 516.15 | 2023-08-30 13:15:00 | 520.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2023-08-29 11:15:00 | 514.75 | 2023-08-30 13:15:00 | 520.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2023-08-29 15:15:00 | 514.25 | 2023-08-30 13:15:00 | 520.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2023-09-01 09:15:00 | 539.15 | 2023-09-04 15:15:00 | 523.60 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2023-09-13 11:45:00 | 563.70 | 2023-09-15 12:15:00 | 550.75 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2023-09-14 13:15:00 | 560.25 | 2023-09-15 12:15:00 | 550.75 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2023-09-14 14:00:00 | 560.10 | 2023-09-15 12:15:00 | 550.75 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2023-09-15 11:00:00 | 560.25 | 2023-09-15 12:15:00 | 550.75 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2023-09-27 13:00:00 | 514.15 | 2023-09-29 09:15:00 | 530.20 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2023-09-27 14:30:00 | 514.10 | 2023-09-29 09:15:00 | 530.20 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2023-09-27 15:00:00 | 513.05 | 2023-09-29 09:15:00 | 530.20 | STOP_HIT | 1.00 | -3.34% |
| SELL | retest2 | 2023-09-28 09:30:00 | 512.45 | 2023-09-29 09:15:00 | 530.20 | STOP_HIT | 1.00 | -3.46% |
| SELL | retest2 | 2023-10-10 15:15:00 | 515.10 | 2023-10-11 09:15:00 | 520.60 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2023-10-17 13:45:00 | 509.45 | 2023-10-20 10:15:00 | 515.75 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2023-10-20 09:45:00 | 508.85 | 2023-10-20 10:15:00 | 515.75 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2023-10-25 12:30:00 | 505.65 | 2023-10-25 14:15:00 | 510.50 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2023-10-26 09:15:00 | 501.70 | 2023-10-30 10:15:00 | 513.75 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2023-11-02 14:45:00 | 514.00 | 2023-11-03 11:15:00 | 508.70 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2023-11-03 09:15:00 | 515.00 | 2023-11-03 11:15:00 | 508.70 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2023-11-09 09:15:00 | 509.50 | 2023-11-09 11:15:00 | 515.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2023-11-09 10:45:00 | 509.95 | 2023-11-09 11:15:00 | 515.50 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2023-11-12 18:15:00 | 516.95 | 2023-11-13 09:15:00 | 511.10 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2023-11-29 13:45:00 | 503.00 | 2023-11-29 15:15:00 | 498.30 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2023-11-30 14:00:00 | 503.65 | 2023-12-06 12:15:00 | 503.20 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2023-12-08 10:45:00 | 499.45 | 2023-12-11 12:15:00 | 507.25 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2023-12-08 11:45:00 | 498.90 | 2023-12-11 12:15:00 | 507.25 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2023-12-08 12:15:00 | 499.00 | 2023-12-11 12:15:00 | 507.25 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-01-10 12:15:00 | 542.35 | 2024-01-17 11:15:00 | 515.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-11 10:45:00 | 543.15 | 2024-01-17 11:15:00 | 515.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-10 12:15:00 | 542.35 | 2024-01-19 09:15:00 | 511.40 | STOP_HIT | 0.50 | 5.71% |
| SELL | retest2 | 2024-01-11 10:45:00 | 543.15 | 2024-01-19 09:15:00 | 511.40 | STOP_HIT | 0.50 | 5.85% |
| SELL | retest2 | 2024-02-12 09:30:00 | 479.35 | 2024-02-19 12:15:00 | 479.90 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2024-02-12 15:15:00 | 480.00 | 2024-02-19 12:15:00 | 479.90 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2024-02-26 13:30:00 | 463.95 | 2024-02-26 14:15:00 | 469.80 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-03-06 09:45:00 | 452.75 | 2024-03-15 10:15:00 | 430.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-06 11:00:00 | 452.00 | 2024-03-15 10:15:00 | 429.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 09:30:00 | 451.95 | 2024-03-15 10:15:00 | 429.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-06 09:45:00 | 452.75 | 2024-03-18 10:15:00 | 429.45 | STOP_HIT | 0.50 | 5.15% |
| SELL | retest2 | 2024-03-06 11:00:00 | 452.00 | 2024-03-18 10:15:00 | 429.45 | STOP_HIT | 0.50 | 4.99% |
| SELL | retest2 | 2024-03-11 09:30:00 | 451.95 | 2024-03-18 10:15:00 | 429.45 | STOP_HIT | 0.50 | 4.98% |
| BUY | retest2 | 2024-04-01 09:15:00 | 436.55 | 2024-04-15 09:15:00 | 445.90 | STOP_HIT | 1.00 | 2.14% |
| BUY | retest2 | 2024-05-24 11:30:00 | 540.80 | 2024-05-27 09:15:00 | 526.70 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2024-05-24 12:00:00 | 542.10 | 2024-05-27 09:15:00 | 526.70 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2024-05-24 14:45:00 | 543.75 | 2024-05-27 09:15:00 | 526.70 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2024-06-12 09:15:00 | 717.70 | 2024-06-12 11:15:00 | 711.45 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-06-14 12:15:00 | 711.85 | 2024-06-18 09:15:00 | 721.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-06-14 14:15:00 | 711.10 | 2024-06-18 09:15:00 | 721.00 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-06-21 14:30:00 | 712.30 | 2024-06-24 09:15:00 | 726.25 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-06-21 15:00:00 | 700.10 | 2024-06-24 09:15:00 | 726.25 | STOP_HIT | 1.00 | -3.74% |
| SELL | retest2 | 2024-06-28 11:30:00 | 690.35 | 2024-07-01 09:15:00 | 762.95 | STOP_HIT | 1.00 | -10.52% |
| BUY | retest2 | 2024-07-16 09:15:00 | 769.70 | 2024-07-25 10:15:00 | 796.20 | STOP_HIT | 1.00 | 3.44% |
| BUY | retest2 | 2024-08-09 09:15:00 | 785.00 | 2024-08-19 13:15:00 | 802.55 | STOP_HIT | 1.00 | 2.24% |
| BUY | retest2 | 2024-08-26 09:15:00 | 813.10 | 2024-08-29 12:15:00 | 814.45 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2024-09-02 09:15:00 | 792.05 | 2024-09-03 09:15:00 | 830.50 | STOP_HIT | 1.00 | -4.85% |
| SELL | retest2 | 2024-09-02 11:45:00 | 804.05 | 2024-09-03 09:15:00 | 830.50 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2024-09-02 14:30:00 | 810.60 | 2024-09-03 09:15:00 | 830.50 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2024-09-20 13:30:00 | 743.90 | 2024-09-23 10:15:00 | 760.00 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2024-10-04 12:45:00 | 741.50 | 2024-10-09 11:15:00 | 748.90 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-10-04 14:15:00 | 741.00 | 2024-10-09 11:15:00 | 748.90 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-10-07 10:15:00 | 741.50 | 2024-10-09 11:15:00 | 748.90 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-10-22 10:15:00 | 682.30 | 2024-10-24 10:15:00 | 648.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 10:15:00 | 682.30 | 2024-10-25 13:15:00 | 645.45 | STOP_HIT | 0.50 | 5.40% |
| BUY | retest2 | 2024-11-05 09:45:00 | 684.05 | 2024-11-07 10:15:00 | 677.55 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-11-06 15:15:00 | 688.45 | 2024-11-07 10:15:00 | 677.55 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest1 | 2024-11-11 09:15:00 | 662.75 | 2024-11-11 09:15:00 | 675.85 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2024-11-11 12:15:00 | 670.95 | 2024-11-12 09:15:00 | 684.20 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2024-11-11 13:30:00 | 670.40 | 2024-11-12 09:15:00 | 684.20 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2024-11-21 09:15:00 | 634.40 | 2024-11-22 13:15:00 | 637.70 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2024-11-27 12:00:00 | 656.10 | 2024-12-02 12:15:00 | 655.60 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2024-11-27 14:15:00 | 656.50 | 2024-12-02 12:15:00 | 655.60 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2024-11-27 14:45:00 | 659.60 | 2024-12-02 12:15:00 | 655.60 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-12-02 09:30:00 | 656.25 | 2024-12-02 12:15:00 | 655.60 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2024-12-17 11:30:00 | 584.50 | 2024-12-19 13:15:00 | 596.50 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-12-18 12:00:00 | 583.20 | 2024-12-19 13:15:00 | 596.50 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-12-19 09:15:00 | 582.10 | 2024-12-19 13:15:00 | 596.50 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-01-01 13:15:00 | 609.30 | 2025-01-06 09:15:00 | 596.90 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-01-01 14:30:00 | 610.10 | 2025-01-06 09:15:00 | 596.90 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-01-01 15:15:00 | 609.90 | 2025-01-06 09:15:00 | 596.90 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-01-02 14:45:00 | 610.40 | 2025-01-06 09:15:00 | 596.90 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-01-17 09:15:00 | 565.20 | 2025-01-17 10:15:00 | 569.50 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-01-17 13:00:00 | 563.60 | 2025-01-23 12:15:00 | 559.40 | STOP_HIT | 1.00 | 0.75% |
| SELL | retest2 | 2025-01-17 15:00:00 | 564.80 | 2025-01-23 12:15:00 | 559.40 | STOP_HIT | 1.00 | 0.96% |
| SELL | retest2 | 2025-01-20 09:15:00 | 563.00 | 2025-01-23 12:15:00 | 559.40 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2025-01-21 10:15:00 | 553.45 | 2025-01-23 12:15:00 | 559.40 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-01-21 12:00:00 | 554.85 | 2025-01-23 12:15:00 | 559.40 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-01-22 15:15:00 | 555.60 | 2025-01-23 12:15:00 | 559.40 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-02-11 09:15:00 | 558.55 | 2025-02-14 10:15:00 | 530.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 558.55 | 2025-02-17 13:15:00 | 523.45 | STOP_HIT | 0.50 | 6.28% |
| SELL | retest2 | 2025-02-28 09:15:00 | 532.00 | 2025-03-06 09:15:00 | 538.55 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-03-12 14:00:00 | 566.75 | 2025-03-17 11:15:00 | 553.75 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-03-12 15:00:00 | 567.05 | 2025-03-17 11:15:00 | 553.75 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-03-20 09:15:00 | 561.65 | 2025-03-25 10:15:00 | 562.35 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2025-03-20 09:45:00 | 561.20 | 2025-03-25 10:15:00 | 562.35 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-03-20 10:45:00 | 562.15 | 2025-03-25 10:15:00 | 562.35 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest1 | 2025-04-11 11:45:00 | 606.80 | 2025-04-21 09:15:00 | 613.05 | STOP_HIT | 1.00 | 1.03% |
| BUY | retest1 | 2025-04-11 13:45:00 | 607.30 | 2025-04-21 09:15:00 | 613.05 | STOP_HIT | 1.00 | 0.95% |
| BUY | retest1 | 2025-04-11 14:30:00 | 608.50 | 2025-04-21 09:15:00 | 613.05 | STOP_HIT | 1.00 | 0.75% |
| BUY | retest1 | 2025-04-15 09:15:00 | 608.05 | 2025-04-21 09:15:00 | 613.05 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2025-04-22 09:15:00 | 634.90 | 2025-04-25 11:15:00 | 624.90 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-04-30 11:15:00 | 633.40 | 2025-04-30 13:15:00 | 627.65 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-05-02 11:30:00 | 626.85 | 2025-05-05 09:15:00 | 633.95 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-05-02 15:15:00 | 628.00 | 2025-05-05 09:15:00 | 633.95 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-05-06 10:15:00 | 642.05 | 2025-05-06 14:15:00 | 627.70 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-05-06 12:00:00 | 635.50 | 2025-05-06 14:15:00 | 627.70 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-05-16 09:15:00 | 634.20 | 2025-05-16 13:15:00 | 625.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-05-23 15:00:00 | 594.45 | 2025-06-04 12:15:00 | 596.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-05-26 12:30:00 | 594.65 | 2025-06-04 12:15:00 | 596.00 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-05-27 10:45:00 | 594.50 | 2025-06-04 12:15:00 | 596.00 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-06-09 14:00:00 | 578.50 | 2025-06-10 14:15:00 | 583.10 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-06-10 09:45:00 | 579.25 | 2025-06-10 14:15:00 | 583.10 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-06-18 14:15:00 | 567.10 | 2025-06-24 10:15:00 | 571.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-06-19 10:30:00 | 566.25 | 2025-06-24 10:15:00 | 571.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-06-30 09:15:00 | 573.75 | 2025-06-30 12:15:00 | 566.35 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-07-09 10:15:00 | 584.10 | 2025-07-14 09:15:00 | 581.55 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-08-12 11:00:00 | 574.80 | 2025-08-13 15:15:00 | 582.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-08-12 15:00:00 | 576.80 | 2025-08-13 15:15:00 | 582.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-08-18 09:15:00 | 584.50 | 2025-08-25 10:15:00 | 608.35 | STOP_HIT | 1.00 | 4.08% |
| SELL | retest2 | 2025-09-01 13:00:00 | 574.15 | 2025-09-01 13:15:00 | 580.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-09-12 11:15:00 | 604.05 | 2025-09-19 10:15:00 | 597.95 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-09-12 12:00:00 | 604.05 | 2025-09-19 10:15:00 | 597.95 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-09-15 09:30:00 | 604.35 | 2025-09-19 10:15:00 | 597.95 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-09-16 14:00:00 | 605.55 | 2025-09-19 10:15:00 | 597.95 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-09-17 13:45:00 | 602.90 | 2025-09-19 10:15:00 | 597.95 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-09-17 15:00:00 | 603.55 | 2025-09-19 10:15:00 | 597.95 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-09-18 10:00:00 | 602.15 | 2025-09-19 10:15:00 | 597.95 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-09-18 10:45:00 | 609.15 | 2025-09-19 10:15:00 | 597.95 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-09-25 11:00:00 | 574.50 | 2025-09-29 12:15:00 | 545.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 11:00:00 | 574.50 | 2025-10-01 13:15:00 | 544.25 | STOP_HIT | 0.50 | 5.27% |
| BUY | retest2 | 2025-10-07 14:15:00 | 552.15 | 2025-10-08 09:15:00 | 543.70 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-10-07 14:45:00 | 551.65 | 2025-10-08 09:15:00 | 543.70 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-10-07 15:15:00 | 552.30 | 2025-10-08 09:15:00 | 543.70 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-10-13 13:30:00 | 538.80 | 2025-10-15 13:15:00 | 541.10 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-10-13 14:45:00 | 537.75 | 2025-10-15 13:15:00 | 541.10 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-10-15 12:15:00 | 538.05 | 2025-10-15 13:15:00 | 541.10 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-10-21 13:45:00 | 545.70 | 2025-10-24 13:15:00 | 541.30 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-10-23 10:00:00 | 544.75 | 2025-10-24 13:15:00 | 541.30 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-10-24 11:30:00 | 542.50 | 2025-10-24 13:15:00 | 541.30 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-11-06 10:15:00 | 530.40 | 2025-11-10 13:15:00 | 503.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 10:15:00 | 530.40 | 2025-11-10 13:15:00 | 518.75 | STOP_HIT | 0.50 | 2.20% |
| SELL | retest2 | 2025-11-17 10:45:00 | 519.25 | 2025-11-20 09:15:00 | 520.50 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-11-17 13:00:00 | 519.45 | 2025-11-20 09:15:00 | 520.50 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-11-18 09:15:00 | 518.00 | 2025-11-20 15:15:00 | 520.00 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-11-18 15:00:00 | 518.95 | 2025-11-20 15:15:00 | 520.00 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-11-19 09:15:00 | 516.50 | 2025-11-20 15:15:00 | 520.00 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-11-19 13:15:00 | 513.40 | 2025-11-20 15:15:00 | 520.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-11-20 14:15:00 | 515.95 | 2025-11-20 15:15:00 | 520.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-12-03 15:15:00 | 519.30 | 2025-12-05 13:15:00 | 527.35 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-12-17 10:45:00 | 541.60 | 2025-12-17 15:15:00 | 535.15 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-12-24 13:15:00 | 517.85 | 2025-12-29 14:15:00 | 536.00 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2026-01-13 09:15:00 | 509.25 | 2026-01-21 13:15:00 | 483.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 13:30:00 | 510.00 | 2026-01-21 13:15:00 | 484.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 15:00:00 | 510.45 | 2026-01-21 13:15:00 | 484.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:15:00 | 509.95 | 2026-01-21 13:15:00 | 484.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 09:15:00 | 509.25 | 2026-01-22 09:15:00 | 493.55 | STOP_HIT | 0.50 | 3.08% |
| SELL | retest2 | 2026-01-13 13:30:00 | 510.00 | 2026-01-22 09:15:00 | 493.55 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2026-01-13 15:00:00 | 510.45 | 2026-01-22 09:15:00 | 493.55 | STOP_HIT | 0.50 | 3.31% |
| SELL | retest2 | 2026-01-14 09:15:00 | 509.95 | 2026-01-22 09:15:00 | 493.55 | STOP_HIT | 0.50 | 3.22% |
| SELL | retest2 | 2026-01-14 13:45:00 | 501.50 | 2026-01-22 13:15:00 | 503.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2026-01-14 15:00:00 | 501.70 | 2026-01-22 13:15:00 | 503.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2026-01-19 09:15:00 | 500.85 | 2026-01-22 13:15:00 | 503.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2026-01-20 09:15:00 | 499.40 | 2026-01-22 13:15:00 | 503.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2026-01-20 14:00:00 | 495.00 | 2026-01-22 13:15:00 | 503.00 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-02-01 12:15:00 | 478.50 | 2026-02-03 14:15:00 | 483.30 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-02-01 13:00:00 | 477.30 | 2026-02-03 14:15:00 | 483.30 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-02-03 12:45:00 | 479.25 | 2026-02-03 14:15:00 | 483.30 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-02-17 12:45:00 | 489.30 | 2026-02-27 10:15:00 | 464.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 14:00:00 | 489.35 | 2026-02-27 10:15:00 | 464.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 15:15:00 | 489.50 | 2026-02-27 10:15:00 | 465.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 12:45:00 | 489.30 | 2026-03-02 14:15:00 | 460.50 | STOP_HIT | 0.50 | 5.89% |
| SELL | retest2 | 2026-02-17 14:00:00 | 489.35 | 2026-03-02 14:15:00 | 460.50 | STOP_HIT | 0.50 | 5.90% |
| SELL | retest2 | 2026-02-17 15:15:00 | 489.50 | 2026-03-02 14:15:00 | 460.50 | STOP_HIT | 0.50 | 5.92% |
| SELL | retest2 | 2026-03-17 09:15:00 | 415.35 | 2026-03-23 09:15:00 | 399.95 | PARTIAL | 0.50 | 3.71% |
| SELL | retest2 | 2026-03-17 13:45:00 | 421.00 | 2026-03-23 09:15:00 | 398.95 | PARTIAL | 0.50 | 5.24% |
| SELL | retest2 | 2026-03-17 14:15:00 | 419.95 | 2026-03-23 09:15:00 | 399.10 | PARTIAL | 0.50 | 4.97% |
| SELL | retest2 | 2026-03-17 09:15:00 | 415.35 | 2026-03-23 11:15:00 | 406.80 | STOP_HIT | 0.50 | 2.06% |
| SELL | retest2 | 2026-03-17 13:45:00 | 421.00 | 2026-03-23 11:15:00 | 406.80 | STOP_HIT | 0.50 | 3.37% |
| SELL | retest2 | 2026-03-17 14:15:00 | 419.95 | 2026-03-23 11:15:00 | 406.80 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2026-03-18 09:45:00 | 420.10 | 2026-03-25 09:15:00 | 416.25 | STOP_HIT | 1.00 | 0.92% |
| SELL | retest2 | 2026-03-24 09:30:00 | 401.40 | 2026-03-25 09:15:00 | 416.25 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2026-03-24 13:45:00 | 401.90 | 2026-03-25 09:15:00 | 416.25 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2026-03-24 14:15:00 | 402.20 | 2026-03-25 10:15:00 | 415.70 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2026-04-01 10:15:00 | 395.90 | 2026-04-06 12:15:00 | 398.60 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2026-04-06 09:15:00 | 392.30 | 2026-04-06 12:15:00 | 398.60 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-04-06 11:00:00 | 394.35 | 2026-04-06 12:15:00 | 398.60 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2026-04-13 10:15:00 | 424.35 | 2026-04-22 11:15:00 | 466.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 15:00:00 | 424.60 | 2026-04-22 11:15:00 | 467.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-15 09:15:00 | 425.55 | 2026-04-22 11:15:00 | 468.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-16 10:15:00 | 424.95 | 2026-04-22 11:15:00 | 467.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-17 09:15:00 | 434.40 | 2026-04-24 13:15:00 | 455.70 | STOP_HIT | 1.00 | 4.90% |
| SELL | retest2 | 2026-04-28 11:15:00 | 456.10 | 2026-04-29 11:15:00 | 465.45 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2026-04-28 12:00:00 | 455.90 | 2026-04-29 11:15:00 | 465.45 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2026-04-29 10:00:00 | 457.20 | 2026-04-29 11:15:00 | 465.45 | STOP_HIT | 1.00 | -1.80% |
