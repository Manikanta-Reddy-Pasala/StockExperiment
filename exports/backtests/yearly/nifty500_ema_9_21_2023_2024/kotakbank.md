# Kotak Mahindra Bank Ltd. (KOTAKBANK)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 381.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 225 |
| ALERT1 | 153 |
| ALERT2 | 151 |
| ALERT2_SKIP | 79 |
| ALERT3 | 442 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 180 |
| PARTIAL | 7 |
| TARGET_HIT | 0 |
| STOP_HIT | 180 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 187 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 44 / 143
- **Target hits / Stop hits / Partials:** 0 / 180 / 7
- **Avg / median % per leg:** -0.19% / -0.66%
- **Sum % (uncompounded):** -35.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 80 | 21 | 26.2% | 0 | 80 | 0 | -0.26% | -21.0% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.60% | 1.6% |
| BUY @ 3rd Alert (retest2) | 79 | 20 | 25.3% | 0 | 79 | 0 | -0.29% | -22.6% |
| SELL (all) | 107 | 23 | 21.5% | 0 | 100 | 7 | -0.13% | -14.4% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.21% | -0.4% |
| SELL @ 3rd Alert (retest2) | 105 | 22 | 21.0% | 0 | 98 | 7 | -0.13% | -14.0% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 3 | 0 | 0.40% | 1.2% |
| retest2 (combined) | 184 | 42 | 22.8% | 0 | 177 | 7 | -0.20% | -36.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 09:15:00 | 393.55 | 391.71 | 391.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-15 11:15:00 | 394.32 | 392.47 | 391.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-16 10:15:00 | 392.48 | 393.57 | 392.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-16 10:15:00 | 392.48 | 393.57 | 392.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 10:15:00 | 392.48 | 393.57 | 392.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-16 10:45:00 | 392.66 | 393.57 | 392.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 11:15:00 | 391.61 | 393.18 | 392.71 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 14:15:00 | 389.68 | 392.05 | 392.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-16 15:15:00 | 388.40 | 391.32 | 391.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-18 09:15:00 | 384.11 | 383.74 | 386.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-18 09:30:00 | 384.52 | 383.74 | 386.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 11:15:00 | 384.30 | 384.19 | 386.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-18 12:15:00 | 383.85 | 384.19 | 386.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-19 13:15:00 | 387.80 | 386.10 | 386.25 | SL hit (close>static) qty=1.00 sl=386.73 alert=retest2 |

### Cycle 3 — BUY (started 2023-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 14:15:00 | 388.42 | 386.56 | 386.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 10:15:00 | 389.15 | 387.52 | 386.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-22 13:15:00 | 387.82 | 388.12 | 387.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 13:15:00 | 387.82 | 388.12 | 387.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 13:15:00 | 387.82 | 388.12 | 387.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-22 13:30:00 | 386.48 | 388.12 | 387.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 14:15:00 | 387.99 | 388.10 | 387.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-22 14:45:00 | 387.19 | 388.10 | 387.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 15:15:00 | 387.50 | 387.98 | 387.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 09:15:00 | 385.95 | 387.98 | 387.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 09:15:00 | 385.63 | 387.51 | 387.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 10:15:00 | 384.77 | 387.51 | 387.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2023-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 10:15:00 | 384.32 | 386.87 | 387.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 12:15:00 | 381.49 | 384.94 | 385.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-25 09:15:00 | 385.10 | 383.72 | 384.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 09:15:00 | 385.10 | 383.72 | 384.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 385.10 | 383.72 | 384.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-25 10:00:00 | 385.10 | 383.72 | 384.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 10:15:00 | 384.54 | 383.89 | 384.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-25 12:45:00 | 383.14 | 383.84 | 384.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-25 14:15:00 | 385.98 | 384.16 | 384.59 | SL hit (close>static) qty=1.00 sl=385.26 alert=retest2 |

### Cycle 5 — BUY (started 2023-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 10:15:00 | 385.87 | 384.96 | 384.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 11:15:00 | 386.57 | 385.28 | 385.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-31 12:15:00 | 392.35 | 393.27 | 391.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 12:15:00 | 392.35 | 393.27 | 391.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 12:15:00 | 392.35 | 393.27 | 391.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 12:45:00 | 392.73 | 393.27 | 391.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 13:15:00 | 393.00 | 393.22 | 391.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 14:45:00 | 395.15 | 395.24 | 392.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-01 09:15:00 | 388.13 | 394.95 | 393.31 | SL hit (close<static) qty=1.00 sl=391.53 alert=retest2 |

### Cycle 6 — SELL (started 2023-06-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-01 11:15:00 | 387.20 | 392.27 | 392.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-01 14:15:00 | 385.99 | 389.72 | 391.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-02 14:15:00 | 388.53 | 388.00 | 389.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-02 15:00:00 | 388.53 | 388.00 | 389.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 09:15:00 | 389.50 | 388.13 | 389.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-05 09:45:00 | 389.27 | 388.13 | 389.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 10:15:00 | 389.25 | 388.35 | 389.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-05 11:30:00 | 389.02 | 388.28 | 389.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-06 10:15:00 | 388.38 | 387.35 | 388.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-06 13:15:00 | 389.56 | 388.72 | 388.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2023-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-06 13:15:00 | 389.56 | 388.72 | 388.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-06 14:15:00 | 392.67 | 389.51 | 389.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-07 09:15:00 | 389.66 | 390.01 | 389.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-07 10:00:00 | 389.66 | 390.01 | 389.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 10:15:00 | 388.78 | 389.76 | 389.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-07 10:45:00 | 389.22 | 389.76 | 389.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 11:15:00 | 388.69 | 389.55 | 389.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-07 12:00:00 | 388.69 | 389.55 | 389.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 12:15:00 | 388.96 | 389.43 | 389.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-07 13:00:00 | 388.96 | 389.43 | 389.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 13:15:00 | 387.81 | 389.11 | 389.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-07 13:30:00 | 387.35 | 389.11 | 389.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2023-06-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-07 14:15:00 | 388.00 | 388.89 | 388.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-08 09:15:00 | 382.57 | 387.46 | 388.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 12:15:00 | 375.56 | 375.47 | 378.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-12 12:30:00 | 375.54 | 375.47 | 378.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 14:15:00 | 372.84 | 371.68 | 373.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-14 15:00:00 | 372.84 | 371.68 | 373.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 09:15:00 | 372.05 | 371.87 | 373.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-15 10:45:00 | 370.63 | 371.68 | 372.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-16 15:15:00 | 371.40 | 369.28 | 370.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-19 09:30:00 | 371.19 | 369.39 | 370.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-21 11:15:00 | 368.45 | 367.51 | 367.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2023-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 11:15:00 | 368.45 | 367.51 | 367.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 13:15:00 | 369.54 | 368.10 | 367.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 09:15:00 | 367.42 | 368.22 | 367.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 09:15:00 | 367.42 | 368.22 | 367.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 09:15:00 | 367.42 | 368.22 | 367.94 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-06-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 14:15:00 | 366.21 | 367.61 | 367.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 365.97 | 367.08 | 367.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-23 10:15:00 | 367.18 | 367.10 | 367.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-23 10:15:00 | 367.18 | 367.10 | 367.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 10:15:00 | 367.18 | 367.10 | 367.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-23 10:45:00 | 367.68 | 367.10 | 367.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 11:15:00 | 366.99 | 367.08 | 367.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-23 15:00:00 | 365.67 | 367.00 | 367.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 12:15:00 | 366.42 | 365.90 | 366.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 14:15:00 | 366.64 | 366.17 | 366.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-27 14:15:00 | 369.13 | 366.76 | 366.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2023-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 14:15:00 | 369.13 | 366.76 | 366.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 14:15:00 | 369.59 | 368.39 | 367.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 10:15:00 | 368.49 | 368.74 | 368.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 10:15:00 | 368.49 | 368.74 | 368.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 10:15:00 | 368.49 | 368.74 | 368.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 10:45:00 | 368.54 | 368.74 | 368.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 11:15:00 | 369.50 | 368.89 | 368.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 11:45:00 | 368.31 | 368.89 | 368.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 14:15:00 | 368.87 | 369.05 | 368.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 15:00:00 | 368.87 | 369.05 | 368.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 15:15:00 | 368.89 | 369.02 | 368.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 09:15:00 | 368.45 | 369.02 | 368.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 368.14 | 368.84 | 368.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 09:45:00 | 367.52 | 368.84 | 368.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 10:15:00 | 368.31 | 368.74 | 368.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 10:30:00 | 367.99 | 368.74 | 368.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 11:15:00 | 370.08 | 369.01 | 368.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 11:30:00 | 369.08 | 369.01 | 368.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 10:15:00 | 373.04 | 373.89 | 372.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 10:45:00 | 372.60 | 373.89 | 372.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 11:15:00 | 373.23 | 373.76 | 372.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 11:30:00 | 372.98 | 373.76 | 372.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 12:15:00 | 373.02 | 373.61 | 372.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 12:30:00 | 372.63 | 373.61 | 372.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 13:15:00 | 372.01 | 373.29 | 372.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 14:00:00 | 372.01 | 373.29 | 372.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 14:15:00 | 370.84 | 372.80 | 372.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 15:00:00 | 370.84 | 372.80 | 372.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2023-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 15:15:00 | 371.00 | 372.44 | 372.46 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 15:15:00 | 374.00 | 372.69 | 372.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 09:15:00 | 378.25 | 373.81 | 373.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-11 13:15:00 | 375.11 | 375.23 | 374.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-11 13:45:00 | 375.56 | 375.23 | 374.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 14:15:00 | 373.01 | 374.79 | 373.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-11 15:00:00 | 373.01 | 374.79 | 373.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 15:15:00 | 373.94 | 374.62 | 373.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-12 09:15:00 | 376.38 | 374.62 | 373.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-12 12:15:00 | 375.27 | 374.96 | 374.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-14 09:15:00 | 376.41 | 375.53 | 375.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-14 10:15:00 | 374.83 | 375.28 | 375.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2023-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 10:15:00 | 374.83 | 375.28 | 375.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-14 11:15:00 | 373.19 | 374.86 | 375.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 15:15:00 | 374.24 | 374.23 | 374.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-17 09:15:00 | 374.20 | 374.23 | 374.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 374.48 | 374.28 | 374.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 09:45:00 | 374.74 | 374.28 | 374.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 10:15:00 | 375.98 | 374.62 | 374.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 11:00:00 | 375.98 | 374.62 | 374.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 11:15:00 | 375.33 | 374.76 | 374.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 12:15:00 | 375.73 | 374.76 | 374.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2023-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 12:15:00 | 375.70 | 374.95 | 374.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 13:15:00 | 377.21 | 375.40 | 375.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 11:15:00 | 376.22 | 377.81 | 376.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-18 11:15:00 | 376.22 | 377.81 | 376.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 11:15:00 | 376.22 | 377.81 | 376.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 12:00:00 | 376.22 | 377.81 | 376.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 12:15:00 | 377.31 | 377.71 | 376.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-19 09:15:00 | 380.31 | 377.71 | 376.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-19 09:45:00 | 380.28 | 378.33 | 377.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-20 11:00:00 | 381.37 | 380.49 | 379.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-24 12:15:00 | 381.01 | 385.23 | 385.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2023-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 12:15:00 | 381.01 | 385.23 | 385.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-24 14:15:00 | 378.72 | 383.25 | 384.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 09:15:00 | 378.37 | 377.23 | 379.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-26 10:00:00 | 378.37 | 377.23 | 379.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 14:15:00 | 378.86 | 378.07 | 379.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 10:45:00 | 378.00 | 378.40 | 379.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-08 10:15:00 | 367.79 | 366.68 | 366.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2023-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 10:15:00 | 367.79 | 366.68 | 366.54 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 14:15:00 | 365.25 | 366.39 | 366.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-09 09:15:00 | 364.20 | 365.90 | 366.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-09 13:15:00 | 365.93 | 365.59 | 365.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 13:15:00 | 365.93 | 365.59 | 365.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 13:15:00 | 365.93 | 365.59 | 365.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 14:00:00 | 365.93 | 365.59 | 365.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 14:15:00 | 366.38 | 365.75 | 365.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 14:30:00 | 366.67 | 365.75 | 365.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 15:15:00 | 365.20 | 365.64 | 365.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-10 09:15:00 | 364.60 | 365.64 | 365.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-10 10:15:00 | 364.22 | 365.78 | 365.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-23 13:15:00 | 355.01 | 353.24 | 353.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2023-08-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 13:15:00 | 355.01 | 353.24 | 353.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-23 14:15:00 | 356.00 | 353.79 | 353.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 12:15:00 | 355.78 | 355.81 | 354.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-24 13:00:00 | 355.78 | 355.81 | 354.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 355.40 | 355.75 | 355.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-25 11:15:00 | 356.30 | 355.54 | 355.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-25 15:00:00 | 356.12 | 355.97 | 355.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-28 09:15:00 | 355.88 | 355.70 | 355.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-28 11:00:00 | 356.20 | 355.78 | 355.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 14:15:00 | 356.43 | 357.66 | 357.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-29 15:00:00 | 356.43 | 357.66 | 357.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 15:15:00 | 356.39 | 357.40 | 357.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-30 09:15:00 | 357.57 | 357.40 | 357.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-30 14:15:00 | 354.91 | 356.53 | 356.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2023-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-30 14:15:00 | 354.91 | 356.53 | 356.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-30 15:15:00 | 354.00 | 356.02 | 356.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 10:15:00 | 353.80 | 353.22 | 354.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-01 11:00:00 | 353.80 | 353.22 | 354.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 12:15:00 | 354.16 | 353.39 | 354.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 13:00:00 | 354.16 | 353.39 | 354.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 13:15:00 | 355.55 | 353.82 | 354.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 14:00:00 | 355.55 | 353.82 | 354.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 14:15:00 | 354.40 | 353.93 | 354.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 14:30:00 | 354.78 | 353.93 | 354.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 15:15:00 | 354.50 | 354.05 | 354.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 09:15:00 | 356.80 | 354.05 | 354.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 352.94 | 353.83 | 354.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-04 10:30:00 | 351.99 | 353.46 | 354.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-04 14:45:00 | 352.30 | 353.28 | 353.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-05 10:15:00 | 355.98 | 354.24 | 354.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2023-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 10:15:00 | 355.98 | 354.24 | 354.15 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-09-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 12:15:00 | 352.65 | 353.99 | 354.15 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-09-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 10:15:00 | 355.60 | 354.44 | 354.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-07 13:15:00 | 356.23 | 354.86 | 354.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-14 10:15:00 | 364.53 | 364.69 | 363.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-14 10:30:00 | 364.31 | 364.69 | 363.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 11:15:00 | 363.81 | 364.52 | 363.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-14 11:45:00 | 363.22 | 364.52 | 363.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 12:15:00 | 364.05 | 364.42 | 363.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-14 12:30:00 | 363.71 | 364.42 | 363.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 13:15:00 | 363.91 | 364.32 | 363.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-14 14:00:00 | 363.91 | 364.32 | 363.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 14:15:00 | 364.20 | 364.30 | 363.43 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 09:15:00 | 360.55 | 362.95 | 363.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 10:15:00 | 359.91 | 362.34 | 362.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-20 15:15:00 | 358.77 | 358.49 | 359.81 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-21 09:15:00 | 356.81 | 358.49 | 359.81 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 11:15:00 | 355.79 | 353.33 | 354.40 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-09-25 11:15:00 | 355.79 | 353.33 | 354.40 | SL hit (close>ema400) qty=1.00 sl=354.40 alert=retest1 |

### Cycle 25 — BUY (started 2023-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 15:15:00 | 357.20 | 355.18 | 355.03 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 09:15:00 | 353.90 | 354.92 | 354.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-26 10:15:00 | 352.60 | 354.46 | 354.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 15:15:00 | 354.00 | 353.68 | 354.15 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-27 09:15:00 | 351.49 | 353.68 | 354.15 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 350.20 | 352.98 | 353.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-09-27 15:15:00 | 353.94 | 352.84 | 353.27 | SL hit (close>ema400) qty=1.00 sl=353.27 alert=retest1 |

### Cycle 27 — BUY (started 2023-10-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 11:15:00 | 347.98 | 346.33 | 346.31 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 12:15:00 | 345.31 | 346.80 | 346.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 14:15:00 | 344.00 | 345.98 | 346.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 347.26 | 345.87 | 346.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 347.26 | 345.87 | 346.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 347.26 | 345.87 | 346.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 10:00:00 | 347.26 | 345.87 | 346.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 10:15:00 | 347.02 | 346.10 | 346.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 10:45:00 | 347.32 | 346.10 | 346.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2023-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 11:15:00 | 350.14 | 346.91 | 346.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 12:15:00 | 352.50 | 348.03 | 347.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-12 09:15:00 | 353.51 | 353.73 | 351.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-12 09:45:00 | 352.92 | 353.73 | 351.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 351.57 | 352.81 | 352.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 11:45:00 | 352.73 | 352.57 | 352.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 12:15:00 | 352.80 | 352.57 | 352.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 13:45:00 | 352.56 | 352.76 | 352.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-16 10:15:00 | 351.13 | 351.95 | 352.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2023-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 10:15:00 | 351.13 | 351.95 | 352.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-16 14:15:00 | 350.05 | 351.48 | 351.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 09:15:00 | 351.69 | 351.29 | 351.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 09:15:00 | 351.69 | 351.29 | 351.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 351.69 | 351.29 | 351.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 10:00:00 | 351.69 | 351.29 | 351.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 10:15:00 | 351.76 | 351.38 | 351.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 10:30:00 | 351.93 | 351.38 | 351.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 11:15:00 | 352.75 | 351.66 | 351.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 12:00:00 | 352.75 | 351.66 | 351.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2023-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 12:15:00 | 352.87 | 351.90 | 351.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-17 14:15:00 | 353.98 | 352.41 | 352.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 10:15:00 | 350.00 | 352.57 | 352.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 10:15:00 | 350.00 | 352.57 | 352.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 10:15:00 | 350.00 | 352.57 | 352.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 11:00:00 | 350.00 | 352.57 | 352.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2023-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 11:15:00 | 350.10 | 352.07 | 352.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 09:15:00 | 348.57 | 350.67 | 351.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 09:15:00 | 350.00 | 349.20 | 350.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 09:15:00 | 350.00 | 349.20 | 350.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 350.00 | 349.20 | 350.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 10:00:00 | 350.00 | 349.20 | 350.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 10:15:00 | 351.32 | 349.63 | 350.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 11:00:00 | 351.32 | 349.63 | 350.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 11:15:00 | 351.44 | 349.99 | 350.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 12:15:00 | 352.07 | 349.99 | 350.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2023-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-20 13:15:00 | 352.59 | 350.89 | 350.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-20 14:15:00 | 354.27 | 351.57 | 351.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-23 09:15:00 | 350.00 | 351.67 | 351.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-23 09:15:00 | 350.00 | 351.67 | 351.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 09:15:00 | 350.00 | 351.67 | 351.19 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 14:15:00 | 347.59 | 350.31 | 350.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 12:15:00 | 347.16 | 349.66 | 350.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 13:15:00 | 340.46 | 340.34 | 342.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-27 14:00:00 | 340.46 | 340.34 | 342.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 11:15:00 | 342.95 | 340.89 | 342.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-30 12:00:00 | 342.95 | 340.89 | 342.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 12:15:00 | 342.62 | 341.23 | 342.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-30 12:45:00 | 342.82 | 341.23 | 342.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 13:15:00 | 342.36 | 341.46 | 342.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-30 13:30:00 | 342.98 | 341.46 | 342.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 14:15:00 | 343.67 | 341.90 | 342.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-30 15:00:00 | 343.67 | 341.90 | 342.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 15:15:00 | 343.84 | 342.29 | 342.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-31 09:15:00 | 344.93 | 342.29 | 342.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2023-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 09:15:00 | 345.11 | 342.85 | 342.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 11:15:00 | 347.95 | 344.25 | 343.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 09:15:00 | 344.23 | 345.90 | 344.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 09:15:00 | 344.23 | 345.90 | 344.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 09:15:00 | 344.23 | 345.90 | 344.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 10:00:00 | 344.23 | 345.90 | 344.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 10:15:00 | 344.40 | 345.60 | 344.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-01 11:30:00 | 344.76 | 345.46 | 344.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-10 11:15:00 | 347.33 | 348.54 | 348.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2023-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 11:15:00 | 347.33 | 348.54 | 348.70 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-11-12 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-12 18:15:00 | 350.40 | 349.01 | 348.85 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 10:15:00 | 347.68 | 348.66 | 348.71 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 11:15:00 | 349.20 | 348.77 | 348.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 09:15:00 | 352.01 | 349.62 | 349.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 14:15:00 | 354.05 | 354.61 | 352.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-16 15:00:00 | 354.05 | 354.61 | 352.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 09:15:00 | 353.91 | 354.43 | 353.18 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2023-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 11:15:00 | 352.92 | 353.11 | 353.13 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 12:15:00 | 354.07 | 353.30 | 353.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-20 13:15:00 | 354.17 | 353.48 | 353.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-21 12:15:00 | 352.48 | 353.56 | 353.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-21 12:15:00 | 352.48 | 353.56 | 353.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 12:15:00 | 352.48 | 353.56 | 353.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-21 13:00:00 | 352.48 | 353.56 | 353.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 13:15:00 | 353.75 | 353.60 | 353.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 14:45:00 | 354.02 | 353.55 | 353.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-22 09:15:00 | 351.28 | 353.10 | 353.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2023-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 09:15:00 | 351.28 | 353.10 | 353.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 11:15:00 | 348.19 | 351.76 | 352.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 10:15:00 | 349.19 | 348.50 | 349.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 10:15:00 | 349.19 | 348.50 | 349.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 10:15:00 | 349.19 | 348.50 | 349.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 11:00:00 | 349.19 | 348.50 | 349.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 11:15:00 | 348.47 | 348.49 | 349.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 11:45:00 | 348.54 | 348.49 | 349.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 14:15:00 | 348.56 | 348.34 | 349.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 15:00:00 | 348.56 | 348.34 | 349.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 15:15:00 | 349.38 | 348.55 | 349.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 09:45:00 | 349.20 | 348.67 | 349.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 10:15:00 | 348.01 | 348.54 | 349.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-28 11:30:00 | 347.71 | 348.41 | 348.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-29 11:15:00 | 350.93 | 349.03 | 348.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2023-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 11:15:00 | 350.93 | 349.03 | 348.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 15:15:00 | 351.80 | 350.29 | 349.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 09:15:00 | 350.13 | 350.26 | 349.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 09:15:00 | 350.13 | 350.26 | 349.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 350.13 | 350.26 | 349.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 09:45:00 | 349.70 | 350.26 | 349.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 10:15:00 | 351.05 | 350.42 | 349.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 10:30:00 | 350.60 | 350.42 | 349.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 13:15:00 | 350.39 | 350.42 | 349.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 13:30:00 | 350.53 | 350.42 | 349.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 14:15:00 | 351.12 | 350.56 | 350.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 14:45:00 | 349.92 | 350.56 | 350.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 11:15:00 | 350.11 | 350.95 | 350.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-01 12:00:00 | 350.11 | 350.95 | 350.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 12:15:00 | 350.45 | 350.85 | 350.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 09:15:00 | 353.89 | 350.43 | 350.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-12 15:15:00 | 364.21 | 366.71 | 366.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2023-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 15:15:00 | 364.21 | 366.71 | 366.95 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-12-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 09:15:00 | 372.00 | 366.99 | 366.75 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2023-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 09:15:00 | 368.43 | 369.63 | 369.67 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2023-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 09:15:00 | 370.60 | 369.58 | 369.52 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 366.17 | 368.88 | 369.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 14:15:00 | 365.47 | 368.19 | 368.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 12:15:00 | 368.80 | 367.33 | 368.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-21 12:15:00 | 368.80 | 367.33 | 368.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 12:15:00 | 368.80 | 367.33 | 368.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 13:00:00 | 368.80 | 367.33 | 368.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 13:15:00 | 368.73 | 367.61 | 368.11 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2023-12-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-21 15:15:00 | 370.98 | 368.84 | 368.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 09:15:00 | 374.80 | 370.03 | 369.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-22 15:15:00 | 371.38 | 371.92 | 370.76 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 09:45:00 | 373.20 | 372.16 | 370.97 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 380.20 | 381.92 | 379.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 10:15:00 | 379.12 | 381.92 | 379.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 10:15:00 | 379.16 | 381.37 | 379.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-12-29 10:15:00 | 379.16 | 381.37 | 379.70 | SL hit (close<ema400) qty=1.00 sl=379.70 alert=retest1 |

### Cycle 50 — SELL (started 2024-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 10:15:00 | 376.80 | 379.94 | 380.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-02 11:15:00 | 375.81 | 379.11 | 379.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-03 14:15:00 | 373.26 | 373.25 | 375.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-03 15:00:00 | 373.26 | 373.25 | 375.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 09:15:00 | 368.95 | 367.10 | 369.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-09 10:00:00 | 368.95 | 367.10 | 369.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 10:15:00 | 371.34 | 367.95 | 369.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-09 11:00:00 | 371.34 | 367.95 | 369.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 11:15:00 | 370.51 | 368.46 | 369.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-09 14:15:00 | 366.23 | 369.14 | 369.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-15 09:15:00 | 368.84 | 366.65 | 366.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2024-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 09:15:00 | 368.84 | 366.65 | 366.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-15 10:15:00 | 369.72 | 367.26 | 366.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 14:15:00 | 369.60 | 370.20 | 369.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-16 15:00:00 | 369.60 | 370.20 | 369.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 15:15:00 | 369.60 | 370.08 | 369.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 09:15:00 | 363.87 | 370.08 | 369.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 362.59 | 368.58 | 368.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 09:45:00 | 363.15 | 368.58 | 368.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2024-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 10:15:00 | 359.88 | 366.84 | 367.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 11:15:00 | 357.49 | 364.97 | 366.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 358.51 | 357.05 | 359.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 09:15:00 | 358.51 | 357.05 | 359.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 358.51 | 357.05 | 359.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 09:30:00 | 358.00 | 357.05 | 359.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 10:15:00 | 356.12 | 354.89 | 356.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-20 11:15:00 | 355.30 | 354.89 | 356.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-20 12:00:00 | 355.54 | 355.02 | 356.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-20 12:15:00 | 363.39 | 356.69 | 357.33 | SL hit (close>static) qty=1.00 sl=358.25 alert=retest2 |

### Cycle 53 — BUY (started 2024-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 13:15:00 | 362.05 | 357.76 | 357.76 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 13:15:00 | 354.52 | 357.98 | 358.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-24 11:15:00 | 354.30 | 357.13 | 357.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 14:15:00 | 358.08 | 356.46 | 357.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 14:15:00 | 358.08 | 356.46 | 357.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 358.08 | 356.46 | 357.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 14:30:00 | 358.87 | 356.46 | 357.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 15:15:00 | 357.01 | 356.57 | 357.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:15:00 | 358.00 | 356.57 | 357.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 355.60 | 356.38 | 357.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 10:15:00 | 354.23 | 356.38 | 357.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-29 09:15:00 | 361.80 | 355.81 | 356.06 | SL hit (close>static) qty=1.00 sl=359.52 alert=retest2 |

### Cycle 55 — BUY (started 2024-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 10:15:00 | 363.30 | 357.31 | 356.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 14:15:00 | 365.18 | 361.24 | 358.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 10:15:00 | 361.95 | 362.00 | 359.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-30 10:30:00 | 361.06 | 362.00 | 359.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 09:15:00 | 365.82 | 363.87 | 361.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 09:45:00 | 362.99 | 363.87 | 361.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 09:15:00 | 364.99 | 364.86 | 363.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 09:30:00 | 363.26 | 364.86 | 363.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 11:15:00 | 365.63 | 365.18 | 363.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 12:00:00 | 365.63 | 365.18 | 363.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 13:15:00 | 364.60 | 364.95 | 363.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 13:30:00 | 364.49 | 364.95 | 363.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 14:15:00 | 364.24 | 364.81 | 364.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 14:30:00 | 363.94 | 364.81 | 364.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 15:15:00 | 364.59 | 364.76 | 364.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 09:15:00 | 366.47 | 364.76 | 364.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 10:30:00 | 365.20 | 365.22 | 364.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 11:00:00 | 366.32 | 365.22 | 364.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 13:45:00 | 365.94 | 365.61 | 364.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 14:15:00 | 364.22 | 365.33 | 364.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 15:00:00 | 364.22 | 365.33 | 364.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 15:15:00 | 364.87 | 365.24 | 364.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 09:15:00 | 362.18 | 365.24 | 364.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 09:15:00 | 363.25 | 364.84 | 364.63 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-02-05 09:15:00 | 363.25 | 364.84 | 364.63 | SL hit (close<static) qty=1.00 sl=363.95 alert=retest2 |

### Cycle 56 — SELL (started 2024-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 10:15:00 | 361.69 | 364.21 | 364.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-06 09:15:00 | 359.42 | 361.55 | 362.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-07 09:15:00 | 361.08 | 359.44 | 360.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-07 09:15:00 | 361.08 | 359.44 | 360.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 09:15:00 | 361.08 | 359.44 | 360.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-07 09:30:00 | 360.62 | 359.44 | 360.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 10:15:00 | 360.99 | 359.75 | 360.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 11:30:00 | 358.83 | 359.38 | 360.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-08 09:15:00 | 359.20 | 359.17 | 360.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-12 12:15:00 | 340.89 | 344.95 | 348.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-12 12:15:00 | 341.24 | 344.95 | 348.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-13 09:15:00 | 345.91 | 343.97 | 346.97 | SL hit (close>ema200) qty=0.50 sl=343.97 alert=retest2 |

### Cycle 57 — BUY (started 2024-02-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 12:15:00 | 350.83 | 347.05 | 347.00 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-02-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 10:15:00 | 345.40 | 346.98 | 347.07 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-02-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 15:15:00 | 348.39 | 347.18 | 347.08 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 14:15:00 | 346.07 | 347.19 | 347.25 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-20 09:15:00 | 351.41 | 347.91 | 347.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-20 14:15:00 | 352.42 | 350.49 | 349.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-21 09:15:00 | 350.04 | 350.86 | 349.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-21 10:00:00 | 350.04 | 350.86 | 349.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 10:15:00 | 350.97 | 350.88 | 349.67 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-02-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 09:15:00 | 344.40 | 348.91 | 349.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-23 13:15:00 | 343.50 | 345.00 | 346.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-23 14:15:00 | 345.18 | 345.03 | 346.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-23 14:45:00 | 345.35 | 345.03 | 346.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 341.87 | 341.62 | 342.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 10:45:00 | 338.66 | 341.11 | 342.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-01 11:15:00 | 344.40 | 340.21 | 339.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2024-03-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 11:15:00 | 344.40 | 340.21 | 339.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 13:15:00 | 344.98 | 341.74 | 340.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 11:15:00 | 344.24 | 344.69 | 343.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-04 12:00:00 | 344.24 | 344.69 | 343.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 09:15:00 | 342.74 | 344.63 | 343.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 10:00:00 | 342.74 | 344.63 | 343.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 10:15:00 | 343.19 | 344.34 | 343.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 10:30:00 | 342.84 | 344.34 | 343.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 12:15:00 | 343.67 | 344.13 | 343.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 12:30:00 | 342.11 | 344.13 | 343.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 13:15:00 | 343.15 | 343.93 | 343.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 14:00:00 | 343.15 | 343.93 | 343.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 14:15:00 | 343.09 | 343.76 | 343.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 14:30:00 | 343.30 | 343.76 | 343.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 15:15:00 | 343.60 | 343.73 | 343.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-06 09:15:00 | 344.99 | 343.73 | 343.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-11 11:15:00 | 345.80 | 348.11 | 348.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2024-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 11:15:00 | 345.80 | 348.11 | 348.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 13:15:00 | 345.26 | 347.12 | 347.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-13 09:15:00 | 347.25 | 345.46 | 346.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-13 09:15:00 | 347.25 | 345.46 | 346.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 347.25 | 345.46 | 346.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-13 10:00:00 | 347.25 | 345.46 | 346.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 10:15:00 | 346.37 | 345.64 | 346.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-13 11:15:00 | 344.94 | 345.64 | 346.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-13 15:00:00 | 345.56 | 346.19 | 346.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-14 10:00:00 | 345.18 | 345.81 | 346.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-14 10:15:00 | 350.00 | 346.64 | 346.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2024-03-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 10:15:00 | 350.00 | 346.64 | 346.48 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 09:15:00 | 345.29 | 347.01 | 347.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-18 10:15:00 | 344.95 | 346.60 | 346.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-18 11:15:00 | 347.89 | 346.85 | 346.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-18 11:15:00 | 347.89 | 346.85 | 346.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 11:15:00 | 347.89 | 346.85 | 346.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 12:00:00 | 347.89 | 346.85 | 346.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 12:15:00 | 347.17 | 346.92 | 346.95 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2024-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 14:15:00 | 347.56 | 347.06 | 347.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-19 11:15:00 | 349.26 | 347.66 | 347.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-20 09:15:00 | 348.09 | 348.89 | 348.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 09:15:00 | 348.09 | 348.89 | 348.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 348.09 | 348.89 | 348.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 10:00:00 | 348.09 | 348.89 | 348.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 10:15:00 | 349.52 | 349.02 | 348.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 10:45:00 | 349.33 | 349.02 | 348.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 11:15:00 | 350.89 | 349.39 | 348.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 12:15:00 | 351.12 | 349.39 | 348.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-26 14:15:00 | 350.84 | 352.98 | 353.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 14:15:00 | 350.84 | 352.98 | 353.14 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2024-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 11:15:00 | 354.08 | 353.30 | 353.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-27 14:15:00 | 355.18 | 353.95 | 353.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-01 14:15:00 | 357.89 | 358.48 | 357.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-01 14:30:00 | 357.92 | 358.48 | 357.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 09:15:00 | 355.21 | 357.75 | 357.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-02 10:00:00 | 355.21 | 357.75 | 357.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 10:15:00 | 354.48 | 357.09 | 356.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-02 11:00:00 | 354.48 | 357.09 | 356.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2024-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-02 11:15:00 | 353.55 | 356.38 | 356.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-02 12:15:00 | 352.29 | 355.57 | 356.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-04 10:15:00 | 348.77 | 348.30 | 350.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-04 11:00:00 | 348.77 | 348.30 | 350.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 13:15:00 | 349.86 | 348.10 | 349.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-04 14:00:00 | 349.86 | 348.10 | 349.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 14:15:00 | 350.15 | 348.51 | 349.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-04 15:00:00 | 350.15 | 348.51 | 349.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 15:15:00 | 349.70 | 348.75 | 349.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-05 09:15:00 | 350.59 | 348.75 | 349.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 351.98 | 349.40 | 350.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-05 09:45:00 | 352.10 | 349.40 | 350.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 10:15:00 | 352.38 | 349.99 | 350.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-05 10:30:00 | 351.81 | 349.99 | 350.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2024-04-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 11:15:00 | 353.18 | 350.63 | 350.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-05 13:15:00 | 354.91 | 351.83 | 351.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-09 10:15:00 | 356.46 | 357.13 | 355.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-09 11:00:00 | 356.46 | 357.13 | 355.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 13:15:00 | 355.45 | 356.69 | 355.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 14:00:00 | 355.45 | 356.69 | 355.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 14:15:00 | 356.81 | 356.71 | 355.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-10 09:15:00 | 359.74 | 356.77 | 355.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-15 15:15:00 | 359.53 | 360.41 | 360.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2024-04-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 15:15:00 | 359.53 | 360.41 | 360.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-16 09:15:00 | 357.40 | 359.81 | 360.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 12:15:00 | 360.02 | 359.52 | 359.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 12:15:00 | 360.02 | 359.52 | 359.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 12:15:00 | 360.02 | 359.52 | 359.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 13:00:00 | 360.02 | 359.52 | 359.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 13:15:00 | 360.03 | 359.62 | 359.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 14:00:00 | 360.03 | 359.62 | 359.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 14:15:00 | 358.80 | 359.46 | 359.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 14:30:00 | 359.56 | 359.46 | 359.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 15:15:00 | 359.98 | 359.56 | 359.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 09:15:00 | 361.98 | 359.56 | 359.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 360.90 | 359.83 | 359.94 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2024-04-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 10:15:00 | 361.50 | 360.16 | 360.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 12:15:00 | 362.40 | 360.72 | 360.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-18 13:15:00 | 360.59 | 360.69 | 360.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 13:15:00 | 360.59 | 360.69 | 360.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 13:15:00 | 360.59 | 360.69 | 360.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 13:45:00 | 360.25 | 360.69 | 360.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2024-04-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 14:15:00 | 356.86 | 359.93 | 360.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 09:15:00 | 354.72 | 358.48 | 359.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 13:15:00 | 359.60 | 357.88 | 358.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 13:15:00 | 359.60 | 357.88 | 358.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 13:15:00 | 359.60 | 357.88 | 358.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 13:45:00 | 360.51 | 357.88 | 358.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 14:15:00 | 358.76 | 358.05 | 358.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 15:15:00 | 357.98 | 358.05 | 358.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 09:15:00 | 362.12 | 358.85 | 358.95 | SL hit (close>static) qty=1.00 sl=359.88 alert=retest2 |

### Cycle 75 — BUY (started 2024-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 10:15:00 | 361.39 | 359.36 | 359.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 364.74 | 361.63 | 360.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 14:15:00 | 362.88 | 363.14 | 361.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-23 15:00:00 | 362.88 | 363.14 | 361.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 09:15:00 | 365.41 | 363.41 | 362.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 10:45:00 | 366.82 | 364.20 | 362.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-25 09:15:00 | 332.78 | 360.06 | 361.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2024-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 09:15:00 | 332.78 | 360.06 | 361.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-02 09:15:00 | 312.70 | 323.62 | 327.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 09:15:00 | 324.40 | 314.54 | 317.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-06 09:15:00 | 324.40 | 314.54 | 317.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 324.40 | 314.54 | 317.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 10:00:00 | 324.40 | 314.54 | 317.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2024-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 12:15:00 | 325.15 | 319.95 | 319.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-07 10:15:00 | 326.84 | 323.58 | 321.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 11:15:00 | 329.19 | 329.28 | 327.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-09 11:15:00 | 329.19 | 329.28 | 327.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 11:15:00 | 329.19 | 329.28 | 327.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 11:30:00 | 327.83 | 329.28 | 327.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 329.41 | 328.95 | 327.96 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2024-05-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-10 12:15:00 | 324.62 | 327.32 | 327.41 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 09:15:00 | 328.75 | 326.81 | 326.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 11:15:00 | 330.00 | 328.88 | 328.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 13:15:00 | 330.97 | 331.00 | 329.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 14:00:00 | 330.97 | 331.00 | 329.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 337.68 | 339.02 | 337.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 10:00:00 | 337.68 | 339.02 | 337.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 337.01 | 338.62 | 337.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 10:45:00 | 336.72 | 338.62 | 337.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 339.28 | 338.75 | 337.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 12:45:00 | 339.52 | 338.95 | 337.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 13:45:00 | 339.65 | 339.03 | 338.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 15:00:00 | 339.98 | 339.22 | 338.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 14:15:00 | 340.10 | 340.33 | 339.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 340.80 | 340.42 | 339.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 09:15:00 | 342.49 | 340.42 | 340.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 11:15:00 | 341.58 | 342.00 | 341.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 12:30:00 | 341.67 | 341.99 | 341.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 15:15:00 | 340.00 | 341.27 | 341.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2024-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 15:15:00 | 340.00 | 341.27 | 341.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 09:15:00 | 339.06 | 340.83 | 341.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 09:15:00 | 339.57 | 339.31 | 340.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 09:15:00 | 339.57 | 339.31 | 340.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 339.57 | 339.31 | 340.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 09:45:00 | 336.19 | 337.86 | 338.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 341.54 | 337.33 | 337.78 | SL hit (close>static) qty=1.00 sl=341.52 alert=retest2 |

### Cycle 81 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 343.20 | 338.51 | 338.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 11:15:00 | 343.98 | 339.60 | 338.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 335.00 | 340.48 | 339.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 335.00 | 340.48 | 339.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 335.00 | 340.48 | 339.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 331.87 | 340.48 | 339.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 327.25 | 337.84 | 338.63 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 340.00 | 336.29 | 336.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 14:15:00 | 344.44 | 338.73 | 337.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 348.54 | 349.22 | 347.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-10 15:15:00 | 348.54 | 349.22 | 347.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 348.54 | 349.22 | 347.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:15:00 | 346.34 | 349.22 | 347.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 346.87 | 348.75 | 347.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:45:00 | 345.56 | 348.75 | 347.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 10:15:00 | 346.15 | 348.23 | 347.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 11:00:00 | 346.15 | 348.23 | 347.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2024-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-11 14:15:00 | 344.15 | 346.51 | 346.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-12 12:15:00 | 343.25 | 345.15 | 345.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-13 09:15:00 | 347.00 | 344.76 | 345.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-13 09:15:00 | 347.00 | 344.76 | 345.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 347.00 | 344.76 | 345.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-13 10:00:00 | 347.00 | 344.76 | 345.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 344.89 | 344.78 | 345.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-13 11:15:00 | 343.98 | 344.78 | 345.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-14 09:15:00 | 342.96 | 344.97 | 345.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-14 12:45:00 | 344.33 | 344.31 | 344.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 348.00 | 344.36 | 344.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 09:15:00 | 348.00 | 344.36 | 344.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 09:15:00 | 353.27 | 348.56 | 346.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 09:15:00 | 350.28 | 352.00 | 349.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 350.28 | 352.00 | 349.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 350.28 | 352.00 | 349.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:30:00 | 349.83 | 352.00 | 349.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 350.50 | 351.70 | 349.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:45:00 | 350.24 | 351.70 | 349.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 351.60 | 352.75 | 351.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:00:00 | 351.60 | 352.75 | 351.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 355.07 | 353.95 | 352.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 09:30:00 | 354.69 | 353.95 | 352.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 363.84 | 360.21 | 357.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 13:30:00 | 364.44 | 362.07 | 359.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 14:30:00 | 364.55 | 362.92 | 360.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 15:00:00 | 366.32 | 362.92 | 360.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 09:15:00 | 354.15 | 360.00 | 360.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 09:15:00 | 354.15 | 360.00 | 360.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 11:15:00 | 348.64 | 357.02 | 359.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 09:15:00 | 357.90 | 355.13 | 357.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 09:15:00 | 357.90 | 355.13 | 357.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 357.90 | 355.13 | 357.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 09:45:00 | 358.69 | 355.13 | 357.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 357.77 | 355.65 | 357.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 10:30:00 | 358.92 | 355.65 | 357.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 11:15:00 | 360.61 | 356.65 | 357.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 11:45:00 | 360.35 | 356.65 | 357.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 12:15:00 | 360.00 | 357.32 | 357.70 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2024-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 14:15:00 | 362.68 | 358.79 | 358.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 367.10 | 360.93 | 359.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 10:15:00 | 367.24 | 369.44 | 367.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 10:15:00 | 367.24 | 369.44 | 367.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 367.24 | 369.44 | 367.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 11:00:00 | 367.24 | 369.44 | 367.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 367.78 | 369.11 | 367.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 12:15:00 | 368.49 | 369.11 | 367.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 10:15:00 | 365.56 | 368.13 | 368.00 | SL hit (close<static) qty=1.00 sl=366.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 11:15:00 | 364.63 | 367.43 | 367.69 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 13:15:00 | 370.05 | 367.17 | 367.17 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 11:15:00 | 366.67 | 367.21 | 367.23 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 12:15:00 | 368.38 | 367.44 | 367.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 13:15:00 | 369.28 | 367.81 | 367.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 14:15:00 | 366.23 | 367.49 | 367.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 14:15:00 | 366.23 | 367.49 | 367.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 14:15:00 | 366.23 | 367.49 | 367.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 15:00:00 | 366.23 | 367.49 | 367.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 366.80 | 367.35 | 367.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:15:00 | 368.64 | 367.35 | 367.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 367.45 | 367.37 | 367.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 13:30:00 | 370.02 | 368.36 | 367.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 09:15:00 | 364.75 | 367.61 | 367.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 09:15:00 | 364.75 | 367.61 | 367.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 10:15:00 | 362.80 | 366.65 | 367.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 12:15:00 | 362.88 | 362.33 | 364.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-18 13:00:00 | 362.88 | 362.33 | 364.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 364.52 | 362.76 | 364.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:00:00 | 364.52 | 362.76 | 364.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 365.47 | 363.31 | 364.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:45:00 | 365.15 | 363.31 | 364.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 365.34 | 363.66 | 364.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:45:00 | 365.08 | 363.66 | 364.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 362.99 | 363.53 | 364.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 09:15:00 | 352.04 | 363.52 | 363.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-25 15:15:00 | 355.64 | 353.23 | 353.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2024-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 15:15:00 | 355.64 | 353.23 | 353.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 09:15:00 | 355.82 | 353.75 | 353.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 11:15:00 | 360.80 | 361.25 | 358.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 12:00:00 | 360.80 | 361.25 | 358.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 12:15:00 | 359.01 | 360.80 | 358.70 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2024-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 09:15:00 | 357.97 | 358.48 | 358.53 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 14:15:00 | 361.83 | 359.07 | 358.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 15:15:00 | 362.21 | 359.69 | 359.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 09:15:00 | 358.84 | 359.52 | 359.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 09:15:00 | 358.84 | 359.52 | 359.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 358.84 | 359.52 | 359.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:00:00 | 358.84 | 359.52 | 359.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 358.95 | 359.41 | 359.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:00:00 | 358.95 | 359.41 | 359.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 358.11 | 359.15 | 358.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:45:00 | 358.00 | 359.15 | 358.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 358.91 | 359.10 | 358.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:30:00 | 358.06 | 359.10 | 358.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 358.99 | 359.08 | 358.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 13:45:00 | 358.88 | 359.08 | 358.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 358.96 | 359.06 | 358.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 15:00:00 | 358.96 | 359.06 | 358.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2024-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 15:15:00 | 358.15 | 358.87 | 358.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 356.67 | 358.43 | 358.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 10:15:00 | 359.02 | 358.55 | 358.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 10:15:00 | 359.02 | 358.55 | 358.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 359.02 | 358.55 | 358.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 10:45:00 | 359.37 | 358.55 | 358.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2024-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-02 11:15:00 | 361.64 | 359.17 | 358.98 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 354.51 | 358.94 | 359.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 13:15:00 | 353.22 | 354.96 | 356.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 14:15:00 | 355.83 | 354.14 | 354.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 14:15:00 | 355.83 | 354.14 | 354.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 355.83 | 354.14 | 354.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 355.83 | 354.14 | 354.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 355.32 | 354.38 | 354.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 354.99 | 354.38 | 354.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 12:30:00 | 354.81 | 354.31 | 354.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:30:00 | 355.05 | 354.64 | 354.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 11:30:00 | 354.80 | 354.73 | 354.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 12:15:00 | 355.10 | 354.81 | 354.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2024-08-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 12:15:00 | 355.10 | 354.81 | 354.80 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-08-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 13:15:00 | 354.50 | 354.75 | 354.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 14:15:00 | 353.98 | 354.59 | 354.70 | Break + close below crossover candle low |

### Cycle 101 — BUY (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 10:15:00 | 357.19 | 354.71 | 354.69 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 12:15:00 | 352.44 | 354.71 | 354.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 351.14 | 353.99 | 354.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 11:15:00 | 355.21 | 351.52 | 352.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 11:15:00 | 355.21 | 351.52 | 352.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 355.21 | 351.52 | 352.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 12:00:00 | 355.21 | 351.52 | 352.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 354.38 | 352.09 | 352.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 12:30:00 | 354.72 | 352.09 | 352.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 13:15:00 | 356.14 | 352.90 | 352.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 12:15:00 | 356.57 | 355.03 | 353.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 09:15:00 | 359.52 | 359.93 | 357.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 09:15:00 | 359.52 | 359.93 | 357.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 359.52 | 359.93 | 357.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:45:00 | 358.06 | 359.93 | 357.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 10:15:00 | 358.49 | 359.64 | 357.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 11:00:00 | 358.49 | 359.64 | 357.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 13:15:00 | 359.78 | 359.71 | 358.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 14:30:00 | 360.46 | 360.26 | 358.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 09:15:00 | 359.60 | 362.30 | 362.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 09:15:00 | 359.60 | 362.30 | 362.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 14:15:00 | 358.14 | 359.78 | 360.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 11:15:00 | 356.98 | 356.84 | 358.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-30 11:45:00 | 356.88 | 356.84 | 358.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 356.00 | 356.61 | 357.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:30:00 | 357.85 | 356.77 | 357.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 357.66 | 356.95 | 357.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 10:30:00 | 358.55 | 356.95 | 357.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 355.43 | 356.64 | 357.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 12:30:00 | 355.03 | 356.38 | 357.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 13:30:00 | 355.07 | 356.41 | 357.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 11:00:00 | 355.08 | 356.13 | 356.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 11:30:00 | 354.99 | 356.10 | 356.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 12:15:00 | 355.36 | 355.95 | 356.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 12:45:00 | 355.82 | 355.95 | 356.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 13:15:00 | 355.88 | 355.93 | 356.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 13:30:00 | 355.80 | 355.93 | 356.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 356.89 | 356.13 | 356.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 356.89 | 356.13 | 356.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 356.78 | 356.26 | 356.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 09:15:00 | 354.33 | 356.26 | 356.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 13:15:00 | 357.01 | 355.72 | 356.06 | SL hit (close>static) qty=1.00 sl=356.99 alert=retest2 |

### Cycle 105 — BUY (started 2024-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 14:15:00 | 358.05 | 355.67 | 355.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 11:15:00 | 359.15 | 357.37 | 356.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 15:15:00 | 357.91 | 358.01 | 357.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 09:15:00 | 357.50 | 358.01 | 357.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 359.54 | 358.32 | 357.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 10:15:00 | 359.85 | 358.32 | 357.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 11:45:00 | 359.62 | 358.91 | 357.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:15:00 | 362.66 | 358.28 | 357.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 14:15:00 | 379.53 | 380.87 | 380.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2024-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 14:15:00 | 379.53 | 380.87 | 380.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 09:15:00 | 378.46 | 380.21 | 380.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 11:15:00 | 381.12 | 380.18 | 380.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 11:15:00 | 381.12 | 380.18 | 380.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 381.12 | 380.18 | 380.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 11:45:00 | 380.66 | 380.18 | 380.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 12:15:00 | 380.57 | 380.26 | 380.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 12:45:00 | 381.01 | 380.26 | 380.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 13:15:00 | 380.37 | 380.28 | 380.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 13:30:00 | 380.43 | 380.28 | 380.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 380.30 | 380.29 | 380.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 14:30:00 | 381.27 | 380.29 | 380.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 380.67 | 380.36 | 380.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 379.72 | 380.36 | 380.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 378.24 | 379.94 | 380.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 10:15:00 | 377.99 | 379.94 | 380.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 10:45:00 | 377.63 | 379.48 | 380.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 12:15:00 | 359.09 | 362.78 | 365.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 12:15:00 | 358.75 | 362.78 | 365.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 12:15:00 | 361.90 | 360.49 | 362.66 | SL hit (close>ema200) qty=0.50 sl=360.49 alert=retest2 |

### Cycle 107 — BUY (started 2024-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 10:15:00 | 365.70 | 362.72 | 362.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 11:15:00 | 369.97 | 364.17 | 363.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 09:15:00 | 378.29 | 380.24 | 376.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-15 10:00:00 | 378.29 | 380.24 | 376.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 379.38 | 379.76 | 377.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 15:00:00 | 379.38 | 379.76 | 377.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 377.56 | 379.21 | 377.89 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2024-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 13:15:00 | 374.28 | 376.72 | 377.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 371.60 | 375.12 | 376.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 09:15:00 | 373.56 | 372.99 | 374.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 09:45:00 | 373.32 | 372.99 | 374.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 372.87 | 372.96 | 374.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 11:15:00 | 372.21 | 372.96 | 374.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 11:15:00 | 375.17 | 373.40 | 374.24 | SL hit (close>static) qty=1.00 sl=374.89 alert=retest2 |

### Cycle 109 — BUY (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 14:15:00 | 351.55 | 347.23 | 347.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 11:15:00 | 352.00 | 349.56 | 348.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 347.15 | 350.41 | 349.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 347.15 | 350.41 | 349.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 347.15 | 350.41 | 349.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 347.15 | 350.41 | 349.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 347.77 | 349.88 | 349.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:45:00 | 348.63 | 349.88 | 349.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 349.01 | 349.39 | 349.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:45:00 | 348.77 | 349.39 | 349.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 349.02 | 349.32 | 349.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 09:15:00 | 350.48 | 349.27 | 349.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 09:45:00 | 350.38 | 349.90 | 349.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 10:15:00 | 350.67 | 349.90 | 349.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 15:15:00 | 348.36 | 349.88 | 349.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 15:15:00 | 348.36 | 349.88 | 349.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 10:15:00 | 347.33 | 349.09 | 349.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 341.16 | 341.14 | 343.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 341.16 | 341.14 | 343.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 343.18 | 341.67 | 342.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 10:00:00 | 343.18 | 341.67 | 342.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 10:15:00 | 343.68 | 342.07 | 342.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 10:45:00 | 343.78 | 342.07 | 342.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 11:15:00 | 344.64 | 342.58 | 342.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:00:00 | 344.64 | 342.58 | 342.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2024-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 13:15:00 | 344.64 | 343.34 | 343.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 345.66 | 344.05 | 343.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 15:15:00 | 344.50 | 345.57 | 344.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 15:15:00 | 344.50 | 345.57 | 344.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 344.50 | 345.57 | 344.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:15:00 | 344.82 | 345.57 | 344.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 345.89 | 345.63 | 344.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 11:45:00 | 346.79 | 345.90 | 345.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 12:30:00 | 347.19 | 346.07 | 345.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 13:00:00 | 346.75 | 346.07 | 345.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 12:15:00 | 351.25 | 355.10 | 355.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 12:15:00 | 351.25 | 355.10 | 355.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 10:15:00 | 350.64 | 352.42 | 353.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 351.63 | 350.41 | 351.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 09:15:00 | 351.63 | 350.41 | 351.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 351.63 | 350.41 | 351.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:45:00 | 351.22 | 350.41 | 351.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 352.77 | 350.88 | 351.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:45:00 | 352.98 | 350.88 | 351.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 351.39 | 351.01 | 351.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:45:00 | 351.71 | 351.01 | 351.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 13:15:00 | 351.60 | 351.13 | 351.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 13:30:00 | 351.83 | 351.13 | 351.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 351.31 | 351.17 | 351.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 15:15:00 | 351.93 | 351.17 | 351.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 351.93 | 351.32 | 351.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:15:00 | 352.41 | 351.32 | 351.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 350.19 | 351.09 | 351.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:30:00 | 352.13 | 351.09 | 351.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 350.60 | 350.99 | 351.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:45:00 | 350.39 | 350.99 | 351.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2024-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 11:15:00 | 353.36 | 351.47 | 351.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 12:15:00 | 355.88 | 352.35 | 351.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 09:15:00 | 353.10 | 353.91 | 352.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-06 09:15:00 | 353.10 | 353.91 | 352.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 353.10 | 353.91 | 352.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:00:00 | 353.10 | 353.91 | 352.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 355.00 | 354.13 | 353.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 13:30:00 | 356.23 | 354.88 | 353.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 09:15:00 | 357.81 | 355.00 | 353.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 11:15:00 | 355.84 | 357.67 | 357.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 11:15:00 | 355.84 | 357.67 | 357.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 12:15:00 | 355.09 | 357.16 | 357.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 11:15:00 | 354.44 | 354.28 | 355.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 12:00:00 | 354.44 | 354.28 | 355.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 356.00 | 354.62 | 355.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:00:00 | 356.00 | 354.62 | 355.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 358.13 | 355.32 | 355.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:45:00 | 358.83 | 355.32 | 355.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2024-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 14:15:00 | 361.15 | 356.49 | 356.43 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 12:15:00 | 355.96 | 357.70 | 357.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 350.55 | 355.11 | 356.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 10:15:00 | 351.63 | 350.50 | 351.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-23 10:15:00 | 351.63 | 350.50 | 351.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 351.63 | 350.50 | 351.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:00:00 | 351.63 | 350.50 | 351.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 351.14 | 350.62 | 351.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:45:00 | 352.07 | 350.62 | 351.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 13:15:00 | 350.78 | 349.60 | 350.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 14:00:00 | 350.78 | 349.60 | 350.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 14:15:00 | 349.68 | 349.62 | 350.35 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2024-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 15:15:00 | 351.83 | 350.62 | 350.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 353.32 | 351.16 | 350.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 12:15:00 | 351.43 | 351.77 | 351.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 12:15:00 | 351.43 | 351.77 | 351.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 351.43 | 351.77 | 351.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:00:00 | 351.43 | 351.77 | 351.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 351.47 | 351.71 | 351.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 14:15:00 | 351.80 | 351.71 | 351.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 351.62 | 351.69 | 351.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 11:15:00 | 353.43 | 351.78 | 351.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 13:15:00 | 348.22 | 351.42 | 351.40 | SL hit (close<static) qty=1.00 sl=351.04 alert=retest2 |

### Cycle 118 — SELL (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 14:15:00 | 348.79 | 350.90 | 351.16 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2024-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 10:15:00 | 356.93 | 352.10 | 351.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 12:15:00 | 358.23 | 354.11 | 352.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 361.03 | 365.46 | 363.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 361.03 | 365.46 | 363.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 361.03 | 365.46 | 363.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 359.87 | 365.46 | 363.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 357.22 | 363.81 | 362.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 357.22 | 363.81 | 362.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 12:15:00 | 355.88 | 361.26 | 361.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 14:15:00 | 355.45 | 359.29 | 360.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 359.91 | 355.30 | 356.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 09:15:00 | 359.91 | 355.30 | 356.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 359.91 | 355.30 | 356.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:30:00 | 358.25 | 355.30 | 356.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 357.31 | 355.70 | 356.44 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2025-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 13:15:00 | 357.99 | 356.87 | 356.85 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 354.31 | 356.74 | 356.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 14:15:00 | 350.80 | 354.42 | 355.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 13:15:00 | 353.02 | 348.99 | 350.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 13:15:00 | 353.02 | 348.99 | 350.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 353.02 | 348.99 | 350.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:00:00 | 353.02 | 348.99 | 350.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 349.72 | 349.14 | 350.39 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 10:15:00 | 357.50 | 352.31 | 351.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 11:15:00 | 358.05 | 353.46 | 352.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 355.60 | 359.27 | 357.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 355.60 | 359.27 | 357.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 355.60 | 359.27 | 357.32 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 12:15:00 | 351.76 | 355.42 | 355.84 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 09:15:00 | 384.04 | 359.64 | 357.43 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-01-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 15:15:00 | 376.68 | 378.63 | 378.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 374.00 | 377.71 | 378.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 12:15:00 | 377.70 | 376.99 | 377.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 12:15:00 | 377.70 | 376.99 | 377.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 12:15:00 | 377.70 | 376.99 | 377.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 12:30:00 | 378.20 | 376.99 | 377.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 13:15:00 | 377.40 | 377.07 | 377.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 14:00:00 | 377.40 | 377.07 | 377.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 14:15:00 | 376.21 | 376.90 | 377.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 15:15:00 | 375.74 | 376.90 | 377.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 10:30:00 | 375.70 | 376.56 | 377.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-28 12:15:00 | 378.78 | 377.19 | 377.44 | SL hit (close>static) qty=1.00 sl=377.99 alert=retest2 |

### Cycle 127 — BUY (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 09:15:00 | 381.37 | 378.15 | 377.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 10:15:00 | 382.20 | 378.96 | 378.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 11:15:00 | 382.90 | 382.96 | 381.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 11:30:00 | 383.40 | 382.96 | 381.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 12:15:00 | 380.82 | 382.53 | 381.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 13:00:00 | 380.82 | 382.53 | 381.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 378.61 | 381.75 | 380.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 13:45:00 | 379.43 | 381.75 | 380.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 380.39 | 381.48 | 380.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-30 15:15:00 | 380.76 | 381.48 | 380.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 09:30:00 | 380.91 | 380.95 | 380.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 10:30:00 | 380.79 | 380.68 | 380.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 11:15:00 | 379.57 | 380.46 | 380.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 11:15:00 | 379.57 | 380.46 | 380.54 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 09:15:00 | 382.17 | 380.54 | 380.50 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 11:15:00 | 377.80 | 380.22 | 380.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 12:15:00 | 376.63 | 379.50 | 380.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 13:15:00 | 380.48 | 379.70 | 380.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 13:15:00 | 380.48 | 379.70 | 380.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 380.48 | 379.70 | 380.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 14:00:00 | 380.48 | 379.70 | 380.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 380.15 | 379.79 | 380.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 14:30:00 | 381.12 | 379.79 | 380.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 378.73 | 379.95 | 380.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:30:00 | 378.65 | 379.95 | 380.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 379.00 | 379.76 | 380.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 11:45:00 | 377.91 | 379.45 | 379.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 13:15:00 | 377.97 | 379.29 | 379.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 09:45:00 | 377.60 | 378.02 | 378.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 12:15:00 | 380.90 | 378.46 | 378.88 | SL hit (close>static) qty=1.00 sl=380.26 alert=retest2 |

### Cycle 131 — BUY (started 2025-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 13:15:00 | 384.00 | 379.57 | 379.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 14:15:00 | 385.35 | 380.72 | 379.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 14:15:00 | 382.68 | 383.23 | 381.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-05 15:00:00 | 382.68 | 383.23 | 381.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 15:15:00 | 381.80 | 382.94 | 381.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:15:00 | 380.26 | 382.94 | 381.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 380.79 | 382.51 | 381.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:30:00 | 380.31 | 382.51 | 381.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 380.30 | 382.07 | 381.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:00:00 | 380.30 | 382.07 | 381.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 382.39 | 382.05 | 381.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 13:45:00 | 382.43 | 382.22 | 381.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 14:15:00 | 382.71 | 382.78 | 382.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-11 14:15:00 | 384.05 | 385.67 | 385.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2025-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 14:15:00 | 384.05 | 385.67 | 385.77 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 12:15:00 | 388.41 | 385.47 | 385.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-12 14:15:00 | 389.38 | 386.36 | 385.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 10:15:00 | 389.52 | 392.19 | 390.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-14 10:15:00 | 389.52 | 392.19 | 390.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 389.52 | 392.19 | 390.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 11:00:00 | 389.52 | 392.19 | 390.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 11:15:00 | 387.84 | 391.32 | 390.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 12:00:00 | 387.84 | 391.32 | 390.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 15:15:00 | 389.96 | 390.10 | 389.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-17 09:15:00 | 389.70 | 390.10 | 389.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2025-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 09:15:00 | 387.17 | 389.51 | 389.60 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2025-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 12:15:00 | 390.00 | 389.20 | 389.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 13:15:00 | 391.07 | 389.57 | 389.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 09:15:00 | 394.61 | 395.30 | 393.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 09:15:00 | 394.61 | 395.30 | 393.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 394.61 | 395.30 | 393.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 09:45:00 | 393.49 | 395.30 | 393.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 10:15:00 | 393.10 | 394.86 | 393.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 10:45:00 | 393.35 | 394.86 | 393.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 11:15:00 | 392.81 | 394.45 | 393.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 11:45:00 | 392.55 | 394.45 | 393.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 12:15:00 | 393.80 | 394.32 | 393.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 13:15:00 | 393.95 | 394.32 | 393.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 09:15:00 | 387.99 | 392.96 | 393.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 09:15:00 | 387.99 | 392.96 | 393.02 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 09:15:00 | 395.87 | 392.33 | 391.86 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 11:15:00 | 389.64 | 391.84 | 392.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 15:15:00 | 388.38 | 390.36 | 391.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 11:15:00 | 382.58 | 382.58 | 385.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 12:00:00 | 382.58 | 382.58 | 385.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 385.14 | 382.65 | 383.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:00:00 | 385.14 | 382.65 | 383.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 387.44 | 383.61 | 383.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:00:00 | 387.44 | 383.61 | 383.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 389.39 | 384.76 | 384.30 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-06 15:15:00 | 384.03 | 385.06 | 385.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 10:15:00 | 383.94 | 384.70 | 384.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-07 12:15:00 | 384.96 | 384.73 | 384.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 12:15:00 | 384.96 | 384.73 | 384.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 384.96 | 384.73 | 384.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:45:00 | 385.65 | 384.73 | 384.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 385.35 | 384.86 | 384.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 14:15:00 | 386.00 | 384.86 | 384.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2025-03-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 14:15:00 | 386.99 | 385.28 | 385.12 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 383.46 | 385.21 | 385.36 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 13:15:00 | 388.34 | 385.81 | 385.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 09:15:00 | 395.52 | 388.14 | 386.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 12:15:00 | 403.60 | 405.30 | 402.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-19 13:00:00 | 403.60 | 405.30 | 402.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 428.53 | 432.48 | 429.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:00:00 | 428.53 | 432.48 | 429.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 12:15:00 | 426.93 | 431.37 | 429.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:00:00 | 426.93 | 431.37 | 429.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 427.21 | 430.54 | 428.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:30:00 | 426.27 | 430.54 | 428.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 15:15:00 | 429.73 | 430.22 | 428.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:15:00 | 428.59 | 430.22 | 428.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 429.04 | 429.99 | 428.99 | EMA400 retest candle locked (from upside) |

### Cycle 144 — SELL (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 14:15:00 | 425.73 | 428.00 | 428.31 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 432.20 | 428.70 | 428.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 11:15:00 | 433.07 | 430.02 | 429.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 430.92 | 432.23 | 430.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 430.92 | 432.23 | 430.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 430.92 | 432.23 | 430.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:15:00 | 428.60 | 432.23 | 430.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 428.32 | 431.45 | 430.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:00:00 | 428.32 | 431.45 | 430.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 427.97 | 430.75 | 430.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:30:00 | 428.11 | 430.75 | 430.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2025-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 13:15:00 | 428.69 | 429.98 | 430.05 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 430.99 | 429.99 | 429.93 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 09:15:00 | 427.74 | 429.64 | 429.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 09:15:00 | 425.20 | 427.05 | 428.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-04 15:15:00 | 426.42 | 426.37 | 427.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 15:15:00 | 426.42 | 426.37 | 427.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 15:15:00 | 426.42 | 426.37 | 427.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 09:15:00 | 411.14 | 426.37 | 427.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 10:15:00 | 425.99 | 415.65 | 414.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 425.99 | 415.65 | 414.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 11:15:00 | 426.62 | 424.80 | 423.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 15:15:00 | 452.00 | 452.64 | 446.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 09:15:00 | 452.06 | 452.64 | 446.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 445.52 | 449.75 | 446.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:30:00 | 444.68 | 449.75 | 446.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 13:15:00 | 445.14 | 448.83 | 446.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 14:00:00 | 445.14 | 448.83 | 446.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 14:15:00 | 445.40 | 448.14 | 446.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 14:30:00 | 445.06 | 448.14 | 446.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 445.62 | 446.89 | 446.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:45:00 | 444.78 | 446.89 | 446.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 11:15:00 | 444.66 | 446.44 | 446.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 12:00:00 | 444.66 | 446.44 | 446.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 446.80 | 446.26 | 446.03 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2025-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 15:15:00 | 443.62 | 445.45 | 445.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 439.44 | 444.25 | 445.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-25 13:15:00 | 442.70 | 441.91 | 443.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 13:15:00 | 442.70 | 441.91 | 443.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 13:15:00 | 442.70 | 441.91 | 443.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 14:00:00 | 442.70 | 441.91 | 443.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 441.32 | 441.79 | 443.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 15:15:00 | 439.32 | 441.79 | 443.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-28 09:15:00 | 445.44 | 442.12 | 443.16 | SL hit (close>static) qty=1.00 sl=443.74 alert=retest2 |

### Cycle 151 — BUY (started 2025-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 14:15:00 | 445.00 | 443.87 | 443.74 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 14:15:00 | 440.74 | 443.60 | 443.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 15:15:00 | 440.42 | 442.97 | 443.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 09:15:00 | 445.02 | 443.38 | 443.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 445.02 | 443.38 | 443.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 445.02 | 443.38 | 443.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 10:00:00 | 445.02 | 443.38 | 443.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 444.46 | 443.59 | 443.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 11:45:00 | 443.46 | 443.36 | 443.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 14:15:00 | 443.16 | 443.59 | 443.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 09:15:00 | 421.29 | 433.46 | 438.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 09:15:00 | 421.00 | 433.46 | 438.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-07 09:15:00 | 415.76 | 415.64 | 420.99 | SL hit (close>ema200) qty=0.50 sl=415.64 alert=retest2 |

### Cycle 153 — BUY (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 12:15:00 | 423.88 | 421.03 | 420.73 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-05-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 13:15:00 | 420.76 | 420.81 | 420.81 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2025-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 15:15:00 | 421.32 | 420.91 | 420.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 431.38 | 423.00 | 421.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 423.16 | 426.40 | 424.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 423.16 | 426.40 | 424.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 423.16 | 426.40 | 424.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 09:45:00 | 421.98 | 426.40 | 424.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 10:15:00 | 422.70 | 425.66 | 424.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 10:30:00 | 421.58 | 425.66 | 424.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 11:15:00 | 423.34 | 425.20 | 424.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:15:00 | 422.76 | 425.20 | 424.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 421.24 | 424.41 | 424.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:45:00 | 421.48 | 424.41 | 424.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2025-05-13 13:15:00)

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

### Cycle 157 — BUY (started 2025-05-16 15:15:00)

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

### Cycle 158 — SELL (started 2025-05-20 12:15:00)

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

### Cycle 159 — BUY (started 2025-05-23 12:15:00)

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

### Cycle 160 — SELL (started 2025-05-27 10:15:00)

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

### Cycle 161 — BUY (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 15:15:00 | 419.10 | 415.33 | 415.21 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-05-30 11:15:00)

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

### Cycle 163 — BUY (started 2025-06-06 10:15:00)

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

### Cycle 164 — SELL (started 2025-06-12 15:15:00)

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

### Cycle 165 — BUY (started 2025-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 12:15:00 | 427.70 | 424.78 | 424.59 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 424.34 | 425.71 | 425.84 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-06-18 15:15:00)

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

### Cycle 168 — SELL (started 2025-06-30 09:15:00)

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

### Cycle 169 — BUY (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 09:15:00 | 446.70 | 431.79 | 429.87 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 13:15:00 | 440.00 | 442.21 | 442.48 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 445.02 | 442.91 | 442.69 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-07-15 12:15:00)

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

### Cycle 173 — BUY (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 10:15:00 | 433.58 | 432.09 | 432.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 11:15:00 | 434.58 | 432.58 | 432.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 15:15:00 | 433.08 | 433.65 | 432.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 09:15:00 | 430.18 | 433.65 | 432.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 429.80 | 432.88 | 432.69 | EMA400 retest candle locked (from upside) |

### Cycle 174 — SELL (started 2025-07-24 10:15:00)

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

### Cycle 175 — BUY (started 2025-08-01 12:15:00)

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

### Cycle 176 — SELL (started 2025-08-07 10:15:00)

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

### Cycle 177 — BUY (started 2025-08-13 11:15:00)

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

### Cycle 178 — SELL (started 2025-08-22 10:15:00)

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

### Cycle 179 — BUY (started 2025-09-09 13:15:00)

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

### Cycle 180 — SELL (started 2025-09-22 12:15:00)

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

### Cycle 181 — BUY (started 2025-09-23 14:15:00)

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

### Cycle 182 — SELL (started 2025-09-25 09:15:00)

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

### Cycle 183 — BUY (started 2025-10-01 09:15:00)

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

### Cycle 184 — SELL (started 2025-10-24 10:15:00)

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

### Cycle 185 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 421.68 | 416.67 | 416.36 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-11-21 15:15:00)

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

### Cycle 187 — BUY (started 2025-11-26 14:15:00)

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

### Cycle 188 — SELL (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 12:15:00 | 425.80 | 427.16 | 427.26 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 14:15:00 | 429.52 | 427.42 | 427.35 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 11:15:00 | 425.86 | 427.38 | 427.42 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 09:15:00 | 429.86 | 427.40 | 427.31 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2025-12-08 15:15:00)

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

### Cycle 193 — BUY (started 2025-12-11 09:15:00)

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

### Cycle 194 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 433.10 | 435.09 | 435.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 12:15:00 | 432.36 | 433.52 | 434.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 433.22 | 432.89 | 433.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 433.22 | 432.89 | 433.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 433.22 | 432.89 | 433.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 12:15:00 | 431.94 | 432.80 | 433.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 10:15:00 | 434.40 | 432.96 | 432.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — BUY (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 10:15:00 | 434.40 | 432.96 | 432.79 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 431.76 | 432.66 | 432.75 | EMA200 below EMA400 |

### Cycle 197 — BUY (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 13:15:00 | 433.84 | 432.70 | 432.70 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 432.34 | 432.65 | 432.68 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 09:15:00 | 433.18 | 432.76 | 432.72 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2025-12-29 10:15:00)

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

### Cycle 201 — BUY (started 2025-12-31 11:15:00)

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

### Cycle 202 — SELL (started 2026-01-05 14:15:00)

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

### Cycle 203 — BUY (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 12:15:00 | 428.20 | 423.72 | 423.56 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 418.80 | 424.04 | 424.70 | EMA200 below EMA400 |

### Cycle 205 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 427.70 | 424.41 | 424.02 | EMA200 above EMA400 |

### Cycle 206 — SELL (started 2026-01-23 14:15:00)

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

### Cycle 207 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 418.50 | 410.11 | 409.69 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 410.40 | 412.61 | 412.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 10:15:00 | 406.80 | 411.45 | 412.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 14:15:00 | 409.15 | 408.81 | 410.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-05 15:00:00 | 409.15 | 408.81 | 410.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 411.65 | 409.38 | 410.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 412.35 | 409.38 | 410.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 411.70 | 409.84 | 410.68 | EMA400 retest candle locked (from downside) |

### Cycle 209 — BUY (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 10:15:00 | 417.05 | 411.29 | 411.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 11:15:00 | 422.70 | 413.57 | 412.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 14:15:00 | 429.35 | 429.88 | 425.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 15:00:00 | 429.35 | 429.88 | 425.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 428.65 | 429.24 | 427.70 | EMA400 retest candle locked (from upside) |

### Cycle 210 — SELL (started 2026-02-13 09:15:00)

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

### Cycle 211 — BUY (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 14:15:00 | 426.30 | 423.84 | 423.69 | EMA200 above EMA400 |

### Cycle 212 — SELL (started 2026-02-19 12:15:00)

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

### Cycle 213 — BUY (started 2026-02-23 09:15:00)

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

### Cycle 214 — SELL (started 2026-02-25 12:15:00)

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

### Cycle 215 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 373.15 | 372.55 | 372.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 375.45 | 373.13 | 372.75 | Break + close above crossover candle high |

### Cycle 216 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 364.40 | 372.00 | 372.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 359.05 | 366.61 | 368.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 364.60 | 361.33 | 364.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 364.60 | 361.33 | 364.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 364.60 | 361.33 | 364.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:00:00 | 364.60 | 361.33 | 364.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 364.40 | 361.94 | 364.31 | EMA400 retest candle locked (from downside) |

### Cycle 217 — BUY (started 2026-03-24 15:15:00)

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

### Cycle 218 — SELL (started 2026-03-27 11:15:00)

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

### Cycle 219 — BUY (started 2026-04-06 14:15:00)

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

### Cycle 220 — SELL (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 10:15:00 | 367.00 | 372.01 | 372.05 | EMA200 below EMA400 |

### Cycle 221 — BUY (started 2026-04-15 09:15:00)

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

### Cycle 222 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 379.55 | 380.80 | 380.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 13:15:00 | 377.55 | 379.47 | 380.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 15:15:00 | 371.00 | 370.77 | 372.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:15:00 | 379.40 | 370.77 | 372.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 379.30 | 372.48 | 373.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 379.30 | 372.48 | 373.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 378.20 | 373.62 | 373.88 | EMA400 retest candle locked (from downside) |

### Cycle 223 — BUY (started 2026-04-27 11:15:00)

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

### Cycle 224 — SELL (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 10:15:00 | 375.10 | 379.50 | 379.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 11:15:00 | 373.15 | 378.23 | 379.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 10:15:00 | 374.55 | 373.66 | 376.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-05 10:45:00 | 374.65 | 373.66 | 376.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 373.25 | 372.77 | 374.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 11:30:00 | 370.95 | 372.28 | 373.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 14:15:00 | 376.05 | 371.70 | 373.13 | SL hit (close>static) qty=1.00 sl=375.10 alert=retest2 |

### Cycle 225 — BUY (started 2026-05-07 09:15:00)

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
| SELL | retest2 | 2023-05-18 12:15:00 | 383.85 | 2023-05-19 13:15:00 | 387.80 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2023-05-25 12:45:00 | 383.14 | 2023-05-25 14:15:00 | 385.98 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2023-05-31 14:45:00 | 395.15 | 2023-06-01 09:15:00 | 388.13 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2023-06-05 11:30:00 | 389.02 | 2023-06-06 13:15:00 | 389.56 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2023-06-06 10:15:00 | 388.38 | 2023-06-06 13:15:00 | 389.56 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2023-06-15 10:45:00 | 370.63 | 2023-06-21 11:15:00 | 368.45 | STOP_HIT | 1.00 | 0.59% |
| SELL | retest2 | 2023-06-16 15:15:00 | 371.40 | 2023-06-21 11:15:00 | 368.45 | STOP_HIT | 1.00 | 0.79% |
| SELL | retest2 | 2023-06-19 09:30:00 | 371.19 | 2023-06-21 11:15:00 | 368.45 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2023-06-23 15:00:00 | 365.67 | 2023-06-27 14:15:00 | 369.13 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2023-06-27 12:15:00 | 366.42 | 2023-06-27 14:15:00 | 369.13 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2023-06-27 14:15:00 | 366.64 | 2023-06-27 14:15:00 | 369.13 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2023-07-12 09:15:00 | 376.38 | 2023-07-14 10:15:00 | 374.83 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2023-07-12 12:15:00 | 375.27 | 2023-07-14 10:15:00 | 374.83 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2023-07-14 09:15:00 | 376.41 | 2023-07-14 10:15:00 | 374.83 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2023-07-19 09:15:00 | 380.31 | 2023-07-24 12:15:00 | 381.01 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2023-07-19 09:45:00 | 380.28 | 2023-07-24 12:15:00 | 381.01 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2023-07-20 11:00:00 | 381.37 | 2023-07-24 12:15:00 | 381.01 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2023-07-27 10:45:00 | 378.00 | 2023-08-08 10:15:00 | 367.79 | STOP_HIT | 1.00 | 2.70% |
| SELL | retest2 | 2023-08-10 09:15:00 | 364.60 | 2023-08-23 13:15:00 | 355.01 | STOP_HIT | 1.00 | 2.63% |
| SELL | retest2 | 2023-08-10 10:15:00 | 364.22 | 2023-08-23 13:15:00 | 355.01 | STOP_HIT | 1.00 | 2.53% |
| BUY | retest2 | 2023-08-25 11:15:00 | 356.30 | 2023-08-30 14:15:00 | 354.91 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2023-08-25 15:00:00 | 356.12 | 2023-08-30 14:15:00 | 354.91 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2023-08-28 09:15:00 | 355.88 | 2023-08-30 14:15:00 | 354.91 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2023-08-28 11:00:00 | 356.20 | 2023-08-30 14:15:00 | 354.91 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2023-08-30 09:15:00 | 357.57 | 2023-08-30 14:15:00 | 354.91 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2023-09-04 10:30:00 | 351.99 | 2023-09-05 10:15:00 | 355.98 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2023-09-04 14:45:00 | 352.30 | 2023-09-05 10:15:00 | 355.98 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest1 | 2023-09-21 09:15:00 | 356.81 | 2023-09-25 11:15:00 | 355.79 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest1 | 2023-09-27 09:15:00 | 351.49 | 2023-09-27 15:15:00 | 353.94 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2023-09-28 10:45:00 | 350.00 | 2023-10-05 11:15:00 | 347.98 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2023-10-13 11:45:00 | 352.73 | 2023-10-16 10:15:00 | 351.13 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2023-10-13 12:15:00 | 352.80 | 2023-10-16 10:15:00 | 351.13 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2023-10-13 13:45:00 | 352.56 | 2023-10-16 10:15:00 | 351.13 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2023-11-01 11:30:00 | 344.76 | 2023-11-10 11:15:00 | 347.33 | STOP_HIT | 1.00 | 0.75% |
| BUY | retest2 | 2023-11-21 14:45:00 | 354.02 | 2023-11-22 09:15:00 | 351.28 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2023-11-28 11:30:00 | 347.71 | 2023-11-29 11:15:00 | 350.93 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2023-12-04 09:15:00 | 353.89 | 2023-12-12 15:15:00 | 364.21 | STOP_HIT | 1.00 | 2.92% |
| BUY | retest1 | 2023-12-26 09:45:00 | 373.20 | 2023-12-29 10:15:00 | 379.16 | STOP_HIT | 1.00 | 1.60% |
| BUY | retest2 | 2024-01-01 11:30:00 | 382.44 | 2024-01-02 09:15:00 | 377.89 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-01-09 14:15:00 | 366.23 | 2024-01-15 09:15:00 | 368.84 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-01-20 11:15:00 | 355.30 | 2024-01-20 12:15:00 | 363.39 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-01-20 12:00:00 | 355.54 | 2024-01-20 12:15:00 | 363.39 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2024-01-25 10:15:00 | 354.23 | 2024-01-29 09:15:00 | 361.80 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-02-02 09:15:00 | 366.47 | 2024-02-05 09:15:00 | 363.25 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-02-02 10:30:00 | 365.20 | 2024-02-05 09:15:00 | 363.25 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-02-02 11:00:00 | 366.32 | 2024-02-05 09:15:00 | 363.25 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-02-02 13:45:00 | 365.94 | 2024-02-05 09:15:00 | 363.25 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-02-07 11:30:00 | 358.83 | 2024-02-12 12:15:00 | 340.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-08 09:15:00 | 359.20 | 2024-02-12 12:15:00 | 341.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-07 11:30:00 | 358.83 | 2024-02-13 09:15:00 | 345.91 | STOP_HIT | 0.50 | 3.60% |
| SELL | retest2 | 2024-02-08 09:15:00 | 359.20 | 2024-02-13 09:15:00 | 345.91 | STOP_HIT | 0.50 | 3.70% |
| SELL | retest2 | 2024-02-28 10:45:00 | 338.66 | 2024-03-01 11:15:00 | 344.40 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-03-06 09:15:00 | 344.99 | 2024-03-11 11:15:00 | 345.80 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2024-03-13 11:15:00 | 344.94 | 2024-03-14 10:15:00 | 350.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-03-13 15:00:00 | 345.56 | 2024-03-14 10:15:00 | 350.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-03-14 10:00:00 | 345.18 | 2024-03-14 10:15:00 | 350.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-03-20 12:15:00 | 351.12 | 2024-03-26 14:15:00 | 350.84 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2024-04-10 09:15:00 | 359.74 | 2024-04-15 15:15:00 | 359.53 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2024-04-19 15:15:00 | 357.98 | 2024-04-22 09:15:00 | 362.12 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-04-24 10:45:00 | 366.82 | 2024-04-25 09:15:00 | 332.78 | STOP_HIT | 1.00 | -9.28% |
| BUY | retest2 | 2024-05-22 12:45:00 | 339.52 | 2024-05-28 15:15:00 | 340.00 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2024-05-22 13:45:00 | 339.65 | 2024-05-28 15:15:00 | 340.00 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2024-05-22 15:00:00 | 339.98 | 2024-05-28 15:15:00 | 340.00 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2024-05-24 14:15:00 | 340.10 | 2024-05-28 15:15:00 | 340.00 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2024-05-27 09:15:00 | 342.49 | 2024-05-28 15:15:00 | 340.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-05-28 11:15:00 | 341.58 | 2024-05-28 15:15:00 | 340.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-05-28 12:30:00 | 341.67 | 2024-05-28 15:15:00 | 340.00 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2024-05-31 09:45:00 | 336.19 | 2024-06-03 09:15:00 | 341.54 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-06-13 11:15:00 | 343.98 | 2024-06-19 09:15:00 | 348.00 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-06-14 09:15:00 | 342.96 | 2024-06-19 09:15:00 | 348.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-06-14 12:45:00 | 344.33 | 2024-06-19 09:15:00 | 348.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-06-27 13:30:00 | 364.44 | 2024-07-02 09:15:00 | 354.15 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2024-06-27 14:30:00 | 364.55 | 2024-07-02 09:15:00 | 354.15 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2024-06-27 15:00:00 | 366.32 | 2024-07-02 09:15:00 | 354.15 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2024-07-09 12:15:00 | 368.49 | 2024-07-10 10:15:00 | 365.56 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-07-15 13:30:00 | 370.02 | 2024-07-16 09:15:00 | 364.75 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-07-22 09:15:00 | 352.04 | 2024-07-25 15:15:00 | 355.64 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-08-08 09:15:00 | 354.99 | 2024-08-09 12:15:00 | 355.10 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2024-08-08 12:30:00 | 354.81 | 2024-08-09 12:15:00 | 355.10 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2024-08-09 10:30:00 | 355.05 | 2024-08-09 12:15:00 | 355.10 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2024-08-09 11:30:00 | 354.80 | 2024-08-09 12:15:00 | 355.10 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2024-08-21 14:30:00 | 360.46 | 2024-08-27 09:15:00 | 359.60 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2024-09-02 12:30:00 | 355.03 | 2024-09-04 13:15:00 | 357.01 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2024-09-02 13:30:00 | 355.07 | 2024-09-09 14:15:00 | 358.05 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-09-03 11:00:00 | 355.08 | 2024-09-09 14:15:00 | 358.05 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-09-03 11:30:00 | 354.99 | 2024-09-09 14:15:00 | 358.05 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-09-04 09:15:00 | 354.33 | 2024-09-09 14:15:00 | 358.05 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-09-05 09:15:00 | 356.04 | 2024-09-09 14:15:00 | 358.05 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-09-11 10:15:00 | 359.85 | 2024-09-25 14:15:00 | 379.53 | STOP_HIT | 1.00 | 5.47% |
| BUY | retest2 | 2024-09-11 11:45:00 | 359.62 | 2024-09-25 14:15:00 | 379.53 | STOP_HIT | 1.00 | 5.54% |
| BUY | retest2 | 2024-09-12 09:15:00 | 362.66 | 2024-09-25 14:15:00 | 379.53 | STOP_HIT | 1.00 | 4.65% |
| SELL | retest2 | 2024-09-27 10:15:00 | 377.99 | 2024-10-07 12:15:00 | 359.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 10:45:00 | 377.63 | 2024-10-07 12:15:00 | 358.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 10:15:00 | 377.99 | 2024-10-08 12:15:00 | 361.90 | STOP_HIT | 0.50 | 4.26% |
| SELL | retest2 | 2024-09-27 10:45:00 | 377.63 | 2024-10-08 12:15:00 | 361.90 | STOP_HIT | 0.50 | 4.17% |
| SELL | retest2 | 2024-10-18 11:15:00 | 372.21 | 2024-10-18 11:15:00 | 375.17 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-10-21 09:15:00 | 362.63 | 2024-11-05 09:15:00 | 344.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 09:15:00 | 362.63 | 2024-11-05 13:15:00 | 349.36 | STOP_HIT | 0.50 | 3.66% |
| BUY | retest2 | 2024-11-08 09:15:00 | 350.48 | 2024-11-11 15:15:00 | 348.36 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-11-11 09:45:00 | 350.38 | 2024-11-11 15:15:00 | 348.36 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-11-11 10:15:00 | 350.67 | 2024-11-11 15:15:00 | 348.36 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-11-21 11:45:00 | 346.79 | 2024-11-28 12:15:00 | 351.25 | STOP_HIT | 1.00 | 1.29% |
| BUY | retest2 | 2024-11-21 12:30:00 | 347.19 | 2024-11-28 12:15:00 | 351.25 | STOP_HIT | 1.00 | 1.17% |
| BUY | retest2 | 2024-11-21 13:00:00 | 346.75 | 2024-11-28 12:15:00 | 351.25 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2024-12-06 13:30:00 | 356.23 | 2024-12-12 11:15:00 | 355.84 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2024-12-09 09:15:00 | 357.81 | 2024-12-12 11:15:00 | 355.84 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-12-30 11:15:00 | 353.43 | 2024-12-30 13:15:00 | 348.22 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-01-27 15:15:00 | 375.74 | 2025-01-28 12:15:00 | 378.78 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-01-28 10:30:00 | 375.70 | 2025-01-28 12:15:00 | 378.78 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-01-30 15:15:00 | 380.76 | 2025-01-31 11:15:00 | 379.57 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-01-31 09:30:00 | 380.91 | 2025-01-31 11:15:00 | 379.57 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-01-31 10:30:00 | 380.79 | 2025-01-31 11:15:00 | 379.57 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-02-03 11:45:00 | 377.91 | 2025-02-04 12:15:00 | 380.90 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-02-03 13:15:00 | 377.97 | 2025-02-04 12:15:00 | 380.90 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-02-04 09:45:00 | 377.60 | 2025-02-04 12:15:00 | 380.90 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-02-06 13:45:00 | 382.43 | 2025-02-11 14:15:00 | 384.05 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2025-02-07 14:15:00 | 382.71 | 2025-02-11 14:15:00 | 384.05 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2025-02-20 13:15:00 | 393.95 | 2025-02-21 09:15:00 | 387.99 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-04-07 09:15:00 | 411.14 | 2025-04-11 10:15:00 | 425.99 | STOP_HIT | 1.00 | -3.61% |
| SELL | retest2 | 2025-04-25 15:15:00 | 439.32 | 2025-04-28 09:15:00 | 445.44 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-04-30 11:45:00 | 443.46 | 2025-05-05 09:15:00 | 421.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 14:15:00 | 443.16 | 2025-05-05 09:15:00 | 421.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 11:45:00 | 443.46 | 2025-05-07 09:15:00 | 415.76 | STOP_HIT | 0.50 | 6.25% |
| SELL | retest2 | 2025-04-30 14:15:00 | 443.16 | 2025-05-07 09:15:00 | 415.76 | STOP_HIT | 0.50 | 6.18% |
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
