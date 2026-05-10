# Indus Towers Ltd. (INDUSTOWER)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 402.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 79 |
| ALERT1 | 51 |
| ALERT2 | 52 |
| ALERT2_SKIP | 26 |
| ALERT3 | 144 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 81 |
| PARTIAL | 14 |
| TARGET_HIT | 2 |
| STOP_HIT | 84 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 100 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 42 / 58
- **Target hits / Stop hits / Partials:** 2 / 84 / 14
- **Avg / median % per leg:** 0.67% / -0.85%
- **Sum % (uncompounded):** 67.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 42 | 14 | 33.3% | 0 | 42 | 0 | -0.65% | -27.5% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.16% | -2.3% |
| BUY @ 3rd Alert (retest2) | 40 | 14 | 35.0% | 0 | 40 | 0 | -0.63% | -25.2% |
| SELL (all) | 58 | 28 | 48.3% | 2 | 42 | 14 | 1.63% | 94.5% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.01% | -3.0% |
| SELL @ 3rd Alert (retest2) | 55 | 28 | 50.9% | 2 | 39 | 14 | 1.77% | 97.5% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.07% | -5.4% |
| retest2 (combined) | 95 | 42 | 44.2% | 2 | 79 | 14 | 0.76% | 72.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 13:15:00 | 389.15 | 396.93 | 397.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 13:15:00 | 381.90 | 383.54 | 385.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 14:15:00 | 383.70 | 383.57 | 385.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 15:00:00 | 383.70 | 383.57 | 385.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 379.40 | 382.64 | 384.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 11:15:00 | 378.30 | 382.04 | 384.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 09:45:00 | 377.55 | 379.77 | 381.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 15:15:00 | 384.50 | 382.48 | 382.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-27 15:15:00 | 384.50 | 382.48 | 382.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 15:15:00 | 384.50 | 382.48 | 382.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 392.35 | 384.46 | 383.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 10:15:00 | 390.35 | 391.58 | 388.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 11:00:00 | 390.35 | 391.58 | 388.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 389.70 | 390.89 | 388.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:45:00 | 389.85 | 390.89 | 388.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 388.80 | 390.48 | 388.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:30:00 | 388.00 | 390.48 | 388.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 392.45 | 390.87 | 389.28 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 11:15:00 | 385.60 | 388.53 | 388.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 384.25 | 387.48 | 388.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 14:15:00 | 380.95 | 380.88 | 382.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-03 14:45:00 | 382.50 | 380.88 | 382.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 379.95 | 380.56 | 382.37 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 386.40 | 383.12 | 382.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 13:15:00 | 389.30 | 386.31 | 384.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 393.00 | 393.51 | 390.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 11:00:00 | 393.00 | 393.51 | 390.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 390.80 | 393.37 | 392.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:00:00 | 390.80 | 393.37 | 392.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 389.90 | 392.67 | 392.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:00:00 | 389.90 | 392.67 | 392.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 387.25 | 390.92 | 391.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 382.95 | 387.32 | 389.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 381.85 | 380.98 | 383.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 10:00:00 | 381.85 | 380.98 | 383.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 385.25 | 381.83 | 383.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 385.25 | 381.83 | 383.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 386.45 | 382.76 | 383.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 386.45 | 382.76 | 383.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 391.55 | 385.70 | 385.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 392.55 | 388.51 | 386.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 10:15:00 | 390.85 | 391.46 | 389.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 10:45:00 | 391.50 | 391.46 | 389.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 391.25 | 391.41 | 389.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:30:00 | 389.70 | 391.41 | 389.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 391.80 | 391.49 | 389.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:45:00 | 391.30 | 391.49 | 389.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 392.05 | 391.58 | 390.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 11:00:00 | 393.60 | 391.67 | 390.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 12:15:00 | 393.05 | 391.86 | 390.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 12:15:00 | 389.70 | 391.43 | 390.69 | SL hit (close<static) qty=1.00 sl=390.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-19 12:15:00 | 389.70 | 391.43 | 390.69 | SL hit (close<static) qty=1.00 sl=390.05 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 09:15:00 | 397.20 | 390.38 | 390.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 09:15:00 | 409.50 | 421.65 | 422.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 409.50 | 421.65 | 422.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 14:15:00 | 408.30 | 413.69 | 417.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 15:15:00 | 407.90 | 407.61 | 411.76 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-09 09:15:00 | 404.45 | 407.61 | 411.76 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-09 13:30:00 | 405.90 | 407.18 | 410.02 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-09 14:00:00 | 405.85 | 407.18 | 410.02 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 409.50 | 407.00 | 409.18 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 409.50 | 407.00 | 409.18 | SL hit (close>ema400) qty=1.00 sl=409.18 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 409.50 | 407.00 | 409.18 | SL hit (close>ema400) qty=1.00 sl=409.18 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 409.50 | 407.00 | 409.18 | SL hit (close>ema400) qty=1.00 sl=409.18 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-07-10 09:45:00 | 408.80 | 407.00 | 409.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 409.35 | 407.47 | 409.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:30:00 | 410.95 | 407.47 | 409.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 407.65 | 407.51 | 409.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:30:00 | 409.50 | 407.51 | 409.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 404.45 | 401.93 | 404.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 15:00:00 | 404.45 | 401.93 | 404.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 403.55 | 402.26 | 404.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 404.70 | 402.26 | 404.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 407.35 | 403.27 | 404.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:00:00 | 407.35 | 403.27 | 404.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 407.40 | 404.10 | 404.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:45:00 | 408.55 | 404.10 | 404.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 403.65 | 404.23 | 404.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:30:00 | 402.85 | 403.94 | 404.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 15:15:00 | 403.00 | 403.93 | 404.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 406.50 | 404.61 | 404.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 406.50 | 404.61 | 404.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 406.50 | 404.61 | 404.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 408.00 | 405.47 | 405.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 11:15:00 | 405.45 | 406.15 | 405.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 11:15:00 | 405.45 | 406.15 | 405.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 405.45 | 406.15 | 405.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:45:00 | 405.65 | 406.15 | 405.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 405.20 | 405.96 | 405.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:00:00 | 405.20 | 405.96 | 405.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 405.95 | 405.96 | 405.56 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 09:15:00 | 404.70 | 405.32 | 405.34 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 10:15:00 | 405.90 | 405.44 | 405.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 11:15:00 | 407.70 | 405.89 | 405.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 12:15:00 | 404.40 | 405.59 | 405.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 12:15:00 | 404.40 | 405.59 | 405.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 404.40 | 405.59 | 405.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 404.40 | 405.59 | 405.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 407.25 | 405.92 | 405.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 14:30:00 | 407.40 | 405.98 | 405.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 402.00 | 405.27 | 405.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 402.00 | 405.27 | 405.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 10:15:00 | 401.00 | 402.92 | 403.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 11:15:00 | 404.20 | 403.18 | 403.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 11:15:00 | 404.20 | 403.18 | 403.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 404.20 | 403.18 | 403.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:00:00 | 404.20 | 403.18 | 403.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 404.85 | 403.51 | 404.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:00:00 | 404.85 | 403.51 | 404.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 402.30 | 403.27 | 403.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:15:00 | 401.00 | 403.30 | 403.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 15:15:00 | 406.50 | 401.86 | 401.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 15:15:00 | 406.50 | 401.86 | 401.37 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 399.15 | 401.27 | 401.28 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 14:15:00 | 401.50 | 401.31 | 401.29 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 15:15:00 | 400.60 | 401.17 | 401.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 394.90 | 399.91 | 400.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 395.90 | 394.97 | 397.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 10:00:00 | 395.90 | 394.97 | 397.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 390.45 | 394.07 | 396.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:45:00 | 388.95 | 392.85 | 395.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 387.85 | 388.61 | 390.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 09:15:00 | 369.50 | 382.07 | 386.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 09:15:00 | 368.46 | 382.07 | 386.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-01 12:15:00 | 350.06 | 359.74 | 369.61 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-08-01 13:15:00 | 349.07 | 357.52 | 367.70 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 16 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 10:15:00 | 338.25 | 336.54 | 336.48 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 332.95 | 337.56 | 337.91 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 13:15:00 | 338.50 | 336.41 | 336.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 13:15:00 | 339.20 | 337.68 | 337.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 11:15:00 | 348.75 | 348.86 | 345.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 12:00:00 | 348.75 | 348.86 | 345.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 343.20 | 352.37 | 351.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 342.75 | 352.37 | 351.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 341.60 | 350.22 | 350.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 340.65 | 344.32 | 346.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 345.60 | 340.41 | 342.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 345.60 | 340.41 | 342.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 345.60 | 340.41 | 342.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 345.60 | 340.41 | 342.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 343.70 | 341.07 | 342.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:45:00 | 345.35 | 341.07 | 342.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 341.85 | 341.41 | 342.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 14:30:00 | 340.50 | 340.84 | 342.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:45:00 | 340.45 | 340.70 | 341.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:45:00 | 340.85 | 340.94 | 341.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 12:15:00 | 340.80 | 340.94 | 341.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 340.55 | 340.86 | 341.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 15:15:00 | 339.00 | 340.50 | 341.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 09:15:00 | 323.47 | 330.01 | 334.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 09:15:00 | 323.43 | 330.01 | 334.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 09:15:00 | 323.81 | 330.01 | 334.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 09:15:00 | 323.76 | 330.01 | 334.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 09:15:00 | 322.05 | 330.01 | 334.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 325.85 | 324.84 | 328.99 | SL hit (close>ema200) qty=0.50 sl=324.84 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 325.85 | 324.84 | 328.99 | SL hit (close>ema200) qty=0.50 sl=324.84 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 325.85 | 324.84 | 328.99 | SL hit (close>ema200) qty=0.50 sl=324.84 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 325.85 | 324.84 | 328.99 | SL hit (close>ema200) qty=0.50 sl=324.84 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 325.85 | 324.84 | 328.99 | SL hit (close>ema200) qty=0.50 sl=324.84 alert=retest2 |

### Cycle 20 — BUY (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 12:15:00 | 333.70 | 328.12 | 328.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 13:15:00 | 341.00 | 330.69 | 329.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 13:15:00 | 359.85 | 362.05 | 358.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 13:15:00 | 359.85 | 362.05 | 358.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 359.85 | 362.05 | 358.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:00:00 | 359.85 | 362.05 | 358.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 359.80 | 361.08 | 358.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 357.95 | 361.08 | 358.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 355.15 | 359.89 | 358.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:00:00 | 355.15 | 359.89 | 358.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 357.70 | 359.45 | 358.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 12:00:00 | 358.40 | 359.24 | 358.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 12:30:00 | 358.00 | 358.91 | 358.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 13:45:00 | 357.90 | 358.68 | 358.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 15:15:00 | 358.05 | 358.49 | 358.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 358.05 | 358.40 | 358.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 356.55 | 358.40 | 358.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 354.70 | 357.66 | 357.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 354.70 | 357.66 | 357.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 354.70 | 357.66 | 357.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 354.70 | 357.66 | 357.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 09:15:00 | 354.70 | 357.66 | 357.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 10:15:00 | 354.00 | 356.93 | 357.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 09:15:00 | 356.40 | 355.23 | 356.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 356.40 | 355.23 | 356.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 356.40 | 355.23 | 356.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:45:00 | 357.60 | 355.23 | 356.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 355.55 | 355.29 | 356.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:00:00 | 355.55 | 355.29 | 356.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 355.80 | 355.39 | 356.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:30:00 | 354.80 | 355.39 | 356.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 353.35 | 354.98 | 355.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:45:00 | 351.65 | 353.94 | 354.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 11:15:00 | 361.55 | 355.46 | 355.57 | SL hit (close>static) qty=1.00 sl=356.25 alert=retest2 |

### Cycle 22 — BUY (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 13:15:00 | 356.35 | 355.66 | 355.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 15:15:00 | 356.80 | 356.02 | 355.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 13:15:00 | 357.80 | 358.05 | 356.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 14:00:00 | 357.80 | 358.05 | 356.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 356.05 | 357.65 | 356.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:45:00 | 356.00 | 357.65 | 356.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 356.65 | 357.45 | 356.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:15:00 | 356.45 | 357.45 | 356.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 355.25 | 357.01 | 356.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:45:00 | 354.20 | 357.01 | 356.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 353.80 | 356.37 | 356.46 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2025-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 11:15:00 | 357.30 | 356.56 | 356.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 15:15:00 | 359.80 | 357.84 | 357.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 14:15:00 | 357.70 | 359.21 | 358.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 14:15:00 | 357.70 | 359.21 | 358.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 357.70 | 359.21 | 358.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 15:00:00 | 357.70 | 359.21 | 358.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 360.00 | 359.37 | 358.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 358.40 | 359.37 | 358.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 359.40 | 359.37 | 358.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:45:00 | 358.25 | 359.37 | 358.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 358.70 | 359.25 | 358.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:45:00 | 359.05 | 359.25 | 358.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 12:15:00 | 357.80 | 358.96 | 358.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 13:00:00 | 357.80 | 358.96 | 358.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 358.90 | 358.95 | 358.61 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 355.85 | 357.91 | 358.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 351.80 | 356.69 | 357.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 348.25 | 348.09 | 350.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:15:00 | 349.70 | 348.09 | 350.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 347.05 | 347.89 | 350.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:30:00 | 348.75 | 347.89 | 350.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 345.90 | 344.70 | 346.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 346.30 | 344.70 | 346.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 345.60 | 344.88 | 346.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:30:00 | 346.50 | 344.88 | 346.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 350.15 | 345.93 | 347.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 350.15 | 345.93 | 347.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 349.70 | 346.69 | 347.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:30:00 | 350.15 | 346.69 | 347.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 352.00 | 348.58 | 348.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 15:15:00 | 352.65 | 349.40 | 348.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 11:15:00 | 348.85 | 349.67 | 348.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 11:15:00 | 348.85 | 349.67 | 348.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 348.85 | 349.67 | 348.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 11:45:00 | 348.75 | 349.67 | 348.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 350.80 | 349.90 | 349.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 14:15:00 | 353.45 | 350.24 | 349.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 10:15:00 | 352.00 | 351.34 | 350.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 14:45:00 | 352.00 | 351.91 | 350.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 12:00:00 | 353.20 | 352.92 | 351.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 354.45 | 355.51 | 353.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 354.45 | 355.51 | 353.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 355.75 | 355.56 | 354.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:30:00 | 353.90 | 355.56 | 354.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 354.75 | 356.07 | 354.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 354.75 | 356.07 | 354.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 355.00 | 355.85 | 354.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 356.10 | 355.85 | 354.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 354.75 | 355.63 | 354.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 11:45:00 | 357.10 | 355.27 | 354.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 10:15:00 | 353.65 | 354.67 | 354.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-13 10:15:00 | 353.65 | 354.67 | 354.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-13 10:15:00 | 353.65 | 354.67 | 354.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-13 10:15:00 | 353.65 | 354.67 | 354.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-13 10:15:00 | 353.65 | 354.67 | 354.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 353.65 | 354.67 | 354.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 11:15:00 | 349.70 | 353.67 | 354.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 344.20 | 343.20 | 346.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 344.20 | 343.20 | 346.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 344.85 | 344.39 | 345.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 13:15:00 | 343.80 | 344.78 | 345.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 15:15:00 | 343.70 | 344.72 | 345.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 347.65 | 344.80 | 345.22 | SL hit (close>static) qty=1.00 sl=346.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 347.65 | 344.80 | 345.22 | SL hit (close>static) qty=1.00 sl=346.55 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 14:15:00 | 343.65 | 344.70 | 345.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 10:15:00 | 350.80 | 345.55 | 345.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 10:15:00 | 350.80 | 345.55 | 345.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 11:15:00 | 352.65 | 346.97 | 345.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 358.10 | 358.16 | 354.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 15:00:00 | 358.10 | 358.16 | 354.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 376.20 | 363.53 | 359.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 384.00 | 368.81 | 363.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 10:15:00 | 382.00 | 371.04 | 365.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 11:15:00 | 383.20 | 372.93 | 366.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 12:30:00 | 384.05 | 376.14 | 369.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 365.30 | 378.42 | 376.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 365.30 | 378.42 | 376.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 370.20 | 376.77 | 375.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 11:15:00 | 371.30 | 376.77 | 375.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 11:15:00 | 368.25 | 375.07 | 375.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-30 11:15:00 | 368.25 | 375.07 | 375.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-30 11:15:00 | 368.25 | 375.07 | 375.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-30 11:15:00 | 368.25 | 375.07 | 375.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-30 11:15:00 | 368.25 | 375.07 | 375.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 368.25 | 375.07 | 375.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 366.40 | 370.17 | 372.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 367.40 | 367.15 | 369.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 367.40 | 367.15 | 369.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 367.40 | 367.15 | 369.62 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 382.90 | 370.45 | 370.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 09:15:00 | 390.90 | 376.51 | 373.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 15:15:00 | 400.00 | 400.48 | 395.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 09:15:00 | 399.50 | 400.48 | 395.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 396.20 | 398.86 | 397.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 396.20 | 398.86 | 397.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 399.20 | 398.93 | 397.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 12:30:00 | 400.05 | 399.05 | 397.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 13:15:00 | 400.90 | 399.05 | 397.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 10:15:00 | 404.35 | 408.53 | 408.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-18 10:15:00 | 404.35 | 408.53 | 408.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 404.35 | 408.53 | 408.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 401.55 | 405.69 | 407.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 12:15:00 | 405.05 | 404.69 | 406.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 12:30:00 | 404.85 | 404.69 | 406.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 400.75 | 403.51 | 405.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 398.50 | 402.18 | 403.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 13:00:00 | 399.45 | 400.23 | 402.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 15:15:00 | 399.20 | 401.07 | 401.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 13:15:00 | 404.95 | 401.71 | 401.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 13:15:00 | 404.95 | 401.71 | 401.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 13:15:00 | 404.95 | 401.71 | 401.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2025-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 13:15:00 | 404.95 | 401.71 | 401.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 406.25 | 403.39 | 402.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 15:15:00 | 405.05 | 405.13 | 403.88 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 09:15:00 | 406.90 | 405.13 | 403.88 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 10:00:00 | 406.55 | 405.41 | 404.12 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 404.40 | 405.32 | 404.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:45:00 | 404.00 | 405.32 | 404.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 402.00 | 404.66 | 404.10 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-27 12:15:00 | 402.00 | 404.66 | 404.10 | SL hit (close<ema400) qty=1.00 sl=404.10 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-11-27 12:15:00 | 402.00 | 404.66 | 404.10 | SL hit (close<ema400) qty=1.00 sl=404.10 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 402.00 | 404.66 | 404.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 403.05 | 404.33 | 404.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 14:30:00 | 406.25 | 404.41 | 404.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 09:15:00 | 401.55 | 403.71 | 403.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 401.55 | 403.71 | 403.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 401.00 | 402.93 | 403.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 403.85 | 402.76 | 403.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 403.85 | 402.76 | 403.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 403.85 | 402.76 | 403.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 10:45:00 | 400.10 | 402.11 | 402.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:45:00 | 400.15 | 398.82 | 400.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 13:00:00 | 400.10 | 400.16 | 400.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 409.00 | 402.51 | 401.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 409.00 | 402.51 | 401.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 409.00 | 402.51 | 401.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 09:15:00 | 409.00 | 402.51 | 401.76 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2025-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 13:15:00 | 399.95 | 403.21 | 403.36 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 411.15 | 404.74 | 403.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 11:15:00 | 411.25 | 406.04 | 404.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 409.15 | 411.16 | 408.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 10:00:00 | 409.15 | 411.16 | 408.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 405.05 | 409.68 | 408.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 405.05 | 409.68 | 408.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 401.80 | 408.11 | 407.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 401.80 | 408.11 | 407.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 400.85 | 406.65 | 406.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 398.60 | 403.72 | 405.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 404.00 | 403.15 | 404.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 14:15:00 | 404.00 | 403.15 | 404.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 404.00 | 403.15 | 404.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 404.00 | 403.15 | 404.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 403.30 | 403.18 | 404.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 406.15 | 403.18 | 404.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 405.05 | 403.55 | 404.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:30:00 | 406.35 | 403.55 | 404.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 405.40 | 403.92 | 404.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:15:00 | 405.40 | 403.92 | 404.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 405.15 | 404.17 | 404.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:30:00 | 406.65 | 404.17 | 404.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 403.45 | 404.41 | 404.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 405.20 | 404.41 | 404.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 410.00 | 405.53 | 405.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 412.90 | 408.97 | 407.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 10:15:00 | 411.00 | 412.64 | 410.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 10:15:00 | 411.00 | 412.64 | 410.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 411.00 | 412.64 | 410.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:45:00 | 411.40 | 412.64 | 410.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 410.40 | 412.19 | 410.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:00:00 | 410.40 | 412.19 | 410.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 410.70 | 411.90 | 410.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 13:00:00 | 410.70 | 411.90 | 410.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 13:15:00 | 407.25 | 410.97 | 410.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:00:00 | 407.25 | 410.97 | 410.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 409.95 | 410.76 | 410.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 10:15:00 | 411.20 | 409.92 | 409.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 10:15:00 | 408.80 | 409.69 | 409.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 408.80 | 409.69 | 409.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 407.55 | 408.94 | 409.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 11:15:00 | 408.05 | 407.88 | 408.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 12:00:00 | 408.05 | 407.88 | 408.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 406.45 | 407.59 | 408.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 13:30:00 | 405.90 | 407.36 | 408.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 14:15:00 | 406.05 | 407.36 | 408.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 11:15:00 | 410.35 | 407.93 | 408.14 | SL hit (close>static) qty=1.00 sl=409.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-18 11:15:00 | 410.35 | 407.93 | 408.14 | SL hit (close>static) qty=1.00 sl=409.20 alert=retest2 |

### Cycle 40 — BUY (started 2025-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 12:15:00 | 410.95 | 408.53 | 408.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 11:15:00 | 412.50 | 410.13 | 409.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 09:15:00 | 410.85 | 411.18 | 410.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 410.85 | 411.18 | 410.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 410.85 | 411.18 | 410.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:30:00 | 409.25 | 411.18 | 410.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 411.15 | 411.17 | 410.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 12:30:00 | 412.00 | 411.25 | 410.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 15:15:00 | 411.95 | 411.22 | 410.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:45:00 | 412.90 | 411.59 | 410.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 10:45:00 | 413.05 | 412.27 | 411.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 414.50 | 414.13 | 412.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 414.45 | 414.13 | 412.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 417.65 | 414.68 | 413.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:15:00 | 420.05 | 414.68 | 413.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:30:00 | 421.00 | 422.24 | 420.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 14:15:00 | 420.85 | 424.42 | 423.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 15:15:00 | 421.00 | 423.02 | 422.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 15:15:00 | 421.00 | 422.62 | 422.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 15:15:00 | 421.00 | 422.62 | 422.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 15:15:00 | 421.00 | 422.62 | 422.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 15:15:00 | 421.00 | 422.62 | 422.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 15:15:00 | 421.00 | 422.62 | 422.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 15:15:00 | 421.00 | 422.62 | 422.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 15:15:00 | 421.00 | 422.62 | 422.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 15:15:00 | 421.00 | 422.62 | 422.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 15:15:00 | 421.00 | 422.62 | 422.73 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 09:15:00 | 434.45 | 424.99 | 423.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 442.40 | 434.53 | 430.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 438.55 | 440.12 | 435.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 438.55 | 440.12 | 435.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 438.55 | 440.12 | 435.82 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 432.00 | 434.94 | 434.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 11:15:00 | 431.05 | 434.16 | 434.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 431.45 | 430.16 | 431.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 431.45 | 430.16 | 431.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 431.45 | 430.16 | 431.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 431.45 | 430.16 | 431.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 429.65 | 430.06 | 431.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:15:00 | 427.70 | 430.06 | 431.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 434.30 | 430.99 | 431.51 | SL hit (close>static) qty=1.00 sl=432.10 alert=retest2 |

### Cycle 44 — BUY (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 09:15:00 | 443.10 | 433.80 | 432.67 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 15:15:00 | 431.95 | 433.87 | 433.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 10:15:00 | 429.40 | 432.45 | 433.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 434.95 | 429.69 | 430.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 434.95 | 429.69 | 430.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 434.95 | 429.69 | 430.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 433.75 | 429.69 | 430.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 441.60 | 432.07 | 431.94 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 430.55 | 433.74 | 433.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 12:15:00 | 425.65 | 430.38 | 432.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 415.85 | 415.41 | 420.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 12:45:00 | 417.05 | 415.41 | 420.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 418.50 | 415.14 | 418.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:45:00 | 414.50 | 414.98 | 418.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:45:00 | 414.50 | 414.79 | 417.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 09:15:00 | 422.10 | 417.84 | 418.33 | SL hit (close>static) qty=1.00 sl=420.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-23 09:15:00 | 422.10 | 417.84 | 418.33 | SL hit (close>static) qty=1.00 sl=420.35 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:45:00 | 414.30 | 416.80 | 417.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 10:15:00 | 414.75 | 414.77 | 416.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 416.30 | 415.07 | 416.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:30:00 | 414.15 | 415.07 | 416.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 418.10 | 415.68 | 416.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 11:30:00 | 418.20 | 415.68 | 416.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 12:15:00 | 416.80 | 415.90 | 416.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 13:45:00 | 415.95 | 415.81 | 416.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 14:15:00 | 422.30 | 417.11 | 416.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 14:15:00 | 422.30 | 417.11 | 416.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 14:15:00 | 422.30 | 417.11 | 416.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 14:15:00 | 422.30 | 417.11 | 416.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 15:15:00 | 424.00 | 418.49 | 417.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-28 13:15:00 | 422.50 | 422.59 | 420.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-28 14:00:00 | 422.50 | 422.59 | 420.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 438.65 | 442.04 | 437.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:45:00 | 436.50 | 442.04 | 437.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 439.20 | 441.48 | 437.62 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 425.05 | 435.32 | 435.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 421.50 | 432.56 | 434.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 431.65 | 428.61 | 431.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 431.65 | 428.61 | 431.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 431.65 | 428.61 | 431.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 431.65 | 428.61 | 431.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 431.70 | 429.23 | 431.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 431.05 | 429.23 | 431.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 428.15 | 429.01 | 430.84 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 444.20 | 433.09 | 432.44 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 13:15:00 | 437.90 | 439.68 | 439.88 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 443.30 | 440.41 | 440.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 450.90 | 442.54 | 441.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 454.30 | 455.42 | 450.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 13:15:00 | 454.30 | 455.42 | 450.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 454.30 | 455.42 | 450.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:00:00 | 454.30 | 455.42 | 450.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 469.25 | 470.55 | 466.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 11:45:00 | 472.80 | 469.75 | 468.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 13:30:00 | 472.45 | 470.55 | 468.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 10:15:00 | 471.55 | 471.44 | 469.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 13:30:00 | 472.65 | 471.37 | 470.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 476.65 | 472.69 | 471.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 10:30:00 | 479.60 | 473.41 | 471.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 09:15:00 | 472.45 | 473.18 | 473.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-20 09:15:00 | 472.45 | 473.18 | 473.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-20 09:15:00 | 472.45 | 473.18 | 473.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-20 09:15:00 | 472.45 | 473.18 | 473.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-20 09:15:00 | 472.45 | 473.18 | 473.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 472.45 | 473.18 | 473.21 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 473.75 | 473.25 | 473.24 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 12:15:00 | 471.75 | 472.95 | 473.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 13:15:00 | 470.35 | 472.43 | 472.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 14:15:00 | 474.50 | 472.85 | 473.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 14:15:00 | 474.50 | 472.85 | 473.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 474.50 | 472.85 | 473.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 15:00:00 | 474.50 | 472.85 | 473.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 472.00 | 472.68 | 472.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 474.30 | 472.68 | 472.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 473.90 | 472.92 | 473.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:30:00 | 469.80 | 472.71 | 472.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:00:00 | 470.60 | 471.77 | 472.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 15:00:00 | 471.45 | 471.46 | 472.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 469.85 | 471.77 | 472.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 469.95 | 471.41 | 472.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 11:45:00 | 465.70 | 469.74 | 471.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 13:15:00 | 465.80 | 469.13 | 470.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 10:30:00 | 466.05 | 468.25 | 469.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 446.31 | 456.15 | 459.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 447.07 | 456.15 | 459.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 447.88 | 456.15 | 459.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 446.36 | 456.15 | 459.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 442.41 | 456.15 | 459.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 442.51 | 456.15 | 459.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 442.75 | 456.15 | 459.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 13:15:00 | 443.45 | 443.10 | 448.68 | SL hit (close>ema200) qty=0.50 sl=443.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 13:15:00 | 443.45 | 443.10 | 448.68 | SL hit (close>ema200) qty=0.50 sl=443.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 13:15:00 | 443.45 | 443.10 | 448.68 | SL hit (close>ema200) qty=0.50 sl=443.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 13:15:00 | 443.45 | 443.10 | 448.68 | SL hit (close>ema200) qty=0.50 sl=443.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 13:15:00 | 443.45 | 443.10 | 448.68 | SL hit (close>ema200) qty=0.50 sl=443.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 13:15:00 | 443.45 | 443.10 | 448.68 | SL hit (close>ema200) qty=0.50 sl=443.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 13:15:00 | 443.45 | 443.10 | 448.68 | SL hit (close>ema200) qty=0.50 sl=443.10 alert=retest2 |

### Cycle 56 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 453.90 | 449.97 | 449.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 12:15:00 | 456.60 | 452.68 | 451.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 14:15:00 | 452.35 | 453.02 | 451.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 14:15:00 | 452.35 | 453.02 | 451.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 452.35 | 453.02 | 451.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 452.35 | 453.02 | 451.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 450.00 | 452.41 | 451.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 435.40 | 452.41 | 451.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 439.25 | 449.78 | 450.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 430.80 | 438.26 | 440.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 426.95 | 425.82 | 430.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 428.70 | 425.82 | 430.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 433.80 | 427.68 | 430.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:30:00 | 434.95 | 427.68 | 430.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 433.45 | 428.84 | 430.65 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 436.70 | 432.41 | 431.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 15:15:00 | 437.95 | 433.52 | 432.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 431.00 | 437.09 | 435.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 431.00 | 437.09 | 435.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 431.00 | 437.09 | 435.58 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 428.65 | 433.52 | 434.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 423.05 | 431.43 | 433.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 436.90 | 431.49 | 432.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 436.90 | 431.49 | 432.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 436.90 | 431.49 | 432.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 436.10 | 431.49 | 432.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 437.95 | 432.78 | 433.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 438.50 | 432.78 | 433.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 11:15:00 | 438.35 | 433.89 | 433.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 12:15:00 | 441.45 | 435.40 | 434.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 14:15:00 | 434.00 | 435.60 | 434.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 14:15:00 | 434.00 | 435.60 | 434.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 434.00 | 435.60 | 434.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 434.00 | 435.60 | 434.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 434.15 | 435.31 | 434.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 421.25 | 435.31 | 434.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 417.50 | 431.75 | 433.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 415.55 | 425.19 | 429.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 420.10 | 419.79 | 424.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:00:00 | 420.10 | 419.79 | 424.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 423.15 | 420.46 | 424.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 11:45:00 | 424.50 | 420.46 | 424.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 429.65 | 422.30 | 424.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 428.30 | 422.30 | 424.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 429.75 | 423.79 | 425.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 429.75 | 423.79 | 425.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 431.00 | 427.02 | 426.65 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 12:15:00 | 425.20 | 426.36 | 426.43 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 428.60 | 426.81 | 426.63 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 424.10 | 426.50 | 426.56 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 12:15:00 | 428.05 | 426.84 | 426.68 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 418.75 | 425.13 | 425.93 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 429.75 | 424.96 | 424.70 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 412.35 | 422.53 | 423.78 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 09:15:00 | 428.40 | 423.20 | 423.02 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2026-04-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 14:15:00 | 422.65 | 423.66 | 423.70 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 433.35 | 425.57 | 424.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 437.60 | 427.98 | 425.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 435.35 | 435.49 | 431.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 435.35 | 435.49 | 431.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 438.10 | 438.29 | 436.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:30:00 | 435.70 | 438.29 | 436.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 432.55 | 437.21 | 435.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 434.20 | 437.21 | 435.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 435.00 | 436.74 | 435.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:30:00 | 436.40 | 436.04 | 435.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 425.00 | 433.83 | 434.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 425.00 | 433.83 | 434.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 425.00 | 433.83 | 434.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 10:15:00 | 425.00 | 433.83 | 434.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-15 13:15:00 | 422.90 | 429.15 | 432.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 415.60 | 409.11 | 412.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 415.60 | 409.11 | 412.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 415.60 | 409.11 | 412.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:00:00 | 415.60 | 409.11 | 412.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 417.75 | 410.84 | 412.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 11:00:00 | 417.75 | 410.84 | 412.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2026-04-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 13:15:00 | 416.10 | 413.89 | 413.85 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 410.20 | 413.34 | 413.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 11:15:00 | 408.65 | 412.40 | 413.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 13:15:00 | 401.60 | 401.53 | 404.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 14:00:00 | 401.60 | 401.53 | 404.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 402.20 | 401.67 | 404.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:30:00 | 403.55 | 401.67 | 404.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 403.35 | 402.04 | 404.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 10:30:00 | 402.30 | 402.21 | 404.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 401.75 | 402.21 | 404.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 12:45:00 | 401.90 | 401.96 | 403.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:30:00 | 402.50 | 402.15 | 403.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 402.15 | 402.15 | 403.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:30:00 | 403.00 | 402.15 | 403.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-28 09:15:00 | 415.75 | 404.84 | 404.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-28 09:15:00 | 415.75 | 404.84 | 404.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-28 09:15:00 | 415.75 | 404.84 | 404.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-28 09:15:00 | 415.75 | 404.84 | 404.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 415.75 | 404.84 | 404.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 420.20 | 413.48 | 409.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 414.00 | 415.07 | 411.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 13:30:00 | 413.90 | 415.07 | 411.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 414.10 | 414.87 | 412.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:30:00 | 415.45 | 414.87 | 412.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 413.00 | 414.50 | 412.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 409.25 | 414.50 | 412.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 408.45 | 413.29 | 411.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 408.45 | 413.29 | 411.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 410.35 | 412.70 | 411.71 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2026-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 14:15:00 | 410.00 | 411.10 | 411.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 09:15:00 | 395.85 | 407.94 | 409.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 13:15:00 | 403.80 | 403.62 | 406.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 13:45:00 | 403.60 | 403.62 | 406.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 403.10 | 402.58 | 405.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:45:00 | 405.85 | 402.58 | 405.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 402.65 | 402.61 | 404.50 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 410.75 | 405.46 | 405.28 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2026-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 12:15:00 | 403.05 | 405.37 | 405.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 09:15:00 | 398.80 | 403.17 | 404.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 13:15:00 | 403.65 | 401.68 | 403.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 13:15:00 | 403.65 | 401.68 | 403.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 403.65 | 401.68 | 403.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:00:00 | 403.65 | 401.68 | 403.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 405.40 | 402.42 | 403.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:30:00 | 405.00 | 402.42 | 403.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-16 10:15:00 | 399.20 | 2025-05-19 13:15:00 | 389.15 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-05-19 10:00:00 | 399.90 | 2025-05-19 13:15:00 | 389.15 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-05-26 11:15:00 | 378.30 | 2025-05-27 15:15:00 | 384.50 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-05-27 09:45:00 | 377.55 | 2025-05-27 15:15:00 | 384.50 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-06-19 11:00:00 | 393.60 | 2025-06-19 12:15:00 | 389.70 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-06-19 12:15:00 | 393.05 | 2025-06-19 12:15:00 | 389.70 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-06-20 09:15:00 | 397.20 | 2025-07-07 09:15:00 | 409.50 | STOP_HIT | 1.00 | 3.10% |
| SELL | retest1 | 2025-07-09 09:15:00 | 404.45 | 2025-07-10 09:15:00 | 409.50 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest1 | 2025-07-09 13:30:00 | 405.90 | 2025-07-10 09:15:00 | 409.50 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest1 | 2025-07-09 14:00:00 | 405.85 | 2025-07-10 09:15:00 | 409.50 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-07-14 13:30:00 | 402.85 | 2025-07-15 10:15:00 | 406.50 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-07-14 15:15:00 | 403.00 | 2025-07-15 10:15:00 | 406.50 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-07-17 14:30:00 | 407.40 | 2025-07-18 09:15:00 | 402.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-07-22 09:15:00 | 401.00 | 2025-07-23 15:15:00 | 406.50 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-07-28 11:45:00 | 388.95 | 2025-07-31 09:15:00 | 369.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 10:15:00 | 387.85 | 2025-07-31 09:15:00 | 368.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-28 11:45:00 | 388.95 | 2025-08-01 12:15:00 | 350.06 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-30 10:15:00 | 387.85 | 2025-08-01 13:15:00 | 349.07 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-29 14:30:00 | 340.50 | 2025-09-03 09:15:00 | 323.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-01 10:45:00 | 340.45 | 2025-09-03 09:15:00 | 323.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-01 11:45:00 | 340.85 | 2025-09-03 09:15:00 | 323.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-01 12:15:00 | 340.80 | 2025-09-03 09:15:00 | 323.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-01 15:15:00 | 339.00 | 2025-09-03 09:15:00 | 322.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-29 14:30:00 | 340.50 | 2025-09-04 09:15:00 | 325.85 | STOP_HIT | 0.50 | 4.30% |
| SELL | retest2 | 2025-09-01 10:45:00 | 340.45 | 2025-09-04 09:15:00 | 325.85 | STOP_HIT | 0.50 | 4.29% |
| SELL | retest2 | 2025-09-01 11:45:00 | 340.85 | 2025-09-04 09:15:00 | 325.85 | STOP_HIT | 0.50 | 4.40% |
| SELL | retest2 | 2025-09-01 12:15:00 | 340.80 | 2025-09-04 09:15:00 | 325.85 | STOP_HIT | 0.50 | 4.39% |
| SELL | retest2 | 2025-09-01 15:15:00 | 339.00 | 2025-09-04 09:15:00 | 325.85 | STOP_HIT | 0.50 | 3.88% |
| BUY | retest2 | 2025-09-16 12:00:00 | 358.40 | 2025-09-17 09:15:00 | 354.70 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-09-16 12:30:00 | 358.00 | 2025-09-17 09:15:00 | 354.70 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-09-16 13:45:00 | 357.90 | 2025-09-17 09:15:00 | 354.70 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-09-16 15:15:00 | 358.05 | 2025-09-17 09:15:00 | 354.70 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-09-19 10:45:00 | 351.65 | 2025-09-19 11:15:00 | 361.55 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2025-10-03 14:15:00 | 353.45 | 2025-10-13 10:15:00 | 353.65 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-10-06 10:15:00 | 352.00 | 2025-10-13 10:15:00 | 353.65 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest2 | 2025-10-06 14:45:00 | 352.00 | 2025-10-13 10:15:00 | 353.65 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest2 | 2025-10-07 12:00:00 | 353.20 | 2025-10-13 10:15:00 | 353.65 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2025-10-10 11:45:00 | 357.10 | 2025-10-13 10:15:00 | 353.65 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-10-16 13:15:00 | 343.80 | 2025-10-17 11:15:00 | 347.65 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-10-16 15:15:00 | 343.70 | 2025-10-17 11:15:00 | 347.65 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-10-17 14:15:00 | 343.65 | 2025-10-20 10:15:00 | 350.80 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-10-28 09:15:00 | 384.00 | 2025-10-30 11:15:00 | 368.25 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2025-10-28 10:15:00 | 382.00 | 2025-10-30 11:15:00 | 368.25 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2025-10-28 11:15:00 | 383.20 | 2025-10-30 11:15:00 | 368.25 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest2 | 2025-10-28 12:30:00 | 384.05 | 2025-10-30 11:15:00 | 368.25 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest2 | 2025-10-30 11:15:00 | 371.30 | 2025-10-30 11:15:00 | 368.25 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-11-11 12:30:00 | 400.05 | 2025-11-18 10:15:00 | 404.35 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2025-11-11 13:15:00 | 400.90 | 2025-11-18 10:15:00 | 404.35 | STOP_HIT | 1.00 | 0.86% |
| SELL | retest2 | 2025-11-21 09:15:00 | 398.50 | 2025-11-25 13:15:00 | 404.95 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-11-21 13:00:00 | 399.45 | 2025-11-25 13:15:00 | 404.95 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-11-24 15:15:00 | 399.20 | 2025-11-25 13:15:00 | 404.95 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest1 | 2025-11-27 09:15:00 | 406.90 | 2025-11-27 12:15:00 | 402.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest1 | 2025-11-27 10:00:00 | 406.55 | 2025-11-27 12:15:00 | 402.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-11-27 14:30:00 | 406.25 | 2025-11-28 09:15:00 | 401.55 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-12-01 10:45:00 | 400.10 | 2025-12-03 09:15:00 | 409.00 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-12-02 09:45:00 | 400.15 | 2025-12-03 09:15:00 | 409.00 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-12-02 13:00:00 | 400.10 | 2025-12-03 09:15:00 | 409.00 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-12-16 10:15:00 | 411.20 | 2025-12-16 10:15:00 | 408.80 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-12-17 13:30:00 | 405.90 | 2025-12-18 11:15:00 | 410.35 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-12-17 14:15:00 | 406.05 | 2025-12-18 11:15:00 | 410.35 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-12-22 12:30:00 | 412.00 | 2025-12-31 15:15:00 | 421.00 | STOP_HIT | 1.00 | 2.18% |
| BUY | retest2 | 2025-12-22 15:15:00 | 411.95 | 2025-12-31 15:15:00 | 421.00 | STOP_HIT | 1.00 | 2.20% |
| BUY | retest2 | 2025-12-23 09:45:00 | 412.90 | 2025-12-31 15:15:00 | 421.00 | STOP_HIT | 1.00 | 1.96% |
| BUY | retest2 | 2025-12-23 10:45:00 | 413.05 | 2025-12-31 15:15:00 | 421.00 | STOP_HIT | 1.00 | 1.92% |
| BUY | retest2 | 2025-12-24 10:15:00 | 420.05 | 2025-12-31 15:15:00 | 421.00 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2025-12-29 09:30:00 | 421.00 | 2025-12-31 15:15:00 | 421.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-12-31 14:15:00 | 420.85 | 2025-12-31 15:15:00 | 421.00 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2025-12-31 15:15:00 | 421.00 | 2025-12-31 15:15:00 | 421.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2026-01-08 11:15:00 | 427.70 | 2026-01-08 12:15:00 | 434.30 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2026-01-22 10:45:00 | 414.50 | 2026-01-23 09:15:00 | 422.10 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-01-22 12:45:00 | 414.50 | 2026-01-23 09:15:00 | 422.10 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-01-23 11:45:00 | 414.30 | 2026-01-27 14:15:00 | 422.30 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2026-01-27 10:15:00 | 414.75 | 2026-01-27 14:15:00 | 422.30 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2026-01-27 13:45:00 | 415.95 | 2026-01-27 14:15:00 | 422.30 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2026-02-16 11:45:00 | 472.80 | 2026-02-20 09:15:00 | 472.45 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2026-02-16 13:30:00 | 472.45 | 2026-02-20 09:15:00 | 472.45 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2026-02-17 10:15:00 | 471.55 | 2026-02-20 09:15:00 | 472.45 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2026-02-17 13:30:00 | 472.65 | 2026-02-20 09:15:00 | 472.45 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2026-02-18 10:30:00 | 479.60 | 2026-02-20 09:15:00 | 472.45 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2026-02-23 10:30:00 | 469.80 | 2026-03-02 09:15:00 | 446.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 13:00:00 | 470.60 | 2026-03-02 09:15:00 | 447.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 15:00:00 | 471.45 | 2026-03-02 09:15:00 | 447.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 469.85 | 2026-03-02 09:15:00 | 446.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 11:45:00 | 465.70 | 2026-03-02 09:15:00 | 442.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 13:15:00 | 465.80 | 2026-03-02 09:15:00 | 442.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 10:30:00 | 466.05 | 2026-03-02 09:15:00 | 442.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 10:30:00 | 469.80 | 2026-03-04 13:15:00 | 443.45 | STOP_HIT | 0.50 | 5.61% |
| SELL | retest2 | 2026-02-23 13:00:00 | 470.60 | 2026-03-04 13:15:00 | 443.45 | STOP_HIT | 0.50 | 5.77% |
| SELL | retest2 | 2026-02-23 15:00:00 | 471.45 | 2026-03-04 13:15:00 | 443.45 | STOP_HIT | 0.50 | 5.94% |
| SELL | retest2 | 2026-02-24 09:15:00 | 469.85 | 2026-03-04 13:15:00 | 443.45 | STOP_HIT | 0.50 | 5.62% |
| SELL | retest2 | 2026-02-24 11:45:00 | 465.70 | 2026-03-04 13:15:00 | 443.45 | STOP_HIT | 0.50 | 4.78% |
| SELL | retest2 | 2026-02-24 13:15:00 | 465.80 | 2026-03-04 13:15:00 | 443.45 | STOP_HIT | 0.50 | 4.80% |
| SELL | retest2 | 2026-02-25 10:30:00 | 466.05 | 2026-03-04 13:15:00 | 443.45 | STOP_HIT | 0.50 | 4.85% |
| BUY | retest2 | 2026-04-13 10:15:00 | 434.20 | 2026-04-15 10:15:00 | 425.00 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2026-04-13 10:45:00 | 435.00 | 2026-04-15 10:15:00 | 425.00 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2026-04-15 09:30:00 | 436.40 | 2026-04-15 10:15:00 | 425.00 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2026-04-27 10:30:00 | 402.30 | 2026-04-28 09:15:00 | 415.75 | STOP_HIT | 1.00 | -3.34% |
| SELL | retest2 | 2026-04-27 11:15:00 | 401.75 | 2026-04-28 09:15:00 | 415.75 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2026-04-27 12:45:00 | 401.90 | 2026-04-28 09:15:00 | 415.75 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2026-04-27 13:30:00 | 402.50 | 2026-04-28 09:15:00 | 415.75 | STOP_HIT | 1.00 | -3.29% |
