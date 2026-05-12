# General Insurance Corporation of India (GICRE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 394.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 18 |
| ALERT1 | 14 |
| ALERT2 | 13 |
| ALERT2_SKIP | 7 |
| ALERT3 | 72 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 63 |
| PARTIAL | 14 |
| TARGET_HIT | 2 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 79 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 50
- **Target hits / Stop hits / Partials:** 2 / 63 / 14
- **Avg / median % per leg:** -0.47% / -1.09%
- **Sum % (uncompounded):** -36.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 33 | 5 | 15.2% | 2 | 31 | 0 | -1.58% | -52.0% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.74% | -5.5% |
| BUY @ 3rd Alert (retest2) | 31 | 5 | 16.1% | 2 | 29 | 0 | -1.50% | -46.6% |
| SELL (all) | 46 | 24 | 52.2% | 0 | 32 | 14 | 0.33% | 15.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 46 | 24 | 52.2% | 0 | 32 | 14 | 0.33% | 15.2% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.74% | -5.5% |
| retest2 (combined) | 77 | 29 | 37.7% | 2 | 61 | 14 | -0.41% | -31.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 15:15:00 | 334.40 | 348.68 | 348.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-10 09:15:00 | 329.25 | 348.49 | 348.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-26 15:15:00 | 343.15 | 338.78 | 342.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-26 15:15:00 | 343.15 | 338.78 | 342.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 15:15:00 | 343.15 | 338.78 | 342.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-29 09:15:00 | 352.80 | 338.78 | 342.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 354.95 | 338.94 | 342.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 09:30:00 | 343.45 | 340.53 | 343.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 12:00:00 | 344.80 | 340.59 | 343.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 09:30:00 | 344.55 | 340.72 | 343.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 10:15:00 | 344.30 | 340.72 | 343.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 10:15:00 | 341.80 | 340.73 | 343.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 11:15:00 | 339.65 | 340.73 | 343.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 14:15:00 | 339.45 | 340.64 | 343.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 15:15:00 | 340.00 | 340.64 | 343.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 09:15:00 | 327.56 | 339.97 | 342.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 09:15:00 | 327.32 | 339.97 | 342.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 09:15:00 | 327.08 | 339.97 | 342.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 10:15:00 | 326.28 | 339.83 | 342.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 12:15:00 | 322.67 | 339.52 | 342.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 12:15:00 | 322.48 | 339.52 | 342.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 12:15:00 | 323.00 | 339.52 | 342.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-15 10:15:00 | 342.20 | 334.71 | 339.32 | SL hit (close>ema200) qty=0.50 sl=334.71 alert=retest2 |

### Cycle 2 — BUY (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 10:15:00 | 374.85 | 342.83 | 342.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 10:15:00 | 388.40 | 351.64 | 348.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 391.35 | 392.63 | 378.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-19 10:00:00 | 391.35 | 392.63 | 378.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 372.45 | 391.92 | 379.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 372.75 | 391.92 | 379.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 374.25 | 391.74 | 379.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:45:00 | 372.90 | 391.74 | 379.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 377.95 | 391.19 | 379.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 11:00:00 | 377.95 | 391.19 | 379.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 11:15:00 | 376.95 | 391.05 | 379.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 12:00:00 | 376.95 | 391.05 | 379.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 10:15:00 | 375.60 | 390.14 | 379.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 10:00:00 | 379.55 | 389.22 | 378.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-26 13:15:00 | 417.51 | 389.83 | 379.38 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 11:15:00 | 364.90 | 392.83 | 392.87 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-17 12:15:00 | 394.70 | 392.70 | 392.70 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 09:15:00 | 387.45 | 392.66 | 392.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 15:15:00 | 380.30 | 391.75 | 392.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 13:15:00 | 379.85 | 377.49 | 383.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-06 14:00:00 | 379.85 | 377.49 | 383.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 382.60 | 377.54 | 383.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 10:30:00 | 378.15 | 377.64 | 383.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 11:45:00 | 378.25 | 377.65 | 383.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 12:45:00 | 378.55 | 377.65 | 383.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 359.24 | 376.76 | 382.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 359.34 | 376.76 | 382.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 359.62 | 376.76 | 382.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-19 10:15:00 | 372.75 | 372.17 | 379.17 | SL hit (close>ema200) qty=0.50 sl=372.17 alert=retest2 |

### Cycle 6 — BUY (started 2024-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 10:15:00 | 413.00 | 383.34 | 383.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 14:15:00 | 428.50 | 391.87 | 387.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 14:15:00 | 434.50 | 436.33 | 418.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 15:00:00 | 434.50 | 436.33 | 418.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 425.60 | 439.19 | 422.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 09:30:00 | 431.85 | 434.88 | 421.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 13:15:00 | 432.70 | 434.84 | 422.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 09:15:00 | 436.75 | 434.75 | 422.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 09:45:00 | 434.85 | 434.74 | 422.22 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 422.95 | 435.86 | 424.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:45:00 | 424.75 | 435.86 | 424.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 421.65 | 435.72 | 424.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 10:30:00 | 422.45 | 435.72 | 424.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 11:15:00 | 418.35 | 435.54 | 424.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:00:00 | 418.35 | 435.54 | 424.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 423.15 | 435.02 | 423.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 09:30:00 | 426.25 | 435.01 | 424.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 14:45:00 | 423.95 | 434.58 | 424.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 15:15:00 | 418.00 | 434.42 | 424.42 | SL hit (close<static) qty=1.00 sl=421.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 389.10 | 417.47 | 417.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 10:15:00 | 387.00 | 417.17 | 417.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 13:15:00 | 389.00 | 387.78 | 397.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-17 13:45:00 | 388.25 | 387.78 | 397.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 397.60 | 388.00 | 396.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:45:00 | 398.00 | 388.00 | 396.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 11:15:00 | 396.80 | 388.09 | 396.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 12:00:00 | 396.80 | 388.09 | 396.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 12:15:00 | 395.80 | 388.17 | 396.94 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 09:15:00 | 417.05 | 403.38 | 403.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 423.60 | 405.05 | 404.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 14:15:00 | 405.95 | 406.43 | 404.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 14:15:00 | 405.95 | 406.43 | 404.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 405.95 | 406.43 | 404.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 14:30:00 | 403.00 | 406.43 | 404.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 15:15:00 | 405.75 | 406.42 | 404.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-07 09:15:00 | 383.30 | 406.42 | 404.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 380.90 | 406.17 | 404.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 11:15:00 | 387.20 | 405.96 | 404.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 15:15:00 | 386.40 | 404.97 | 404.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 12:15:00 | 392.50 | 403.55 | 403.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 12:15:00 | 392.50 | 403.55 | 403.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 13:15:00 | 391.15 | 403.43 | 403.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 09:15:00 | 404.00 | 402.94 | 403.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-15 09:15:00 | 404.00 | 402.94 | 403.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 404.00 | 402.94 | 403.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 10:00:00 | 404.00 | 402.94 | 403.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 10:15:00 | 408.30 | 402.99 | 403.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 11:00:00 | 408.30 | 402.99 | 403.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2025-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 10:15:00 | 417.80 | 403.61 | 403.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 09:15:00 | 429.35 | 404.53 | 404.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 15:15:00 | 415.05 | 415.29 | 410.38 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-02 09:15:00 | 422.90 | 415.29 | 410.38 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 11:45:00 | 418.90 | 416.88 | 411.61 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 409.35 | 416.76 | 411.63 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-06 14:15:00 | 409.35 | 416.76 | 411.63 | SL hit (close<ema400) qty=1.00 sl=411.63 alert=retest1 |

### Cycle 11 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 401.65 | 412.60 | 412.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 398.60 | 412.47 | 412.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 386.25 | 385.67 | 394.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 11:15:00 | 396.25 | 385.90 | 394.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 396.25 | 385.90 | 394.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:00:00 | 396.25 | 385.90 | 394.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 395.60 | 386.00 | 394.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:45:00 | 395.70 | 386.00 | 394.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 393.30 | 386.13 | 394.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:30:00 | 395.00 | 386.13 | 394.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 389.35 | 386.23 | 394.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 10:30:00 | 388.25 | 386.25 | 394.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 10:15:00 | 368.84 | 383.26 | 390.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-30 13:15:00 | 386.40 | 382.55 | 389.77 | SL hit (close>ema200) qty=0.50 sl=382.55 alert=retest2 |

### Cycle 12 — BUY (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 14:15:00 | 383.50 | 378.23 | 378.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 10:15:00 | 385.35 | 378.41 | 378.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 377.85 | 379.04 | 378.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 10:15:00 | 377.85 | 379.04 | 378.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 377.85 | 379.04 | 378.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 377.85 | 379.04 | 378.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 379.70 | 379.04 | 378.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 12:15:00 | 381.20 | 379.04 | 378.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 13:15:00 | 377.25 | 379.02 | 378.64 | SL hit (close<static) qty=1.00 sl=377.60 alert=retest2 |

### Cycle 13 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 10:15:00 | 371.70 | 378.31 | 378.32 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 11:15:00 | 381.40 | 378.32 | 378.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 12:15:00 | 382.05 | 378.36 | 378.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 382.00 | 383.08 | 381.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 382.00 | 383.08 | 381.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 382.00 | 383.08 | 381.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:45:00 | 382.10 | 383.08 | 381.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 379.70 | 383.04 | 381.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 379.70 | 383.04 | 381.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 378.80 | 382.99 | 381.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 382.75 | 382.99 | 381.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 376.25 | 382.76 | 380.97 | SL hit (close<static) qty=1.00 sl=378.05 alert=retest2 |

### Cycle 15 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 364.05 | 380.87 | 380.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 14:15:00 | 362.40 | 380.02 | 380.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 377.35 | 377.35 | 378.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 10:45:00 | 378.35 | 377.35 | 378.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 377.20 | 377.35 | 378.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:30:00 | 378.30 | 377.35 | 378.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 378.05 | 376.41 | 378.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:45:00 | 378.00 | 376.41 | 378.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 386.25 | 376.51 | 378.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 376.30 | 377.56 | 378.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 10:30:00 | 374.60 | 377.51 | 378.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:15:00 | 371.15 | 372.06 | 375.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:30:00 | 374.90 | 371.87 | 374.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 375.15 | 371.99 | 374.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:00:00 | 375.15 | 371.99 | 374.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 377.20 | 372.04 | 374.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:00:00 | 377.20 | 372.04 | 374.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 376.95 | 372.09 | 374.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:30:00 | 377.60 | 372.09 | 374.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 379.85 | 372.47 | 374.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 11:30:00 | 378.00 | 372.55 | 374.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 377.00 | 372.55 | 374.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 14:00:00 | 378.25 | 372.68 | 374.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 09:15:00 | 384.60 | 373.16 | 374.65 | SL hit (close>static) qty=1.00 sl=381.00 alert=retest2 |

### Cycle 16 — BUY (started 2026-02-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 13:15:00 | 393.55 | 376.12 | 376.06 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2026-03-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 13:15:00 | 364.60 | 377.68 | 377.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-10 11:15:00 | 363.50 | 377.07 | 377.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 10:15:00 | 375.10 | 373.37 | 375.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 10:15:00 | 375.10 | 373.37 | 375.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 375.10 | 373.37 | 375.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:45:00 | 375.25 | 373.37 | 375.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 374.40 | 373.38 | 375.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:30:00 | 375.40 | 373.38 | 375.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 375.65 | 373.41 | 375.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:00:00 | 375.65 | 373.41 | 375.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 13:15:00 | 373.90 | 373.41 | 375.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 15:00:00 | 371.45 | 373.39 | 375.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 15:15:00 | 371.05 | 372.45 | 374.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 352.88 | 369.36 | 372.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 352.50 | 369.36 | 372.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 371.35 | 368.92 | 372.36 | SL hit (close>ema200) qty=0.50 sl=368.92 alert=retest2 |

### Cycle 18 — BUY (started 2026-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 13:15:00 | 397.50 | 375.01 | 374.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 417.00 | 378.33 | 376.70 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-02 09:30:00 | 343.45 | 2024-05-07 09:15:00 | 327.56 | PARTIAL | 0.50 | 4.63% |
| SELL | retest2 | 2024-05-02 12:00:00 | 344.80 | 2024-05-07 09:15:00 | 327.32 | PARTIAL | 0.50 | 5.07% |
| SELL | retest2 | 2024-05-03 09:30:00 | 344.55 | 2024-05-07 09:15:00 | 327.08 | PARTIAL | 0.50 | 5.07% |
| SELL | retest2 | 2024-05-03 10:15:00 | 344.30 | 2024-05-07 10:15:00 | 326.28 | PARTIAL | 0.50 | 5.23% |
| SELL | retest2 | 2024-05-03 11:15:00 | 339.65 | 2024-05-07 12:15:00 | 322.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 14:15:00 | 339.45 | 2024-05-07 12:15:00 | 322.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 15:15:00 | 340.00 | 2024-05-07 12:15:00 | 323.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-02 09:30:00 | 343.45 | 2024-05-15 10:15:00 | 342.20 | STOP_HIT | 0.50 | 0.36% |
| SELL | retest2 | 2024-05-02 12:00:00 | 344.80 | 2024-05-15 10:15:00 | 342.20 | STOP_HIT | 0.50 | 0.75% |
| SELL | retest2 | 2024-05-03 09:30:00 | 344.55 | 2024-05-15 10:15:00 | 342.20 | STOP_HIT | 0.50 | 0.68% |
| SELL | retest2 | 2024-05-03 10:15:00 | 344.30 | 2024-05-15 10:15:00 | 342.20 | STOP_HIT | 0.50 | 0.61% |
| SELL | retest2 | 2024-05-03 11:15:00 | 339.65 | 2024-05-15 10:15:00 | 342.20 | STOP_HIT | 0.50 | -0.75% |
| SELL | retest2 | 2024-05-03 14:15:00 | 339.45 | 2024-05-15 10:15:00 | 342.20 | STOP_HIT | 0.50 | -0.81% |
| SELL | retest2 | 2024-05-03 15:15:00 | 340.00 | 2024-05-15 10:15:00 | 342.20 | STOP_HIT | 0.50 | -0.65% |
| SELL | retest2 | 2024-05-15 12:15:00 | 339.55 | 2024-05-17 10:15:00 | 343.05 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-05-16 13:45:00 | 336.30 | 2024-05-17 10:15:00 | 343.05 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2024-05-17 09:15:00 | 337.20 | 2024-05-22 09:15:00 | 371.65 | STOP_HIT | 1.00 | -10.22% |
| BUY | retest2 | 2024-07-26 10:00:00 | 379.55 | 2024-07-26 13:15:00 | 417.51 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-06 15:15:00 | 377.35 | 2024-08-21 09:15:00 | 415.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-07 10:00:00 | 376.45 | 2024-10-07 10:15:00 | 367.80 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2024-11-07 10:30:00 | 378.15 | 2024-11-11 09:15:00 | 359.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 11:45:00 | 378.25 | 2024-11-11 09:15:00 | 359.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 12:45:00 | 378.55 | 2024-11-11 09:15:00 | 359.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 10:30:00 | 378.15 | 2024-11-19 10:15:00 | 372.75 | STOP_HIT | 0.50 | 1.43% |
| SELL | retest2 | 2024-11-07 11:45:00 | 378.25 | 2024-11-19 10:15:00 | 372.75 | STOP_HIT | 0.50 | 1.45% |
| SELL | retest2 | 2024-11-07 12:45:00 | 378.55 | 2024-11-19 10:15:00 | 372.75 | STOP_HIT | 0.50 | 1.53% |
| SELL | retest2 | 2024-11-22 10:15:00 | 376.85 | 2024-11-25 09:15:00 | 391.95 | STOP_HIT | 1.00 | -4.01% |
| BUY | retest2 | 2025-01-16 09:30:00 | 431.85 | 2025-01-24 15:15:00 | 418.00 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2025-01-16 13:15:00 | 432.70 | 2025-01-24 15:15:00 | 418.00 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2025-01-17 09:15:00 | 436.75 | 2025-01-27 09:15:00 | 402.70 | STOP_HIT | 1.00 | -7.80% |
| BUY | retest2 | 2025-01-17 09:45:00 | 434.85 | 2025-01-27 09:15:00 | 402.70 | STOP_HIT | 1.00 | -7.39% |
| BUY | retest2 | 2025-01-23 09:30:00 | 426.25 | 2025-01-27 09:15:00 | 402.70 | STOP_HIT | 1.00 | -5.52% |
| BUY | retest2 | 2025-01-24 14:45:00 | 423.95 | 2025-01-27 09:15:00 | 402.70 | STOP_HIT | 1.00 | -5.01% |
| BUY | retest2 | 2025-04-07 11:15:00 | 387.20 | 2025-04-09 12:15:00 | 392.50 | STOP_HIT | 1.00 | 1.37% |
| BUY | retest2 | 2025-04-07 15:15:00 | 386.40 | 2025-04-09 12:15:00 | 392.50 | STOP_HIT | 1.00 | 1.58% |
| BUY | retest1 | 2025-05-02 09:15:00 | 422.90 | 2025-05-06 14:15:00 | 409.35 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest1 | 2025-05-06 11:45:00 | 418.90 | 2025-05-06 14:15:00 | 409.35 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-05-07 11:15:00 | 415.25 | 2025-05-08 13:15:00 | 402.60 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2025-05-07 14:30:00 | 414.00 | 2025-05-08 13:15:00 | 402.60 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-05-13 09:30:00 | 414.75 | 2025-05-28 11:15:00 | 414.40 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-05-13 14:00:00 | 414.35 | 2025-05-28 11:15:00 | 414.40 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-05-28 09:30:00 | 417.85 | 2025-05-30 09:15:00 | 402.10 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest2 | 2025-05-28 11:00:00 | 417.75 | 2025-05-30 09:15:00 | 402.10 | STOP_HIT | 1.00 | -3.75% |
| BUY | retest2 | 2025-06-11 12:30:00 | 419.75 | 2025-06-11 13:15:00 | 406.65 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2025-07-17 10:30:00 | 388.25 | 2025-07-29 10:15:00 | 368.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-17 10:30:00 | 388.25 | 2025-07-30 13:15:00 | 386.40 | STOP_HIT | 0.50 | 0.48% |
| SELL | retest2 | 2025-07-30 13:45:00 | 385.05 | 2025-08-08 12:15:00 | 393.95 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-07-31 09:45:00 | 387.65 | 2025-08-08 12:15:00 | 393.95 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-07-31 14:00:00 | 387.55 | 2025-08-08 13:15:00 | 396.25 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-08-05 11:30:00 | 386.35 | 2025-08-08 13:15:00 | 396.25 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-08-05 12:15:00 | 386.00 | 2025-08-08 13:15:00 | 396.25 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-08-14 14:45:00 | 386.40 | 2025-08-18 09:15:00 | 398.55 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2025-08-21 11:30:00 | 385.00 | 2025-08-28 15:15:00 | 365.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 11:30:00 | 385.00 | 2025-10-06 10:15:00 | 370.05 | STOP_HIT | 0.50 | 3.88% |
| BUY | retest2 | 2025-10-31 12:15:00 | 381.20 | 2025-10-31 13:15:00 | 377.25 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-11-03 15:15:00 | 380.10 | 2025-11-04 09:15:00 | 376.25 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-11-04 11:00:00 | 380.75 | 2025-11-04 14:15:00 | 376.50 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-11-04 12:30:00 | 380.00 | 2025-11-04 14:15:00 | 376.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-11-24 09:15:00 | 382.75 | 2025-11-24 14:15:00 | 376.25 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-11-24 14:45:00 | 381.05 | 2025-11-24 15:15:00 | 376.90 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-11-25 09:45:00 | 381.00 | 2025-12-08 09:15:00 | 376.05 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-11-25 11:00:00 | 380.50 | 2025-12-08 09:15:00 | 376.05 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-11-26 09:15:00 | 385.45 | 2025-12-08 09:15:00 | 376.05 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-12-05 10:00:00 | 384.20 | 2025-12-08 09:15:00 | 376.05 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-12-05 13:15:00 | 382.80 | 2025-12-08 09:15:00 | 376.05 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-12-10 09:45:00 | 382.60 | 2025-12-10 10:15:00 | 378.75 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-12-15 13:45:00 | 383.00 | 2025-12-16 15:15:00 | 376.50 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-01-06 09:15:00 | 376.30 | 2026-02-09 09:15:00 | 384.60 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2026-01-06 10:30:00 | 374.60 | 2026-02-09 09:15:00 | 384.60 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2026-01-23 09:15:00 | 371.15 | 2026-02-09 09:15:00 | 384.60 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2026-01-29 09:30:00 | 374.90 | 2026-02-10 11:15:00 | 396.55 | STOP_HIT | 1.00 | -5.77% |
| SELL | retest2 | 2026-02-01 11:30:00 | 378.00 | 2026-02-10 11:15:00 | 396.55 | STOP_HIT | 1.00 | -4.91% |
| SELL | retest2 | 2026-02-01 12:15:00 | 377.00 | 2026-02-10 11:15:00 | 396.55 | STOP_HIT | 1.00 | -5.19% |
| SELL | retest2 | 2026-02-01 14:00:00 | 378.25 | 2026-02-10 11:15:00 | 396.55 | STOP_HIT | 1.00 | -4.84% |
| SELL | retest2 | 2026-03-18 15:00:00 | 371.45 | 2026-03-30 09:15:00 | 352.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 15:15:00 | 371.05 | 2026-03-30 09:15:00 | 352.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 15:00:00 | 371.45 | 2026-04-01 09:15:00 | 371.35 | STOP_HIT | 0.50 | 0.03% |
| SELL | retest2 | 2026-03-20 15:15:00 | 371.05 | 2026-04-01 09:15:00 | 371.35 | STOP_HIT | 0.50 | -0.08% |
| SELL | retest2 | 2026-04-02 09:15:00 | 366.50 | 2026-04-02 12:15:00 | 376.40 | STOP_HIT | 1.00 | -2.70% |
