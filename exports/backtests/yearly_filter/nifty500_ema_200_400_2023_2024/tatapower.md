# Tata Power Co. Ltd. (TATAPOWER)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 435.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 3 |
| ALERT3 | 71 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 55 |
| PARTIAL | 11 |
| TARGET_HIT | 9 |
| STOP_HIT | 46 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 66 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 40
- **Target hits / Stop hits / Partials:** 9 / 46 / 11
- **Avg / median % per leg:** 0.79% / -1.24%
- **Sum % (uncompounded):** 52.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 5 | 20.8% | 5 | 19 | 0 | -0.30% | -7.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 24 | 5 | 20.8% | 5 | 19 | 0 | -0.30% | -7.2% |
| SELL (all) | 42 | 21 | 50.0% | 4 | 27 | 11 | 1.41% | 59.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 42 | 21 | 50.0% | 4 | 27 | 11 | 1.41% | 59.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 66 | 26 | 39.4% | 9 | 46 | 11 | 0.79% | 52.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 10:15:00 | 408.75 | 430.96 | 431.07 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-09-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 15:15:00 | 442.20 | 429.53 | 429.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 09:15:00 | 446.55 | 429.70 | 429.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 10:15:00 | 447.35 | 451.86 | 442.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-07 10:15:00 | 447.35 | 451.86 | 442.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 447.35 | 451.86 | 442.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 11:00:00 | 447.35 | 451.86 | 442.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 12:15:00 | 443.20 | 451.74 | 442.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 12:45:00 | 443.00 | 451.74 | 442.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 13:15:00 | 440.40 | 451.63 | 442.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 13:30:00 | 441.00 | 451.63 | 442.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 14:15:00 | 440.60 | 451.52 | 442.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 09:15:00 | 447.55 | 451.41 | 442.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-22 12:45:00 | 444.60 | 455.06 | 447.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-22 14:15:00 | 434.55 | 454.75 | 447.64 | SL hit (close<static) qty=1.00 sl=438.70 alert=retest2 |

### Cycle 3 — SELL (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 13:15:00 | 432.80 | 442.63 | 442.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 09:15:00 | 427.75 | 442.27 | 442.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 426.10 | 424.53 | 431.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-03 10:00:00 | 426.10 | 424.53 | 431.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 429.40 | 424.70 | 431.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 10:30:00 | 427.65 | 424.72 | 431.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 14:45:00 | 426.65 | 424.78 | 430.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 15:15:00 | 432.95 | 424.86 | 430.95 | SL hit (close>static) qty=1.00 sl=432.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 10:15:00 | 392.65 | 370.22 | 370.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 14:15:00 | 394.50 | 372.14 | 371.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 12:15:00 | 378.35 | 379.28 | 375.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 13:00:00 | 378.35 | 379.28 | 375.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 375.90 | 379.22 | 375.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:45:00 | 373.40 | 379.22 | 375.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 374.70 | 379.17 | 375.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 377.60 | 379.17 | 375.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 371.55 | 379.10 | 375.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:30:00 | 379.65 | 379.08 | 375.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 12:15:00 | 379.40 | 379.08 | 375.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 13:15:00 | 379.05 | 379.07 | 375.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 09:15:00 | 379.70 | 379.04 | 375.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 372.85 | 379.01 | 375.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 372.85 | 379.01 | 375.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 371.25 | 378.93 | 375.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 371.25 | 378.93 | 375.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 365.75 | 378.69 | 375.52 | SL hit (close<static) qty=1.00 sl=366.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 11:15:00 | 382.80 | 395.15 | 395.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 380.95 | 391.14 | 392.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 11:15:00 | 388.10 | 387.64 | 390.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-02 12:00:00 | 388.10 | 387.64 | 390.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 387.50 | 387.60 | 390.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 12:15:00 | 384.65 | 387.61 | 390.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:45:00 | 385.80 | 387.19 | 390.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 10:30:00 | 386.05 | 387.17 | 390.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 14:30:00 | 386.50 | 386.89 | 389.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 387.85 | 386.93 | 389.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 10:15:00 | 387.40 | 386.93 | 389.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 10:45:00 | 387.15 | 386.88 | 389.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 14:00:00 | 387.50 | 386.92 | 389.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 09:15:00 | 398.00 | 387.05 | 389.45 | SL hit (close>static) qty=1.00 sl=392.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 399.60 | 390.68 | 390.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 401.45 | 392.04 | 391.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 395.65 | 397.50 | 394.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 10:00:00 | 395.65 | 397.50 | 394.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 395.25 | 397.47 | 394.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 394.75 | 397.47 | 394.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 395.25 | 397.45 | 394.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:30:00 | 394.65 | 397.45 | 394.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 392.35 | 397.38 | 394.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:45:00 | 392.30 | 397.38 | 394.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 391.80 | 397.32 | 394.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 391.80 | 397.32 | 394.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 394.65 | 396.79 | 394.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:45:00 | 394.25 | 396.79 | 394.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 396.20 | 396.78 | 394.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:30:00 | 395.10 | 396.78 | 394.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 391.80 | 396.72 | 394.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 391.80 | 396.72 | 394.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 392.10 | 396.67 | 394.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:30:00 | 390.95 | 396.67 | 394.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 395.45 | 396.58 | 394.52 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 12:15:00 | 384.00 | 392.99 | 393.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 15:15:00 | 382.00 | 392.69 | 392.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 09:15:00 | 394.70 | 391.79 | 392.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 394.70 | 391.79 | 392.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 394.70 | 391.79 | 392.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:30:00 | 397.00 | 391.79 | 392.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 394.05 | 391.81 | 392.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 11:45:00 | 392.60 | 391.82 | 392.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 12:15:00 | 391.85 | 391.82 | 392.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:00:00 | 392.15 | 391.83 | 392.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 14:45:00 | 392.40 | 391.83 | 392.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 392.60 | 391.84 | 392.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 392.60 | 391.84 | 392.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 391.85 | 391.84 | 392.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 12:30:00 | 390.85 | 391.84 | 392.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 13:15:00 | 390.90 | 391.84 | 392.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 14:15:00 | 372.97 | 389.23 | 390.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 14:15:00 | 372.54 | 389.23 | 390.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 14:15:00 | 372.78 | 389.23 | 390.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 372.26 | 388.94 | 390.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 371.31 | 388.94 | 390.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 371.35 | 388.94 | 390.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 384.50 | 382.20 | 385.58 | SL hit (close>ema200) qty=0.50 sl=382.20 alert=retest2 |

### Cycle 8 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 405.10 | 375.14 | 375.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 09:15:00 | 415.00 | 382.67 | 379.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 15:15:00 | 382.90 | 384.22 | 380.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 15:15:00 | 382.90 | 384.22 | 380.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 15:15:00 | 382.90 | 384.22 | 380.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 392.05 | 384.22 | 380.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 12:30:00 | 389.20 | 384.34 | 380.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 13:15:00 | 388.70 | 384.34 | 380.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 09:15:00 | 391.60 | 384.38 | 380.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 384.85 | 385.06 | 381.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:15:00 | 380.30 | 385.06 | 381.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 382.10 | 385.03 | 381.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:30:00 | 380.65 | 385.03 | 381.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 378.40 | 384.92 | 381.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 15:00:00 | 378.40 | 384.92 | 381.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 379.30 | 384.87 | 381.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 386.70 | 384.87 | 381.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 15:00:00 | 380.35 | 384.71 | 381.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 372.15 | 384.54 | 381.09 | SL hit (close<static) qty=1.00 sl=373.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-10-08 09:15:00 | 447.55 | 2024-10-22 14:15:00 | 434.55 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2024-10-22 12:45:00 | 444.60 | 2024-10-22 14:15:00 | 434.55 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2024-10-31 13:00:00 | 443.65 | 2024-11-04 09:15:00 | 432.70 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2024-11-01 18:00:00 | 447.15 | 2024-11-04 09:15:00 | 432.70 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2024-11-06 14:15:00 | 451.35 | 2024-11-08 10:15:00 | 439.75 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2024-12-04 10:30:00 | 427.65 | 2024-12-05 15:15:00 | 432.95 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-12-05 14:45:00 | 426.65 | 2024-12-05 15:15:00 | 432.95 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-12-13 09:15:00 | 427.40 | 2024-12-20 13:15:00 | 406.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-13 15:15:00 | 427.70 | 2024-12-20 13:15:00 | 406.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-13 09:15:00 | 427.40 | 2025-01-06 10:15:00 | 384.66 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-13 15:15:00 | 427.70 | 2025-01-06 10:15:00 | 384.93 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-07 09:15:00 | 353.75 | 2025-04-07 09:15:00 | 336.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-07 09:15:00 | 353.75 | 2025-04-11 10:15:00 | 365.45 | STOP_HIT | 0.50 | -3.31% |
| SELL | retest2 | 2025-04-11 11:45:00 | 366.55 | 2025-04-15 09:15:00 | 378.90 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2025-04-11 13:15:00 | 366.40 | 2025-04-15 09:15:00 | 378.90 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2025-05-07 11:30:00 | 379.65 | 2025-05-09 09:15:00 | 365.75 | STOP_HIT | 1.00 | -3.66% |
| BUY | retest2 | 2025-05-07 12:15:00 | 379.40 | 2025-05-09 09:15:00 | 365.75 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2025-05-07 13:15:00 | 379.05 | 2025-05-09 09:15:00 | 365.75 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest2 | 2025-05-08 09:15:00 | 379.70 | 2025-05-09 09:15:00 | 365.75 | STOP_HIT | 1.00 | -3.67% |
| BUY | retest2 | 2025-06-23 13:30:00 | 393.30 | 2025-08-01 14:15:00 | 388.60 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-08-01 10:00:00 | 393.30 | 2025-08-01 14:15:00 | 388.60 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-08-01 10:30:00 | 393.85 | 2025-08-01 14:15:00 | 388.60 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-08-01 12:15:00 | 392.50 | 2025-08-01 14:15:00 | 388.60 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-09-04 12:15:00 | 384.65 | 2025-09-16 09:15:00 | 398.00 | STOP_HIT | 1.00 | -3.47% |
| SELL | retest2 | 2025-09-08 09:45:00 | 385.80 | 2025-09-16 09:15:00 | 398.00 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2025-09-08 10:30:00 | 386.05 | 2025-09-16 09:15:00 | 398.00 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-09-10 14:30:00 | 386.50 | 2025-09-16 09:15:00 | 398.00 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2025-09-12 10:15:00 | 387.40 | 2025-09-16 09:15:00 | 398.00 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2025-09-15 10:45:00 | 387.15 | 2025-09-16 09:15:00 | 398.00 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2025-09-15 14:00:00 | 387.50 | 2025-09-16 09:15:00 | 398.00 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-09-25 15:00:00 | 385.90 | 2025-09-30 10:15:00 | 391.25 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-09-30 13:45:00 | 386.70 | 2025-10-01 09:15:00 | 391.05 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-10-08 14:30:00 | 386.95 | 2025-10-10 09:15:00 | 395.60 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-10-09 12:15:00 | 386.50 | 2025-10-10 09:15:00 | 395.60 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-10-13 09:30:00 | 386.65 | 2025-10-13 15:15:00 | 391.65 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-11-27 11:45:00 | 392.60 | 2025-12-08 14:15:00 | 372.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 12:15:00 | 391.85 | 2025-12-08 14:15:00 | 372.54 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2025-11-27 13:00:00 | 392.15 | 2025-12-08 14:15:00 | 372.78 | PARTIAL | 0.50 | 4.94% |
| SELL | retest2 | 2025-11-27 14:45:00 | 392.40 | 2025-12-09 09:15:00 | 372.26 | PARTIAL | 0.50 | 5.13% |
| SELL | retest2 | 2025-11-28 12:30:00 | 390.85 | 2025-12-09 09:15:00 | 371.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-28 13:15:00 | 390.90 | 2025-12-09 09:15:00 | 371.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 11:45:00 | 392.60 | 2026-01-02 09:15:00 | 384.50 | STOP_HIT | 0.50 | 2.06% |
| SELL | retest2 | 2025-11-27 12:15:00 | 391.85 | 2026-01-02 09:15:00 | 384.50 | STOP_HIT | 0.50 | 1.88% |
| SELL | retest2 | 2025-11-27 13:00:00 | 392.15 | 2026-01-02 09:15:00 | 384.50 | STOP_HIT | 0.50 | 1.95% |
| SELL | retest2 | 2025-11-27 14:45:00 | 392.40 | 2026-01-02 09:15:00 | 384.50 | STOP_HIT | 0.50 | 2.01% |
| SELL | retest2 | 2025-11-28 12:30:00 | 390.85 | 2026-01-02 09:15:00 | 384.50 | STOP_HIT | 0.50 | 1.62% |
| SELL | retest2 | 2025-11-28 13:15:00 | 390.90 | 2026-01-02 09:15:00 | 384.50 | STOP_HIT | 0.50 | 1.64% |
| SELL | retest2 | 2026-01-05 09:15:00 | 391.00 | 2026-01-09 09:15:00 | 371.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 10:30:00 | 390.45 | 2026-01-09 09:15:00 | 370.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 09:15:00 | 391.00 | 2026-01-20 15:15:00 | 351.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-05 10:30:00 | 390.45 | 2026-01-20 15:15:00 | 351.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-19 13:15:00 | 372.10 | 2026-02-20 10:15:00 | 374.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2026-03-02 09:15:00 | 370.40 | 2026-03-05 09:15:00 | 375.05 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-03-02 10:00:00 | 371.65 | 2026-03-05 09:15:00 | 375.05 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-03-09 09:15:00 | 366.75 | 2026-03-09 15:15:00 | 373.90 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2026-03-24 09:15:00 | 392.05 | 2026-04-02 09:15:00 | 372.15 | STOP_HIT | 1.00 | -5.08% |
| BUY | retest2 | 2026-03-24 12:30:00 | 389.20 | 2026-04-02 09:15:00 | 372.15 | STOP_HIT | 1.00 | -4.38% |
| BUY | retest2 | 2026-03-24 13:15:00 | 388.70 | 2026-04-02 09:15:00 | 372.15 | STOP_HIT | 1.00 | -4.26% |
| BUY | retest2 | 2026-03-25 09:15:00 | 391.60 | 2026-04-02 09:15:00 | 372.15 | STOP_HIT | 1.00 | -4.97% |
| BUY | retest2 | 2026-04-01 09:15:00 | 386.70 | 2026-04-02 09:15:00 | 372.15 | STOP_HIT | 1.00 | -3.76% |
| BUY | retest2 | 2026-04-01 15:00:00 | 380.35 | 2026-04-02 09:15:00 | 372.15 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2026-04-02 14:00:00 | 380.10 | 2026-04-15 09:15:00 | 418.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 14:45:00 | 382.40 | 2026-04-15 10:15:00 | 420.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 13:30:00 | 383.60 | 2026-04-15 10:15:00 | 421.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 10:15:00 | 383.60 | 2026-04-15 10:15:00 | 421.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 11:00:00 | 384.25 | 2026-04-15 10:15:00 | 422.68 | TARGET_HIT | 1.00 | 10.00% |
