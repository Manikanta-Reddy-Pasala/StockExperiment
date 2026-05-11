# Power Finance Corporation Ltd. (PFC)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 461.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 3 |
| ALERT3 | 53 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 52 |
| PARTIAL | 4 |
| TARGET_HIT | 5 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 56 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 44
- **Target hits / Stop hits / Partials:** 5 / 47 / 4
- **Avg / median % per leg:** -0.48% / -1.42%
- **Sum % (uncompounded):** -27.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 4 | 18.2% | 3 | 19 | 0 | -0.11% | -2.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 22 | 4 | 18.2% | 3 | 19 | 0 | -0.11% | -2.5% |
| SELL (all) | 34 | 8 | 23.5% | 2 | 28 | 4 | -0.72% | -24.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 34 | 8 | 23.5% | 2 | 28 | 4 | -0.72% | -24.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 56 | 12 | 21.4% | 5 | 47 | 4 | -0.48% | -27.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 15:15:00 | 489.75 | 509.33 | 509.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 09:15:00 | 483.95 | 509.07 | 509.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 15:15:00 | 473.40 | 471.21 | 484.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-30 09:15:00 | 472.75 | 471.21 | 484.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 486.15 | 466.03 | 478.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:00:00 | 486.15 | 466.03 | 478.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 484.40 | 466.21 | 478.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:30:00 | 487.30 | 466.21 | 478.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 479.55 | 467.23 | 478.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:30:00 | 481.75 | 467.23 | 478.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 475.70 | 466.23 | 476.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 475.70 | 466.23 | 476.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 477.30 | 466.34 | 476.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:15:00 | 476.60 | 466.34 | 476.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 472.85 | 466.40 | 476.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:00:00 | 470.70 | 466.52 | 476.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 15:00:00 | 470.15 | 466.56 | 476.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 09:15:00 | 447.16 | 466.29 | 476.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 09:15:00 | 446.64 | 466.29 | 476.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-22 09:15:00 | 469.45 | 465.49 | 475.37 | SL hit (close>ema200) qty=0.50 sl=465.49 alert=retest2 |

### Cycle 2 — BUY (started 2024-12-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 10:15:00 | 519.35 | 481.72 | 481.66 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 12:15:00 | 461.10 | 484.30 | 484.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-27 11:15:00 | 457.95 | 482.97 | 483.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 10:15:00 | 394.45 | 393.17 | 413.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-05 11:00:00 | 394.45 | 393.17 | 413.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 398.90 | 395.28 | 409.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 09:15:00 | 382.45 | 405.30 | 410.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 397.65 | 404.08 | 409.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 13:45:00 | 397.50 | 403.93 | 409.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 14:30:00 | 398.05 | 403.88 | 409.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 416.25 | 403.20 | 408.77 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 416.25 | 403.20 | 408.77 | SL hit (close>static) qty=1.00 sl=410.95 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 13:15:00 | 432.40 | 413.14 | 413.11 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 401.80 | 413.20 | 413.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 13:15:00 | 399.45 | 412.93 | 413.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 11:15:00 | 409.95 | 409.30 | 411.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-12 11:45:00 | 409.75 | 409.30 | 411.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 13:15:00 | 411.10 | 409.31 | 411.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 14:00:00 | 411.10 | 409.31 | 411.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 14:15:00 | 414.05 | 409.36 | 411.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 15:00:00 | 414.05 | 409.36 | 411.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 15:15:00 | 414.65 | 409.41 | 411.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-13 09:15:00 | 412.50 | 409.41 | 411.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 10:15:00 | 413.85 | 409.49 | 411.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-13 10:45:00 | 414.55 | 409.49 | 411.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 409.75 | 409.58 | 411.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-14 12:00:00 | 408.90 | 409.78 | 411.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 09:15:00 | 421.60 | 409.47 | 411.02 | SL hit (close>static) qty=1.00 sl=412.50 alert=retest2 |

### Cycle 6 — BUY (started 2025-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 09:15:00 | 431.25 | 411.71 | 411.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 09:15:00 | 434.65 | 413.08 | 412.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 405.10 | 414.44 | 413.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 405.10 | 414.44 | 413.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 405.10 | 414.44 | 413.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 12:00:00 | 406.90 | 414.26 | 413.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 400.40 | 413.79 | 412.80 | SL hit (close<static) qty=1.00 sl=401.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 15:15:00 | 396.90 | 411.86 | 411.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 395.15 | 411.69 | 411.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 411.50 | 410.50 | 411.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 411.50 | 410.50 | 411.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 411.50 | 410.50 | 411.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:00:00 | 411.50 | 410.50 | 411.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 412.10 | 410.52 | 411.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:15:00 | 414.20 | 410.52 | 411.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 410.15 | 410.51 | 411.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:30:00 | 412.25 | 410.51 | 411.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 410.00 | 410.50 | 411.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:45:00 | 410.40 | 410.50 | 411.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 412.50 | 410.39 | 411.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:00:00 | 412.50 | 410.39 | 411.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 410.75 | 410.39 | 411.10 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 11:15:00 | 424.10 | 411.82 | 411.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 11:15:00 | 426.25 | 412.55 | 412.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 12:15:00 | 414.15 | 414.38 | 413.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 12:15:00 | 414.15 | 414.38 | 413.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 414.15 | 414.38 | 413.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:45:00 | 413.35 | 414.38 | 413.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 412.85 | 414.36 | 413.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:45:00 | 413.00 | 414.36 | 413.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 413.35 | 414.35 | 413.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 14:45:00 | 412.90 | 414.35 | 413.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 414.35 | 414.35 | 413.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 412.70 | 414.35 | 413.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 412.90 | 414.34 | 413.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 09:30:00 | 414.90 | 414.17 | 413.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 11:15:00 | 415.45 | 414.17 | 413.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 13:00:00 | 414.90 | 414.16 | 413.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 15:15:00 | 418.00 | 418.84 | 416.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 418.00 | 418.83 | 416.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:30:00 | 413.80 | 418.79 | 416.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 414.25 | 418.74 | 416.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:45:00 | 413.95 | 418.74 | 416.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 416.10 | 418.64 | 416.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 14:30:00 | 417.70 | 418.64 | 416.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 10:15:00 | 417.35 | 418.64 | 416.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 12:15:00 | 415.15 | 418.68 | 416.43 | SL hit (close<static) qty=1.00 sl=415.35 alert=retest2 |

### Cycle 9 — SELL (started 2025-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 13:15:00 | 405.40 | 414.75 | 414.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 14:15:00 | 403.45 | 413.63 | 414.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 401.00 | 400.83 | 405.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 09:45:00 | 401.60 | 400.83 | 405.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 405.35 | 400.80 | 405.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 10:30:00 | 405.65 | 400.80 | 405.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 405.00 | 400.84 | 405.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:45:00 | 405.95 | 400.84 | 405.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 405.20 | 400.88 | 405.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:45:00 | 405.30 | 400.88 | 405.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 403.20 | 400.90 | 405.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 399.80 | 404.14 | 406.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 10:00:00 | 402.80 | 403.72 | 405.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 402.65 | 403.71 | 405.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 12:00:00 | 402.60 | 403.71 | 405.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 406.40 | 403.75 | 405.83 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-29 13:15:00 | 406.40 | 403.75 | 405.83 | SL hit (close>static) qty=1.00 sl=405.75 alert=retest2 |

### Cycle 10 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 403.40 | 370.33 | 370.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 11:15:00 | 407.65 | 371.04 | 370.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 397.55 | 401.89 | 391.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-04 10:00:00 | 397.55 | 401.89 | 391.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 12:15:00 | 396.90 | 401.70 | 391.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 14:00:00 | 397.50 | 401.66 | 391.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 09:15:00 | 400.85 | 401.51 | 391.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 388.35 | 402.62 | 392.57 | SL hit (close<static) qty=1.00 sl=391.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-19 14:00:00 | 470.70 | 2024-11-21 09:15:00 | 447.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-19 15:00:00 | 470.15 | 2024-11-21 09:15:00 | 446.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-19 14:00:00 | 470.70 | 2024-11-22 09:15:00 | 469.45 | STOP_HIT | 0.50 | 0.27% |
| SELL | retest2 | 2024-11-19 15:00:00 | 470.15 | 2024-11-22 09:15:00 | 469.45 | STOP_HIT | 0.50 | 0.15% |
| SELL | retest2 | 2025-04-07 09:15:00 | 382.45 | 2025-04-15 09:15:00 | 416.25 | STOP_HIT | 1.00 | -8.84% |
| SELL | retest2 | 2025-04-08 10:30:00 | 397.65 | 2025-04-15 09:15:00 | 416.25 | STOP_HIT | 1.00 | -4.68% |
| SELL | retest2 | 2025-04-08 13:45:00 | 397.50 | 2025-04-15 09:15:00 | 416.25 | STOP_HIT | 1.00 | -4.72% |
| SELL | retest2 | 2025-04-08 14:30:00 | 398.05 | 2025-04-15 09:15:00 | 416.25 | STOP_HIT | 1.00 | -4.57% |
| SELL | retest2 | 2025-05-14 12:00:00 | 408.90 | 2025-05-16 09:15:00 | 421.60 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-05-20 14:00:00 | 409.20 | 2025-05-21 13:15:00 | 415.00 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-05-21 10:15:00 | 408.90 | 2025-05-21 13:15:00 | 415.00 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-05-21 11:00:00 | 409.00 | 2025-05-21 13:15:00 | 415.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-05-22 09:15:00 | 407.20 | 2025-06-06 11:15:00 | 418.25 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-05-27 09:45:00 | 408.85 | 2025-06-06 11:15:00 | 418.25 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-05-27 13:45:00 | 409.85 | 2025-06-09 09:15:00 | 427.20 | STOP_HIT | 1.00 | -4.23% |
| SELL | retest2 | 2025-05-29 09:30:00 | 410.05 | 2025-06-09 09:15:00 | 427.20 | STOP_HIT | 1.00 | -4.18% |
| SELL | retest2 | 2025-06-03 11:00:00 | 407.60 | 2025-06-09 09:15:00 | 427.20 | STOP_HIT | 1.00 | -4.81% |
| SELL | retest2 | 2025-06-05 13:00:00 | 409.50 | 2025-06-09 09:15:00 | 427.20 | STOP_HIT | 1.00 | -4.32% |
| BUY | retest2 | 2025-06-13 12:00:00 | 406.90 | 2025-06-16 09:15:00 | 400.40 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-06-16 13:30:00 | 406.95 | 2025-06-18 11:15:00 | 399.75 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-06-16 14:00:00 | 407.85 | 2025-06-18 11:15:00 | 399.75 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-06-16 15:15:00 | 407.05 | 2025-06-18 11:15:00 | 399.75 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-07-08 09:30:00 | 414.90 | 2025-07-25 12:15:00 | 415.15 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-07-08 11:15:00 | 415.45 | 2025-07-25 12:15:00 | 415.15 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-07-08 13:00:00 | 414.90 | 2025-07-28 12:15:00 | 410.35 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-07-22 15:15:00 | 418.00 | 2025-07-28 12:15:00 | 410.35 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-07-23 14:30:00 | 417.70 | 2025-07-28 12:15:00 | 410.35 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-07-24 10:15:00 | 417.35 | 2025-07-28 12:15:00 | 410.35 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-09-26 09:15:00 | 399.80 | 2025-09-29 13:15:00 | 406.40 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-09-29 10:00:00 | 402.80 | 2025-09-29 13:15:00 | 406.40 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-09-29 11:30:00 | 402.65 | 2025-09-29 13:15:00 | 406.40 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-09-29 12:00:00 | 402.60 | 2025-09-29 13:15:00 | 406.40 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-09-30 09:15:00 | 405.30 | 2025-09-30 09:15:00 | 407.45 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-10-06 11:45:00 | 406.00 | 2025-10-07 09:15:00 | 407.60 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-10-06 12:15:00 | 405.70 | 2025-10-07 09:15:00 | 407.60 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-10-08 09:15:00 | 404.70 | 2025-10-10 09:15:00 | 407.60 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-10-10 14:30:00 | 405.25 | 2025-10-29 15:15:00 | 410.50 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-10-15 13:45:00 | 405.20 | 2025-10-29 15:15:00 | 410.50 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-10-16 12:00:00 | 403.90 | 2025-10-29 15:15:00 | 410.50 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-10-29 12:45:00 | 405.05 | 2025-10-29 15:15:00 | 410.50 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-11-03 11:45:00 | 401.95 | 2025-11-07 14:15:00 | 381.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 10:15:00 | 402.25 | 2025-11-07 14:15:00 | 382.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 11:45:00 | 401.95 | 2025-11-24 14:15:00 | 362.03 | TARGET_HIT | 0.50 | 9.93% |
| SELL | retest2 | 2025-11-04 10:15:00 | 402.25 | 2025-11-25 13:15:00 | 361.75 | TARGET_HIT | 0.50 | 10.07% |
| BUY | retest2 | 2026-03-04 14:00:00 | 397.50 | 2026-03-09 09:15:00 | 388.35 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2026-03-05 09:15:00 | 400.85 | 2026-03-09 09:15:00 | 388.35 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2026-03-10 09:15:00 | 399.60 | 2026-03-23 09:15:00 | 390.25 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2026-03-23 14:00:00 | 397.10 | 2026-03-27 10:15:00 | 395.80 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2026-03-24 09:15:00 | 405.05 | 2026-03-27 10:15:00 | 395.80 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2026-03-24 12:30:00 | 400.70 | 2026-03-27 10:15:00 | 395.80 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2026-03-24 14:45:00 | 399.60 | 2026-03-27 10:15:00 | 395.80 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2026-03-25 09:15:00 | 407.90 | 2026-03-27 14:15:00 | 395.80 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2026-03-27 12:45:00 | 402.85 | 2026-03-30 09:15:00 | 388.80 | STOP_HIT | 1.00 | -3.49% |
| BUY | retest2 | 2026-04-02 15:00:00 | 403.35 | 2026-04-15 09:15:00 | 443.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 09:15:00 | 404.05 | 2026-04-15 09:15:00 | 444.46 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 10:30:00 | 403.50 | 2026-04-15 09:15:00 | 443.85 | TARGET_HIT | 1.00 | 10.00% |
