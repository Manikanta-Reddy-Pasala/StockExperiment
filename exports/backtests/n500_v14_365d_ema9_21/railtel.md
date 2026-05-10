# Railtel Corporation Of India Ltd. (RAILTEL)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 343.35
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 77 |
| ALERT1 | 46 |
| ALERT2 | 45 |
| ALERT2_SKIP | 23 |
| ALERT3 | 115 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 68 |
| PARTIAL | 24 |
| TARGET_HIT | 8 |
| STOP_HIT | 62 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 94 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 58 / 36
- **Target hits / Stop hits / Partials:** 8 / 62 / 24
- **Avg / median % per leg:** 2.26% / 3.17%
- **Sum % (uncompounded):** 212.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 7 | 36.8% | 7 | 12 | 0 | 2.27% | 43.2% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.61% | -2.6% |
| BUY @ 3rd Alert (retest2) | 18 | 7 | 38.9% | 7 | 11 | 0 | 2.55% | 45.8% |
| SELL (all) | 75 | 51 | 68.0% | 1 | 50 | 24 | 2.26% | 169.4% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.66% | -0.7% |
| SELL @ 3rd Alert (retest2) | 74 | 51 | 68.9% | 1 | 49 | 24 | 2.30% | 170.0% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.64% | -3.3% |
| retest2 (combined) | 92 | 58 | 63.0% | 8 | 60 | 24 | 2.35% | 215.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 317.95 | 305.92 | 305.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 323.05 | 315.94 | 311.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 381.00 | 389.86 | 379.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 381.00 | 389.86 | 379.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 381.00 | 389.86 | 379.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:15:00 | 376.50 | 389.86 | 379.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 382.30 | 388.35 | 380.06 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 14:15:00 | 378.20 | 379.00 | 379.01 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 379.50 | 379.01 | 379.01 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 10:15:00 | 379.00 | 379.01 | 379.01 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 380.45 | 379.30 | 379.14 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 13:15:00 | 378.00 | 378.91 | 378.98 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 14:15:00 | 381.00 | 379.33 | 379.17 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 09:15:00 | 374.50 | 378.42 | 378.79 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 381.40 | 378.47 | 378.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 13:15:00 | 383.50 | 380.17 | 379.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 13:15:00 | 383.90 | 383.95 | 382.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 14:00:00 | 383.90 | 383.95 | 382.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 395.70 | 399.16 | 395.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 395.70 | 399.16 | 395.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 395.35 | 398.39 | 395.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 11:15:00 | 396.35 | 398.39 | 395.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 13:15:00 | 397.00 | 397.40 | 395.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 09:30:00 | 396.35 | 397.40 | 396.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 13:15:00 | 396.15 | 397.87 | 397.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 396.70 | 397.64 | 397.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 14:45:00 | 399.20 | 398.00 | 397.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 399.40 | 397.89 | 397.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-04 11:15:00 | 435.99 | 409.74 | 403.28 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-06-04 11:15:00 | 436.70 | 409.74 | 403.28 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-06-04 11:15:00 | 435.99 | 409.74 | 403.28 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-06-04 11:15:00 | 435.76 | 409.74 | 403.28 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-06-04 11:15:00 | 439.12 | 409.74 | 403.28 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-06-04 11:15:00 | 439.34 | 409.74 | 403.28 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 436.10 | 451.40 | 451.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 432.75 | 443.98 | 447.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 426.70 | 426.30 | 433.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:45:00 | 426.65 | 426.30 | 433.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 432.00 | 427.83 | 433.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 432.00 | 427.83 | 433.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 433.20 | 428.90 | 433.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 426.25 | 428.90 | 433.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 09:15:00 | 442.50 | 430.51 | 431.43 | SL hit (close>static) qty=1.00 sl=436.10 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:15:00 | 431.15 | 430.91 | 431.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:45:00 | 431.05 | 431.13 | 431.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:15:00 | 430.95 | 431.13 | 431.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 430.20 | 430.94 | 431.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 14:15:00 | 427.60 | 430.79 | 431.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 15:00:00 | 428.15 | 430.26 | 431.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:45:00 | 426.15 | 428.15 | 429.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 409.59 | 419.66 | 425.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 409.50 | 419.66 | 425.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 409.40 | 419.66 | 425.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 406.22 | 419.66 | 425.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 406.74 | 419.66 | 425.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 404.84 | 419.66 | 425.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 416.55 | 414.35 | 420.49 | SL hit (close>ema200) qty=0.50 sl=414.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 416.55 | 414.35 | 420.49 | SL hit (close>ema200) qty=0.50 sl=414.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 416.55 | 414.35 | 420.49 | SL hit (close>ema200) qty=0.50 sl=414.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 416.55 | 414.35 | 420.49 | SL hit (close>ema200) qty=0.50 sl=414.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 416.55 | 414.35 | 420.49 | SL hit (close>ema200) qty=0.50 sl=414.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 416.55 | 414.35 | 420.49 | SL hit (close>ema200) qty=0.50 sl=414.35 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 426.75 | 420.92 | 420.56 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 421.50 | 425.09 | 425.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 13:15:00 | 420.20 | 422.35 | 423.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 411.75 | 410.08 | 413.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 15:00:00 | 411.75 | 410.08 | 413.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 410.70 | 410.50 | 413.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:15:00 | 409.95 | 411.18 | 412.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:45:00 | 408.95 | 410.54 | 411.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:30:00 | 410.15 | 410.14 | 411.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:45:00 | 409.00 | 408.72 | 410.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 409.65 | 408.91 | 410.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 409.65 | 408.91 | 410.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 410.00 | 409.12 | 410.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:45:00 | 409.85 | 409.12 | 410.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 409.65 | 409.23 | 410.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:30:00 | 410.35 | 409.23 | 410.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 410.05 | 409.39 | 410.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:00:00 | 410.05 | 409.39 | 410.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 411.10 | 409.73 | 410.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 15:00:00 | 411.10 | 409.73 | 410.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 410.90 | 409.97 | 410.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 410.65 | 409.97 | 410.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 410.00 | 410.00 | 410.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:15:00 | 409.50 | 410.00 | 410.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 12:45:00 | 409.20 | 409.74 | 410.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 14:15:00 | 409.20 | 409.78 | 410.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 15:15:00 | 409.50 | 409.92 | 410.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 409.50 | 409.84 | 410.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 423.35 | 409.84 | 410.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 419.05 | 411.68 | 410.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 419.05 | 411.68 | 410.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 419.05 | 411.68 | 410.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 419.05 | 411.68 | 410.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 419.05 | 411.68 | 410.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 419.05 | 411.68 | 410.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 419.05 | 411.68 | 410.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 419.05 | 411.68 | 410.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 419.05 | 411.68 | 410.82 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 10:15:00 | 411.45 | 413.39 | 413.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 11:15:00 | 410.55 | 412.82 | 413.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 402.55 | 401.72 | 403.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 15:00:00 | 402.55 | 401.72 | 403.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 404.05 | 402.18 | 403.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 403.60 | 402.18 | 403.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 402.65 | 402.28 | 403.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 11:45:00 | 401.25 | 402.10 | 403.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 13:00:00 | 400.70 | 401.82 | 402.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:30:00 | 398.70 | 401.02 | 402.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 12:15:00 | 381.19 | 389.87 | 394.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 380.66 | 387.70 | 393.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 378.76 | 387.70 | 393.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 382.85 | 382.56 | 387.75 | SL hit (close>ema200) qty=0.50 sl=382.56 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 382.85 | 382.56 | 387.75 | SL hit (close>ema200) qty=0.50 sl=382.56 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 382.85 | 382.56 | 387.75 | SL hit (close>ema200) qty=0.50 sl=382.56 alert=retest2 |

### Cycle 15 — BUY (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 12:15:00 | 368.50 | 367.20 | 367.15 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 13:15:00 | 365.90 | 366.94 | 367.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 363.85 | 366.10 | 366.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 358.45 | 356.75 | 360.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 358.45 | 356.75 | 360.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 360.00 | 357.40 | 360.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 356.55 | 357.40 | 360.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 09:30:00 | 353.40 | 349.00 | 349.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 354.95 | 349.47 | 349.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 354.95 | 349.47 | 349.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 354.95 | 349.47 | 349.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 11:15:00 | 355.45 | 353.32 | 351.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 15:15:00 | 358.65 | 358.79 | 356.47 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:15:00 | 367.30 | 358.79 | 356.47 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 357.70 | 361.58 | 359.93 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-22 09:15:00 | 357.70 | 361.58 | 359.93 | SL hit (close<ema400) qty=1.00 sl=359.93 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 356.80 | 361.58 | 359.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 356.65 | 360.60 | 359.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 356.65 | 360.60 | 359.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 12:15:00 | 355.05 | 358.84 | 358.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 354.70 | 357.56 | 358.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 10:15:00 | 333.70 | 333.04 | 337.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 11:00:00 | 333.70 | 333.04 | 337.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 339.15 | 334.81 | 337.11 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 347.05 | 339.78 | 338.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 352.30 | 342.28 | 340.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 348.85 | 349.09 | 345.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 10:30:00 | 348.50 | 349.09 | 345.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 348.55 | 348.52 | 346.61 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 343.50 | 345.98 | 346.08 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 347.20 | 345.76 | 345.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 09:15:00 | 364.50 | 349.94 | 347.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 363.15 | 363.81 | 359.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 10:00:00 | 363.15 | 363.81 | 359.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 395.25 | 397.31 | 395.85 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 15:15:00 | 394.50 | 395.39 | 395.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 10:15:00 | 391.50 | 394.32 | 394.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 14:15:00 | 376.55 | 375.83 | 379.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 15:00:00 | 376.55 | 375.83 | 379.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 388.25 | 378.31 | 380.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 375.90 | 378.31 | 380.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 13:30:00 | 375.60 | 377.08 | 378.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 12:00:00 | 376.45 | 377.89 | 378.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 15:00:00 | 373.00 | 377.31 | 378.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 374.45 | 372.79 | 374.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:45:00 | 373.90 | 372.79 | 374.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 373.30 | 372.89 | 374.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:30:00 | 374.25 | 372.89 | 374.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 374.30 | 373.33 | 374.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:00:00 | 374.30 | 373.33 | 374.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 374.70 | 373.60 | 374.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 374.70 | 373.60 | 374.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 384.60 | 375.80 | 375.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 384.60 | 375.80 | 375.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 384.60 | 375.80 | 375.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 384.60 | 375.80 | 375.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 384.60 | 375.80 | 375.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 15:15:00 | 386.00 | 377.84 | 376.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 384.90 | 385.52 | 382.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 09:45:00 | 385.00 | 385.52 | 382.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 383.55 | 385.13 | 382.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 382.80 | 385.13 | 382.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 384.80 | 384.79 | 383.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:45:00 | 384.20 | 384.79 | 383.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 390.75 | 385.98 | 384.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:00:00 | 393.25 | 387.44 | 384.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 11:15:00 | 384.55 | 387.35 | 387.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 384.55 | 387.35 | 387.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 14:15:00 | 383.35 | 385.99 | 386.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 388.00 | 386.08 | 386.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 388.00 | 386.08 | 386.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 388.00 | 386.08 | 386.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 13:00:00 | 384.60 | 386.10 | 386.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 13:30:00 | 384.85 | 385.76 | 386.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 15:15:00 | 365.37 | 370.30 | 375.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 15:15:00 | 365.61 | 370.30 | 375.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 12:15:00 | 370.90 | 369.95 | 373.50 | SL hit (close>ema200) qty=0.50 sl=369.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 12:15:00 | 370.90 | 369.95 | 373.50 | SL hit (close>ema200) qty=0.50 sl=369.95 alert=retest2 |

### Cycle 25 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 370.75 | 369.71 | 369.64 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 13:15:00 | 368.85 | 369.55 | 369.59 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 09:15:00 | 370.00 | 369.67 | 369.64 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 368.90 | 369.51 | 369.57 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 12:15:00 | 370.25 | 369.71 | 369.65 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 13:15:00 | 368.55 | 369.48 | 369.55 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 15:15:00 | 371.50 | 369.94 | 369.75 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 09:15:00 | 368.10 | 369.57 | 369.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 10:15:00 | 366.55 | 368.97 | 369.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 15:15:00 | 366.00 | 365.42 | 366.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 15:15:00 | 366.00 | 365.42 | 366.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 366.00 | 365.42 | 366.64 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 372.50 | 367.32 | 367.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 14:15:00 | 376.60 | 370.21 | 368.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 10:15:00 | 371.90 | 372.03 | 370.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 11:00:00 | 371.90 | 372.03 | 370.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 369.90 | 371.57 | 370.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:45:00 | 369.85 | 371.57 | 370.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 371.70 | 371.60 | 370.44 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 11:15:00 | 368.65 | 369.75 | 369.87 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 369.85 | 369.58 | 369.55 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 367.05 | 369.14 | 369.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 10:15:00 | 366.20 | 368.55 | 369.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 12:15:00 | 349.80 | 349.76 | 352.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-11 12:30:00 | 349.50 | 349.76 | 352.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 351.00 | 350.00 | 352.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:30:00 | 350.50 | 350.00 | 352.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 353.90 | 350.78 | 352.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 15:00:00 | 353.90 | 350.78 | 352.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 353.70 | 351.37 | 352.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 357.35 | 351.37 | 352.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 355.35 | 353.08 | 353.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 12:15:00 | 358.25 | 354.76 | 353.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 354.25 | 355.78 | 355.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 12:15:00 | 354.25 | 355.78 | 355.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 354.25 | 355.78 | 355.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:45:00 | 354.35 | 355.78 | 355.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 354.00 | 355.43 | 354.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 354.00 | 355.43 | 354.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 353.85 | 355.11 | 354.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:30:00 | 353.15 | 355.11 | 354.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 15:15:00 | 351.90 | 354.47 | 354.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 351.25 | 353.82 | 354.26 | Break + close below crossover candle low |

### Cycle 39 — BUY (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 12:15:00 | 357.70 | 354.48 | 354.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 368.30 | 358.56 | 356.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 15:15:00 | 363.50 | 363.51 | 360.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 09:15:00 | 359.80 | 363.51 | 360.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 356.95 | 362.20 | 360.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 356.95 | 362.20 | 360.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 357.15 | 361.19 | 359.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 11:15:00 | 356.70 | 361.19 | 359.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 14:15:00 | 356.05 | 358.54 | 358.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 15:15:00 | 355.60 | 357.95 | 358.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 10:15:00 | 352.20 | 351.94 | 354.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 11:00:00 | 352.20 | 351.94 | 354.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 354.30 | 352.41 | 354.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:30:00 | 355.10 | 352.41 | 354.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 12:15:00 | 352.35 | 352.40 | 354.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 13:15:00 | 351.90 | 352.40 | 354.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 14:15:00 | 334.30 | 338.85 | 343.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 11:15:00 | 338.10 | 337.47 | 341.25 | SL hit (close>ema200) qty=0.50 sl=337.47 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 343.40 | 341.39 | 341.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 09:15:00 | 347.85 | 343.25 | 342.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 15:15:00 | 345.25 | 345.33 | 343.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 09:15:00 | 341.10 | 345.33 | 343.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 339.00 | 344.06 | 343.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 339.00 | 344.06 | 343.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 10:15:00 | 339.30 | 343.11 | 343.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 15:15:00 | 338.75 | 340.49 | 341.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 341.75 | 340.74 | 341.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 341.75 | 340.74 | 341.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 341.75 | 340.74 | 341.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:45:00 | 338.90 | 340.27 | 341.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 13:00:00 | 338.90 | 339.99 | 341.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 15:15:00 | 337.50 | 339.97 | 340.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 336.50 | 337.28 | 338.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 334.25 | 336.67 | 338.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:30:00 | 333.75 | 336.14 | 337.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 11:15:00 | 331.70 | 336.14 | 337.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 11:00:00 | 333.90 | 333.83 | 335.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 332.60 | 332.94 | 333.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 330.40 | 332.44 | 333.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 11:45:00 | 329.30 | 331.07 | 332.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:15:00 | 321.95 | 329.37 | 331.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:15:00 | 321.95 | 329.37 | 331.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 320.62 | 327.70 | 330.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 14:15:00 | 319.68 | 325.86 | 329.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 14:15:00 | 317.06 | 325.86 | 329.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 14:15:00 | 317.20 | 325.86 | 329.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 15:15:00 | 315.11 | 324.08 | 328.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 15:15:00 | 315.97 | 324.08 | 328.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 15:15:00 | 312.83 | 324.08 | 328.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 13:15:00 | 323.30 | 321.88 | 325.33 | SL hit (close>ema200) qty=0.50 sl=321.88 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 13:15:00 | 323.30 | 321.88 | 325.33 | SL hit (close>ema200) qty=0.50 sl=321.88 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 13:15:00 | 323.30 | 321.88 | 325.33 | SL hit (close>ema200) qty=0.50 sl=321.88 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 13:15:00 | 323.30 | 321.88 | 325.33 | SL hit (close>ema200) qty=0.50 sl=321.88 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 13:15:00 | 323.30 | 321.88 | 325.33 | SL hit (close>ema200) qty=0.50 sl=321.88 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 13:15:00 | 323.30 | 321.88 | 325.33 | SL hit (close>ema200) qty=0.50 sl=321.88 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 13:15:00 | 323.30 | 321.88 | 325.33 | SL hit (close>ema200) qty=0.50 sl=321.88 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 13:15:00 | 323.30 | 321.88 | 325.33 | SL hit (close>ema200) qty=0.50 sl=321.88 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 13:15:00 | 323.30 | 321.88 | 325.33 | SL hit (close>ema200) qty=0.50 sl=321.88 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 10:45:00 | 328.55 | 322.74 | 323.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 12:15:00 | 331.85 | 325.41 | 324.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 331.85 | 325.41 | 324.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 13:15:00 | 336.25 | 327.58 | 325.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 14:15:00 | 330.00 | 330.58 | 328.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 14:15:00 | 330.00 | 330.58 | 328.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 330.00 | 330.58 | 328.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 15:00:00 | 330.00 | 330.58 | 328.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 332.15 | 330.79 | 329.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 12:15:00 | 334.20 | 331.51 | 329.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 12:45:00 | 333.75 | 332.23 | 330.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 11:30:00 | 334.75 | 333.69 | 332.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 13:30:00 | 333.55 | 333.61 | 332.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 332.40 | 333.37 | 332.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:30:00 | 332.15 | 333.37 | 332.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 331.40 | 332.98 | 332.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 332.85 | 332.98 | 332.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 10:15:00 | 330.05 | 332.51 | 332.13 | SL hit (close<static) qty=1.00 sl=330.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 11:15:00 | 327.15 | 331.44 | 331.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 11:15:00 | 327.15 | 331.44 | 331.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 11:15:00 | 327.15 | 331.44 | 331.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 11:15:00 | 327.15 | 331.44 | 331.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 11:15:00 | 327.15 | 331.44 | 331.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 325.55 | 329.48 | 330.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 15:15:00 | 329.30 | 329.21 | 330.35 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:15:00 | 326.40 | 329.21 | 330.35 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 323.90 | 328.15 | 329.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 328.55 | 326.31 | 327.19 | SL hit (close>ema400) qty=1.00 sl=327.19 alert=retest1 |

### Cycle 45 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 332.60 | 328.60 | 328.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 342.00 | 331.28 | 329.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 355.85 | 356.12 | 348.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 10:00:00 | 355.85 | 356.12 | 348.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 379.65 | 361.28 | 354.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 388.00 | 361.28 | 354.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 13:15:00 | 367.85 | 369.38 | 369.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 13:15:00 | 367.85 | 369.38 | 369.56 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 14:15:00 | 371.00 | 369.71 | 369.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 373.85 | 370.69 | 370.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 372.15 | 373.90 | 372.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 372.15 | 373.90 | 372.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 372.15 | 373.90 | 372.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 372.60 | 373.90 | 372.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 371.40 | 373.40 | 372.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:00:00 | 371.40 | 373.40 | 372.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 371.25 | 372.97 | 372.28 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 369.35 | 371.85 | 371.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 367.80 | 370.42 | 371.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 12:15:00 | 370.20 | 369.93 | 370.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 12:15:00 | 370.20 | 369.93 | 370.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 370.20 | 369.93 | 370.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:00:00 | 370.20 | 369.93 | 370.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 369.10 | 369.77 | 370.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:45:00 | 370.25 | 369.77 | 370.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 370.70 | 369.95 | 370.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:45:00 | 372.20 | 369.95 | 370.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 370.50 | 370.06 | 370.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 370.35 | 370.06 | 370.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 368.90 | 369.83 | 370.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 366.40 | 369.20 | 369.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 348.08 | 357.18 | 362.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 14:15:00 | 343.15 | 342.84 | 348.71 | SL hit (close>ema200) qty=0.50 sl=342.84 alert=retest2 |

### Cycle 49 — BUY (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 13:15:00 | 351.10 | 345.79 | 345.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 14:15:00 | 352.15 | 347.06 | 346.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 12:15:00 | 348.55 | 349.22 | 347.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 12:45:00 | 348.75 | 349.22 | 347.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 347.25 | 348.83 | 347.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 347.25 | 348.83 | 347.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 347.75 | 348.61 | 347.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 15:15:00 | 347.20 | 348.61 | 347.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 347.20 | 348.33 | 347.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 346.10 | 348.33 | 347.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 349.45 | 348.55 | 347.88 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 344.40 | 347.32 | 347.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 339.15 | 344.36 | 345.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 334.35 | 331.14 | 334.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 334.35 | 331.14 | 334.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 334.35 | 331.14 | 334.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 336.35 | 331.14 | 334.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 333.60 | 331.63 | 334.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 332.70 | 331.63 | 334.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:15:00 | 332.60 | 332.48 | 334.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:45:00 | 333.15 | 332.48 | 334.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 336.00 | 333.19 | 334.61 | SL hit (close>static) qty=1.00 sl=335.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 336.00 | 333.19 | 334.61 | SL hit (close>static) qty=1.00 sl=335.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 336.00 | 333.19 | 334.61 | SL hit (close>static) qty=1.00 sl=335.95 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 09:15:00 | 341.75 | 335.51 | 335.45 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 331.00 | 334.77 | 335.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 327.95 | 333.00 | 334.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 334.20 | 331.88 | 333.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 334.20 | 331.88 | 333.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 334.20 | 331.88 | 333.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:00:00 | 334.20 | 331.88 | 333.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 331.10 | 331.73 | 333.10 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 337.95 | 333.65 | 333.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 342.00 | 335.32 | 334.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 339.30 | 340.90 | 338.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 10:15:00 | 339.30 | 340.90 | 338.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 339.30 | 340.90 | 338.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:30:00 | 339.15 | 340.90 | 338.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 337.85 | 340.29 | 338.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:00:00 | 337.85 | 340.29 | 338.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 340.00 | 340.23 | 338.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 13:30:00 | 340.60 | 340.23 | 338.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 14:15:00 | 340.90 | 340.23 | 338.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:30:00 | 342.05 | 342.26 | 339.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 12:45:00 | 340.95 | 350.21 | 347.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 334.60 | 347.09 | 346.06 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-01 13:15:00 | 334.60 | 347.09 | 346.06 | SL hit (close<static) qty=1.00 sl=337.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 13:15:00 | 334.60 | 347.09 | 346.06 | SL hit (close<static) qty=1.00 sl=337.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 13:15:00 | 334.60 | 347.09 | 346.06 | SL hit (close<static) qty=1.00 sl=337.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 13:15:00 | 334.60 | 347.09 | 346.06 | SL hit (close<static) qty=1.00 sl=337.60 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 334.60 | 347.09 | 346.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 332.70 | 344.21 | 344.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 331.35 | 341.64 | 343.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 336.30 | 333.07 | 337.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 336.30 | 333.07 | 337.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 336.30 | 333.07 | 337.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 336.30 | 333.07 | 337.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 338.50 | 334.16 | 337.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 338.70 | 334.16 | 337.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 334.90 | 334.30 | 337.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 330.75 | 335.01 | 336.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 12:30:00 | 334.00 | 334.51 | 335.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 09:15:00 | 332.95 | 334.68 | 335.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 15:15:00 | 331.40 | 329.90 | 329.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-09 15:15:00 | 331.40 | 329.90 | 329.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-09 15:15:00 | 331.40 | 329.90 | 329.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-02-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 15:15:00 | 331.40 | 329.90 | 329.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 353.00 | 334.52 | 331.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 343.50 | 343.60 | 338.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:15:00 | 339.15 | 343.60 | 338.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 339.10 | 342.70 | 338.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 338.05 | 342.70 | 338.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 339.30 | 342.02 | 338.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 339.30 | 342.02 | 338.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 338.70 | 341.35 | 338.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:00:00 | 338.70 | 341.35 | 338.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 339.30 | 340.94 | 338.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:30:00 | 338.20 | 340.94 | 338.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 339.00 | 340.55 | 338.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:45:00 | 338.60 | 340.55 | 338.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 339.60 | 340.36 | 338.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 14:30:00 | 338.85 | 340.36 | 338.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 339.00 | 340.09 | 338.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 334.00 | 340.09 | 338.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 334.80 | 339.03 | 338.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 334.50 | 339.03 | 338.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 334.65 | 338.16 | 338.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:15:00 | 334.15 | 338.16 | 338.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 334.50 | 337.42 | 337.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 330.05 | 335.18 | 336.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 15:15:00 | 333.20 | 331.33 | 332.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 15:15:00 | 333.20 | 331.33 | 332.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 333.20 | 331.33 | 332.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:00:00 | 329.45 | 330.84 | 331.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:30:00 | 330.25 | 330.18 | 331.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 13:15:00 | 334.10 | 331.91 | 331.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 13:15:00 | 334.10 | 331.91 | 331.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 13:15:00 | 334.10 | 331.91 | 331.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 09:15:00 | 336.00 | 333.14 | 332.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 332.85 | 333.45 | 332.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 11:15:00 | 332.85 | 333.45 | 332.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 332.85 | 333.45 | 332.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 332.85 | 333.45 | 332.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 331.10 | 332.98 | 332.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:00:00 | 331.10 | 332.98 | 332.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 330.15 | 332.41 | 332.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:00:00 | 330.15 | 332.41 | 332.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 328.85 | 331.70 | 332.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 328.30 | 331.02 | 331.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 327.70 | 326.51 | 328.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 327.70 | 326.51 | 328.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 327.70 | 326.51 | 328.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 324.30 | 328.68 | 328.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 308.08 | 310.56 | 314.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-02 13:15:00 | 291.87 | 301.87 | 308.42 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 13:15:00 | 296.05 | 290.49 | 289.89 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 279.15 | 288.45 | 289.13 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 289.15 | 286.16 | 285.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 290.25 | 286.98 | 286.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 11:15:00 | 287.75 | 287.76 | 286.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 12:00:00 | 287.75 | 287.76 | 286.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 285.15 | 287.25 | 286.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:00:00 | 285.15 | 287.25 | 286.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 284.05 | 286.61 | 286.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 284.05 | 286.61 | 286.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 283.50 | 285.99 | 286.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 279.15 | 284.62 | 285.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 286.45 | 284.99 | 285.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 10:15:00 | 286.45 | 284.99 | 285.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 286.45 | 284.99 | 285.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 286.45 | 284.99 | 285.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 288.40 | 285.67 | 285.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:45:00 | 288.55 | 285.67 | 285.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 288.55 | 286.25 | 286.18 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 279.55 | 285.52 | 285.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 278.20 | 282.46 | 284.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 271.40 | 271.30 | 275.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 14:15:00 | 272.50 | 271.70 | 274.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 272.50 | 271.70 | 274.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:45:00 | 274.05 | 271.70 | 274.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 276.20 | 272.65 | 274.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 278.40 | 272.65 | 274.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 279.75 | 274.07 | 274.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:45:00 | 279.80 | 274.07 | 274.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 281.30 | 276.24 | 275.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 283.20 | 277.63 | 276.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 274.75 | 278.37 | 277.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 274.75 | 278.37 | 277.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 274.75 | 278.37 | 277.05 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 273.50 | 276.28 | 276.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 270.80 | 275.18 | 275.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 277.80 | 274.51 | 275.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 277.80 | 274.51 | 275.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 277.80 | 274.51 | 275.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 277.20 | 274.51 | 275.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 277.00 | 275.01 | 275.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 277.15 | 275.01 | 275.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 275.15 | 275.24 | 275.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:30:00 | 275.95 | 275.24 | 275.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 271.45 | 274.48 | 275.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 264.55 | 274.14 | 274.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 15:15:00 | 251.32 | 259.68 | 265.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 262.35 | 259.00 | 263.46 | SL hit (close>ema200) qty=0.50 sl=259.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 10:15:00 | 269.15 | 262.44 | 263.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 10:45:00 | 270.15 | 264.44 | 264.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 11:15:00 | 271.95 | 265.94 | 265.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-25 11:15:00 | 271.95 | 265.94 | 265.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 271.95 | 265.94 | 265.21 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 259.00 | 265.30 | 265.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 254.35 | 260.10 | 262.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 263.50 | 254.04 | 257.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 263.50 | 254.04 | 257.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 263.50 | 254.04 | 257.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 264.44 | 254.04 | 257.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 261.97 | 255.63 | 257.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:45:00 | 262.42 | 255.63 | 257.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 264.08 | 259.96 | 259.43 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 254.18 | 258.72 | 259.12 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 263.97 | 259.85 | 259.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 266.99 | 262.78 | 261.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 282.72 | 282.96 | 277.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 282.48 | 282.96 | 277.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 281.48 | 286.14 | 283.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 283.19 | 286.14 | 283.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 311.51 | 291.94 | 287.25 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 15:15:00 | 326.75 | 327.59 | 327.65 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 10:15:00 | 329.25 | 327.83 | 327.74 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 320.82 | 327.37 | 328.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 316.80 | 322.50 | 325.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 321.34 | 319.55 | 322.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 321.34 | 319.55 | 322.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 321.34 | 319.55 | 322.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 322.89 | 319.55 | 322.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 324.31 | 320.50 | 323.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 324.05 | 320.50 | 323.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 322.83 | 320.96 | 323.01 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 332.97 | 324.86 | 324.23 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 321.10 | 326.49 | 326.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 320.91 | 325.38 | 326.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 333.15 | 325.37 | 325.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 333.15 | 325.37 | 325.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 333.15 | 325.37 | 325.65 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 334.50 | 327.20 | 326.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 337.50 | 332.16 | 329.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 344.45 | 345.99 | 342.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 344.45 | 345.99 | 342.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 344.45 | 345.99 | 342.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 343.40 | 345.99 | 342.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 345.95 | 345.65 | 343.11 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-30 11:15:00 | 396.35 | 2025-06-04 11:15:00 | 435.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-30 13:15:00 | 397.00 | 2025-06-04 11:15:00 | 436.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-02 09:30:00 | 396.35 | 2025-06-04 11:15:00 | 435.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-03 13:15:00 | 396.15 | 2025-06-04 11:15:00 | 435.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-03 14:45:00 | 399.20 | 2025-06-04 11:15:00 | 439.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-04 09:15:00 | 399.40 | 2025-06-04 11:15:00 | 439.34 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-17 09:15:00 | 426.25 | 2025-06-18 09:15:00 | 442.50 | STOP_HIT | 1.00 | -3.81% |
| SELL | retest2 | 2025-06-18 11:15:00 | 431.15 | 2025-06-19 12:15:00 | 409.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 11:45:00 | 431.05 | 2025-06-19 12:15:00 | 409.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 12:15:00 | 430.95 | 2025-06-19 12:15:00 | 409.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 14:15:00 | 427.60 | 2025-06-19 12:15:00 | 406.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 15:00:00 | 428.15 | 2025-06-19 12:15:00 | 406.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-19 09:45:00 | 426.15 | 2025-06-19 12:15:00 | 404.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 11:15:00 | 431.15 | 2025-06-20 09:15:00 | 416.55 | STOP_HIT | 0.50 | 3.39% |
| SELL | retest2 | 2025-06-18 11:45:00 | 431.05 | 2025-06-20 09:15:00 | 416.55 | STOP_HIT | 0.50 | 3.36% |
| SELL | retest2 | 2025-06-18 12:15:00 | 430.95 | 2025-06-20 09:15:00 | 416.55 | STOP_HIT | 0.50 | 3.34% |
| SELL | retest2 | 2025-06-18 14:15:00 | 427.60 | 2025-06-20 09:15:00 | 416.55 | STOP_HIT | 0.50 | 2.58% |
| SELL | retest2 | 2025-06-18 15:00:00 | 428.15 | 2025-06-20 09:15:00 | 416.55 | STOP_HIT | 0.50 | 2.71% |
| SELL | retest2 | 2025-06-19 09:45:00 | 426.15 | 2025-06-20 09:15:00 | 416.55 | STOP_HIT | 0.50 | 2.25% |
| SELL | retest2 | 2025-07-09 14:15:00 | 409.95 | 2025-07-15 09:15:00 | 419.05 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-07-10 09:45:00 | 408.95 | 2025-07-15 09:15:00 | 419.05 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-07-10 10:30:00 | 410.15 | 2025-07-15 09:15:00 | 419.05 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-07-11 09:45:00 | 409.00 | 2025-07-15 09:15:00 | 419.05 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-07-14 11:15:00 | 409.50 | 2025-07-15 09:15:00 | 419.05 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-07-14 12:45:00 | 409.20 | 2025-07-15 09:15:00 | 419.05 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2025-07-14 14:15:00 | 409.20 | 2025-07-15 09:15:00 | 419.05 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2025-07-14 15:15:00 | 409.50 | 2025-07-15 09:15:00 | 419.05 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-07-24 11:45:00 | 401.25 | 2025-07-28 12:15:00 | 381.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 13:00:00 | 400.70 | 2025-07-28 13:15:00 | 380.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 09:30:00 | 398.70 | 2025-07-28 13:15:00 | 378.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 11:45:00 | 401.25 | 2025-07-29 12:15:00 | 382.85 | STOP_HIT | 0.50 | 4.59% |
| SELL | retest2 | 2025-07-24 13:00:00 | 400.70 | 2025-07-29 12:15:00 | 382.85 | STOP_HIT | 0.50 | 4.45% |
| SELL | retest2 | 2025-07-25 09:30:00 | 398.70 | 2025-07-29 12:15:00 | 382.85 | STOP_HIT | 0.50 | 3.98% |
| SELL | retest2 | 2025-08-08 09:15:00 | 356.55 | 2025-08-18 09:15:00 | 354.95 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2025-08-14 09:30:00 | 353.40 | 2025-08-18 09:15:00 | 354.95 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-08-21 09:15:00 | 367.30 | 2025-08-22 09:15:00 | 357.70 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2025-09-26 09:15:00 | 375.90 | 2025-10-01 14:15:00 | 384.60 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-09-26 13:30:00 | 375.60 | 2025-10-01 14:15:00 | 384.60 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2025-09-29 12:00:00 | 376.45 | 2025-10-01 14:15:00 | 384.60 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-09-29 15:00:00 | 373.00 | 2025-10-01 14:15:00 | 384.60 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest2 | 2025-10-07 14:00:00 | 393.25 | 2025-10-09 11:15:00 | 384.55 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-10-10 13:00:00 | 384.60 | 2025-10-14 15:15:00 | 365.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 13:30:00 | 384.85 | 2025-10-14 15:15:00 | 365.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 13:00:00 | 384.60 | 2025-10-15 12:15:00 | 370.90 | STOP_HIT | 0.50 | 3.56% |
| SELL | retest2 | 2025-10-10 13:30:00 | 384.85 | 2025-10-15 12:15:00 | 370.90 | STOP_HIT | 0.50 | 3.62% |
| SELL | retest2 | 2025-11-20 13:15:00 | 351.90 | 2025-11-24 14:15:00 | 334.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 13:15:00 | 351.90 | 2025-11-25 11:15:00 | 338.10 | STOP_HIT | 0.50 | 3.92% |
| SELL | retest2 | 2025-12-01 11:45:00 | 338.90 | 2025-12-08 12:15:00 | 321.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 13:00:00 | 338.90 | 2025-12-08 12:15:00 | 321.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 15:15:00 | 337.50 | 2025-12-08 13:15:00 | 320.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 09:15:00 | 336.50 | 2025-12-08 14:15:00 | 319.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 10:30:00 | 333.75 | 2025-12-08 14:15:00 | 317.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 11:15:00 | 331.70 | 2025-12-08 14:15:00 | 317.20 | PARTIAL | 0.50 | 4.37% |
| SELL | retest2 | 2025-12-04 11:00:00 | 333.90 | 2025-12-08 15:15:00 | 315.11 | PARTIAL | 0.50 | 5.63% |
| SELL | retest2 | 2025-12-08 09:15:00 | 332.60 | 2025-12-08 15:15:00 | 315.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 11:45:00 | 329.30 | 2025-12-08 15:15:00 | 312.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 11:45:00 | 338.90 | 2025-12-09 13:15:00 | 323.30 | STOP_HIT | 0.50 | 4.60% |
| SELL | retest2 | 2025-12-01 13:00:00 | 338.90 | 2025-12-09 13:15:00 | 323.30 | STOP_HIT | 0.50 | 4.60% |
| SELL | retest2 | 2025-12-01 15:15:00 | 337.50 | 2025-12-09 13:15:00 | 323.30 | STOP_HIT | 0.50 | 4.21% |
| SELL | retest2 | 2025-12-03 09:15:00 | 336.50 | 2025-12-09 13:15:00 | 323.30 | STOP_HIT | 0.50 | 3.92% |
| SELL | retest2 | 2025-12-03 10:30:00 | 333.75 | 2025-12-09 13:15:00 | 323.30 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2025-12-03 11:15:00 | 331.70 | 2025-12-09 13:15:00 | 323.30 | STOP_HIT | 0.50 | 2.53% |
| SELL | retest2 | 2025-12-04 11:00:00 | 333.90 | 2025-12-09 13:15:00 | 323.30 | STOP_HIT | 0.50 | 3.17% |
| SELL | retest2 | 2025-12-08 09:15:00 | 332.60 | 2025-12-09 13:15:00 | 323.30 | STOP_HIT | 0.50 | 2.80% |
| SELL | retest2 | 2025-12-08 11:45:00 | 329.30 | 2025-12-09 13:15:00 | 323.30 | STOP_HIT | 0.50 | 1.82% |
| SELL | retest2 | 2025-12-11 10:45:00 | 328.55 | 2025-12-11 12:15:00 | 331.85 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-12-15 12:15:00 | 334.20 | 2025-12-17 10:15:00 | 330.05 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-12-15 12:45:00 | 333.75 | 2025-12-17 11:15:00 | 327.15 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-12-16 11:30:00 | 334.75 | 2025-12-17 11:15:00 | 327.15 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-12-16 13:30:00 | 333.55 | 2025-12-17 11:15:00 | 327.15 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-12-17 09:15:00 | 332.85 | 2025-12-17 11:15:00 | 327.15 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest1 | 2025-12-18 09:15:00 | 326.40 | 2025-12-19 13:15:00 | 328.55 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-12-26 10:15:00 | 388.00 | 2026-01-01 13:15:00 | 367.85 | STOP_HIT | 1.00 | -5.19% |
| SELL | retest2 | 2026-01-08 09:15:00 | 366.40 | 2026-01-09 09:15:00 | 348.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 366.40 | 2026-01-12 14:15:00 | 343.15 | STOP_HIT | 0.50 | 6.35% |
| SELL | retest2 | 2026-01-22 11:15:00 | 332.70 | 2026-01-22 14:15:00 | 336.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2026-01-22 13:15:00 | 332.60 | 2026-01-22 14:15:00 | 336.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2026-01-22 13:45:00 | 333.15 | 2026-01-22 14:15:00 | 336.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-01-29 13:30:00 | 340.60 | 2026-02-01 13:15:00 | 334.60 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-01-29 14:15:00 | 340.90 | 2026-02-01 13:15:00 | 334.60 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2026-01-30 09:30:00 | 342.05 | 2026-02-01 13:15:00 | 334.60 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2026-02-01 12:45:00 | 340.95 | 2026-02-01 13:15:00 | 334.60 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2026-02-04 09:15:00 | 330.75 | 2026-02-09 15:15:00 | 331.40 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2026-02-04 12:30:00 | 334.00 | 2026-02-09 15:15:00 | 331.40 | STOP_HIT | 1.00 | 0.78% |
| SELL | retest2 | 2026-02-05 09:15:00 | 332.95 | 2026-02-09 15:15:00 | 331.40 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2026-02-17 14:00:00 | 329.45 | 2026-02-18 13:15:00 | 334.10 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2026-02-18 09:30:00 | 330.25 | 2026-02-18 13:15:00 | 334.10 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2026-02-24 09:15:00 | 324.30 | 2026-03-02 09:15:00 | 308.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 324.30 | 2026-03-02 13:15:00 | 291.87 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-23 09:15:00 | 264.55 | 2026-03-23 15:15:00 | 251.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-23 09:15:00 | 264.55 | 2026-03-24 12:15:00 | 262.35 | STOP_HIT | 0.50 | 0.83% |
| SELL | retest2 | 2026-03-25 10:15:00 | 269.15 | 2026-03-25 11:15:00 | 271.95 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2026-03-25 10:45:00 | 270.15 | 2026-03-25 11:15:00 | 271.95 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2026-04-13 10:15:00 | 283.19 | 2026-04-15 09:15:00 | 311.51 | TARGET_HIT | 1.00 | 10.00% |
