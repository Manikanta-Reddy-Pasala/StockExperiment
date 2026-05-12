# Coal India Ltd. (COALINDIA)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 456.55
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 11 |
| ALERT2_SKIP | 6 |
| ALERT3 | 65 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 58 |
| PARTIAL | 4 |
| TARGET_HIT | 15 |
| STOP_HIT | 43 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 62 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 42
- **Target hits / Stop hits / Partials:** 15 / 43 / 4
- **Avg / median % per leg:** 1.86% / -0.52%
- **Sum % (uncompounded):** 115.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 44 | 12 | 27.3% | 12 | 32 | 0 | 1.74% | 76.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 44 | 12 | 27.3% | 12 | 32 | 0 | 1.74% | 76.7% |
| SELL (all) | 18 | 8 | 44.4% | 3 | 11 | 4 | 2.14% | 38.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 8 | 44.4% | 3 | 11 | 4 | 2.14% | 38.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 62 | 20 | 32.3% | 15 | 43 | 4 | 1.86% | 115.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 09:15:00 | 229.85 | 230.95 | 230.95 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 10:15:00 | 236.20 | 231.00 | 230.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 14:15:00 | 236.85 | 231.19 | 231.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 12:15:00 | 430.60 | 431.36 | 405.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-28 12:30:00 | 432.30 | 431.36 | 405.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 14:15:00 | 415.45 | 439.97 | 417.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 14:30:00 | 411.90 | 439.97 | 417.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 15:15:00 | 418.50 | 439.76 | 417.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-14 09:15:00 | 426.05 | 439.76 | 417.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 416.35 | 439.53 | 417.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-14 09:45:00 | 415.90 | 439.53 | 417.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 10:15:00 | 423.15 | 439.36 | 417.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-14 12:15:00 | 426.35 | 439.22 | 417.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-14 13:00:00 | 426.30 | 439.10 | 417.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-14 15:00:00 | 428.55 | 438.86 | 417.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-15 09:15:00 | 414.10 | 438.50 | 417.87 | SL hit (close<static) qty=1.00 sl=415.40 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 09:15:00 | 489.95 | 500.35 | 500.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 13:15:00 | 488.00 | 499.92 | 500.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 09:15:00 | 501.70 | 499.20 | 499.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 09:15:00 | 501.70 | 499.20 | 499.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 501.70 | 499.20 | 499.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:00:00 | 501.70 | 499.20 | 499.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 500.20 | 499.21 | 499.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 12:30:00 | 499.30 | 499.22 | 499.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 13:45:00 | 499.30 | 499.23 | 499.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 14:45:00 | 499.00 | 499.22 | 499.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 13:15:00 | 474.33 | 496.01 | 497.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 13:15:00 | 474.33 | 496.01 | 497.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 13:15:00 | 474.05 | 496.01 | 497.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-28 09:15:00 | 449.37 | 490.30 | 494.74 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 398.00 | 386.80 | 386.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 398.80 | 386.92 | 386.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 385.55 | 387.51 | 387.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 385.55 | 387.51 | 387.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 385.55 | 387.51 | 387.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 385.55 | 387.51 | 387.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 387.15 | 387.50 | 387.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:30:00 | 386.80 | 387.50 | 387.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 11:15:00 | 386.45 | 387.49 | 387.12 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2025-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 12:15:00 | 369.80 | 386.69 | 386.73 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 11:15:00 | 398.35 | 386.76 | 386.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 13:15:00 | 399.60 | 387.01 | 386.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 11:15:00 | 391.25 | 391.58 | 389.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-29 12:00:00 | 391.25 | 391.58 | 389.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 12:15:00 | 390.35 | 391.57 | 389.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 13:00:00 | 390.35 | 391.57 | 389.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 389.40 | 391.54 | 389.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 15:00:00 | 389.40 | 391.54 | 389.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 388.70 | 391.51 | 389.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:15:00 | 388.20 | 391.51 | 389.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 387.45 | 391.44 | 389.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 10:30:00 | 387.45 | 391.44 | 389.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 384.95 | 391.09 | 389.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 11:00:00 | 384.95 | 391.09 | 389.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 11:15:00 | 382.65 | 391.01 | 389.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 12:00:00 | 382.65 | 391.01 | 389.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 386.70 | 389.20 | 388.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 11:00:00 | 386.70 | 389.20 | 388.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 11:15:00 | 384.00 | 389.15 | 388.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 12:00:00 | 384.00 | 389.15 | 388.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 394.50 | 397.08 | 393.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 10:00:00 | 394.50 | 397.08 | 393.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 393.30 | 397.04 | 393.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:00:00 | 393.30 | 397.04 | 393.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 392.65 | 397.00 | 393.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:30:00 | 392.50 | 397.00 | 393.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 394.10 | 396.66 | 393.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:30:00 | 393.65 | 396.66 | 393.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 394.25 | 396.64 | 393.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 15:00:00 | 394.70 | 396.62 | 393.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 395.95 | 396.59 | 393.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 11:15:00 | 394.80 | 396.56 | 393.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 13:15:00 | 393.00 | 396.52 | 393.88 | SL hit (close<static) qty=1.00 sl=393.65 alert=retest2 |

### Cycle 7 — SELL (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 11:15:00 | 387.05 | 393.34 | 393.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 10:15:00 | 386.00 | 392.94 | 393.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 14:15:00 | 388.80 | 388.77 | 390.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-18 15:00:00 | 388.80 | 388.77 | 390.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 389.85 | 388.60 | 390.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:45:00 | 391.10 | 388.60 | 390.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 389.70 | 388.61 | 390.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 14:15:00 | 389.20 | 388.61 | 390.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 390.80 | 388.64 | 390.38 | SL hit (close>static) qty=1.00 sl=390.60 alert=retest2 |

### Cycle 8 — BUY (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 15:15:00 | 394.95 | 385.98 | 385.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 396.40 | 386.08 | 385.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 389.40 | 389.80 | 388.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 14:00:00 | 389.40 | 389.80 | 388.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 389.10 | 389.79 | 388.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 09:15:00 | 389.85 | 389.79 | 388.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 12:15:00 | 386.30 | 389.78 | 388.22 | SL hit (close<static) qty=1.00 sl=388.05 alert=retest2 |

### Cycle 9 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 382.50 | 387.25 | 387.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 381.40 | 387.14 | 387.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 12:15:00 | 387.40 | 386.75 | 386.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 12:15:00 | 387.40 | 386.75 | 386.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 387.40 | 386.75 | 386.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:45:00 | 387.10 | 386.75 | 386.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 387.05 | 386.75 | 386.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:00:00 | 387.05 | 386.75 | 386.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 387.75 | 386.76 | 386.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:30:00 | 387.75 | 386.76 | 386.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 388.05 | 386.78 | 387.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:00:00 | 388.05 | 386.78 | 387.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 388.45 | 386.80 | 387.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:00:00 | 388.45 | 386.80 | 387.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 388.60 | 386.80 | 387.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 15:00:00 | 388.60 | 386.80 | 387.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 388.70 | 386.82 | 387.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:15:00 | 390.05 | 386.82 | 387.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 392.15 | 387.20 | 387.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 11:15:00 | 393.70 | 387.32 | 387.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 13:15:00 | 382.60 | 389.11 | 388.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 13:15:00 | 382.60 | 389.11 | 388.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 382.60 | 389.11 | 388.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:45:00 | 384.80 | 389.11 | 388.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 381.85 | 389.04 | 388.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:45:00 | 381.15 | 389.04 | 388.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 388.00 | 388.80 | 388.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 390.05 | 388.80 | 388.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 13:15:00 | 388.30 | 388.91 | 388.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 15:00:00 | 388.85 | 388.90 | 388.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 379.30 | 388.81 | 388.16 | SL hit (close<static) qty=1.00 sl=387.45 alert=retest2 |

### Cycle 11 — SELL (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 13:15:00 | 374.35 | 387.52 | 387.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 372.95 | 387.38 | 387.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 12:15:00 | 386.80 | 385.73 | 386.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 12:15:00 | 386.80 | 385.73 | 386.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 386.80 | 385.73 | 386.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 13:00:00 | 386.80 | 385.73 | 386.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 386.35 | 385.73 | 386.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 13:30:00 | 387.20 | 385.73 | 386.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 386.50 | 385.74 | 386.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 15:00:00 | 386.50 | 385.74 | 386.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 386.20 | 385.74 | 386.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:15:00 | 385.45 | 385.74 | 386.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 385.65 | 385.74 | 386.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 14:00:00 | 383.85 | 385.74 | 386.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 10:00:00 | 383.90 | 385.67 | 386.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 388.20 | 385.70 | 386.48 | SL hit (close>static) qty=1.00 sl=387.00 alert=retest2 |

### Cycle 12 — BUY (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 12:15:00 | 402.70 | 384.05 | 384.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 09:15:00 | 405.35 | 384.80 | 384.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 419.30 | 419.57 | 408.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-02 09:15:00 | 418.55 | 419.57 | 408.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 412.45 | 422.91 | 413.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 09:45:00 | 418.90 | 422.11 | 413.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 14:45:00 | 418.25 | 421.66 | 413.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 10:45:00 | 417.90 | 421.52 | 413.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 14:45:00 | 417.85 | 421.33 | 413.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 424.80 | 423.96 | 416.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 431.20 | 423.97 | 416.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-12 10:15:00 | 460.79 | 431.12 | 422.24 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-06-02 11:15:00 | 231.95 | 2023-06-06 09:15:00 | 228.80 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2023-06-02 13:30:00 | 232.25 | 2023-06-06 09:15:00 | 228.80 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2023-07-03 10:45:00 | 232.05 | 2023-07-17 14:15:00 | 230.45 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2023-07-03 14:15:00 | 232.05 | 2023-07-18 11:15:00 | 229.40 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2023-07-17 09:30:00 | 232.35 | 2023-07-18 11:15:00 | 229.40 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2023-08-01 10:15:00 | 232.95 | 2023-08-03 12:15:00 | 230.45 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2023-08-04 11:15:00 | 232.80 | 2023-08-08 10:15:00 | 229.60 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2023-08-09 09:15:00 | 232.60 | 2023-08-18 09:15:00 | 227.40 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2023-08-17 13:30:00 | 232.05 | 2023-08-18 09:15:00 | 227.40 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2023-08-24 09:15:00 | 232.40 | 2023-08-24 10:15:00 | 230.50 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-03-14 12:15:00 | 426.35 | 2024-03-15 09:15:00 | 414.10 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2024-03-14 13:00:00 | 426.30 | 2024-03-15 09:15:00 | 414.10 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2024-03-14 15:00:00 | 428.55 | 2024-03-15 09:15:00 | 414.10 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest2 | 2024-03-21 09:30:00 | 428.90 | 2024-05-03 10:15:00 | 471.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-22 09:15:00 | 440.25 | 2024-05-21 09:15:00 | 484.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-13 10:00:00 | 440.35 | 2024-05-21 09:15:00 | 484.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-13 11:00:00 | 438.90 | 2024-05-21 09:15:00 | 482.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 12:45:00 | 438.75 | 2024-06-06 09:15:00 | 482.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-06 09:15:00 | 474.00 | 2024-07-16 09:15:00 | 510.24 | TARGET_HIT | 1.00 | 7.64% |
| BUY | retest2 | 2024-06-27 14:30:00 | 463.85 | 2024-07-30 09:15:00 | 521.40 | TARGET_HIT | 1.00 | 12.41% |
| SELL | retest2 | 2024-10-14 12:30:00 | 499.30 | 2024-10-22 13:15:00 | 474.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-14 13:45:00 | 499.30 | 2024-10-22 13:15:00 | 474.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-14 14:45:00 | 499.00 | 2024-10-22 13:15:00 | 474.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-14 12:30:00 | 499.30 | 2024-10-28 09:15:00 | 449.37 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-14 13:45:00 | 499.30 | 2024-10-28 09:15:00 | 449.37 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-14 14:45:00 | 499.00 | 2024-10-28 09:15:00 | 449.10 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-06-04 15:00:00 | 394.70 | 2025-06-05 13:15:00 | 393.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-06-05 09:15:00 | 395.95 | 2025-06-05 13:15:00 | 393.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-06-05 11:15:00 | 394.80 | 2025-06-05 13:15:00 | 393.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-06-05 15:00:00 | 395.00 | 2025-06-12 13:15:00 | 393.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-06-27 09:15:00 | 396.15 | 2025-06-30 10:15:00 | 392.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-06-27 11:00:00 | 395.95 | 2025-06-30 10:15:00 | 392.50 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-06-27 11:30:00 | 395.55 | 2025-06-30 10:15:00 | 392.50 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-06-27 12:00:00 | 395.70 | 2025-06-30 10:15:00 | 392.50 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-06-27 15:00:00 | 394.55 | 2025-06-30 10:15:00 | 392.50 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-06-30 10:00:00 | 395.25 | 2025-06-30 10:15:00 | 392.50 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-07-22 14:15:00 | 389.20 | 2025-07-23 09:15:00 | 390.80 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-07-23 13:30:00 | 389.10 | 2025-07-23 14:15:00 | 390.80 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-07-24 09:15:00 | 389.50 | 2025-08-28 09:15:00 | 370.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 09:15:00 | 389.50 | 2025-09-02 09:15:00 | 383.75 | STOP_HIT | 0.50 | 1.48% |
| SELL | retest2 | 2025-09-03 14:00:00 | 389.25 | 2025-09-04 12:15:00 | 391.10 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-09-29 09:15:00 | 389.85 | 2025-09-29 12:15:00 | 386.30 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-09-29 14:00:00 | 389.70 | 2025-09-29 14:15:00 | 388.00 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-09-29 15:15:00 | 389.40 | 2025-09-30 13:15:00 | 387.80 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-09-30 09:45:00 | 390.00 | 2025-09-30 13:15:00 | 387.80 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-10-01 09:30:00 | 392.15 | 2025-10-03 09:15:00 | 383.00 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-10-01 14:30:00 | 390.50 | 2025-10-03 09:15:00 | 383.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-10-31 09:15:00 | 390.05 | 2025-11-04 09:15:00 | 379.30 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2025-11-03 13:15:00 | 388.30 | 2025-11-04 09:15:00 | 379.30 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-11-03 15:00:00 | 388.85 | 2025-11-04 09:15:00 | 379.30 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-11-13 14:00:00 | 383.85 | 2025-11-17 09:15:00 | 388.20 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-11-14 10:00:00 | 383.90 | 2025-11-17 09:15:00 | 388.20 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-11-18 15:00:00 | 384.10 | 2025-12-15 11:15:00 | 384.35 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-12-11 13:00:00 | 384.30 | 2025-12-15 11:15:00 | 384.35 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-12-15 09:15:00 | 380.65 | 2025-12-17 10:15:00 | 384.85 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-12-15 10:30:00 | 382.00 | 2025-12-23 09:15:00 | 397.40 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2025-12-16 10:15:00 | 381.40 | 2025-12-23 09:15:00 | 397.40 | STOP_HIT | 1.00 | -4.20% |
| BUY | retest2 | 2026-02-16 09:45:00 | 418.90 | 2026-03-12 10:15:00 | 460.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-18 14:45:00 | 418.25 | 2026-03-12 10:15:00 | 460.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-19 10:45:00 | 417.90 | 2026-03-12 10:15:00 | 459.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-19 14:45:00 | 417.85 | 2026-03-12 10:15:00 | 459.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-04 09:15:00 | 431.20 | 2026-03-13 09:15:00 | 474.32 | TARGET_HIT | 1.00 | 10.00% |
