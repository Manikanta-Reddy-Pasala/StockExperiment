# Balrampur Chini Mills Ltd. (BALRAMCHIN)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 522.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT2_SKIP | 3 |
| ALERT3 | 80 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 46 |
| PARTIAL | 3 |
| TARGET_HIT | 6 |
| STOP_HIT | 40 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 49 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 37
- **Target hits / Stop hits / Partials:** 6 / 40 / 3
- **Avg / median % per leg:** -0.07% / -1.57%
- **Sum % (uncompounded):** -3.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 7 | 25.9% | 6 | 21 | 0 | 0.78% | 21.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 27 | 7 | 25.9% | 6 | 21 | 0 | 0.78% | 21.1% |
| SELL (all) | 22 | 5 | 22.7% | 0 | 19 | 3 | -1.11% | -24.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 22 | 5 | 22.7% | 0 | 19 | 3 | -1.11% | -24.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 49 | 12 | 24.5% | 6 | 40 | 3 | -0.07% | -3.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 11:15:00 | 386.50 | 394.79 | 394.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-30 13:15:00 | 385.60 | 394.62 | 394.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-12 09:15:00 | 390.45 | 390.10 | 392.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-12 10:00:00 | 390.45 | 390.10 | 392.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 09:15:00 | 389.35 | 387.32 | 390.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 09:30:00 | 390.00 | 387.32 | 390.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 10:15:00 | 392.80 | 387.38 | 390.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 10:45:00 | 393.70 | 387.38 | 390.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 11:15:00 | 392.50 | 387.43 | 390.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 12:00:00 | 392.50 | 387.43 | 390.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 13:15:00 | 389.90 | 387.49 | 390.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 13:30:00 | 389.55 | 387.49 | 390.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 14:15:00 | 390.70 | 387.52 | 390.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 15:00:00 | 390.70 | 387.52 | 390.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 15:15:00 | 389.50 | 387.54 | 390.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-21 09:15:00 | 385.80 | 387.54 | 390.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-21 14:15:00 | 391.30 | 387.55 | 390.18 | SL hit (close>static) qty=1.00 sl=390.70 alert=retest2 |

### Cycle 2 — BUY (started 2023-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-01 14:15:00 | 406.15 | 392.27 | 392.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 15:15:00 | 408.45 | 392.43 | 392.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 10:15:00 | 392.95 | 395.92 | 394.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 10:15:00 | 392.95 | 395.92 | 394.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 10:15:00 | 392.95 | 395.92 | 394.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 11:00:00 | 392.95 | 395.92 | 394.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 11:15:00 | 394.75 | 395.91 | 394.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 11:45:00 | 394.80 | 395.91 | 394.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 12:15:00 | 394.80 | 395.90 | 394.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-09 09:45:00 | 395.65 | 395.85 | 394.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-09 10:15:00 | 392.20 | 395.81 | 394.20 | SL hit (close<static) qty=1.00 sl=394.10 alert=retest2 |

### Cycle 3 — SELL (started 2023-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 15:15:00 | 383.50 | 393.13 | 393.17 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 10:15:00 | 402.10 | 393.21 | 393.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 12:15:00 | 404.35 | 393.39 | 393.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 14:15:00 | 399.30 | 399.60 | 396.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-12 15:00:00 | 399.30 | 399.60 | 396.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 09:15:00 | 411.45 | 422.05 | 415.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-23 10:00:00 | 411.45 | 422.05 | 415.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 10:15:00 | 409.05 | 421.92 | 415.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-23 11:00:00 | 409.05 | 421.92 | 415.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 10:15:00 | 412.10 | 420.95 | 414.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-25 11:00:00 | 412.10 | 420.95 | 414.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 11:15:00 | 410.75 | 420.85 | 414.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-25 11:45:00 | 412.15 | 420.85 | 414.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 14:15:00 | 413.50 | 418.38 | 414.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-27 14:45:00 | 412.55 | 418.38 | 414.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 15:15:00 | 413.00 | 418.32 | 414.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-30 09:15:00 | 411.50 | 418.32 | 414.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 09:15:00 | 413.50 | 418.27 | 414.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 10:15:00 | 416.95 | 417.23 | 413.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-11-22 14:15:00 | 458.65 | 429.37 | 422.29 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-18 10:15:00 | 404.20 | 424.80 | 424.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-18 14:15:00 | 401.95 | 423.97 | 424.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-29 11:15:00 | 414.05 | 411.76 | 417.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 11:15:00 | 414.05 | 411.76 | 417.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 11:15:00 | 414.05 | 411.76 | 417.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-29 11:30:00 | 415.50 | 411.76 | 417.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 09:15:00 | 418.00 | 411.80 | 417.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-01 09:45:00 | 420.00 | 411.80 | 417.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 10:15:00 | 416.45 | 411.85 | 417.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-01 10:30:00 | 418.15 | 411.85 | 417.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 11:15:00 | 417.00 | 411.90 | 417.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-01 11:45:00 | 417.65 | 411.90 | 417.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 12:15:00 | 417.40 | 411.96 | 417.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-01 12:45:00 | 417.95 | 411.96 | 417.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 13:15:00 | 418.00 | 412.02 | 417.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-01 13:45:00 | 418.30 | 412.02 | 417.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 14:15:00 | 414.30 | 412.04 | 417.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-02 09:15:00 | 409.00 | 412.06 | 417.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-10 12:15:00 | 388.55 | 408.29 | 414.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-30 10:15:00 | 396.00 | 395.76 | 404.46 | SL hit (close>ema200) qty=0.50 sl=395.76 alert=retest2 |

### Cycle 6 — BUY (started 2024-05-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 12:15:00 | 390.35 | 382.54 | 382.54 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 10:15:00 | 376.05 | 382.57 | 382.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 13:15:00 | 374.20 | 382.38 | 382.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 11:15:00 | 380.80 | 380.55 | 381.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-16 12:00:00 | 380.80 | 380.55 | 381.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 379.15 | 380.54 | 381.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 14:00:00 | 378.75 | 380.52 | 381.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 15:15:00 | 378.40 | 380.51 | 381.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-17 14:15:00 | 384.70 | 380.55 | 381.41 | SL hit (close>static) qty=1.00 sl=381.80 alert=retest2 |

### Cycle 8 — BUY (started 2024-06-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 12:15:00 | 404.10 | 381.82 | 381.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 13:15:00 | 405.10 | 382.05 | 381.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 09:15:00 | 609.75 | 622.84 | 583.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-22 10:00:00 | 609.75 | 622.84 | 583.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 595.00 | 620.84 | 595.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 588.00 | 620.84 | 595.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 584.00 | 620.48 | 595.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:45:00 | 583.05 | 620.48 | 595.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 582.05 | 620.10 | 595.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 11:15:00 | 582.05 | 620.10 | 595.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 586.30 | 618.30 | 595.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:15:00 | 580.80 | 618.30 | 595.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2024-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 12:15:00 | 555.55 | 579.69 | 579.79 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 14:15:00 | 609.00 | 579.61 | 579.61 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2024-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 14:15:00 | 538.00 | 580.12 | 580.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 15:15:00 | 529.95 | 579.62 | 579.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 13:15:00 | 470.25 | 468.44 | 496.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-20 14:00:00 | 470.25 | 468.44 | 496.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 489.15 | 460.92 | 481.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 489.15 | 460.92 | 481.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 477.70 | 461.08 | 481.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 09:15:00 | 472.80 | 461.08 | 481.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 15:15:00 | 473.85 | 462.36 | 480.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 14:15:00 | 501.90 | 464.17 | 480.93 | SL hit (close>static) qty=1.00 sl=497.15 alert=retest2 |

### Cycle 12 — BUY (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 13:15:00 | 545.40 | 493.79 | 493.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 14:15:00 | 546.60 | 494.32 | 493.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 545.15 | 545.90 | 528.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 10:00:00 | 545.15 | 545.90 | 528.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 539.05 | 547.35 | 530.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 09:45:00 | 551.35 | 546.21 | 530.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:30:00 | 549.10 | 550.89 | 536.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 11:00:00 | 549.25 | 550.96 | 537.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-03 09:15:00 | 606.49 | 560.28 | 545.73 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 550.00 | 581.55 | 581.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 09:15:00 | 547.20 | 579.92 | 580.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 15:15:00 | 578.00 | 574.90 | 577.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 15:15:00 | 578.00 | 574.90 | 577.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 578.00 | 574.90 | 577.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 577.70 | 574.91 | 577.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 582.70 | 574.99 | 578.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:45:00 | 583.05 | 574.99 | 578.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 582.85 | 575.07 | 578.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:00:00 | 582.85 | 575.07 | 578.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 585.05 | 576.08 | 578.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:30:00 | 585.70 | 576.08 | 578.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 577.70 | 570.81 | 575.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:30:00 | 578.55 | 570.81 | 575.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 578.85 | 570.89 | 575.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:30:00 | 579.65 | 570.89 | 575.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 436.50 | 424.08 | 438.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:00:00 | 433.50 | 424.18 | 438.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 14:15:00 | 411.82 | 423.79 | 437.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 431.00 | 423.82 | 437.45 | SL hit (close>ema200) qty=0.50 sl=423.82 alert=retest2 |

### Cycle 14 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 454.75 | 444.83 | 444.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 14:15:00 | 465.30 | 445.63 | 445.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-17 12:15:00 | 469.30 | 470.94 | 460.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-17 13:00:00 | 469.30 | 470.94 | 460.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 452.00 | 472.74 | 463.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:00:00 | 452.00 | 472.74 | 463.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 451.15 | 472.53 | 463.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:45:00 | 446.60 | 472.53 | 463.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 472.15 | 480.03 | 470.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:00:00 | 476.85 | 480.00 | 470.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 15:00:00 | 475.95 | 479.86 | 470.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-21 09:15:00 | 523.55 | 482.66 | 473.55 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-26 09:30:00 | 393.80 | 2023-05-26 10:15:00 | 389.85 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2023-05-29 09:30:00 | 394.55 | 2023-06-06 09:15:00 | 387.80 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2023-05-29 11:15:00 | 394.35 | 2023-06-06 09:15:00 | 387.80 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2023-05-29 13:45:00 | 394.85 | 2023-06-06 09:15:00 | 387.80 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2023-06-07 09:45:00 | 398.00 | 2023-06-14 14:15:00 | 389.40 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2023-06-14 13:30:00 | 394.90 | 2023-06-14 14:15:00 | 389.40 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2023-06-16 09:30:00 | 396.15 | 2023-06-20 09:15:00 | 391.60 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2023-06-20 14:00:00 | 399.55 | 2023-06-23 14:15:00 | 390.60 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2023-07-21 09:15:00 | 385.80 | 2023-07-21 14:15:00 | 391.30 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2023-07-28 13:30:00 | 388.70 | 2023-07-28 14:15:00 | 395.00 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2023-08-09 09:45:00 | 395.65 | 2023-08-09 10:15:00 | 392.20 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2023-08-09 14:15:00 | 395.55 | 2023-08-11 09:15:00 | 393.70 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2023-08-10 09:15:00 | 397.30 | 2023-08-11 09:15:00 | 393.70 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2023-08-10 12:15:00 | 397.30 | 2023-08-11 09:15:00 | 393.70 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2023-08-16 11:30:00 | 394.45 | 2023-08-18 09:15:00 | 386.75 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2023-08-16 13:00:00 | 394.35 | 2023-08-18 09:15:00 | 386.75 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2023-08-16 14:15:00 | 394.60 | 2023-08-18 09:15:00 | 386.75 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2023-08-17 09:30:00 | 394.25 | 2023-08-18 09:15:00 | 386.75 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2023-08-24 10:15:00 | 396.70 | 2023-08-25 14:15:00 | 384.15 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2023-11-02 10:15:00 | 416.95 | 2023-11-22 14:15:00 | 458.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-07 10:30:00 | 423.85 | 2023-12-07 15:15:00 | 402.70 | STOP_HIT | 1.00 | -4.99% |
| BUY | retest2 | 2023-12-07 12:45:00 | 417.00 | 2023-12-07 15:15:00 | 402.70 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest2 | 2023-12-07 14:30:00 | 416.95 | 2023-12-07 15:15:00 | 402.70 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2024-01-02 09:15:00 | 409.00 | 2024-01-10 12:15:00 | 388.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-02 09:15:00 | 409.00 | 2024-01-30 10:15:00 | 396.00 | STOP_HIT | 0.50 | 3.18% |
| SELL | retest2 | 2024-05-16 14:00:00 | 378.75 | 2024-05-17 14:15:00 | 384.70 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-05-16 15:15:00 | 378.40 | 2024-05-17 14:15:00 | 384.70 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-05-21 09:15:00 | 376.90 | 2024-05-24 12:15:00 | 384.25 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2024-05-21 12:45:00 | 378.20 | 2024-05-24 12:15:00 | 384.25 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-05-28 11:15:00 | 380.10 | 2024-05-29 12:15:00 | 382.15 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-05-28 13:30:00 | 379.90 | 2024-05-29 12:15:00 | 382.15 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2024-05-29 09:15:00 | 379.25 | 2024-05-29 12:15:00 | 382.15 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-05-29 10:00:00 | 380.25 | 2024-05-29 12:15:00 | 382.15 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-05-30 14:00:00 | 377.90 | 2024-06-03 09:15:00 | 387.10 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-05-31 09:45:00 | 379.15 | 2024-06-03 09:15:00 | 387.10 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2024-05-31 15:00:00 | 379.00 | 2024-06-03 09:15:00 | 387.10 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-06-04 09:30:00 | 379.10 | 2024-06-04 11:15:00 | 360.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:30:00 | 379.10 | 2024-06-05 10:15:00 | 387.85 | STOP_HIT | 0.50 | -2.31% |
| SELL | retest2 | 2024-06-04 12:00:00 | 353.10 | 2024-06-05 10:15:00 | 387.85 | STOP_HIT | 1.00 | -9.84% |
| SELL | retest2 | 2025-03-13 09:15:00 | 472.80 | 2025-03-18 14:15:00 | 501.90 | STOP_HIT | 1.00 | -6.15% |
| SELL | retest2 | 2025-03-17 15:15:00 | 473.85 | 2025-03-18 14:15:00 | 501.90 | STOP_HIT | 1.00 | -5.92% |
| BUY | retest2 | 2025-05-12 09:45:00 | 551.35 | 2025-06-03 09:15:00 | 606.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-21 09:30:00 | 549.10 | 2025-06-03 09:15:00 | 604.01 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-23 11:00:00 | 549.25 | 2025-06-03 09:15:00 | 604.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-07 11:15:00 | 549.25 | 2025-08-12 10:15:00 | 550.00 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2026-02-01 12:00:00 | 433.50 | 2026-02-02 14:15:00 | 411.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 12:00:00 | 433.50 | 2026-02-03 09:15:00 | 431.00 | STOP_HIT | 0.50 | 0.58% |
| BUY | retest2 | 2026-04-13 11:00:00 | 476.85 | 2026-04-21 09:15:00 | 523.55 | TARGET_HIT | 1.00 | 9.79% |
| BUY | retest2 | 2026-04-13 15:00:00 | 475.95 | 2026-04-22 13:15:00 | 524.54 | TARGET_HIT | 1.00 | 10.21% |
