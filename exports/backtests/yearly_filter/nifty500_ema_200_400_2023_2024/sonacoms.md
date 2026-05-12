# Sona BLW Precision Forgings Ltd. (SONACOMS)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 579.65
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT2_SKIP | 5 |
| ALERT3 | 76 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 61 |
| PARTIAL | 3 |
| TARGET_HIT | 7 |
| STOP_HIT | 55 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 65 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 54
- **Target hits / Stop hits / Partials:** 7 / 55 / 3
- **Avg / median % per leg:** -0.79% / -1.95%
- **Sum % (uncompounded):** -51.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 36 | 4 | 11.1% | 4 | 32 | 0 | -1.22% | -43.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 36 | 4 | 11.1% | 4 | 32 | 0 | -1.22% | -43.8% |
| SELL (all) | 29 | 7 | 24.1% | 3 | 23 | 3 | -0.26% | -7.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.83% | -2.8% |
| SELL @ 3rd Alert (retest2) | 28 | 7 | 25.0% | 3 | 22 | 3 | -0.17% | -4.6% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.83% | -2.8% |
| retest2 (combined) | 64 | 11 | 17.2% | 7 | 54 | 3 | -0.76% | -48.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 14:15:00 | 510.05 | 565.93 | 565.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 09:15:00 | 507.50 | 564.80 | 565.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 560.60 | 560.04 | 562.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 560.60 | 560.04 | 562.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 560.60 | 560.04 | 562.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 10:00:00 | 560.60 | 560.04 | 562.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 15:15:00 | 557.20 | 555.23 | 559.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-07 09:15:00 | 558.40 | 555.23 | 559.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 556.35 | 555.24 | 559.57 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 10:15:00 | 578.00 | 562.52 | 562.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-20 14:15:00 | 580.10 | 563.14 | 562.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-28 09:15:00 | 562.05 | 566.85 | 564.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 09:15:00 | 562.05 | 566.85 | 564.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 562.05 | 566.85 | 564.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 10:00:00 | 562.05 | 566.85 | 564.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 10:15:00 | 559.45 | 566.78 | 564.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 11:00:00 | 559.45 | 566.78 | 564.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 10:15:00 | 561.45 | 566.25 | 564.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-29 11:00:00 | 561.45 | 566.25 | 564.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 13:15:00 | 560.20 | 564.91 | 564.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-01 13:30:00 | 560.75 | 564.91 | 564.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 15:15:00 | 562.80 | 564.84 | 564.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 09:15:00 | 565.50 | 564.84 | 564.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 10:30:00 | 563.95 | 564.79 | 564.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 15:15:00 | 563.40 | 564.66 | 563.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-12 13:15:00 | 558.40 | 567.42 | 565.67 | SL hit (close<static) qty=1.00 sl=559.00 alert=retest2 |

### Cycle 3 — SELL (started 2023-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 15:15:00 | 537.70 | 564.08 | 564.16 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-12-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-21 15:15:00 | 570.00 | 564.27 | 564.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 09:15:00 | 579.70 | 564.43 | 564.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-15 09:15:00 | 598.50 | 604.53 | 589.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-15 09:45:00 | 595.05 | 604.53 | 589.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 11:15:00 | 587.95 | 603.88 | 589.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 12:00:00 | 587.95 | 603.88 | 589.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 12:15:00 | 591.10 | 603.75 | 589.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 11:30:00 | 594.75 | 596.39 | 588.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 12:30:00 | 595.10 | 596.32 | 588.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-24 14:15:00 | 585.45 | 596.13 | 588.28 | SL hit (close<static) qty=1.00 sl=587.65 alert=retest2 |

### Cycle 5 — SELL (started 2024-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 15:15:00 | 603.95 | 644.65 | 644.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-10 09:15:00 | 596.85 | 644.18 | 644.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-21 09:15:00 | 628.00 | 626.73 | 634.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-21 10:00:00 | 628.00 | 626.73 | 634.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 637.55 | 626.84 | 634.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:30:00 | 636.50 | 626.84 | 634.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 11:15:00 | 637.25 | 626.95 | 634.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-21 11:45:00 | 636.25 | 626.95 | 634.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 640.10 | 627.62 | 634.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-22 12:30:00 | 635.00 | 627.79 | 634.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-22 13:30:00 | 636.75 | 627.84 | 634.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 10:15:00 | 636.45 | 628.07 | 634.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 13:30:00 | 636.40 | 628.41 | 634.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 14:15:00 | 626.70 | 628.39 | 634.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 12:30:00 | 625.85 | 629.30 | 634.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 13:45:00 | 625.25 | 629.24 | 634.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 09:15:00 | 622.45 | 628.69 | 634.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 14:15:00 | 637.25 | 628.69 | 633.92 | SL hit (close>static) qty=1.00 sl=634.80 alert=retest2 |

### Cycle 6 — BUY (started 2024-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-10 14:15:00 | 660.45 | 637.79 | 637.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 09:15:00 | 665.25 | 638.30 | 638.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-18 13:15:00 | 640.60 | 642.37 | 640.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 13:15:00 | 640.60 | 642.37 | 640.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 13:15:00 | 640.60 | 642.37 | 640.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 13:45:00 | 639.70 | 642.37 | 640.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 14:15:00 | 640.85 | 642.36 | 640.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 15:00:00 | 640.85 | 642.36 | 640.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 637.00 | 642.42 | 640.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:45:00 | 637.05 | 642.42 | 640.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 638.55 | 642.38 | 640.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:45:00 | 637.30 | 642.38 | 640.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 633.35 | 641.92 | 640.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:45:00 | 633.00 | 641.92 | 640.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 14:15:00 | 638.05 | 641.11 | 639.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 15:15:00 | 634.25 | 641.11 | 639.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 15:15:00 | 634.25 | 641.04 | 639.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 10:00:00 | 634.05 | 640.97 | 639.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 15:15:00 | 639.80 | 640.36 | 639.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 09:15:00 | 645.80 | 640.36 | 639.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 12:15:00 | 636.95 | 640.71 | 639.85 | SL hit (close<static) qty=1.00 sl=637.75 alert=retest2 |

### Cycle 7 — SELL (started 2024-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 13:15:00 | 645.95 | 694.31 | 694.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 640.60 | 692.89 | 693.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 09:15:00 | 708.30 | 680.25 | 686.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 09:15:00 | 708.30 | 680.25 | 686.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 708.30 | 680.25 | 686.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:00:00 | 708.30 | 680.25 | 686.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 704.15 | 680.49 | 686.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:15:00 | 700.00 | 680.49 | 686.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-24 11:15:00 | 724.00 | 680.92 | 687.17 | SL hit (close>static) qty=1.00 sl=710.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 527.25 | 501.11 | 501.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 531.70 | 502.43 | 501.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 09:15:00 | 512.30 | 521.20 | 513.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 512.30 | 521.20 | 513.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 512.30 | 521.20 | 513.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 527.00 | 520.39 | 513.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 11:00:00 | 524.95 | 520.56 | 513.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 502.55 | 520.01 | 513.70 | SL hit (close<static) qty=1.00 sl=509.25 alert=retest2 |

### Cycle 9 — SELL (started 2025-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 15:15:00 | 483.05 | 508.75 | 508.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-23 09:15:00 | 480.30 | 508.47 | 508.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 11:15:00 | 488.45 | 477.59 | 489.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-17 12:00:00 | 488.45 | 477.59 | 489.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 478.75 | 477.60 | 489.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:30:00 | 486.20 | 477.60 | 489.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 486.85 | 477.75 | 489.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:15:00 | 479.85 | 477.90 | 489.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:45:00 | 480.30 | 478.75 | 488.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 12:00:00 | 479.80 | 479.58 | 488.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 12:15:00 | 455.86 | 477.25 | 485.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 12:15:00 | 456.28 | 477.25 | 485.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 12:15:00 | 455.81 | 477.25 | 485.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-07 13:15:00 | 431.87 | 465.97 | 478.07 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 10 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 481.85 | 449.24 | 449.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 12:15:00 | 487.35 | 461.00 | 455.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 11:15:00 | 489.30 | 490.69 | 477.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-05 12:00:00 | 489.30 | 490.69 | 477.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 480.50 | 490.55 | 478.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 476.95 | 490.55 | 478.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 479.15 | 490.22 | 478.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:45:00 | 478.45 | 490.22 | 478.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 480.00 | 490.12 | 478.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 480.25 | 490.12 | 478.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 478.95 | 490.01 | 478.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:15:00 | 485.00 | 489.01 | 478.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 486.85 | 488.87 | 478.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 10:15:00 | 486.00 | 488.83 | 478.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 11:00:00 | 486.50 | 488.86 | 479.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 482.90 | 488.43 | 481.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 477.45 | 488.43 | 481.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 476.00 | 488.30 | 481.45 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 476.00 | 488.30 | 481.45 | SL hit (close<static) qty=1.00 sl=477.70 alert=retest2 |

### Cycle 11 — SELL (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 10:15:00 | 460.65 | 477.74 | 477.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 10:15:00 | 452.40 | 475.45 | 476.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 15:15:00 | 470.00 | 469.69 | 473.38 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 09:15:00 | 464.30 | 469.69 | 473.38 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 470.80 | 469.06 | 472.91 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-27 10:15:00 | 477.45 | 469.14 | 472.94 | SL hit (close>ema400) qty=1.00 sl=472.94 alert=retest1 |

### Cycle 12 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 500.00 | 476.28 | 476.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 528.40 | 477.03 | 476.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 11:15:00 | 513.15 | 517.45 | 503.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-02 12:00:00 | 513.15 | 517.45 | 503.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 503.60 | 517.17 | 503.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 09:45:00 | 498.65 | 517.17 | 503.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 503.45 | 517.03 | 503.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:45:00 | 503.65 | 517.03 | 503.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 500.10 | 516.87 | 503.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 11:30:00 | 500.35 | 516.87 | 503.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 501.25 | 516.44 | 503.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:30:00 | 502.70 | 516.44 | 503.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 499.00 | 516.27 | 503.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 09:15:00 | 510.00 | 516.27 | 503.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 12:00:00 | 503.10 | 515.98 | 503.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 497.65 | 515.56 | 504.04 | SL hit (close<static) qty=1.00 sl=499.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-12-04 09:15:00 | 565.50 | 2023-12-12 13:15:00 | 558.40 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2023-12-04 10:30:00 | 563.95 | 2023-12-12 13:15:00 | 558.40 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2023-12-04 15:15:00 | 563.40 | 2023-12-12 13:15:00 | 558.40 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2023-12-18 09:30:00 | 566.60 | 2023-12-20 13:15:00 | 542.30 | STOP_HIT | 1.00 | -4.29% |
| BUY | retest2 | 2023-12-19 09:45:00 | 565.40 | 2023-12-20 13:15:00 | 542.30 | STOP_HIT | 1.00 | -4.09% |
| BUY | retest2 | 2023-12-19 13:15:00 | 565.65 | 2023-12-20 13:15:00 | 542.30 | STOP_HIT | 1.00 | -4.13% |
| BUY | retest2 | 2023-12-20 11:00:00 | 565.80 | 2023-12-20 13:15:00 | 542.30 | STOP_HIT | 1.00 | -4.15% |
| BUY | retest2 | 2023-12-20 12:30:00 | 566.20 | 2023-12-20 13:15:00 | 542.30 | STOP_HIT | 1.00 | -4.22% |
| BUY | retest2 | 2024-01-24 11:30:00 | 594.75 | 2024-01-24 14:15:00 | 585.45 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-01-24 12:30:00 | 595.10 | 2024-01-24 14:15:00 | 585.45 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2024-01-29 12:30:00 | 595.75 | 2024-02-13 12:15:00 | 585.65 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-02-13 09:45:00 | 594.00 | 2024-02-13 12:15:00 | 585.65 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-02-14 14:45:00 | 598.45 | 2024-02-26 09:15:00 | 658.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-05-22 12:30:00 | 635.00 | 2024-05-30 14:15:00 | 637.25 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2024-05-22 13:30:00 | 636.75 | 2024-05-30 14:15:00 | 637.25 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2024-05-23 10:15:00 | 636.45 | 2024-05-30 14:15:00 | 637.25 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2024-05-23 13:30:00 | 636.40 | 2024-05-31 12:15:00 | 636.50 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2024-05-28 12:30:00 | 625.85 | 2024-05-31 13:15:00 | 645.15 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2024-05-28 13:45:00 | 625.25 | 2024-05-31 13:15:00 | 645.15 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2024-05-30 09:15:00 | 622.45 | 2024-05-31 13:15:00 | 645.15 | STOP_HIT | 1.00 | -3.65% |
| SELL | retest2 | 2024-05-31 09:30:00 | 624.65 | 2024-05-31 13:15:00 | 645.15 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2024-06-04 11:45:00 | 616.40 | 2024-06-05 11:15:00 | 654.30 | STOP_HIT | 1.00 | -6.15% |
| BUY | retest2 | 2024-06-27 09:15:00 | 645.80 | 2024-06-28 12:15:00 | 636.95 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-06-28 15:00:00 | 641.25 | 2024-07-11 09:15:00 | 705.38 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-24 11:15:00 | 700.00 | 2024-10-24 11:15:00 | 724.00 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2024-10-25 12:30:00 | 701.45 | 2024-11-05 14:15:00 | 689.25 | STOP_HIT | 1.00 | 1.74% |
| SELL | retest2 | 2024-10-25 13:00:00 | 701.75 | 2024-11-06 13:15:00 | 710.40 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-10-25 14:45:00 | 699.95 | 2024-11-06 13:15:00 | 710.40 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-11-05 13:00:00 | 677.35 | 2024-11-06 13:15:00 | 710.40 | STOP_HIT | 1.00 | -4.88% |
| SELL | retest2 | 2024-11-12 15:00:00 | 680.65 | 2024-11-19 09:15:00 | 693.90 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2024-11-13 09:15:00 | 672.80 | 2024-11-19 09:15:00 | 693.90 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2024-11-19 13:15:00 | 681.05 | 2024-11-21 11:15:00 | 692.00 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-11-25 15:00:00 | 672.45 | 2024-12-04 12:15:00 | 695.50 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2024-11-27 09:15:00 | 673.40 | 2024-12-04 12:15:00 | 695.50 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2024-11-27 10:00:00 | 672.70 | 2024-12-04 12:15:00 | 695.50 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2024-11-28 09:15:00 | 672.55 | 2024-12-04 12:15:00 | 695.50 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2025-06-11 09:15:00 | 527.00 | 2025-06-13 09:15:00 | 502.55 | STOP_HIT | 1.00 | -4.64% |
| BUY | retest2 | 2025-06-12 11:00:00 | 524.95 | 2025-06-13 09:15:00 | 502.55 | STOP_HIT | 1.00 | -4.27% |
| SELL | retest2 | 2025-07-18 10:15:00 | 479.85 | 2025-07-30 12:15:00 | 455.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 10:45:00 | 480.30 | 2025-07-30 12:15:00 | 456.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 12:00:00 | 479.80 | 2025-07-30 12:15:00 | 455.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 10:15:00 | 479.85 | 2025-08-07 13:15:00 | 431.87 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-22 10:45:00 | 480.30 | 2025-08-07 13:15:00 | 432.27 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-25 12:00:00 | 479.80 | 2025-08-07 13:15:00 | 431.82 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-23 12:00:00 | 480.75 | 2025-10-29 09:15:00 | 481.85 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-12-11 10:15:00 | 485.00 | 2025-12-29 09:15:00 | 476.00 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-12-12 09:15:00 | 486.85 | 2025-12-29 09:15:00 | 476.00 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-12-12 10:15:00 | 486.00 | 2025-12-29 09:15:00 | 476.00 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-12-15 11:00:00 | 486.50 | 2025-12-29 09:15:00 | 476.00 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2026-01-02 10:30:00 | 485.15 | 2026-01-06 09:15:00 | 474.10 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest1 | 2026-01-23 09:15:00 | 464.30 | 2026-01-27 10:15:00 | 477.45 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-03-05 09:15:00 | 510.00 | 2026-03-09 09:15:00 | 497.65 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2026-03-05 12:00:00 | 503.10 | 2026-03-09 09:15:00 | 497.65 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2026-03-09 11:30:00 | 502.30 | 2026-03-09 13:15:00 | 497.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2026-03-10 09:15:00 | 508.30 | 2026-03-13 09:15:00 | 501.65 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2026-03-12 12:15:00 | 512.30 | 2026-03-13 09:15:00 | 501.65 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2026-03-12 13:00:00 | 511.40 | 2026-03-13 09:15:00 | 501.65 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2026-03-12 15:00:00 | 510.80 | 2026-03-13 12:15:00 | 490.90 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest2 | 2026-03-18 09:15:00 | 515.80 | 2026-03-19 12:15:00 | 502.25 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2026-03-20 11:15:00 | 509.90 | 2026-03-23 09:15:00 | 486.40 | STOP_HIT | 1.00 | -4.61% |
| BUY | retest2 | 2026-03-20 12:15:00 | 507.60 | 2026-03-23 09:15:00 | 486.40 | STOP_HIT | 1.00 | -4.18% |
| BUY | retest2 | 2026-03-20 15:15:00 | 509.80 | 2026-03-23 09:15:00 | 486.40 | STOP_HIT | 1.00 | -4.59% |
| BUY | retest2 | 2026-03-25 09:15:00 | 507.85 | 2026-03-27 10:15:00 | 492.95 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2026-04-06 11:15:00 | 507.50 | 2026-04-10 13:15:00 | 558.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 10:00:00 | 508.55 | 2026-04-10 13:15:00 | 559.41 | TARGET_HIT | 1.00 | 10.00% |
