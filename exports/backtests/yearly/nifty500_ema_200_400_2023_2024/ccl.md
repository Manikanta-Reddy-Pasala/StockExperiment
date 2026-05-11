# CCL Products (I) Ltd. (CCL)

## Backtest Summary

- **Window:** 2022-04-07 13:15:00 → 2026-05-08 15:15:00 (7050 bars)
- **Last close:** 1122.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 12 |
| ALERT2 | 12 |
| ALERT2_SKIP | 5 |
| ALERT3 | 84 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 81 |
| PARTIAL | 1 |
| TARGET_HIT | 10 |
| STOP_HIT | 75 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 85 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 75
- **Target hits / Stop hits / Partials:** 9 / 75 / 1
- **Avg / median % per leg:** -1.26% / -2.19%
- **Sum % (uncompounded):** -107.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 53 | 8 | 15.1% | 8 | 45 | 0 | -0.63% | -33.3% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.46% | -16.4% |
| BUY @ 3rd Alert (retest2) | 50 | 8 | 16.0% | 8 | 42 | 0 | -0.34% | -17.0% |
| SELL (all) | 32 | 2 | 6.2% | 1 | 30 | 1 | -2.31% | -73.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 32 | 2 | 6.2% | 1 | 30 | 1 | -2.31% | -73.9% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.46% | -16.4% |
| retest2 (combined) | 82 | 10 | 12.2% | 9 | 72 | 1 | -1.11% | -90.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 09:15:00 | 601.00 | 631.21 | 631.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-28 09:15:00 | 590.35 | 620.95 | 625.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-04 14:15:00 | 615.70 | 615.69 | 621.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-04 14:30:00 | 615.00 | 615.69 | 621.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 09:15:00 | 621.25 | 615.72 | 621.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-05 10:00:00 | 621.25 | 615.72 | 621.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 10:15:00 | 623.90 | 615.80 | 621.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-05 10:30:00 | 623.15 | 615.80 | 621.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 11:15:00 | 623.60 | 615.88 | 621.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-05 11:30:00 | 623.40 | 615.88 | 621.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2023-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 13:15:00 | 683.00 | 626.26 | 626.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-11 14:15:00 | 685.05 | 626.85 | 626.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-20 10:15:00 | 630.00 | 638.90 | 633.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-20 10:15:00 | 630.00 | 638.90 | 633.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 10:15:00 | 630.00 | 638.90 | 633.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 10:45:00 | 630.75 | 638.90 | 633.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 11:15:00 | 625.00 | 638.76 | 633.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 11:45:00 | 624.90 | 638.76 | 633.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 10:15:00 | 624.55 | 638.08 | 632.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-25 11:45:00 | 635.00 | 635.83 | 632.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-28 09:15:00 | 633.85 | 636.49 | 632.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-28 12:30:00 | 631.50 | 636.15 | 632.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-28 13:45:00 | 630.95 | 636.09 | 632.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 635.65 | 636.09 | 632.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 15:15:00 | 630.45 | 636.09 | 632.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 630.45 | 636.03 | 632.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 11:00:00 | 638.10 | 636.05 | 632.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-18 09:30:00 | 636.70 | 643.11 | 638.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-18 10:00:00 | 636.25 | 643.11 | 638.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-18 11:45:00 | 636.00 | 642.98 | 638.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 14:15:00 | 629.35 | 644.30 | 639.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-20 15:00:00 | 629.35 | 644.30 | 639.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 15:15:00 | 630.00 | 644.16 | 639.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-23 09:15:00 | 632.35 | 644.16 | 639.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-23 09:15:00 | 624.15 | 643.96 | 639.16 | SL hit (close<static) qty=1.00 sl=628.95 alert=retest2 |

### Cycle 3 — SELL (started 2023-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 14:15:00 | 592.50 | 635.25 | 635.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 15:15:00 | 591.00 | 634.81 | 635.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 11:15:00 | 631.10 | 630.48 | 632.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-06 11:15:00 | 631.10 | 630.48 | 632.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 11:15:00 | 631.10 | 630.48 | 632.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 11:45:00 | 632.65 | 630.48 | 632.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 12:15:00 | 632.00 | 630.49 | 632.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 12:45:00 | 631.45 | 630.49 | 632.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 13:15:00 | 632.20 | 630.51 | 632.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 13:45:00 | 632.45 | 630.51 | 632.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 14:15:00 | 636.10 | 630.56 | 632.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 15:00:00 | 636.10 | 630.56 | 632.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 15:15:00 | 634.15 | 630.60 | 632.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-07 09:15:00 | 629.30 | 630.60 | 632.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 11:15:00 | 634.20 | 630.03 | 632.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-08 12:00:00 | 634.20 | 630.03 | 632.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 12:15:00 | 635.15 | 630.08 | 632.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-08 13:00:00 | 635.15 | 630.08 | 632.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 633.05 | 630.22 | 632.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-09 10:30:00 | 621.00 | 630.12 | 632.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-10 13:15:00 | 630.05 | 629.77 | 632.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-10 13:45:00 | 630.00 | 629.73 | 632.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-12 18:45:00 | 630.00 | 629.70 | 631.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 09:15:00 | 631.60 | 629.72 | 631.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-13 10:00:00 | 631.60 | 629.72 | 631.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 10:15:00 | 627.00 | 629.70 | 631.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-20 10:30:00 | 623.95 | 630.55 | 632.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-21 10:30:00 | 625.00 | 629.89 | 631.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-21 11:00:00 | 625.00 | 629.89 | 631.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-22 13:15:00 | 637.85 | 629.92 | 631.64 | SL hit (close>static) qty=1.00 sl=632.20 alert=retest2 |

### Cycle 4 — BUY (started 2023-11-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 13:15:00 | 656.60 | 633.02 | 633.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 14:15:00 | 661.85 | 633.31 | 633.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 12:15:00 | 641.60 | 642.89 | 638.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-12 13:00:00 | 641.60 | 642.89 | 638.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 13:15:00 | 637.10 | 642.83 | 638.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 14:00:00 | 637.10 | 642.83 | 638.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 14:15:00 | 644.00 | 642.84 | 638.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-12 15:15:00 | 652.95 | 642.84 | 638.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-13 13:00:00 | 646.55 | 642.96 | 638.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-18 14:15:00 | 629.10 | 642.60 | 639.16 | SL hit (close<static) qty=1.00 sl=633.10 alert=retest2 |

### Cycle 5 — SELL (started 2024-01-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-04 09:15:00 | 625.50 | 636.92 | 636.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 10:15:00 | 624.65 | 635.99 | 636.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 12:15:00 | 636.00 | 635.47 | 636.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 12:15:00 | 636.00 | 635.47 | 636.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 12:15:00 | 636.00 | 635.47 | 636.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-10 09:15:00 | 630.75 | 635.50 | 636.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-10 11:00:00 | 632.10 | 635.43 | 636.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-11 09:15:00 | 640.60 | 635.50 | 636.13 | SL hit (close>static) qty=1.00 sl=640.30 alert=retest2 |

### Cycle 6 — BUY (started 2024-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 13:15:00 | 666.50 | 636.33 | 636.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 15:15:00 | 669.00 | 636.97 | 636.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-02 15:15:00 | 639.05 | 644.64 | 640.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-02 15:15:00 | 639.05 | 644.64 | 640.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 15:15:00 | 639.05 | 644.64 | 640.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 09:15:00 | 637.05 | 644.64 | 640.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 09:15:00 | 634.80 | 644.55 | 640.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 09:30:00 | 632.85 | 644.55 | 640.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 10:15:00 | 632.10 | 644.42 | 640.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 11:00:00 | 632.10 | 644.42 | 640.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 13:15:00 | 637.55 | 644.19 | 640.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 14:00:00 | 637.55 | 644.19 | 640.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 10:15:00 | 643.60 | 646.44 | 642.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 11:00:00 | 643.60 | 646.44 | 642.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 11:15:00 | 647.50 | 646.45 | 642.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 12:45:00 | 650.05 | 646.49 | 642.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 13:30:00 | 649.65 | 646.54 | 642.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-12 13:30:00 | 650.85 | 646.77 | 642.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-13 11:15:00 | 637.00 | 646.56 | 642.64 | SL hit (close<static) qty=1.00 sl=641.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 09:15:00 | 628.20 | 641.34 | 641.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 10:15:00 | 625.70 | 641.18 | 641.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-29 14:15:00 | 590.15 | 587.60 | 601.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-29 15:00:00 | 590.15 | 587.60 | 601.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 587.65 | 575.34 | 587.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:30:00 | 589.15 | 575.34 | 587.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 591.75 | 575.50 | 587.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 11:00:00 | 591.75 | 575.50 | 587.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 11:15:00 | 594.00 | 575.69 | 587.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 12:00:00 | 594.00 | 575.69 | 587.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 590.00 | 577.51 | 587.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 10:00:00 | 585.05 | 577.58 | 587.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 11:15:00 | 592.00 | 577.85 | 587.85 | SL hit (close>static) qty=1.00 sl=591.50 alert=retest2 |

### Cycle 8 — BUY (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 10:15:00 | 601.70 | 591.56 | 591.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 11:15:00 | 608.15 | 592.80 | 592.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 13:15:00 | 593.55 | 594.85 | 593.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 13:15:00 | 593.55 | 594.85 | 593.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 593.55 | 594.85 | 593.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 13:45:00 | 593.20 | 594.85 | 593.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 596.00 | 594.86 | 593.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 10:30:00 | 597.95 | 594.89 | 593.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 11:15:00 | 600.00 | 594.89 | 593.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 14:15:00 | 589.30 | 595.57 | 593.92 | SL hit (close<static) qty=1.00 sl=592.60 alert=retest2 |

### Cycle 9 — SELL (started 2024-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 09:15:00 | 628.45 | 679.13 | 679.26 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 09:15:00 | 708.75 | 675.93 | 675.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 09:15:00 | 741.15 | 684.53 | 680.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 13:15:00 | 751.30 | 754.89 | 730.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 14:00:00 | 751.30 | 754.89 | 730.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 725.60 | 752.89 | 731.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 10:00:00 | 725.60 | 752.89 | 731.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 725.00 | 752.61 | 731.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 10:45:00 | 724.25 | 752.61 | 731.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 732.15 | 751.28 | 731.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:30:00 | 732.90 | 751.28 | 731.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 730.35 | 751.07 | 731.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:00:00 | 730.35 | 751.07 | 731.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 729.40 | 750.85 | 731.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:30:00 | 730.95 | 750.85 | 731.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 733.60 | 750.50 | 731.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 15:15:00 | 735.00 | 750.33 | 731.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 09:30:00 | 739.45 | 750.11 | 731.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-31 09:15:00 | 726.20 | 749.27 | 731.49 | SL hit (close<static) qty=1.00 sl=729.60 alert=retest2 |

### Cycle 11 — SELL (started 2025-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 09:15:00 | 635.25 | 720.36 | 720.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 10:15:00 | 627.00 | 703.87 | 711.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 14:15:00 | 659.05 | 657.69 | 680.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-03 15:00:00 | 659.05 | 657.69 | 680.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 677.25 | 657.89 | 680.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:45:00 | 679.65 | 657.89 | 680.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 675.15 | 658.26 | 680.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 09:15:00 | 656.00 | 660.45 | 680.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 10:15:00 | 623.20 | 659.28 | 678.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-11 09:15:00 | 590.40 | 657.31 | 676.98 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 12 — BUY (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 15:15:00 | 773.00 | 612.53 | 611.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 13:15:00 | 819.25 | 673.45 | 646.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 15:15:00 | 803.70 | 803.76 | 747.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 09:15:00 | 803.05 | 803.76 | 747.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 849.25 | 863.10 | 830.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 15:15:00 | 870.00 | 861.96 | 832.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 14:30:00 | 870.00 | 862.04 | 833.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 871.20 | 861.22 | 836.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:45:00 | 889.75 | 861.52 | 836.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-03 10:15:00 | 957.00 | 877.50 | 854.23 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 09:15:00 | 837.85 | 865.64 | 865.68 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 11:15:00 | 988.15 | 864.26 | 863.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 09:15:00 | 1012.45 | 870.19 | 866.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 970.40 | 971.04 | 934.78 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 14:45:00 | 988.70 | 971.44 | 935.87 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-03 09:30:00 | 992.80 | 971.84 | 936.43 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 11:15:00 | 990.90 | 973.71 | 938.77 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 936.70 | 973.30 | 939.94 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-05 11:15:00 | 936.70 | 973.30 | 939.94 | SL hit (close<ema400) qty=1.00 sl=939.94 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-09-25 11:45:00 | 635.00 | 2023-10-23 09:15:00 | 624.15 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2023-09-28 09:15:00 | 633.85 | 2023-10-23 09:15:00 | 624.15 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2023-09-28 12:30:00 | 631.50 | 2023-10-23 09:15:00 | 624.15 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2023-09-28 13:45:00 | 630.95 | 2023-10-23 09:15:00 | 624.15 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2023-09-29 11:00:00 | 638.10 | 2023-10-23 09:15:00 | 624.15 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2023-10-18 09:30:00 | 636.70 | 2023-10-23 11:15:00 | 620.45 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2023-10-18 10:00:00 | 636.25 | 2023-10-23 11:15:00 | 620.45 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2023-10-18 11:45:00 | 636.00 | 2023-10-23 11:15:00 | 620.45 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2023-10-23 09:15:00 | 632.35 | 2023-10-23 11:15:00 | 620.45 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2023-10-25 15:00:00 | 633.90 | 2023-10-26 09:15:00 | 615.20 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2023-11-09 10:30:00 | 621.00 | 2023-11-22 13:15:00 | 637.85 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2023-11-10 13:15:00 | 630.05 | 2023-11-22 13:15:00 | 637.85 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2023-11-10 13:45:00 | 630.00 | 2023-11-22 13:15:00 | 637.85 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2023-11-12 18:45:00 | 630.00 | 2023-11-28 14:15:00 | 635.10 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2023-11-20 10:30:00 | 623.95 | 2023-11-29 09:15:00 | 643.75 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2023-11-21 10:30:00 | 625.00 | 2023-11-29 09:15:00 | 643.75 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2023-11-21 11:00:00 | 625.00 | 2023-11-29 09:15:00 | 643.75 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2023-11-28 09:15:00 | 623.75 | 2023-11-29 09:15:00 | 643.75 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2023-12-12 15:15:00 | 652.95 | 2023-12-18 14:15:00 | 629.10 | STOP_HIT | 1.00 | -3.65% |
| BUY | retest2 | 2023-12-13 13:00:00 | 646.55 | 2023-12-18 14:15:00 | 629.10 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2023-12-29 12:00:00 | 648.50 | 2024-01-01 15:15:00 | 633.00 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2023-12-29 14:00:00 | 646.40 | 2024-01-01 15:15:00 | 633.00 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-01-10 09:15:00 | 630.75 | 2024-01-11 09:15:00 | 640.60 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-01-10 11:00:00 | 632.10 | 2024-01-11 09:15:00 | 640.60 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-01-15 14:15:00 | 632.15 | 2024-01-24 12:15:00 | 638.20 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-01-17 09:15:00 | 628.00 | 2024-01-24 12:15:00 | 638.20 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-01-20 10:45:00 | 629.00 | 2024-01-24 12:15:00 | 638.20 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-01-20 12:00:00 | 628.35 | 2024-01-24 12:15:00 | 638.20 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-01-20 13:15:00 | 628.30 | 2024-01-24 13:15:00 | 642.60 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-01-23 14:45:00 | 628.30 | 2024-01-24 13:15:00 | 642.60 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2024-02-09 12:45:00 | 650.05 | 2024-02-13 11:15:00 | 637.00 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2024-02-09 13:30:00 | 649.65 | 2024-02-13 11:15:00 | 637.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2024-02-12 13:30:00 | 650.85 | 2024-02-13 11:15:00 | 637.00 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-02-14 14:15:00 | 649.65 | 2024-02-19 13:15:00 | 641.10 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-02-19 09:15:00 | 651.30 | 2024-02-20 09:15:00 | 639.75 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-02-20 14:00:00 | 651.00 | 2024-02-22 09:15:00 | 639.10 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2024-02-20 15:15:00 | 650.00 | 2024-02-22 09:15:00 | 639.10 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-02-21 11:00:00 | 652.00 | 2024-02-22 09:15:00 | 639.10 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-02-27 15:00:00 | 648.75 | 2024-02-28 11:15:00 | 638.90 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-05-29 10:00:00 | 585.05 | 2024-05-29 11:15:00 | 592.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-05-29 15:00:00 | 585.95 | 2024-06-07 09:15:00 | 613.00 | STOP_HIT | 1.00 | -4.62% |
| SELL | retest2 | 2024-05-30 11:00:00 | 585.70 | 2024-06-07 09:15:00 | 613.00 | STOP_HIT | 1.00 | -4.66% |
| SELL | retest2 | 2024-05-30 14:15:00 | 584.85 | 2024-06-07 09:15:00 | 613.00 | STOP_HIT | 1.00 | -4.81% |
| SELL | retest2 | 2024-06-03 10:30:00 | 574.65 | 2024-06-07 09:15:00 | 613.00 | STOP_HIT | 1.00 | -6.67% |
| SELL | retest2 | 2024-06-03 12:45:00 | 575.30 | 2024-06-07 09:15:00 | 613.00 | STOP_HIT | 1.00 | -6.55% |
| SELL | retest2 | 2024-06-03 13:30:00 | 575.05 | 2024-06-07 09:15:00 | 613.00 | STOP_HIT | 1.00 | -6.60% |
| SELL | retest2 | 2024-06-03 14:00:00 | 574.90 | 2024-06-07 09:15:00 | 613.00 | STOP_HIT | 1.00 | -6.63% |
| SELL | retest2 | 2024-06-07 15:15:00 | 598.00 | 2024-06-18 09:15:00 | 615.45 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2024-06-10 11:15:00 | 602.25 | 2024-06-18 09:15:00 | 615.45 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-06-11 12:30:00 | 602.10 | 2024-06-18 09:15:00 | 615.45 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2024-06-11 13:30:00 | 601.40 | 2024-06-18 09:15:00 | 615.45 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2024-07-18 10:30:00 | 597.95 | 2024-07-19 14:15:00 | 589.30 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-07-18 11:15:00 | 600.00 | 2024-07-19 14:15:00 | 589.30 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2024-07-23 10:15:00 | 596.90 | 2024-07-31 12:15:00 | 656.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-23 12:30:00 | 598.90 | 2024-07-31 13:15:00 | 658.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-27 15:15:00 | 735.00 | 2024-12-31 09:15:00 | 726.20 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-12-30 09:30:00 | 739.45 | 2024-12-31 09:15:00 | 726.20 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-12-31 14:30:00 | 738.95 | 2025-01-01 15:15:00 | 727.00 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-01-01 10:00:00 | 734.55 | 2025-01-01 15:15:00 | 727.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-02-06 09:15:00 | 656.00 | 2025-02-10 10:15:00 | 623.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 09:15:00 | 656.00 | 2025-02-11 09:15:00 | 590.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-06 11:00:00 | 673.85 | 2025-05-06 14:15:00 | 694.25 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2025-05-06 13:00:00 | 673.70 | 2025-05-06 14:15:00 | 694.25 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2025-08-08 15:15:00 | 870.00 | 2025-09-03 10:15:00 | 957.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-11 14:30:00 | 870.00 | 2025-09-03 10:15:00 | 957.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-18 09:15:00 | 871.20 | 2025-09-03 10:15:00 | 958.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-18 09:45:00 | 889.75 | 2025-10-13 10:15:00 | 820.60 | STOP_HIT | 1.00 | -7.77% |
| BUY | retest1 | 2025-12-02 14:45:00 | 988.70 | 2025-12-05 11:15:00 | 936.70 | STOP_HIT | 1.00 | -5.26% |
| BUY | retest1 | 2025-12-03 09:30:00 | 992.80 | 2025-12-05 11:15:00 | 936.70 | STOP_HIT | 1.00 | -5.65% |
| BUY | retest1 | 2025-12-04 11:15:00 | 990.90 | 2025-12-05 11:15:00 | 936.70 | STOP_HIT | 1.00 | -5.47% |
| BUY | retest2 | 2025-12-05 14:45:00 | 951.10 | 2025-12-08 12:15:00 | 932.80 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-12-08 09:30:00 | 965.20 | 2025-12-08 12:15:00 | 932.80 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2025-12-09 10:15:00 | 950.50 | 2025-12-29 10:15:00 | 929.20 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-12-23 14:15:00 | 950.30 | 2025-12-29 10:15:00 | 929.20 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2026-01-13 15:15:00 | 975.00 | 2026-01-21 09:15:00 | 943.90 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2026-01-14 10:30:00 | 972.75 | 2026-01-21 09:15:00 | 943.90 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2026-01-16 09:30:00 | 973.20 | 2026-01-21 09:15:00 | 943.90 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2026-01-16 14:45:00 | 972.40 | 2026-01-21 09:15:00 | 943.90 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2026-01-21 15:15:00 | 953.00 | 2026-01-22 15:15:00 | 930.00 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2026-01-27 14:45:00 | 951.10 | 2026-01-29 09:15:00 | 922.45 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2026-01-28 10:30:00 | 952.10 | 2026-01-29 09:15:00 | 922.45 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest2 | 2026-01-28 11:00:00 | 950.80 | 2026-01-29 09:15:00 | 922.45 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2026-02-02 14:00:00 | 972.15 | 2026-02-23 13:15:00 | 1069.37 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-02 14:45:00 | 966.75 | 2026-02-23 13:15:00 | 1063.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-03 11:30:00 | 971.80 | 2026-02-23 13:15:00 | 1068.98 | TARGET_HIT | 1.00 | 10.00% |
