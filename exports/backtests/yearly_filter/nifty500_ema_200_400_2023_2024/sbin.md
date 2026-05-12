# State Bank of India (SBIN)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1018.50
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
| ALERT2_SKIP | 2 |
| ALERT3 | 64 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 66 |
| PARTIAL | 0 |
| TARGET_HIT | 17 |
| STOP_HIT | 49 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 66 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 40
- **Target hits / Stop hits / Partials:** 17 / 49 / 0
- **Avg / median % per leg:** 1.60% / -0.72%
- **Sum % (uncompounded):** 105.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 26 | 65.0% | 17 | 23 | 0 | 3.57% | 142.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 40 | 26 | 65.0% | 17 | 23 | 0 | 3.57% | 142.9% |
| SELL (all) | 26 | 0 | 0.0% | 0 | 26 | 0 | -1.42% | -37.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 26 | 0 | 0.0% | 0 | 26 | 0 | -1.42% | -37.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 66 | 26 | 39.4% | 17 | 49 | 0 | 1.60% | 105.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 10:15:00 | 571.35 | 582.25 | 582.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 10:15:00 | 565.50 | 579.06 | 580.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-07 14:15:00 | 578.25 | 576.54 | 578.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-07 14:15:00 | 578.25 | 576.54 | 578.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 14:15:00 | 578.25 | 576.54 | 578.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-07 15:00:00 | 578.25 | 576.54 | 578.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 15:15:00 | 580.60 | 576.58 | 578.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-08 09:15:00 | 580.80 | 576.58 | 578.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 09:15:00 | 583.20 | 576.65 | 578.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-08 09:45:00 | 582.85 | 576.65 | 578.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2023-09-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 15:15:00 | 596.75 | 580.80 | 580.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 10:15:00 | 598.85 | 581.14 | 580.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 11:15:00 | 587.20 | 589.04 | 585.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 11:15:00 | 587.20 | 589.04 | 585.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 11:15:00 | 587.20 | 589.04 | 585.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 12:00:00 | 587.20 | 589.04 | 585.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 13:15:00 | 587.80 | 589.01 | 585.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 13:30:00 | 586.20 | 589.01 | 585.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 15:15:00 | 585.20 | 588.95 | 585.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 09:15:00 | 589.15 | 588.95 | 585.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-09 12:15:00 | 582.95 | 589.35 | 586.29 | SL hit (close<static) qty=1.00 sl=584.45 alert=retest2 |

### Cycle 3 — SELL (started 2023-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 12:15:00 | 562.70 | 584.28 | 584.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 09:15:00 | 560.80 | 583.43 | 583.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-03 09:15:00 | 576.90 | 574.44 | 578.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-03 10:00:00 | 576.90 | 574.44 | 578.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 10:15:00 | 577.75 | 574.47 | 578.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-03 10:45:00 | 577.80 | 574.47 | 578.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 12:15:00 | 578.15 | 574.54 | 578.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-03 12:45:00 | 577.95 | 574.54 | 578.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 13:15:00 | 578.15 | 574.57 | 578.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-03 13:30:00 | 577.80 | 574.57 | 578.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 15:15:00 | 579.00 | 574.65 | 578.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 09:15:00 | 576.75 | 574.65 | 578.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 09:15:00 | 576.20 | 574.66 | 578.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-06 11:00:00 | 573.75 | 574.66 | 578.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-06 11:45:00 | 575.05 | 574.66 | 578.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-06 12:30:00 | 574.90 | 574.65 | 578.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-06 14:15:00 | 574.80 | 574.66 | 578.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 10:15:00 | 576.75 | 574.70 | 578.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-07 11:30:00 | 575.60 | 574.72 | 578.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-07 12:15:00 | 575.60 | 574.72 | 578.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-07 13:15:00 | 579.20 | 574.77 | 578.39 | SL hit (close>static) qty=1.00 sl=578.45 alert=retest2 |

### Cycle 4 — BUY (started 2023-12-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 12:15:00 | 609.90 | 578.29 | 578.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 14:15:00 | 611.95 | 578.94 | 578.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-09 14:15:00 | 625.10 | 625.83 | 610.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-09 15:00:00 | 625.10 | 625.83 | 610.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 11:15:00 | 613.00 | 627.58 | 615.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 12:00:00 | 613.00 | 627.58 | 615.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 12:15:00 | 610.15 | 627.41 | 615.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 13:00:00 | 610.15 | 627.41 | 615.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 13:15:00 | 606.80 | 627.21 | 615.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 14:00:00 | 606.80 | 627.21 | 615.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 10:15:00 | 621.30 | 626.63 | 615.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-24 10:45:00 | 619.40 | 626.63 | 615.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 11:15:00 | 613.60 | 626.50 | 615.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-24 12:00:00 | 613.60 | 626.50 | 615.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 12:15:00 | 613.55 | 626.37 | 615.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-24 12:30:00 | 612.40 | 626.37 | 615.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 13:15:00 | 614.70 | 626.25 | 615.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-24 13:45:00 | 612.20 | 626.25 | 615.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 10:15:00 | 611.40 | 625.85 | 615.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 11:00:00 | 611.40 | 625.85 | 615.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 11:15:00 | 609.50 | 625.69 | 615.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 11:45:00 | 610.35 | 625.69 | 615.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 772.00 | 817.37 | 789.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 772.00 | 817.37 | 789.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 791.70 | 817.12 | 789.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 13:15:00 | 792.00 | 817.12 | 789.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 13:45:00 | 793.65 | 816.84 | 789.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 09:15:00 | 816.15 | 813.37 | 789.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-15 10:15:00 | 871.20 | 841.93 | 822.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 09:15:00 | 814.55 | 829.77 | 829.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 14:15:00 | 810.35 | 828.08 | 828.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 12:15:00 | 804.65 | 803.68 | 813.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-23 13:15:00 | 804.20 | 803.68 | 813.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 797.50 | 800.86 | 809.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 13:15:00 | 796.20 | 800.86 | 809.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:00:00 | 796.50 | 800.70 | 809.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 10:15:00 | 796.60 | 798.15 | 807.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:45:00 | 796.45 | 798.18 | 807.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 805.80 | 798.28 | 806.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:00:00 | 805.80 | 798.28 | 806.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 806.20 | 798.36 | 806.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 11:30:00 | 805.45 | 798.42 | 806.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 09:30:00 | 805.35 | 798.66 | 806.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 10:45:00 | 804.30 | 798.72 | 806.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 12:15:00 | 805.00 | 798.80 | 806.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 805.35 | 798.86 | 806.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 12:45:00 | 806.55 | 798.86 | 806.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 807.10 | 799.11 | 806.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:45:00 | 808.00 | 799.11 | 806.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 805.55 | 799.17 | 806.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 11:15:00 | 804.80 | 799.17 | 806.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 12:00:00 | 804.75 | 799.23 | 806.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 13:30:00 | 805.15 | 799.36 | 806.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 14:15:00 | 804.60 | 799.36 | 806.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 805.90 | 799.48 | 806.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:15:00 | 815.10 | 799.48 | 806.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 811.00 | 799.59 | 806.31 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 811.00 | 799.59 | 806.31 | SL hit (close>static) qty=1.00 sl=809.85 alert=retest2 |

### Cycle 6 — BUY (started 2024-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 12:15:00 | 858.90 | 808.76 | 808.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 12:15:00 | 860.45 | 811.89 | 810.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 10:15:00 | 814.95 | 818.93 | 814.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-13 11:00:00 | 814.95 | 818.93 | 814.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 11:15:00 | 816.60 | 818.91 | 814.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-13 13:00:00 | 817.40 | 818.89 | 814.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-13 13:30:00 | 819.90 | 818.85 | 814.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-13 14:15:00 | 808.05 | 818.75 | 814.33 | SL hit (close<static) qty=1.00 sl=813.50 alert=retest2 |

### Cycle 7 — SELL (started 2025-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 14:15:00 | 801.20 | 824.21 | 824.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 13:15:00 | 795.65 | 822.79 | 823.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 10:15:00 | 775.90 | 774.39 | 790.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-01 10:45:00 | 775.80 | 774.39 | 790.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 13:15:00 | 746.85 | 732.08 | 748.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-19 15:00:00 | 744.75 | 732.20 | 748.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 09:45:00 | 745.40 | 732.45 | 748.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-20 14:15:00 | 749.35 | 733.19 | 748.75 | SL hit (close>static) qty=1.00 sl=749.30 alert=retest2 |

### Cycle 8 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 819.10 | 757.64 | 757.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 10:15:00 | 822.65 | 758.28 | 757.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 779.35 | 781.81 | 771.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 09:45:00 | 779.15 | 781.81 | 771.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 776.25 | 781.71 | 772.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:30:00 | 773.00 | 781.71 | 772.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 772.80 | 781.44 | 772.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 781.00 | 781.44 | 772.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 775.55 | 781.38 | 772.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 10:00:00 | 783.60 | 781.09 | 772.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 11:15:00 | 782.60 | 781.09 | 772.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 09:15:00 | 799.50 | 780.40 | 772.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 12:30:00 | 783.85 | 786.92 | 778.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 793.30 | 800.71 | 790.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 11:15:00 | 798.05 | 797.62 | 790.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 15:00:00 | 797.00 | 797.54 | 790.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 801.00 | 797.06 | 790.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 797.05 | 797.25 | 790.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 805.40 | 811.64 | 803.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:45:00 | 804.55 | 811.64 | 803.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 801.40 | 811.47 | 803.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 13:00:00 | 801.40 | 811.47 | 803.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 798.25 | 811.34 | 803.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 14:00:00 | 798.25 | 811.34 | 803.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 800.75 | 808.98 | 803.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:45:00 | 800.30 | 808.98 | 803.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 797.10 | 806.51 | 802.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 14:00:00 | 797.10 | 806.51 | 802.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 800.70 | 806.28 | 802.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:45:00 | 801.85 | 806.28 | 802.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 803.25 | 806.25 | 802.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 12:15:00 | 804.20 | 806.25 | 802.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 10:15:00 | 798.60 | 806.09 | 802.63 | SL hit (close<static) qty=1.00 sl=799.00 alert=retest2 |

### Cycle 9 — SELL (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 10:15:00 | 1017.35 | 1070.19 | 1070.41 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 1107.60 | 1069.70 | 1069.65 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-10-05 09:15:00 | 589.15 | 2023-10-09 12:15:00 | 582.95 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2023-10-10 09:30:00 | 587.25 | 2023-10-12 15:15:00 | 585.65 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2023-10-10 10:00:00 | 587.50 | 2023-10-12 15:15:00 | 585.65 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2023-10-10 11:15:00 | 587.05 | 2023-10-13 09:15:00 | 575.70 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2023-10-10 12:30:00 | 590.95 | 2023-10-13 09:15:00 | 575.70 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2023-10-12 09:15:00 | 592.25 | 2023-10-13 09:15:00 | 575.70 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2023-11-06 11:00:00 | 573.75 | 2023-11-07 13:15:00 | 579.20 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2023-11-06 11:45:00 | 575.05 | 2023-11-07 13:15:00 | 579.20 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2023-11-06 12:30:00 | 574.90 | 2023-11-15 09:15:00 | 583.60 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2023-11-06 14:15:00 | 574.80 | 2023-11-15 09:15:00 | 583.60 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2023-11-07 11:30:00 | 575.60 | 2023-11-15 09:15:00 | 583.60 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2023-11-07 12:15:00 | 575.60 | 2023-11-15 09:15:00 | 583.60 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2023-11-17 09:15:00 | 571.50 | 2023-12-04 09:15:00 | 588.75 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2024-06-04 13:15:00 | 792.00 | 2024-07-15 10:15:00 | 871.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 13:45:00 | 793.65 | 2024-07-15 10:15:00 | 873.02 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-06 09:15:00 | 816.15 | 2024-07-19 13:15:00 | 897.77 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-04 13:15:00 | 796.20 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-10-07 10:00:00 | 796.50 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-10-09 10:15:00 | 796.60 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-10-09 11:45:00 | 796.45 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-10-14 11:30:00 | 805.45 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-10-15 09:30:00 | 805.35 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-10-15 10:45:00 | 804.30 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-10-15 12:15:00 | 805.00 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-10-16 11:15:00 | 804.80 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-10-16 12:00:00 | 804.75 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-10-16 13:30:00 | 805.15 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-10-16 14:15:00 | 804.60 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-10-18 09:15:00 | 804.15 | 2024-10-18 12:15:00 | 818.10 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-10-22 11:30:00 | 801.00 | 2024-10-29 12:15:00 | 817.95 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-11-13 13:00:00 | 817.40 | 2024-11-13 14:15:00 | 808.05 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-11-13 13:30:00 | 819.90 | 2024-11-13 14:15:00 | 808.05 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-11-19 09:30:00 | 818.60 | 2024-11-19 11:15:00 | 809.70 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-11-19 10:15:00 | 818.75 | 2024-11-19 11:15:00 | 809.70 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-12-19 10:30:00 | 830.20 | 2024-12-20 12:15:00 | 819.75 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-12-19 13:30:00 | 830.55 | 2024-12-20 12:15:00 | 819.75 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-12-20 09:30:00 | 833.50 | 2024-12-20 12:15:00 | 819.75 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-12-20 11:00:00 | 830.20 | 2024-12-20 12:15:00 | 819.75 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-03-19 15:00:00 | 744.75 | 2025-03-20 14:15:00 | 749.35 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-03-20 09:45:00 | 745.40 | 2025-03-20 14:15:00 | 749.35 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-04-07 09:30:00 | 743.85 | 2025-04-08 09:15:00 | 764.15 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-04-07 11:15:00 | 745.35 | 2025-04-08 09:15:00 | 764.15 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-04-11 10:30:00 | 748.00 | 2025-04-15 09:15:00 | 769.60 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2025-05-08 10:00:00 | 783.60 | 2025-08-07 10:15:00 | 798.60 | STOP_HIT | 1.00 | 1.91% |
| BUY | retest2 | 2025-05-08 11:15:00 | 782.60 | 2025-08-29 09:15:00 | 799.15 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest2 | 2025-05-12 09:15:00 | 799.50 | 2025-08-29 13:15:00 | 801.75 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2025-05-21 12:30:00 | 783.85 | 2025-08-29 13:15:00 | 801.75 | STOP_HIT | 1.00 | 2.28% |
| BUY | retest2 | 2025-06-20 11:15:00 | 798.05 | 2025-08-29 13:15:00 | 801.75 | STOP_HIT | 1.00 | 0.46% |
| BUY | retest2 | 2025-06-20 15:00:00 | 797.00 | 2025-09-05 09:15:00 | 807.50 | STOP_HIT | 1.00 | 1.32% |
| BUY | retest2 | 2025-06-24 09:15:00 | 801.00 | 2025-09-09 09:15:00 | 806.50 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2025-06-25 09:15:00 | 797.05 | 2025-09-09 09:15:00 | 806.50 | STOP_HIT | 1.00 | 1.19% |
| BUY | retest2 | 2025-08-06 12:15:00 | 804.20 | 2025-09-09 09:15:00 | 806.50 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-08-07 14:30:00 | 804.45 | 2025-09-19 10:15:00 | 861.96 | TARGET_HIT | 1.00 | 7.15% |
| BUY | retest2 | 2025-08-08 14:00:00 | 805.25 | 2025-09-19 10:15:00 | 860.86 | TARGET_HIT | 1.00 | 6.91% |
| BUY | retest2 | 2025-08-08 14:30:00 | 806.00 | 2025-09-19 10:15:00 | 862.24 | TARGET_HIT | 1.00 | 6.98% |
| BUY | retest2 | 2025-08-11 09:15:00 | 818.00 | 2025-09-24 09:15:00 | 879.45 | TARGET_HIT | 1.00 | 7.51% |
| BUY | retest2 | 2025-08-29 11:30:00 | 806.10 | 2025-09-24 09:15:00 | 877.86 | TARGET_HIT | 1.00 | 8.90% |
| BUY | retest2 | 2025-08-29 12:00:00 | 805.50 | 2025-09-24 09:15:00 | 876.70 | TARGET_HIT | 1.00 | 8.84% |
| BUY | retest2 | 2025-08-29 12:30:00 | 805.40 | 2025-09-24 09:15:00 | 876.75 | TARGET_HIT | 1.00 | 8.86% |
| BUY | retest2 | 2025-09-05 09:15:00 | 811.10 | 2025-10-10 10:15:00 | 881.10 | TARGET_HIT | 1.00 | 8.63% |
| BUY | retest2 | 2025-09-08 09:15:00 | 810.80 | 2025-10-13 09:15:00 | 884.90 | TARGET_HIT | 1.00 | 9.14% |
| BUY | retest2 | 2025-09-08 10:30:00 | 811.10 | 2025-10-13 09:15:00 | 885.78 | TARGET_HIT | 1.00 | 9.21% |
| BUY | retest2 | 2025-09-08 12:00:00 | 810.30 | 2025-10-13 10:15:00 | 886.60 | TARGET_HIT | 1.00 | 9.42% |
| BUY | retest2 | 2025-09-09 11:30:00 | 808.60 | 2025-10-16 09:15:00 | 889.46 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-09 14:00:00 | 809.00 | 2025-10-16 13:15:00 | 889.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-10 09:15:00 | 812.70 | 2025-10-17 10:15:00 | 893.97 | TARGET_HIT | 1.00 | 10.00% |
