# Aegis Logistics Ltd. (AEGISLOG)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 725.00
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
| ALERT2 | 10 |
| ALERT2_SKIP | 3 |
| ALERT3 | 71 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 61 |
| PARTIAL | 10 |
| TARGET_HIT | 18 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 71 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 40
- **Target hits / Stop hits / Partials:** 18 / 43 / 10
- **Avg / median % per leg:** 1.57% / -0.99%
- **Sum % (uncompounded):** 111.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 11 | 40.7% | 11 | 16 | 0 | 2.95% | 79.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 27 | 11 | 40.7% | 11 | 16 | 0 | 2.95% | 79.8% |
| SELL (all) | 44 | 20 | 45.5% | 7 | 27 | 10 | 0.71% | 31.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 44 | 20 | 45.5% | 7 | 27 | 10 | 0.71% | 31.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 71 | 31 | 43.7% | 18 | 43 | 10 | 1.57% | 111.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-06-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-01 12:15:00 | 349.75 | 374.84 | 374.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-01 14:15:00 | 345.45 | 374.30 | 374.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-05 10:15:00 | 342.55 | 342.24 | 353.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-05 10:45:00 | 341.30 | 342.24 | 353.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 12:15:00 | 352.75 | 342.40 | 353.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-05 12:45:00 | 353.45 | 342.40 | 353.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 13:15:00 | 352.85 | 342.51 | 353.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-05 13:30:00 | 352.75 | 342.51 | 353.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 14:15:00 | 354.10 | 342.62 | 353.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-05 14:30:00 | 356.30 | 342.62 | 353.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 15:15:00 | 353.00 | 342.72 | 353.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-06 09:15:00 | 347.55 | 342.72 | 353.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-10 14:15:00 | 357.50 | 343.26 | 352.46 | SL hit (close>static) qty=1.00 sl=355.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 10:15:00 | 383.90 | 357.02 | 356.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 09:15:00 | 386.00 | 358.56 | 357.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-14 11:15:00 | 363.75 | 366.22 | 362.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-14 11:15:00 | 363.75 | 366.22 | 362.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 11:15:00 | 363.75 | 366.22 | 362.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 12:00:00 | 363.75 | 366.22 | 362.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 09:15:00 | 363.50 | 366.14 | 362.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-16 09:30:00 | 363.85 | 366.14 | 362.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 10:15:00 | 362.25 | 366.10 | 362.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-16 10:45:00 | 362.15 | 366.10 | 362.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 11:15:00 | 362.50 | 366.06 | 362.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-16 12:00:00 | 362.50 | 366.06 | 362.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 12:15:00 | 361.65 | 366.02 | 362.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-16 12:30:00 | 361.70 | 366.02 | 362.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 10:15:00 | 357.80 | 365.68 | 362.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 11:00:00 | 357.80 | 365.68 | 362.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 10:15:00 | 352.10 | 364.06 | 361.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-21 11:00:00 | 352.10 | 364.06 | 361.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 15:15:00 | 369.25 | 365.17 | 362.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-01 09:15:00 | 372.85 | 366.28 | 363.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-04 12:15:00 | 360.00 | 366.32 | 363.84 | SL hit (close<static) qty=1.00 sl=361.15 alert=retest2 |

### Cycle 3 — SELL (started 2023-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 15:15:00 | 341.85 | 361.82 | 361.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 09:15:00 | 336.45 | 361.57 | 361.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-09 13:15:00 | 314.05 | 308.67 | 323.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-09 14:00:00 | 314.05 | 308.67 | 323.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 323.20 | 311.55 | 321.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-22 09:30:00 | 325.00 | 311.55 | 321.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 10:15:00 | 325.50 | 311.69 | 321.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-22 10:30:00 | 326.30 | 311.69 | 321.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 13:15:00 | 324.25 | 313.07 | 321.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 14:00:00 | 324.25 | 313.07 | 321.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 14:15:00 | 326.20 | 313.20 | 321.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 15:00:00 | 326.20 | 313.20 | 321.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2023-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-05 09:15:00 | 361.10 | 328.70 | 328.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 11:15:00 | 369.60 | 333.67 | 331.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 09:15:00 | 355.90 | 357.63 | 347.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-28 10:00:00 | 355.90 | 357.63 | 347.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 09:15:00 | 349.45 | 356.97 | 347.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 10:00:00 | 349.45 | 356.97 | 347.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 11:15:00 | 348.25 | 356.80 | 347.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 11:30:00 | 348.00 | 356.80 | 347.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 12:15:00 | 347.40 | 356.71 | 347.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 13:45:00 | 348.35 | 356.63 | 347.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 11:15:00 | 348.35 | 356.34 | 347.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 14:15:00 | 348.50 | 356.06 | 347.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 15:15:00 | 350.50 | 355.98 | 347.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 15:15:00 | 350.50 | 355.93 | 347.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 09:15:00 | 352.45 | 355.93 | 347.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-04 13:15:00 | 344.90 | 355.27 | 347.99 | SL hit (close<static) qty=1.00 sl=346.20 alert=retest2 |

### Cycle 5 — SELL (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 09:15:00 | 743.00 | 781.06 | 781.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 10:15:00 | 732.40 | 777.64 | 779.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 11:15:00 | 747.00 | 739.03 | 755.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-21 11:30:00 | 741.65 | 739.03 | 755.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 745.95 | 739.33 | 755.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 10:30:00 | 737.30 | 739.33 | 755.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 12:30:00 | 721.50 | 739.29 | 754.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 13:45:00 | 734.35 | 739.28 | 754.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 14:15:00 | 736.10 | 739.28 | 754.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 750.05 | 739.72 | 754.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 13:45:00 | 739.95 | 739.95 | 754.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:30:00 | 736.00 | 740.09 | 753.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 12:15:00 | 740.30 | 740.15 | 753.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 14:00:00 | 739.50 | 740.18 | 753.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 757.50 | 740.87 | 753.39 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-29 09:15:00 | 757.50 | 740.87 | 753.39 | SL hit (close>static) qty=1.00 sl=755.80 alert=retest2 |

### Cycle 6 — BUY (started 2024-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 12:15:00 | 800.90 | 763.03 | 763.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-14 09:15:00 | 809.40 | 766.13 | 764.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 15:15:00 | 806.95 | 811.35 | 793.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-09 09:15:00 | 777.60 | 811.35 | 793.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 777.55 | 811.01 | 793.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:45:00 | 778.00 | 811.01 | 793.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 777.00 | 810.67 | 793.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:30:00 | 776.50 | 810.67 | 793.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 791.45 | 798.42 | 789.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-17 11:15:00 | 794.05 | 798.42 | 789.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-17 12:45:00 | 793.55 | 798.26 | 789.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 09:15:00 | 778.45 | 797.74 | 789.57 | SL hit (close<static) qty=1.00 sl=784.80 alert=retest2 |

### Cycle 7 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 636.30 | 800.94 | 801.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 10:15:00 | 624.70 | 799.19 | 800.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 14:15:00 | 759.25 | 750.65 | 772.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 15:00:00 | 759.25 | 750.65 | 772.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 755.50 | 750.70 | 772.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 11:15:00 | 750.25 | 763.74 | 776.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 12:45:00 | 751.70 | 763.59 | 776.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 14:00:00 | 751.10 | 763.46 | 776.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 09:15:00 | 738.85 | 763.48 | 776.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 769.60 | 763.27 | 775.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:45:00 | 776.85 | 763.27 | 775.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 766.25 | 763.30 | 775.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:45:00 | 769.15 | 763.30 | 775.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 796.00 | 763.58 | 775.68 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-12 14:15:00 | 796.00 | 763.58 | 775.68 | SL hit (close>static) qty=1.00 sl=778.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 902.00 | 765.18 | 765.00 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 746.90 | 796.98 | 797.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 742.30 | 778.50 | 786.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 11:15:00 | 732.75 | 729.40 | 748.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 11:45:00 | 732.15 | 729.40 | 748.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 741.65 | 713.35 | 730.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:00:00 | 741.65 | 713.35 | 730.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 740.25 | 713.62 | 730.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:30:00 | 744.50 | 713.62 | 730.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2025-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 15:15:00 | 764.90 | 742.97 | 742.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 09:15:00 | 766.45 | 743.21 | 743.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 745.45 | 745.48 | 744.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 13:15:00 | 745.45 | 745.48 | 744.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 745.45 | 745.48 | 744.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:00:00 | 745.45 | 745.48 | 744.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 742.05 | 746.00 | 744.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:00:00 | 742.05 | 746.00 | 744.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 740.95 | 745.95 | 744.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:45:00 | 741.70 | 745.95 | 744.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 741.90 | 745.83 | 744.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 746.15 | 745.83 | 744.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 770.00 | 789.62 | 774.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:45:00 | 768.70 | 789.62 | 774.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 771.50 | 789.44 | 774.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:30:00 | 770.50 | 789.44 | 774.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 771.25 | 786.27 | 774.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:00:00 | 771.25 | 786.27 | 774.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 774.90 | 786.16 | 774.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:30:00 | 771.10 | 786.16 | 774.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 768.90 | 785.99 | 774.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:00:00 | 768.90 | 785.99 | 774.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 769.90 | 785.83 | 774.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 774.50 | 785.47 | 774.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 10:15:00 | 765.50 | 785.12 | 773.98 | SL hit (close<static) qty=1.00 sl=768.90 alert=retest2 |

### Cycle 11 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 739.20 | 772.47 | 772.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 725.55 | 771.67 | 772.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 09:15:00 | 754.05 | 739.83 | 751.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 754.05 | 739.83 | 751.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 754.05 | 739.83 | 751.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 747.10 | 739.83 | 751.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 751.40 | 739.94 | 751.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 12:15:00 | 746.70 | 740.06 | 751.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 11:30:00 | 747.95 | 739.67 | 751.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 14:45:00 | 748.00 | 739.85 | 751.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 15:00:00 | 747.50 | 740.12 | 750.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 751.00 | 740.23 | 750.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 748.70 | 740.23 | 750.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 753.00 | 740.35 | 750.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 753.00 | 740.35 | 750.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 747.15 | 740.42 | 750.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:30:00 | 756.80 | 740.42 | 750.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 750.00 | 740.59 | 750.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:45:00 | 750.15 | 740.59 | 750.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 750.05 | 740.69 | 750.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:45:00 | 750.00 | 740.69 | 750.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 748.00 | 740.76 | 750.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 14:30:00 | 751.20 | 740.76 | 750.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 14:15:00 | 709.37 | 737.41 | 747.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 14:15:00 | 710.55 | 737.41 | 747.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 14:15:00 | 710.60 | 737.41 | 747.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 14:15:00 | 710.12 | 737.41 | 747.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 12:15:00 | 673.16 | 731.81 | 744.07 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 12 — BUY (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 11:15:00 | 699.90 | 672.79 | 672.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 15:15:00 | 703.90 | 673.89 | 673.26 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-07-06 09:15:00 | 347.55 | 2023-07-10 14:15:00 | 357.50 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2023-07-13 12:00:00 | 348.05 | 2023-07-17 09:15:00 | 356.50 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2023-07-14 11:15:00 | 350.15 | 2023-07-17 09:15:00 | 356.50 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2023-07-14 11:45:00 | 350.00 | 2023-07-17 09:15:00 | 356.50 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2023-07-17 12:30:00 | 352.60 | 2023-07-18 10:15:00 | 359.05 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2023-07-18 09:30:00 | 352.55 | 2023-07-18 10:15:00 | 359.05 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2023-09-01 09:15:00 | 372.85 | 2023-09-04 12:15:00 | 360.00 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2024-01-01 13:45:00 | 348.35 | 2024-01-04 13:15:00 | 344.90 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-01-02 11:15:00 | 348.35 | 2024-01-04 13:15:00 | 344.90 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-01-02 14:15:00 | 348.50 | 2024-01-04 13:15:00 | 344.90 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-01-02 15:15:00 | 350.50 | 2024-01-04 13:15:00 | 344.90 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-01-03 09:15:00 | 352.45 | 2024-01-04 13:15:00 | 344.90 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-01-10 09:15:00 | 353.40 | 2024-01-23 09:15:00 | 388.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-10 09:45:00 | 352.25 | 2024-01-23 09:15:00 | 387.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-10 13:45:00 | 352.30 | 2024-01-23 09:15:00 | 387.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-13 10:30:00 | 388.60 | 2024-02-15 10:15:00 | 427.46 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-13 11:00:00 | 388.20 | 2024-02-15 10:15:00 | 427.02 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-13 11:45:00 | 393.50 | 2024-02-15 10:15:00 | 429.39 | TARGET_HIT | 1.00 | 9.12% |
| BUY | retest2 | 2024-02-14 11:15:00 | 390.35 | 2024-02-16 09:15:00 | 432.85 | TARGET_HIT | 1.00 | 10.89% |
| BUY | retest2 | 2024-03-28 09:15:00 | 410.05 | 2024-03-28 15:15:00 | 451.06 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-22 10:30:00 | 737.30 | 2024-10-29 09:15:00 | 757.50 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2024-10-22 12:30:00 | 721.50 | 2024-10-29 09:15:00 | 757.50 | STOP_HIT | 1.00 | -4.99% |
| SELL | retest2 | 2024-10-22 13:45:00 | 734.35 | 2024-10-29 09:15:00 | 757.50 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2024-10-22 14:15:00 | 736.10 | 2024-10-29 09:15:00 | 757.50 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2024-10-24 13:45:00 | 739.95 | 2024-10-29 11:15:00 | 760.95 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2024-10-25 09:30:00 | 736.00 | 2024-10-29 11:15:00 | 760.95 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2024-10-25 12:15:00 | 740.30 | 2024-10-29 11:15:00 | 760.95 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2024-10-25 14:00:00 | 739.50 | 2024-10-29 11:15:00 | 760.95 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2024-12-17 11:15:00 | 794.05 | 2024-12-18 09:15:00 | 778.45 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2024-12-17 12:45:00 | 793.55 | 2024-12-18 09:15:00 | 778.45 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-12-20 10:00:00 | 807.00 | 2025-01-06 09:15:00 | 874.50 | TARGET_HIT | 1.00 | 8.36% |
| BUY | retest2 | 2024-12-23 10:30:00 | 795.00 | 2025-01-06 10:15:00 | 887.70 | TARGET_HIT | 1.00 | 11.66% |
| BUY | retest2 | 2024-12-24 10:15:00 | 811.00 | 2025-01-06 10:15:00 | 892.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-13 14:45:00 | 805.45 | 2025-01-20 09:15:00 | 793.65 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-01-14 10:30:00 | 813.70 | 2025-01-21 09:15:00 | 781.95 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest2 | 2025-01-15 09:45:00 | 805.45 | 2025-01-21 09:15:00 | 781.95 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2025-01-15 11:15:00 | 816.20 | 2025-01-21 09:15:00 | 781.95 | STOP_HIT | 1.00 | -4.20% |
| SELL | retest2 | 2025-02-11 11:15:00 | 750.25 | 2025-02-12 14:15:00 | 796.00 | STOP_HIT | 1.00 | -6.10% |
| SELL | retest2 | 2025-02-11 12:45:00 | 751.70 | 2025-02-12 14:15:00 | 796.00 | STOP_HIT | 1.00 | -5.89% |
| SELL | retest2 | 2025-02-11 14:00:00 | 751.10 | 2025-02-12 14:15:00 | 796.00 | STOP_HIT | 1.00 | -5.98% |
| SELL | retest2 | 2025-02-12 09:15:00 | 738.85 | 2025-02-12 14:15:00 | 796.00 | STOP_HIT | 1.00 | -7.73% |
| SELL | retest2 | 2025-02-13 09:15:00 | 776.75 | 2025-02-13 15:15:00 | 737.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 09:15:00 | 776.75 | 2025-02-14 09:15:00 | 699.08 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-21 10:45:00 | 793.55 | 2025-03-24 09:15:00 | 812.00 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-03-21 15:15:00 | 794.50 | 2025-03-24 09:15:00 | 812.00 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-03-24 15:00:00 | 787.50 | 2025-03-27 14:15:00 | 925.00 | STOP_HIT | 1.00 | -17.46% |
| BUY | retest2 | 2025-11-04 09:15:00 | 774.50 | 2025-11-04 10:15:00 | 765.50 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-11-07 11:30:00 | 775.00 | 2025-11-07 12:15:00 | 766.65 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-11-10 12:45:00 | 775.00 | 2025-11-21 10:15:00 | 768.80 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-11-11 13:00:00 | 774.15 | 2025-11-21 10:15:00 | 768.80 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2026-01-05 12:15:00 | 746.70 | 2026-01-16 14:15:00 | 709.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 11:30:00 | 747.95 | 2026-01-16 14:15:00 | 710.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 14:45:00 | 748.00 | 2026-01-16 14:15:00 | 710.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 15:00:00 | 747.50 | 2026-01-16 14:15:00 | 710.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 12:15:00 | 746.70 | 2026-01-20 12:15:00 | 673.16 | TARGET_HIT | 0.50 | 9.85% |
| SELL | retest2 | 2026-01-06 11:30:00 | 747.95 | 2026-01-20 12:15:00 | 673.20 | TARGET_HIT | 0.50 | 9.99% |
| SELL | retest2 | 2026-01-06 14:45:00 | 748.00 | 2026-01-20 13:15:00 | 672.03 | TARGET_HIT | 0.50 | 10.16% |
| SELL | retest2 | 2026-01-07 15:00:00 | 747.50 | 2026-01-20 13:15:00 | 672.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-01 09:15:00 | 709.85 | 2026-02-19 14:15:00 | 682.00 | PARTIAL | 0.50 | 3.92% |
| SELL | retest2 | 2026-02-09 14:30:00 | 717.90 | 2026-02-19 14:15:00 | 681.77 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2026-02-10 09:15:00 | 716.15 | 2026-02-19 15:15:00 | 680.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 09:15:00 | 709.85 | 2026-02-23 15:15:00 | 707.00 | STOP_HIT | 0.50 | 0.40% |
| SELL | retest2 | 2026-02-09 14:30:00 | 717.90 | 2026-02-23 15:15:00 | 707.00 | STOP_HIT | 0.50 | 1.52% |
| SELL | retest2 | 2026-02-10 09:15:00 | 716.15 | 2026-02-23 15:15:00 | 707.00 | STOP_HIT | 0.50 | 1.28% |
| SELL | retest2 | 2026-02-10 13:30:00 | 717.65 | 2026-02-25 10:15:00 | 722.55 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2026-02-24 13:30:00 | 709.55 | 2026-03-02 09:15:00 | 674.36 | PARTIAL | 0.50 | 4.96% |
| SELL | retest2 | 2026-02-26 09:45:00 | 713.70 | 2026-03-02 09:15:00 | 678.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 13:30:00 | 709.55 | 2026-03-04 15:15:00 | 642.33 | TARGET_HIT | 0.50 | 9.47% |
| SELL | retest2 | 2026-02-26 09:45:00 | 713.70 | 2026-03-11 11:15:00 | 638.87 | TARGET_HIT | 0.50 | 10.49% |
| SELL | retest2 | 2026-04-17 09:45:00 | 712.45 | 2026-04-17 11:15:00 | 723.10 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2026-04-17 12:45:00 | 708.40 | 2026-04-20 09:15:00 | 726.00 | STOP_HIT | 1.00 | -2.48% |
