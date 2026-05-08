# NTPC (NTPC)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 402.15
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty booked @ 5% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 3 |
| ALERT3 | 11 |
| PENDING | 31 |
| PENDING_CANCEL | 6 |
| ENTRY1 | 3 |
| ENTRY2 | 22 |
| PARTIAL | 2 |
| TARGET_HIT | 3 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 22
- **Target hits / Stop hits / Partials:** 3 / 22 / 2
- **Avg / median % per leg:** -0.22% / -1.52%
- **Sum % (uncompounded):** -5.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 3 | 25.0% | 2 | 9 | 1 | 0.11% | 1.3% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| BUY @ 3rd Alert (retest2) | 10 | 1 | 10.0% | 1 | 9 | 0 | -1.37% | -13.7% |
| SELL (all) | 15 | 2 | 13.3% | 1 | 13 | 1 | -0.48% | -7.2% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 4.33% | 13.0% |
| SELL @ 3rd Alert (retest2) | 12 | 0 | 0.0% | 0 | 12 | 0 | -1.68% | -20.2% |
| retest1 (combined) | 5 | 4 | 80.0% | 2 | 1 | 2 | 5.60% | 28.0% |
| retest2 (combined) | 22 | 1 | 4.5% | 1 | 21 | 0 | -1.54% | -33.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 10:15:00 | 385.85 | 408.34 | 408.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 381.65 | 407.39 | 407.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 11:15:00 | 335.75 | 335.74 | 353.33 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-01-21 09:15:00 | 331.90 | 335.69 | 352.87 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:15:00 | 331.40 | 335.65 | 352.76 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 314.83 | 331.97 | 348.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-02-17 09:15:00 | 298.26 | 319.29 | 334.31 | Target hit (10%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 324.75 | 317.68 | 328.11 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 11:15:00 | 360.25 | 333.46 | 333.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 15:15:00 | 364.00 | 335.92 | 334.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 339.65 | 341.40 | 337.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 339.65 | 341.40 | 337.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 339.65 | 341.40 | 337.82 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-04-07 12:15:00 | 345.25 | 341.45 | 337.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 13:15:00 | 348.25 | 341.52 | 337.95 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 344.80 | 349.26 | 345.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 347.25 | 349.24 | 345.74 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-14 09:15:00 | 343.70 | 348.73 | 345.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-14 10:15:00 | 342.45 | 348.67 | 345.68 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-14 11:15:00 | 342.85 | 348.61 | 345.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-14 12:15:00 | 341.15 | 348.53 | 345.65 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-15 13:15:00 | 343.30 | 347.69 | 345.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-15 14:15:00 | 341.95 | 347.64 | 345.32 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-16 09:15:00 | 345.15 | 347.56 | 345.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:15:00 | 344.95 | 347.53 | 345.30 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-23 10:15:00 | 344.00 | 346.53 | 345.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 11:15:00 | 343.00 | 346.49 | 345.11 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 343.65 | 346.37 | 345.09 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-02 10:15:00 | 331.45 | 344.05 | 344.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-02 10:15:00 | 331.45 | 344.05 | 344.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-02 10:15:00 | 331.45 | 344.05 | 344.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-02 10:15:00 | 331.45 | 344.05 | 344.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 10:15:00 | 331.45 | 344.05 | 344.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 11:15:00 | 331.15 | 343.92 | 343.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 12:15:00 | 339.90 | 339.69 | 341.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 341.25 | 339.70 | 341.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 341.25 | 339.70 | 341.56 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-06-11 13:15:00 | 339.35 | 339.76 | 341.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 14:15:00 | 338.15 | 339.74 | 341.54 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-12 10:15:00 | 337.40 | 339.71 | 341.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 11:15:00 | 338.00 | 339.69 | 341.48 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-27 12:15:00 | 338.50 | 336.11 | 338.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 13:15:00 | 338.95 | 336.14 | 338.75 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-08 11:15:00 | 339.85 | 335.81 | 338.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-08 12:15:00 | 340.55 | 335.86 | 338.01 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-07-08 13:15:00 | 343.30 | 335.93 | 338.04 | SL hit (close>static) qty=1.00 sl=341.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 13:15:00 | 343.30 | 335.93 | 338.04 | SL hit (close>static) qty=1.00 sl=341.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 13:15:00 | 343.30 | 335.93 | 338.04 | SL hit (close>static) qty=1.00 sl=341.80 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-24 12:15:00 | 337.15 | 339.46 | 339.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 13:15:00 | 338.20 | 339.44 | 339.46 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 335.65 | 339.39 | 339.44 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-07-25 10:15:00 | 335.30 | 339.35 | 339.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 11:15:00 | 335.05 | 339.31 | 339.40 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-30 11:15:00 | 340.55 | 338.43 | 338.92 | SL hit (close>static) qty=1.00 sl=339.80 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-31 14:15:00 | 334.05 | 338.43 | 338.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 15:15:00 | 333.70 | 338.38 | 338.87 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-08 13:15:00 | 334.75 | 336.33 | 337.65 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:15:00 | 334.40 | 336.31 | 337.64 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-11 11:15:00 | 334.90 | 336.29 | 337.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 12:15:00 | 334.65 | 336.27 | 337.58 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 339.05 | 336.29 | 337.57 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-12 12:15:00 | 341.65 | 336.40 | 337.61 | SL hit (close>static) qty=1.00 sl=339.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 12:15:00 | 341.65 | 336.40 | 337.61 | SL hit (close>static) qty=1.00 sl=339.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 12:15:00 | 341.65 | 336.40 | 337.61 | SL hit (close>static) qty=1.00 sl=339.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 342.00 | 336.58 | 337.67 | SL hit (close>static) qty=1.00 sl=341.80 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-18 14:15:00 | 335.80 | 336.98 | 337.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-18 15:15:00 | 336.30 | 336.97 | 337.78 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-19 10:15:00 | 335.25 | 336.95 | 337.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 11:15:00 | 335.15 | 336.93 | 337.74 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-20 09:15:00 | 339.60 | 336.89 | 337.70 | SL hit (close>static) qty=1.00 sl=339.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-26 09:15:00 | 335.10 | 337.32 | 337.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 10:15:00 | 335.60 | 337.30 | 337.82 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-26 13:15:00 | 335.75 | 337.26 | 337.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 14:15:00 | 332.70 | 337.21 | 337.77 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-02 13:15:00 | 335.30 | 335.80 | 336.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-02 14:15:00 | 336.25 | 335.80 | 336.93 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-03 12:15:00 | 334.65 | 335.81 | 336.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 13:15:00 | 334.70 | 335.80 | 336.89 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 334.60 | 333.00 | 335.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-23 09:15:00 | 339.95 | 334.24 | 335.33 | SL hit (close>static) qty=1.00 sl=339.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 09:15:00 | 339.95 | 334.24 | 335.33 | SL hit (close>static) qty=1.00 sl=339.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 09:15:00 | 339.95 | 334.24 | 335.33 | SL hit (close>static) qty=1.00 sl=339.40 alert=retest2 |

### Cycle 4 — BUY (started 2025-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 09:15:00 | 340.25 | 336.35 | 336.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 10:15:00 | 341.25 | 336.40 | 336.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 10:15:00 | 337.25 | 337.44 | 336.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 11:15:00 | 337.50 | 337.44 | 336.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 337.50 | 337.44 | 336.94 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-10-07 12:15:00 | 338.50 | 337.45 | 336.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:15:00 | 338.90 | 337.47 | 336.96 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 336.85 | 337.47 | 336.97 | SL hit (close<static) qty=1.00 sl=336.90 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-10 09:15:00 | 339.80 | 337.12 | 336.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 10:15:00 | 340.75 | 337.15 | 336.84 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-13 10:15:00 | 338.60 | 337.31 | 336.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 11:15:00 | 339.00 | 337.33 | 336.94 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-14 12:15:00 | 335.65 | 337.50 | 337.04 | SL hit (close<static) qty=1.00 sl=336.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 12:15:00 | 335.65 | 337.50 | 337.04 | SL hit (close<static) qty=1.00 sl=336.90 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-15 09:15:00 | 339.95 | 337.50 | 337.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:15:00 | 340.45 | 337.53 | 337.07 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 338.40 | 339.19 | 338.11 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-10-29 09:15:00 | 343.60 | 339.22 | 338.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 10:15:00 | 346.75 | 339.29 | 338.20 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-31 10:15:00 | 337.05 | 340.07 | 338.68 | SL hit (close<static) qty=1.00 sl=338.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 336.45 | 339.96 | 338.65 | SL hit (close<static) qty=1.00 sl=336.90 alert=retest2 |

### Cycle 5 — SELL (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 13:15:00 | 325.95 | 337.52 | 337.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 13:15:00 | 325.20 | 336.74 | 337.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 325.40 | 324.54 | 328.16 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-12-24 13:15:00 | 323.40 | 324.53 | 328.08 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 14:15:00 | 322.45 | 324.51 | 328.05 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 328.90 | 324.57 | 327.70 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 328.90 | 324.57 | 327.70 | SL hit (close>ema400) qty=1.00 sl=327.70 alert=retest1 |

### Cycle 6 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 348.15 | 330.28 | 330.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 351.60 | 334.41 | 332.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 336.35 | 336.97 | 334.31 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-27 09:15:00 | 342.40 | 337.02 | 334.36 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 10:15:00 | 342.85 | 337.08 | 334.40 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 11:15:00 | 359.99 | 338.70 | 335.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-02-24 09:15:00 | 377.14 | 358.39 | 349.36 | Target hit (10%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 367.20 | 374.15 | 365.52 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-04-08 09:15:00 | 370.60 | 371.66 | 365.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 10:15:00 | 370.85 | 371.65 | 365.26 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2026-04-27 09:15:00 | 407.94 | 383.76 | 374.25 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-01-21 10:15:00 | 331.40 | 2025-01-28 09:15:00 | 314.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-01-21 10:15:00 | 331.40 | 2025-02-17 09:15:00 | 298.26 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-04-07 13:15:00 | 348.25 | 2025-06-02 10:15:00 | 331.45 | STOP_HIT | 1.00 | -4.82% |
| BUY | retest2 | 2025-05-12 10:15:00 | 347.25 | 2025-06-02 10:15:00 | 331.45 | STOP_HIT | 1.00 | -4.55% |
| BUY | retest2 | 2025-05-16 10:15:00 | 344.95 | 2025-06-02 10:15:00 | 331.45 | STOP_HIT | 1.00 | -3.91% |
| BUY | retest2 | 2025-05-23 11:15:00 | 343.00 | 2025-06-02 10:15:00 | 331.45 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2025-06-11 14:15:00 | 338.15 | 2025-07-08 13:15:00 | 343.30 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-06-12 11:15:00 | 338.00 | 2025-07-08 13:15:00 | 343.30 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-06-27 13:15:00 | 338.95 | 2025-07-08 13:15:00 | 343.30 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-07-24 13:15:00 | 338.20 | 2025-07-30 11:15:00 | 340.55 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-07-25 11:15:00 | 335.05 | 2025-08-12 12:15:00 | 341.65 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-07-31 15:15:00 | 333.70 | 2025-08-12 12:15:00 | 341.65 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2025-08-08 14:15:00 | 334.40 | 2025-08-12 12:15:00 | 341.65 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-08-11 12:15:00 | 334.65 | 2025-08-13 09:15:00 | 342.00 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-08-19 11:15:00 | 335.15 | 2025-08-20 09:15:00 | 339.60 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-08-26 10:15:00 | 335.60 | 2025-09-23 09:15:00 | 339.95 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-08-26 14:15:00 | 332.70 | 2025-09-23 09:15:00 | 339.95 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-09-03 13:15:00 | 334.70 | 2025-09-23 09:15:00 | 339.95 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-10-07 13:15:00 | 338.90 | 2025-10-08 09:15:00 | 336.85 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-10-10 10:15:00 | 340.75 | 2025-10-14 12:15:00 | 335.65 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-10-13 11:15:00 | 339.00 | 2025-10-14 12:15:00 | 335.65 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-10-15 10:15:00 | 340.45 | 2025-10-31 10:15:00 | 337.05 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-10-29 10:15:00 | 346.75 | 2025-10-31 14:15:00 | 336.45 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest1 | 2025-12-24 14:15:00 | 322.45 | 2025-12-31 09:15:00 | 328.90 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest1 | 2026-01-27 10:15:00 | 342.85 | 2026-01-29 11:15:00 | 359.99 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-01-27 10:15:00 | 342.85 | 2026-02-24 09:15:00 | 377.14 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-04-08 10:15:00 | 370.85 | 2026-04-27 09:15:00 | 407.94 | TARGET_HIT | 1.00 | 10.00% |
