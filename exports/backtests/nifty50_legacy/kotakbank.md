# KOTAKBANK (KOTAKBANK)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 380.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 6 |
| ALERT3 | 14 |
| PENDING | 47 |
| PENDING_CANCEL | 8 |
| ENTRY1 | 5 |
| ENTRY2 | 34 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 43 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 35
- **Target hits / Stop hits / Partials:** 0 / 39 / 4
- **Avg / median % per leg:** -0.78% / -1.93%
- **Sum % (uncompounded):** -33.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 6 | 22.2% | 0 | 24 | 3 | -0.80% | -21.7% |
| BUY @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 0 | 5 | 3 | 7.58% | 60.7% |
| BUY @ 3rd Alert (retest2) | 19 | 0 | 0.0% | 0 | 19 | 0 | -4.34% | -82.4% |
| SELL (all) | 16 | 2 | 12.5% | 0 | 15 | 1 | -0.73% | -11.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 2 | 12.5% | 0 | 15 | 1 | -0.73% | -11.8% |
| retest1 (combined) | 8 | 6 | 75.0% | 0 | 5 | 3 | 7.58% | 60.7% |
| retest2 (combined) | 35 | 2 | 5.7% | 0 | 34 | 1 | -2.69% | -94.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 13:15:00 | 364.98 | 354.08 | 354.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 14:15:00 | 368.11 | 354.22 | 354.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 09:15:00 | 367.74 | 368.65 | 363.53 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-01-09 10:15:00 | 371.34 | 368.50 | 363.65 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 11:15:00 | 370.51 | 368.52 | 363.69 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 365.78 | 368.47 | 363.78 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-01-11 09:15:00 | 366.72 | 368.21 | 363.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-11 10:15:00 | 367.72 | 368.20 | 363.83 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-12 12:15:00 | 366.89 | 367.95 | 363.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 13:15:00 | 367.00 | 367.94 | 363.91 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-17 09:15:00 | 362.52 | 368.15 | 364.35 | SL hit (close<ema400) qty=1.00 sl=364.35 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-01-17 09:15:00 | 362.52 | 368.15 | 364.35 | SL hit (close<static) qty=1.00 sl=362.89 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-17 09:15:00 | 362.52 | 368.15 | 364.35 | SL hit (close<static) qty=1.00 sl=362.89 alert=retest2 |
| Cross detected — sustain check pending | 2024-02-02 10:15:00 | 366.32 | 364.12 | 363.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 11:15:00 | 367.92 | 364.16 | 363.03 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-02-02 13:15:00 | 366.25 | 364.18 | 363.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-02-02 14:15:00 | 364.22 | 364.18 | 363.05 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2024-02-05 10:15:00 | 361.71 | 364.15 | 363.06 | SL hit (close<static) qty=1.00 sl=362.89 alert=retest2 |

### Cycle 2 — SELL (started 2024-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 11:15:00 | 348.24 | 362.05 | 362.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 12:15:00 | 347.39 | 361.90 | 362.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 09:15:00 | 350.55 | 349.45 | 354.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 13:15:00 | 353.02 | 349.55 | 353.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 13:15:00 | 353.02 | 349.55 | 353.98 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-03-07 11:15:00 | 350.20 | 349.63 | 353.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 12:15:00 | 350.06 | 349.63 | 353.89 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-19 14:15:00 | 349.56 | 348.64 | 352.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 15:15:00 | 349.93 | 348.66 | 352.35 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-03-21 09:15:00 | 357.49 | 348.92 | 352.34 | SL hit (close>static) qty=1.00 sl=354.31 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-21 09:15:00 | 357.49 | 348.92 | 352.34 | SL hit (close>static) qty=1.00 sl=354.31 alert=retest2 |
| Cross detected — sustain check pending | 2024-04-03 09:15:00 | 348.10 | 351.29 | 352.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-03 10:15:00 | 347.96 | 351.26 | 352.90 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-04-05 13:15:00 | 354.90 | 350.98 | 352.62 | SL hit (close>static) qty=1.00 sl=354.31 alert=retest2 |

### Cycle 3 — BUY (started 2024-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 12:15:00 | 360.07 | 353.98 | 353.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 09:15:00 | 360.89 | 354.22 | 354.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 09:15:00 | 332.76 | 356.34 | 355.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-25 09:15:00 | 332.76 | 356.34 | 355.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 332.76 | 356.34 | 355.24 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2024-04-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 11:15:00 | 324.16 | 353.90 | 354.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-26 13:15:00 | 324.01 | 353.33 | 353.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-17 10:15:00 | 338.00 | 337.07 | 343.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 341.06 | 337.74 | 342.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 341.06 | 337.74 | 342.94 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-05-24 10:15:00 | 340.20 | 337.76 | 342.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 11:15:00 | 340.48 | 337.79 | 342.92 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-27 10:15:00 | 343.56 | 338.00 | 342.87 | SL hit (close>static) qty=1.00 sl=343.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-28 14:15:00 | 340.75 | 338.43 | 342.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 15:15:00 | 340.00 | 338.44 | 342.82 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-06-03 11:15:00 | 343.98 | 338.39 | 342.30 | SL hit (close>static) qty=1.00 sl=343.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-04 09:15:00 | 335.00 | 338.55 | 342.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:15:00 | 327.25 | 338.44 | 342.21 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-05 11:15:00 | 340.07 | 337.94 | 341.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-05 12:15:00 | 340.00 | 337.96 | 341.79 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 344.44 | 338.06 | 341.80 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-05 14:15:00 | 344.44 | 338.06 | 341.80 | SL hit (close>static) qty=1.00 sl=343.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-05 14:15:00 | 344.44 | 338.06 | 341.80 | SL hit (close>static) qty=1.00 sl=343.50 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 355.07 | 344.06 | 344.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 12:15:00 | 357.62 | 344.42 | 344.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 348.64 | 349.11 | 346.79 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-07-03 09:15:00 | 357.90 | 349.32 | 346.95 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 10:15:00 | 357.77 | 349.40 | 347.01 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 354.88 | 358.66 | 353.45 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-22 10:15:00 | 353.15 | 358.60 | 353.45 | SL hit (close<ema400) qty=1.00 sl=353.45 alert=retest1 |
| Cross detected — sustain check pending | 2024-07-26 12:15:00 | 365.80 | 357.22 | 353.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-07-26 13:15:00 | 363.94 | 357.29 | 353.46 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-08-22 14:15:00 | 364.46 | 356.96 | 355.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 15:15:00 | 364.30 | 357.03 | 355.07 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-08-26 09:15:00 | 364.10 | 357.58 | 355.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-08-26 10:15:00 | 362.84 | 357.63 | 355.46 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-09-12 14:15:00 | 366.03 | 357.70 | 356.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 15:15:00 | 366.20 | 357.79 | 356.37 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-10-03 14:15:00 | 364.60 | 368.30 | 363.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 15:15:00 | 364.56 | 368.27 | 363.38 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-10-07 09:15:00 | 365.45 | 367.94 | 363.40 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-07 10:15:00 | 364.17 | 367.90 | 363.41 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 11:15:00 | 361.72 | 367.84 | 363.40 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-10-10 11:15:00 | 369.97 | 366.60 | 363.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 12:15:00 | 374.28 | 366.68 | 363.24 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-10-21 09:15:00 | 352.53 | 369.93 | 365.79 | SL hit (close<static) qty=1.00 sl=361.43 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-21 10:15:00 | 349.19 | 369.72 | 365.71 | SL hit (close<static) qty=1.00 sl=351.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-21 10:15:00 | 349.19 | 369.72 | 365.71 | SL hit (close<static) qty=1.00 sl=351.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-21 10:15:00 | 349.19 | 369.72 | 365.71 | SL hit (close<static) qty=1.00 sl=351.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-21 10:15:00 | 349.19 | 369.72 | 365.71 | SL hit (close<static) qty=1.00 sl=351.25 alert=retest2 |

### Cycle 6 — SELL (started 2024-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 13:15:00 | 348.29 | 362.65 | 362.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 14:15:00 | 346.74 | 362.50 | 362.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 13:15:00 | 353.00 | 352.61 | 356.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 09:15:00 | 354.92 | 352.62 | 356.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 354.92 | 352.62 | 356.42 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-11-28 11:15:00 | 353.13 | 353.48 | 356.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 12:15:00 | 351.25 | 353.45 | 356.43 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-06 09:15:00 | 353.10 | 352.96 | 355.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-06 10:15:00 | 355.00 | 352.98 | 355.63 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2024-12-09 09:15:00 | 357.40 | 353.15 | 355.63 | SL hit (close>static) qty=1.00 sl=356.99 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-12 14:15:00 | 353.33 | 354.25 | 355.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 15:15:00 | 353.74 | 354.24 | 355.91 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-13 13:15:00 | 358.13 | 354.25 | 355.87 | SL hit (close>static) qty=1.00 sl=356.99 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-19 09:15:00 | 350.56 | 354.94 | 356.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 10:15:00 | 350.57 | 354.90 | 356.04 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-27 11:15:00 | 352.00 | 353.59 | 355.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 12:15:00 | 351.45 | 353.57 | 355.12 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 11:15:00 | 354.61 | 353.49 | 355.03 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-12-30 12:15:00 | 351.74 | 353.48 | 355.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 13:15:00 | 348.22 | 353.42 | 354.98 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-31 10:15:00 | 356.75 | 353.36 | 354.92 | SL hit (close>static) qty=1.00 sl=356.19 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-31 12:15:00 | 358.23 | 353.44 | 354.95 | SL hit (close>static) qty=1.00 sl=356.99 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-31 12:15:00 | 358.23 | 353.44 | 354.95 | SL hit (close>static) qty=1.00 sl=356.99 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-10 14:15:00 | 350.80 | 355.59 | 355.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 15:15:00 | 351.90 | 355.55 | 355.83 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-14 14:15:00 | 349.72 | 354.65 | 355.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-14 15:15:00 | 352.40 | 354.62 | 355.33 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-01-15 09:15:00 | 356.31 | 354.64 | 355.33 | SL hit (close>static) qty=1.00 sl=356.19 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-17 10:15:00 | 351.67 | 355.22 | 355.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 11:15:00 | 350.82 | 355.17 | 355.56 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-20 09:15:00 | 384.04 | 355.32 | 355.62 | SL hit (close>static) qty=1.00 sl=356.19 alert=retest2 |

### Cycle 7 — BUY (started 2025-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 12:15:00 | 384.92 | 356.16 | 356.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 14:15:00 | 385.35 | 369.09 | 363.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 10:15:00 | 383.08 | 383.34 | 375.08 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-03-05 10:15:00 | 387.44 | 383.17 | 375.81 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 11:15:00 | 389.40 | 383.23 | 375.88 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-07 14:15:00 | 386.98 | 383.58 | 376.67 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 15:15:00 | 387.08 | 383.62 | 376.72 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-11 12:15:00 | 385.96 | 383.82 | 377.19 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 13:15:00 | 388.37 | 383.86 | 377.24 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 11:15:00 | 447.81 | 414.55 | 401.59 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 11:15:00 | 445.14 | 414.55 | 401.59 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 11:15:00 | 446.63 | 414.55 | 401.59 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 413.00 | 427.84 | 412.75 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 413.00 | 427.84 | 412.75 | SL hit (close<ema200) qty=0.50 sl=427.84 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 413.00 | 427.84 | 412.75 | SL hit (close<ema200) qty=0.50 sl=427.84 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 413.00 | 427.84 | 412.75 | SL hit (close<ema200) qty=0.50 sl=427.84 alert=retest1 |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 431.30 | 425.19 | 413.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 431.86 | 425.25 | 413.80 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 411.30 | 422.95 | 415.29 | SL hit (close<static) qty=1.00 sl=411.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-09 10:15:00 | 426.78 | 417.55 | 414.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 11:15:00 | 426.96 | 417.64 | 414.79 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-16 11:15:00 | 427.40 | 420.28 | 416.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 12:15:00 | 427.82 | 420.35 | 416.76 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-19 09:15:00 | 429.60 | 421.40 | 417.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 10:15:00 | 430.36 | 421.49 | 417.68 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 423.00 | 428.73 | 423.31 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-07-04 14:15:00 | 425.98 | 428.61 | 423.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 15:15:00 | 426.00 | 428.58 | 423.34 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-25 14:15:00 | 424.98 | 432.56 | 428.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 15:15:00 | 424.92 | 432.48 | 428.21 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 398.08 | 432.14 | 428.06 | SL hit (close<static) qty=1.00 sl=411.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 398.08 | 432.14 | 428.06 | SL hit (close<static) qty=1.00 sl=411.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 398.08 | 432.14 | 428.06 | SL hit (close<static) qty=1.00 sl=411.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 398.08 | 432.14 | 428.06 | SL hit (close<static) qty=1.00 sl=422.22 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 398.08 | 432.14 | 428.06 | SL hit (close<static) qty=1.00 sl=422.22 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 390.78 | 424.39 | 424.40 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 11:15:00 | 428.30 | 407.84 | 407.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 12:15:00 | 429.18 | 408.06 | 407.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 09:15:00 | 423.78 | 424.06 | 417.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 419.94 | 423.87 | 418.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 419.94 | 423.87 | 418.01 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-11-03 12:15:00 | 422.00 | 423.77 | 418.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 13:15:00 | 423.52 | 423.77 | 418.08 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 417.38 | 423.33 | 418.18 | SL hit (close<static) qty=1.00 sl=417.58 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-21 13:15:00 | 421.92 | 420.64 | 418.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-21 14:15:00 | 417.54 | 420.61 | 418.21 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-11-27 10:15:00 | 423.68 | 419.97 | 418.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 11:15:00 | 425.02 | 420.02 | 418.17 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-16 09:15:00 | 424.20 | 429.58 | 427.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 10:15:00 | 423.90 | 429.53 | 427.03 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-19 09:15:00 | 422.30 | 428.98 | 426.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 10:15:00 | 422.60 | 428.92 | 426.81 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 427.20 | 428.90 | 426.81 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-19 12:15:00 | 428.20 | 428.90 | 426.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-19 13:15:00 | 427.70 | 428.88 | 426.82 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-20 09:15:00 | 429.60 | 428.86 | 426.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 10:15:00 | 428.60 | 428.86 | 426.85 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-21 10:15:00 | 419.00 | 428.57 | 426.77 | SL hit (close<static) qty=1.00 sl=421.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 09:15:00 | 406.00 | 427.50 | 426.38 | SL hit (close<static) qty=1.00 sl=417.58 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 09:15:00 | 406.00 | 427.50 | 426.38 | SL hit (close<static) qty=1.00 sl=417.58 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 09:15:00 | 406.00 | 427.50 | 426.38 | SL hit (close<static) qty=1.00 sl=417.58 alert=retest2 |

### Cycle 10 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 410.40 | 425.23 | 425.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 10:15:00 | 407.00 | 424.13 | 424.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 11:15:00 | 422.70 | 420.38 | 422.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 11:15:00 | 422.70 | 420.38 | 422.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 422.70 | 420.38 | 422.57 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-19 14:15:00 | 416.15 | 422.54 | 423.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-19 15:15:00 | 416.40 | 422.48 | 423.24 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-27 14:15:00 | 414.65 | 423.15 | 423.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 15:15:00 | 415.20 | 423.07 | 423.44 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 352.92 | 388.77 | 401.95 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-08 14:15:00 | 380.15 | 379.44 | 394.37 | SL hit (close>ema200) qty=0.50 sl=379.44 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-09 11:15:00 | 370.51 | 2024-01-17 09:15:00 | 362.52 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2024-01-11 10:15:00 | 367.72 | 2024-01-17 09:15:00 | 362.52 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-01-12 13:15:00 | 367.00 | 2024-01-17 09:15:00 | 362.52 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-02-02 11:15:00 | 367.92 | 2024-02-05 10:15:00 | 361.71 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-03-07 12:15:00 | 350.06 | 2024-03-21 09:15:00 | 357.49 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2024-03-19 15:15:00 | 349.93 | 2024-03-21 09:15:00 | 357.49 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2024-04-03 10:15:00 | 347.96 | 2024-04-05 13:15:00 | 354.90 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2024-05-24 11:15:00 | 340.48 | 2024-05-27 10:15:00 | 343.56 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-05-28 15:15:00 | 340.00 | 2024-06-03 11:15:00 | 343.98 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-06-04 10:15:00 | 327.25 | 2024-06-05 14:15:00 | 344.44 | STOP_HIT | 1.00 | -5.25% |
| SELL | retest2 | 2024-06-05 12:15:00 | 340.00 | 2024-06-05 14:15:00 | 344.44 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest1 | 2024-07-03 10:15:00 | 357.77 | 2024-07-22 10:15:00 | 353.15 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-08-22 15:15:00 | 364.30 | 2024-10-21 09:15:00 | 352.53 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2024-09-12 15:15:00 | 366.20 | 2024-10-21 10:15:00 | 349.19 | STOP_HIT | 1.00 | -4.65% |
| BUY | retest2 | 2024-10-03 15:15:00 | 364.56 | 2024-10-21 10:15:00 | 349.19 | STOP_HIT | 1.00 | -4.22% |
| BUY | retest2 | 2024-10-07 10:15:00 | 364.17 | 2024-10-21 10:15:00 | 349.19 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest2 | 2024-10-10 12:15:00 | 374.28 | 2024-10-21 10:15:00 | 349.19 | STOP_HIT | 1.00 | -6.70% |
| SELL | retest2 | 2024-11-28 12:15:00 | 351.25 | 2024-12-09 09:15:00 | 357.40 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-12-12 15:15:00 | 353.74 | 2024-12-13 13:15:00 | 358.13 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-12-19 10:15:00 | 350.57 | 2024-12-31 10:15:00 | 356.75 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-12-27 12:15:00 | 351.45 | 2024-12-31 12:15:00 | 358.23 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2024-12-30 13:15:00 | 348.22 | 2024-12-31 12:15:00 | 358.23 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2025-01-10 15:15:00 | 351.90 | 2025-01-15 09:15:00 | 356.31 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-01-17 11:15:00 | 350.82 | 2025-01-20 09:15:00 | 384.04 | STOP_HIT | 1.00 | -9.47% |
| BUY | retest1 | 2025-03-05 11:15:00 | 389.40 | 2025-04-21 11:15:00 | 447.81 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2025-03-07 15:15:00 | 387.08 | 2025-04-21 11:15:00 | 445.14 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2025-03-11 13:15:00 | 388.37 | 2025-04-21 11:15:00 | 446.63 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2025-03-05 11:15:00 | 389.40 | 2025-05-05 09:15:00 | 413.00 | STOP_HIT | 0.50 | 6.06% |
| BUY | retest1 | 2025-03-07 15:15:00 | 387.08 | 2025-05-05 09:15:00 | 413.00 | STOP_HIT | 0.50 | 6.70% |
| BUY | retest1 | 2025-03-11 13:15:00 | 388.37 | 2025-05-05 09:15:00 | 413.00 | STOP_HIT | 0.50 | 6.34% |
| BUY | retest2 | 2025-05-12 10:15:00 | 431.86 | 2025-05-22 09:15:00 | 411.30 | STOP_HIT | 1.00 | -4.76% |
| BUY | retest2 | 2025-06-09 11:15:00 | 426.96 | 2025-07-28 09:15:00 | 398.08 | STOP_HIT | 1.00 | -6.76% |
| BUY | retest2 | 2025-06-16 12:15:00 | 427.82 | 2025-07-28 09:15:00 | 398.08 | STOP_HIT | 1.00 | -6.95% |
| BUY | retest2 | 2025-06-19 10:15:00 | 430.36 | 2025-07-28 09:15:00 | 398.08 | STOP_HIT | 1.00 | -7.50% |
| BUY | retest2 | 2025-07-04 15:15:00 | 426.00 | 2025-07-28 09:15:00 | 398.08 | STOP_HIT | 1.00 | -6.55% |
| BUY | retest2 | 2025-07-25 15:15:00 | 424.92 | 2025-07-28 09:15:00 | 398.08 | STOP_HIT | 1.00 | -6.32% |
| BUY | retest2 | 2025-11-03 13:15:00 | 423.52 | 2025-11-06 11:15:00 | 417.38 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-11-27 11:15:00 | 425.02 | 2026-01-21 10:15:00 | 419.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2026-01-16 10:15:00 | 423.90 | 2026-01-27 09:15:00 | 406.00 | STOP_HIT | 1.00 | -4.22% |
| BUY | retest2 | 2026-01-19 10:15:00 | 422.60 | 2026-01-27 09:15:00 | 406.00 | STOP_HIT | 1.00 | -3.93% |
| BUY | retest2 | 2026-01-20 10:15:00 | 428.60 | 2026-01-27 09:15:00 | 406.00 | STOP_HIT | 1.00 | -5.27% |
| SELL | retest2 | 2026-02-27 15:15:00 | 415.20 | 2026-03-30 09:15:00 | 352.92 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2026-02-27 15:15:00 | 415.20 | 2026-04-08 14:15:00 | 380.15 | STOP_HIT | 0.50 | 8.44% |
