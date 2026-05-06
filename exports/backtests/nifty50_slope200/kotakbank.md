# KOTAKBANK (KOTAKBANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 376.60
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 3 |
| ALERT3 | 10 |
| PENDING | 42 |
| PENDING_CANCEL | 6 |
| ENTRY1 | 5 |
| ENTRY2 | 30 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 32 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 35 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 7 / 28
- **Target hits / Stop hits / Partials:** 0 / 32 / 3
- **Avg / median % per leg:** -0.45% / -1.35%
- **Sum % (uncompounded):** -15.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 7 | 22.6% | 0 | 28 | 3 | -0.25% | -7.9% |
| BUY @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 0 | 5 | 3 | 7.50% | 60.0% |
| BUY @ 3rd Alert (retest2) | 23 | 1 | 4.3% | 0 | 23 | 0 | -2.95% | -67.9% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.98% | -7.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.98% | -7.9% |
| retest1 (combined) | 8 | 6 | 75.0% | 0 | 5 | 3 | 7.50% | 60.0% |
| retest2 (combined) | 27 | 1 | 3.7% | 0 | 27 | 0 | -2.81% | -75.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 14:15:00 | 368.11 | 354.22 | 354.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 09:15:00 | 372.00 | 354.53 | 354.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 09:15:00 | 367.74 | 368.65 | 363.58 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-01-09 10:15:00 | 371.34 | 368.50 | 363.70 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 11:15:00 | 370.51 | 368.52 | 363.73 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 365.78 | 368.47 | 363.82 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-01-10 09:15:00 | 363.82 | 368.47 | 363.82 | SL hit qty=1.00 sl=363.82 alert=retest1 |
| Cross detected — sustain check pending | 2024-01-11 09:15:00 | 366.72 | 368.21 | 363.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-11 10:15:00 | 367.72 | 368.20 | 363.87 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-12 12:15:00 | 366.89 | 367.95 | 363.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 13:15:00 | 367.00 | 367.94 | 363.94 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-17 09:15:00 | 362.89 | 368.15 | 364.38 | SL hit qty=1.00 sl=362.89 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-17 09:15:00 | 362.89 | 368.15 | 364.38 | SL hit qty=1.00 sl=362.89 alert=retest2 |
| Cross detected — sustain check pending | 2024-02-02 10:15:00 | 366.32 | 364.12 | 363.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 11:15:00 | 367.92 | 364.16 | 363.05 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-02-02 13:15:00 | 366.25 | 364.18 | 363.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-02-02 14:15:00 | 364.22 | 364.18 | 363.08 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2024-02-05 09:15:00 | 362.89 | 364.17 | 363.09 | SL hit qty=1.00 sl=362.89 alert=retest2 |
| CROSSOVER_SKIP | 2024-02-09 10:15:00 | 347.48 | 362.19 | 362.19 | slope filter: EMA200 not falling 2.00% over 1400 bars |

### Cycle 2 — BUY (started 2024-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 12:15:00 | 360.07 | 353.98 | 353.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 09:15:00 | 360.89 | 354.22 | 354.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 09:15:00 | 332.76 | 356.34 | 355.24 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-25 09:15:00 | 332.76 | 356.34 | 355.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 332.76 | 356.34 | 355.24 | EMA400 retest candle locked |

### Cycle 3 — SELL (started 2024-04-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 11:15:00 | 324.16 | 353.90 | 354.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-26 13:15:00 | 324.01 | 353.33 | 353.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-17 10:15:00 | 338.00 | 337.07 | 343.39 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 341.06 | 337.74 | 342.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 341.06 | 337.74 | 342.95 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-05-24 10:15:00 | 340.20 | 337.76 | 342.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 11:15:00 | 340.48 | 337.79 | 342.92 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-27 09:15:00 | 343.50 | 337.94 | 342.87 | SL hit qty=1.00 sl=343.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-28 14:15:00 | 340.75 | 338.43 | 342.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 15:15:00 | 340.00 | 338.44 | 342.82 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 343.50 | 338.29 | 342.29 | SL hit qty=1.00 sl=343.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-04 09:15:00 | 335.00 | 338.55 | 342.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:15:00 | 327.25 | 338.44 | 342.21 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-05 11:15:00 | 340.07 | 337.94 | 341.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-05 12:15:00 | 340.00 | 337.96 | 341.79 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 344.44 | 338.06 | 341.80 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-06-05 14:15:00 | 343.50 | 338.06 | 341.80 | SL hit qty=1.00 sl=343.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-05 14:15:00 | 343.50 | 338.06 | 341.80 | SL hit qty=1.00 sl=343.50 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 355.07 | 344.06 | 344.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 12:15:00 | 357.62 | 344.42 | 344.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 348.64 | 349.11 | 346.79 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-07-03 09:15:00 | 357.90 | 349.32 | 346.95 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 10:15:00 | 357.77 | 349.40 | 347.01 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 354.88 | 358.66 | 353.45 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-07-22 09:15:00 | 353.45 | 358.66 | 353.45 | SL hit qty=1.00 sl=353.45 alert=retest1 |
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
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 11:15:00 | 361.72 | 367.84 | 363.40 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-10-10 11:15:00 | 369.97 | 366.60 | 363.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 12:15:00 | 374.28 | 366.68 | 363.24 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-10-21 09:15:00 | 361.43 | 369.93 | 365.79 | SL hit qty=1.00 sl=361.43 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-21 10:15:00 | 351.25 | 369.72 | 365.71 | SL hit qty=1.00 sl=351.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-21 10:15:00 | 351.25 | 369.72 | 365.71 | SL hit qty=1.00 sl=351.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-21 10:15:00 | 351.25 | 369.72 | 365.71 | SL hit qty=1.00 sl=351.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-21 10:15:00 | 351.25 | 369.72 | 365.71 | SL hit qty=1.00 sl=351.25 alert=retest2 |
| CROSSOVER_SKIP | 2024-10-30 13:15:00 | 348.29 | 362.65 | 362.67 | slope filter: EMA200 not falling 2.00% over 1400 bars |
| Cross detected — sustain check pending | 2025-01-02 14:15:00 | 367.82 | 354.53 | 355.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 15:15:00 | 367.43 | 354.66 | 355.45 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-03 12:15:00 | 367.93 | 355.12 | 355.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 13:15:00 | 367.51 | 355.24 | 355.73 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-06 09:15:00 | 361.43 | 355.55 | 355.87 | SL hit qty=1.00 sl=361.43 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-06 09:15:00 | 361.43 | 355.55 | 355.87 | SL hit qty=1.00 sl=361.43 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-20 09:15:00 | 384.04 | 355.32 | 355.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 10:15:00 | 382.99 | 355.59 | 355.76 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-20 12:15:00 | 384.92 | 356.16 | 356.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 12:15:00 | 384.92 | 356.16 | 356.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 14:15:00 | 385.35 | 369.09 | 363.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 10:15:00 | 383.08 | 383.34 | 375.08 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-03-05 10:15:00 | 387.44 | 383.17 | 375.81 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 11:15:00 | 389.40 | 383.23 | 375.88 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-07 14:15:00 | 386.98 | 383.58 | 376.67 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 15:15:00 | 387.08 | 383.62 | 376.72 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-11 12:15:00 | 385.96 | 383.82 | 377.19 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 13:15:00 | 388.37 | 383.86 | 377.24 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-04-21 11:15:00 | 447.81 | 414.55 | 401.59 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-04-21 11:15:00 | 445.14 | 414.55 | 401.59 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-04-21 11:15:00 | 446.63 | 414.55 | 401.59 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 413.00 | 427.84 | 412.75 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 431.30 | 425.19 | 413.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 431.86 | 425.25 | 413.80 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 411.60 | 422.95 | 415.29 | SL hit qty=1.00 sl=411.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-09 10:15:00 | 426.78 | 417.55 | 414.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 11:15:00 | 426.96 | 417.64 | 414.79 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-16 11:15:00 | 427.40 | 420.28 | 416.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 12:15:00 | 427.82 | 420.35 | 416.76 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-19 09:15:00 | 429.60 | 421.40 | 417.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 10:15:00 | 430.36 | 421.49 | 417.68 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 423.00 | 428.73 | 423.31 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-07-04 14:15:00 | 425.98 | 428.61 | 423.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 15:15:00 | 426.00 | 428.58 | 423.34 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-25 14:15:00 | 424.98 | 432.56 | 428.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 15:15:00 | 424.92 | 432.48 | 428.21 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 411.60 | 432.14 | 428.06 | SL hit qty=1.00 sl=411.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 411.60 | 432.14 | 428.06 | SL hit qty=1.00 sl=411.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 411.60 | 432.14 | 428.06 | SL hit qty=1.00 sl=411.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 422.22 | 432.14 | 428.06 | SL hit qty=1.00 sl=422.22 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 422.22 | 432.14 | 428.06 | SL hit qty=1.00 sl=422.22 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 389.40 | 424.73 | 424.57 | SL hit qty=0.50 sl=389.40 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 387.08 | 424.73 | 424.57 | SL hit qty=0.50 sl=387.08 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 388.37 | 424.73 | 424.57 | SL hit qty=0.50 sl=388.37 alert=retest1 |
| CROSSOVER_SKIP | 2025-07-31 10:15:00 | 390.78 | 424.39 | 424.40 | slope filter: EMA200 not falling 2.00% over 1400 bars |
| Cross detected — sustain check pending | 2025-10-06 09:15:00 | 425.12 | 403.10 | 405.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 10:15:00 | 428.00 | 403.35 | 405.64 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-06 10:15:00 | 422.22 | 403.35 | 405.64 | SL hit qty=1.00 sl=422.22 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-09 10:15:00 | 426.64 | 407.64 | 407.68 | ENTRY2 cross detected — sustain check pending (15m) |

### Cycle 6 — BUY (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 11:15:00 | 428.30 | 407.84 | 407.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 12:15:00 | 429.18 | 408.06 | 407.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 09:15:00 | 423.78 | 424.06 | 417.90 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 419.94 | 423.87 | 418.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 419.94 | 423.87 | 418.01 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-11-03 12:15:00 | 422.00 | 423.77 | 418.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 13:15:00 | 423.52 | 423.77 | 418.08 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 417.58 | 423.43 | 418.18 | SL hit qty=1.00 sl=417.58 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-21 13:15:00 | 421.92 | 420.64 | 418.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-21 14:15:00 | 417.54 | 420.61 | 418.21 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-11-27 10:15:00 | 423.68 | 419.97 | 418.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 11:15:00 | 425.02 | 420.02 | 418.17 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-16 09:15:00 | 424.20 | 429.58 | 427.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 10:15:00 | 423.90 | 429.53 | 427.03 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-16 14:15:00 | 417.58 | 429.16 | 426.89 | SL hit qty=1.00 sl=417.58 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-16 14:15:00 | 417.58 | 429.16 | 426.89 | SL hit qty=1.00 sl=417.58 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-19 09:15:00 | 422.30 | 428.98 | 426.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 10:15:00 | 422.60 | 428.92 | 426.81 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 427.20 | 428.90 | 426.81 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-19 12:15:00 | 428.20 | 428.90 | 426.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-19 13:15:00 | 427.70 | 428.88 | 426.82 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-20 09:15:00 | 429.60 | 428.86 | 426.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 10:15:00 | 428.60 | 428.86 | 426.85 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-21 10:15:00 | 421.90 | 428.57 | 426.77 | SL hit qty=1.00 sl=421.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 11:15:00 | 417.58 | 428.46 | 426.73 | SL hit qty=1.00 sl=417.58 alert=retest2 |
| CROSSOVER_SKIP | 2026-01-29 09:15:00 | 410.40 | 425.23 | 425.28 | slope filter: EMA200 not falling 2.00% over 1400 bars |
| Cross detected — sustain check pending | 2026-02-09 09:15:00 | 429.50 | 420.51 | 422.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-09 10:15:00 | 425.55 | 420.56 | 422.60 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-09 13:15:00 | 428.35 | 420.76 | 422.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 14:15:00 | 428.45 | 420.84 | 422.70 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-10 09:15:00 | 430.65 | 421.00 | 422.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 10:15:00 | 433.05 | 421.12 | 422.81 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-12 12:15:00 | 429.45 | 422.36 | 423.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:15:00 | 428.00 | 422.41 | 423.35 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| ALERT3_SKIP | 2026-02-13 10:15:00 | 423.80 | 422.51 | 423.38 | max_alert3_locks_per_cycle=2 reached — end cycle |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-09 11:15:00 | 370.51 | 2024-01-10 09:15:00 | 363.82 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2024-01-11 10:15:00 | 367.72 | 2024-01-17 09:15:00 | 362.89 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-01-12 13:15:00 | 367.00 | 2024-01-17 09:15:00 | 362.89 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-02-02 11:15:00 | 367.92 | 2024-02-05 09:15:00 | 362.89 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-05-24 11:15:00 | 340.48 | 2024-05-27 09:15:00 | 343.50 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-05-28 15:15:00 | 340.00 | 2024-06-03 09:15:00 | 343.50 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-06-04 10:15:00 | 327.25 | 2024-06-05 14:15:00 | 343.50 | STOP_HIT | 1.00 | -4.97% |
| SELL | retest2 | 2024-06-05 12:15:00 | 340.00 | 2024-06-05 14:15:00 | 343.50 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest1 | 2024-07-03 10:15:00 | 357.77 | 2024-07-22 09:15:00 | 353.45 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-08-22 15:15:00 | 364.30 | 2024-10-21 09:15:00 | 361.43 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-09-12 15:15:00 | 366.20 | 2024-10-21 10:15:00 | 351.25 | STOP_HIT | 1.00 | -4.08% |
| BUY | retest2 | 2024-10-03 15:15:00 | 364.56 | 2024-10-21 10:15:00 | 351.25 | STOP_HIT | 1.00 | -3.65% |
| BUY | retest2 | 2024-10-07 10:15:00 | 364.17 | 2024-10-21 10:15:00 | 351.25 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2024-10-10 12:15:00 | 374.28 | 2024-10-21 10:15:00 | 351.25 | STOP_HIT | 1.00 | -6.15% |
| BUY | retest2 | 2025-01-02 15:15:00 | 367.43 | 2025-01-06 09:15:00 | 361.43 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-01-03 13:15:00 | 367.51 | 2025-01-06 09:15:00 | 361.43 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-01-20 10:15:00 | 382.99 | 2025-01-20 12:15:00 | 384.92 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest1 | 2025-03-05 11:15:00 | 389.40 | 2025-04-21 11:15:00 | 447.81 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2025-03-07 15:15:00 | 387.08 | 2025-04-21 11:15:00 | 445.14 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2025-03-11 13:15:00 | 388.37 | 2025-04-21 11:15:00 | 446.63 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2025-03-05 11:15:00 | 389.40 | 2025-05-22 09:15:00 | 411.60 | STOP_HIT | 0.50 | 5.70% |
| BUY | retest1 | 2025-03-07 15:15:00 | 387.08 | 2025-07-28 09:15:00 | 411.60 | STOP_HIT | 0.50 | 6.33% |
| BUY | retest1 | 2025-03-11 13:15:00 | 388.37 | 2025-07-28 09:15:00 | 411.60 | STOP_HIT | 0.50 | 5.98% |
| BUY | retest2 | 2025-05-12 10:15:00 | 431.86 | 2025-07-28 09:15:00 | 411.60 | STOP_HIT | 1.00 | -4.69% |
| BUY | retest2 | 2025-06-09 11:15:00 | 426.96 | 2025-07-28 09:15:00 | 422.22 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-06-16 12:15:00 | 427.82 | 2025-07-28 09:15:00 | 422.22 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-06-19 10:15:00 | 430.36 | 2025-07-31 09:15:00 | 389.40 | STOP_HIT | 1.00 | -9.52% |
| BUY | retest2 | 2025-07-04 15:15:00 | 426.00 | 2025-07-31 09:15:00 | 387.08 | STOP_HIT | 1.00 | -9.14% |
| BUY | retest2 | 2025-07-25 15:15:00 | 424.92 | 2025-07-31 09:15:00 | 388.37 | STOP_HIT | 1.00 | -8.60% |
| BUY | retest2 | 2025-10-06 10:15:00 | 428.00 | 2025-10-06 10:15:00 | 422.22 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-11-03 13:15:00 | 423.52 | 2025-11-06 09:15:00 | 417.58 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-11-27 11:15:00 | 425.02 | 2026-01-16 14:15:00 | 417.58 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2026-01-16 10:15:00 | 423.90 | 2026-01-16 14:15:00 | 417.58 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2026-01-19 10:15:00 | 422.60 | 2026-01-21 10:15:00 | 421.90 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2026-01-20 10:15:00 | 428.60 | 2026-01-21 11:15:00 | 417.58 | STOP_HIT | 1.00 | -2.57% |
