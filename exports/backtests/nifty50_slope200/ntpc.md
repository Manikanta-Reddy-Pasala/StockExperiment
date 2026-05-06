# NTPC (NTPC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 394.85
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 5 |
| PENDING | 17 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 1 |
| ENTRY2 | 15 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 5
- **Winners / losers:** 2 / 10
- **Target hits / Stop hits / Partials:** 0 / 11 / 1
- **Avg / median % per leg:** 0.30% / -0.84%
- **Sum % (uncompounded):** 3.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 0 | 6 | 1 | 1.31% | 9.1% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 0 | 1 | 15.00% | 15.0% |
| BUY @ 3rd Alert (retest2) | 6 | 1 | 16.7% | 0 | 6 | 0 | -0.98% | -5.9% |
| SELL (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.11% | -5.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.11% | -5.5% |
| retest1 (combined) | 1 | 1 | 100.0% | 0 | 0 | 1 | 15.00% | 15.0% |
| retest2 (combined) | 11 | 1 | 9.1% | 0 | 11 | 0 | -1.04% | -11.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 10:15:00 | 331.45 | 344.05 | 344.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 11:15:00 | 331.15 | 343.92 | 343.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 12:15:00 | 339.90 | 339.69 | 341.60 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 341.25 | 339.70 | 341.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 341.25 | 339.70 | 341.56 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-11 13:15:00 | 339.35 | 339.76 | 341.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 14:15:00 | 338.15 | 339.74 | 341.54 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-12 10:15:00 | 337.40 | 339.71 | 341.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 11:15:00 | 338.00 | 339.69 | 341.48 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-27 12:15:00 | 338.50 | 336.11 | 338.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 13:15:00 | 338.95 | 336.14 | 338.75 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 341.80 | 335.72 | 337.98 | SL hit qty=1.00 sl=341.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 341.80 | 335.72 | 337.98 | SL hit qty=1.00 sl=341.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 341.80 | 335.72 | 337.98 | SL hit qty=1.00 sl=341.80 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-08 11:15:00 | 339.85 | 335.81 | 338.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-08 12:15:00 | 340.55 | 335.86 | 338.01 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-24 12:15:00 | 337.15 | 339.46 | 339.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 13:15:00 | 338.20 | 339.44 | 339.46 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 335.65 | 339.39 | 339.44 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-07-25 10:15:00 | 335.30 | 339.35 | 339.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 11:15:00 | 335.05 | 339.31 | 339.40 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-30 09:15:00 | 339.80 | 338.40 | 338.91 | SL hit qty=1.00 sl=339.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-30 12:15:00 | 341.80 | 338.45 | 338.93 | SL hit qty=1.00 sl=341.80 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-31 14:15:00 | 334.05 | 338.43 | 338.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 15:15:00 | 333.70 | 338.38 | 338.87 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-08 13:15:00 | 334.75 | 336.33 | 337.65 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:15:00 | 334.40 | 336.31 | 337.64 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-11 11:15:00 | 334.90 | 336.29 | 337.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 12:15:00 | 334.65 | 336.27 | 337.58 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| ALERT3_SKIP | 2025-08-12 09:15:00 | 339.05 | 336.29 | 337.57 | max_alert3_locks_per_cycle=2 reached — end cycle |

### Cycle 2 — BUY (started 2025-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 09:15:00 | 340.25 | 336.35 | 336.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 10:15:00 | 341.25 | 336.40 | 336.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 10:15:00 | 337.25 | 337.44 | 336.94 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 11:15:00 | 337.50 | 337.44 | 336.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 337.50 | 337.44 | 336.94 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-07 12:15:00 | 338.50 | 337.45 | 336.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:15:00 | 338.90 | 337.47 | 336.96 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 336.90 | 337.47 | 336.97 | SL hit qty=1.00 sl=336.90 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-10 09:15:00 | 339.80 | 337.12 | 336.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 10:15:00 | 340.75 | 337.15 | 336.84 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 336.90 | 337.30 | 336.92 | SL hit qty=1.00 sl=336.90 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-13 10:15:00 | 338.60 | 337.31 | 336.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 11:15:00 | 339.00 | 337.33 | 336.94 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-14 12:15:00 | 336.90 | 337.50 | 337.04 | SL hit qty=1.00 sl=336.90 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-15 09:15:00 | 339.95 | 337.50 | 337.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:15:00 | 340.45 | 337.53 | 337.07 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 338.40 | 339.19 | 338.11 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-29 09:15:00 | 343.60 | 339.22 | 338.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 10:15:00 | 346.75 | 339.29 | 338.20 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-31 09:15:00 | 336.90 | 340.10 | 338.69 | SL hit qty=1.00 sl=336.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-31 09:15:00 | 338.00 | 340.10 | 338.69 | SL hit qty=1.00 sl=338.00 alert=retest2 |
| CROSSOVER_SKIP | 2025-11-07 13:15:00 | 325.95 | 337.52 | 337.53 | slope filter: EMA200 not falling 2.00% over 1400 bars |
| Cross detected — sustain check pending | 2026-01-02 10:15:00 | 345.40 | 325.90 | 328.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 11:15:00 | 348.00 | 326.12 | 328.25 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 348.15 | 330.28 | 330.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 348.15 | 330.28 | 330.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 351.60 | 334.41 | 332.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 336.35 | 336.97 | 334.31 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-01-27 09:15:00 | 342.40 | 337.02 | 334.36 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 10:15:00 | 342.85 | 337.08 | 334.40 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-03-13 11:15:00 | 394.28 | 370.79 | 359.96 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 367.20 | 374.15 | 365.52 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-08 09:15:00 | 370.60 | 371.66 | 365.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 10:15:00 | 370.85 | 371.65 | 365.26 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-06-11 14:15:00 | 338.15 | 2025-07-08 09:15:00 | 341.80 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-06-12 11:15:00 | 338.00 | 2025-07-08 09:15:00 | 341.80 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-06-27 13:15:00 | 338.95 | 2025-07-08 09:15:00 | 341.80 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-07-24 13:15:00 | 338.20 | 2025-07-30 09:15:00 | 339.80 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-07-25 11:15:00 | 335.05 | 2025-07-30 12:15:00 | 341.80 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-10-07 13:15:00 | 338.90 | 2025-10-08 09:15:00 | 336.90 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-10-10 10:15:00 | 340.75 | 2025-10-13 09:15:00 | 336.90 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-10-13 11:15:00 | 339.00 | 2025-10-14 12:15:00 | 336.90 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-10-15 10:15:00 | 340.45 | 2025-10-31 09:15:00 | 336.90 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-10-29 10:15:00 | 346.75 | 2025-10-31 09:15:00 | 338.00 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2026-01-02 11:15:00 | 348.00 | 2026-01-07 09:15:00 | 348.15 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest1 | 2026-01-27 10:15:00 | 342.85 | 2026-03-13 11:15:00 | 394.28 | PARTIAL | 0.50 | 15.00% |
