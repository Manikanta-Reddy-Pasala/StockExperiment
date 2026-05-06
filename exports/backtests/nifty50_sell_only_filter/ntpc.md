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
| PENDING | 18 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 1 |
| ENTRY2 | 13 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 2 / 11
- **Target hits / Stop hits / Partials:** 0 / 12 / 1
- **Avg / median % per leg:** -1.38% / -1.62%
- **Sum % (uncompounded):** -17.95%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 2 | 15.4% | 0 | 12 | 1 | -1.38% | -18.0% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 0 | 1 | 15.00% | 15.0% |
| BUY @ 3rd Alert (retest2) | 12 | 1 | 8.3% | 0 | 12 | 0 | -2.75% | -33.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 1 | 1 | 100.0% | 0 | 0 | 1 | 15.00% | 15.0% |
| retest2 (combined) | 12 | 1 | 8.3% | 0 | 12 | 0 | -2.75% | -33.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 11:15:00 | 360.25 | 333.46 | 333.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 15:15:00 | 364.00 | 335.92 | 334.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 339.65 | 341.40 | 337.82 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 339.65 | 341.40 | 337.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 339.65 | 341.40 | 337.82 | EMA400 retest candle locked |
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
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 343.75 | 346.47 | 345.10 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-26 09:15:00 | 346.70 | 346.40 | 345.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 10:15:00 | 345.65 | 346.40 | 345.10 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-27 09:15:00 | 342.60 | 346.20 | 345.04 | SL hit qty=1.00 sl=342.60 alert=retest2 |
| CROSSOVER_SKIP | 2025-06-02 10:15:00 | 331.45 | 344.05 | 344.05 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 324.70 | 336.79 | 339.42 | SL hit qty=1.00 sl=324.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 324.70 | 336.79 | 339.42 | SL hit qty=1.00 sl=324.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 324.70 | 336.79 | 339.42 | SL hit qty=1.00 sl=324.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 324.70 | 336.79 | 339.42 | SL hit qty=1.00 sl=324.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-09 12:15:00 | 344.80 | 336.40 | 338.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-09 13:15:00 | 344.15 | 336.48 | 338.25 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-24 09:15:00 | 345.95 | 334.80 | 335.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 10:15:00 | 348.55 | 334.94 | 335.65 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-25 13:15:00 | 342.60 | 336.09 | 336.21 | SL hit qty=1.00 sl=342.60 alert=retest2 |

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
| CROSSOVER_SKIP | 2025-11-07 13:15:00 | 325.95 | 337.52 | 337.53 | slope filter: EMA200 not falling 0.50% over 350 bars |
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
| BUY | retest2 | 2025-04-07 13:15:00 | 348.25 | 2025-05-27 09:15:00 | 342.60 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-05-12 10:15:00 | 347.25 | 2025-06-24 09:15:00 | 324.70 | STOP_HIT | 1.00 | -6.49% |
| BUY | retest2 | 2025-05-16 10:15:00 | 344.95 | 2025-06-24 09:15:00 | 324.70 | STOP_HIT | 1.00 | -5.87% |
| BUY | retest2 | 2025-05-23 11:15:00 | 343.00 | 2025-06-24 09:15:00 | 324.70 | STOP_HIT | 1.00 | -5.34% |
| BUY | retest2 | 2025-05-26 10:15:00 | 345.65 | 2025-06-24 09:15:00 | 324.70 | STOP_HIT | 1.00 | -6.06% |
| BUY | retest2 | 2025-09-24 10:15:00 | 348.55 | 2025-09-25 13:15:00 | 342.60 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-10-07 13:15:00 | 338.90 | 2025-10-08 09:15:00 | 336.90 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-10-10 10:15:00 | 340.75 | 2025-10-13 09:15:00 | 336.90 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-10-13 11:15:00 | 339.00 | 2025-10-14 12:15:00 | 336.90 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-10-15 10:15:00 | 340.45 | 2025-10-31 09:15:00 | 336.90 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-10-29 10:15:00 | 346.75 | 2025-10-31 09:15:00 | 338.00 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2026-01-02 11:15:00 | 348.00 | 2026-01-07 09:15:00 | 348.15 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest1 | 2026-01-27 10:15:00 | 342.85 | 2026-03-13 11:15:00 | 394.28 | PARTIAL | 0.50 | 15.00% |
