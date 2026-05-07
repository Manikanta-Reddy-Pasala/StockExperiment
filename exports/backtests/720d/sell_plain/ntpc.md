# NTPC (NTPC)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 400.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 5 |
| PENDING | 19 |
| PENDING_CANCEL | 7 |
| ENTRY1 | 2 |
| ENTRY2 | 10 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 10
- **Target hits / Stop hits / Partials:** 0 / 12 / 0
- **Avg / median % per leg:** -1.51% / -1.76%
- **Sum % (uncompounded):** -18.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 2 | 16.7% | 0 | 12 | 0 | -1.51% | -18.1% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.66% | -1.3% |
| SELL @ 3rd Alert (retest2) | 10 | 1 | 10.0% | 0 | 10 | 0 | -1.68% | -16.8% |
| retest1 (combined) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.66% | -1.3% |
| retest2 (combined) | 10 | 1 | 10.0% | 0 | 10 | 0 | -1.68% | -16.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 14:15:00 | 380.10 | 409.06 | 409.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-14 10:15:00 | 377.70 | 406.56 | 407.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 14:15:00 | 336.15 | 335.75 | 353.14 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-01-21 09:15:00 | 331.90 | 335.70 | 352.94 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 11:15:00 | 331.65 | 335.62 | 352.73 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 324.70 | 317.58 | 327.78 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-03-06 11:15:00 | 330.15 | 317.80 | 327.79 | SL hit (close>ema400) qty=1.00 sl=327.79 alert=retest1 |

### Cycle 2 — SELL (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 11:15:00 | 331.15 | 343.92 | 343.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 10:15:00 | 328.85 | 343.21 | 343.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 12:15:00 | 339.90 | 339.69 | 341.56 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 341.25 | 339.69 | 341.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 341.25 | 339.69 | 341.53 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-11 13:15:00 | 339.25 | 339.75 | 341.52 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 15:15:00 | 338.15 | 339.72 | 341.49 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-06-12 10:15:00 | 337.35 | 339.70 | 341.46 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 12:15:00 | 337.65 | 339.67 | 341.43 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-06-27 12:15:00 | 338.50 | 336.12 | 338.73 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 14:15:00 | 337.95 | 336.16 | 338.73 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-07-08 11:15:00 | 339.95 | 335.82 | 337.99 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-07-08 12:15:00 | 340.50 | 335.86 | 338.00 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-07-08 13:15:00 | 343.20 | 335.94 | 338.03 | SL hit (close>static) qty=1.00 sl=341.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 13:15:00 | 343.20 | 335.94 | 338.03 | SL hit (close>static) qty=1.00 sl=341.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 13:15:00 | 343.20 | 335.94 | 338.03 | SL hit (close>static) qty=1.00 sl=341.80 alert=retest2 |

### Cycle 3 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 337.15 | 339.46 | 339.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 335.65 | 339.40 | 339.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 09:15:00 | 339.00 | 338.40 | 338.90 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 09:15:00 | 339.00 | 338.40 | 338.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 339.00 | 338.40 | 338.90 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-01 09:15:00 | 332.35 | 338.32 | 338.83 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-08-01 10:15:00 | 333.85 | 338.27 | 338.80 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-01 11:15:00 | 331.65 | 338.21 | 338.77 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 13:15:00 | 332.05 | 338.09 | 338.70 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-08-04 13:15:00 | 332.00 | 337.60 | 338.43 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 15:15:00 | 332.15 | 337.49 | 338.37 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-08-05 13:15:00 | 331.80 | 337.24 | 338.22 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-08-05 14:15:00 | 333.80 | 337.20 | 338.20 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-06 14:15:00 | 332.20 | 336.97 | 338.04 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-08-06 15:15:00 | 332.55 | 336.92 | 338.02 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-07 09:15:00 | 330.60 | 336.86 | 337.98 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 11:15:00 | 330.85 | 336.73 | 337.90 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-08-12 12:15:00 | 341.70 | 336.41 | 337.60 | SL hit (close>static) qty=1.00 sl=340.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 12:15:00 | 341.70 | 336.41 | 337.60 | SL hit (close>static) qty=1.00 sl=340.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 12:15:00 | 341.70 | 336.41 | 337.60 | SL hit (close>static) qty=1.00 sl=340.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-28 09:15:00 | 331.10 | 337.11 | 337.70 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 11:15:00 | 330.70 | 337.00 | 337.64 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 11:15:00 | 336.55 | 335.78 | 336.93 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-09-02 13:15:00 | 335.15 | 335.80 | 336.93 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-09-02 14:15:00 | 336.25 | 335.80 | 336.93 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-03 12:15:00 | 334.65 | 335.81 | 336.90 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 14:15:00 | 333.65 | 335.78 | 336.88 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-09-18 12:15:00 | 334.85 | 333.42 | 335.05 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-09-18 14:15:00 | 337.00 | 333.47 | 335.06 | ENTRY2 sustain failed after 120m |
| Stop hit — per-position SL triggered | 2025-09-19 09:15:00 | 337.45 | 333.55 | 335.08 | SL hit (close>static) qty=1.00 sl=337.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 10:15:00 | 340.35 | 334.30 | 335.36 | SL hit (close>static) qty=1.00 sl=340.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-08 10:15:00 | 334.35 | 337.46 | 336.97 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 12:15:00 | 334.05 | 337.39 | 336.94 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 339.95 | 337.14 | 336.83 | SL hit (close>static) qty=1.00 sl=337.15 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-03 11:15:00 | 334.95 | 339.81 | 338.60 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-11-03 12:15:00 | 335.65 | 339.77 | 338.59 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-11-03 13:15:00 | 335.10 | 339.72 | 338.57 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 15:15:00 | 335.40 | 339.63 | 338.54 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-11-07 13:15:00 | 326.00 | 337.52 | 337.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 13:15:00 | 326.00 | 337.52 | 337.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 13:15:00 | 325.10 | 336.73 | 337.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 325.45 | 324.54 | 328.16 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-12-24 14:15:00 | 322.40 | 324.51 | 328.05 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 09:15:00 | 323.20 | 324.48 | 328.00 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 2580m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 328.90 | 324.57 | 327.70 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 328.90 | 324.57 | 327.70 | SL hit (close>ema400) qty=1.00 sl=327.70 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-01-21 11:15:00 | 331.65 | 2025-03-06 11:15:00 | 330.15 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2025-06-11 15:15:00 | 338.15 | 2025-07-08 13:15:00 | 343.20 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-06-12 12:15:00 | 337.65 | 2025-07-08 13:15:00 | 343.20 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-06-27 14:15:00 | 337.95 | 2025-07-08 13:15:00 | 343.20 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-08-01 13:15:00 | 332.05 | 2025-08-12 12:15:00 | 341.70 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2025-08-04 15:15:00 | 332.15 | 2025-08-12 12:15:00 | 341.70 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-08-07 11:15:00 | 330.85 | 2025-08-12 12:15:00 | 341.70 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2025-08-28 11:15:00 | 330.70 | 2025-09-19 09:15:00 | 337.45 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-09-03 14:15:00 | 333.65 | 2025-09-23 10:15:00 | 340.35 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-10-08 12:15:00 | 334.05 | 2025-10-10 09:15:00 | 339.95 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-11-03 15:15:00 | 335.40 | 2025-11-07 13:15:00 | 326.00 | STOP_HIT | 1.00 | 2.80% |
| SELL | retest1 | 2025-12-26 09:15:00 | 323.20 | 2025-12-31 09:15:00 | 328.90 | STOP_HIT | 1.00 | -1.76% |
