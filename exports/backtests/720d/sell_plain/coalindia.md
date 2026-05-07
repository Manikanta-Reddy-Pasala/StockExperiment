# COALINDIA (COALINDIA)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 465.95
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 7 |
| PENDING | 29 |
| PENDING_CANCEL | 10 |
| ENTRY1 | 3 |
| ENTRY2 | 16 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 16
- **Target hits / Stop hits / Partials:** 0 / 19 / 0
- **Avg / median % per leg:** -1.31% / -1.23%
- **Sum % (uncompounded):** -24.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 19 | 3 | 15.8% | 0 | 19 | 0 | -1.31% | -24.9% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.65% | -5.0% |
| SELL @ 3rd Alert (retest2) | 16 | 3 | 18.8% | 0 | 16 | 0 | -1.25% | -20.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.65% | -5.0% |
| retest2 (combined) | 16 | 3 | 18.8% | 0 | 16 | 0 | -1.25% | -20.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-20 11:15:00 | 492.50 | 502.09 | 502.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-20 14:15:00 | 490.00 | 501.80 | 501.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 14:15:00 | 501.50 | 501.48 | 501.81 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-23 14:15:00 | 501.50 | 501.48 | 501.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 14:15:00 | 501.50 | 501.48 | 501.81 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-10-07 09:15:00 | 491.40 | 503.43 | 502.86 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 11:15:00 | 483.20 | 503.01 | 502.66 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2024-10-07 14:15:00 | 479.35 | 502.29 | 502.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 14:15:00 | 479.35 | 502.29 | 502.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 13:15:00 | 474.35 | 496.02 | 498.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 10:15:00 | 387.00 | 385.32 | 401.95 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-01-31 09:15:00 | 379.65 | 385.26 | 401.43 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-31 10:15:00 | 384.55 | 385.25 | 401.35 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-03 09:15:00 | 370.70 | 385.68 | 400.56 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-03 11:15:00 | 371.70 | 385.42 | 400.28 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2025-02-05 15:15:00 | 382.35 | 384.32 | 398.42 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 09:15:00 | 379.20 | 384.27 | 398.32 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 12:15:00 | 383.05 | 369.62 | 381.67 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-03-06 12:15:00 | 383.05 | 369.62 | 381.67 | SL hit (close>ema400) qty=1.00 sl=381.67 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-03-06 12:15:00 | 383.05 | 369.62 | 381.67 | SL hit (close>ema400) qty=1.00 sl=381.67 alert=retest1 |
| Cross detected — sustain check pending | 2025-03-10 14:15:00 | 375.20 | 371.35 | 381.65 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-11 09:15:00 | 372.65 | 371.40 | 381.57 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 1140m) |
| Cross detected — sustain check pending | 2025-03-12 10:15:00 | 376.40 | 371.79 | 381.37 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-03-12 12:15:00 | 377.30 | 371.90 | 381.33 | ENTRY2 sustain failed after 120m |
| Stop hit — per-position SL triggered | 2025-03-17 13:15:00 | 385.00 | 373.19 | 381.32 | SL hit (close>static) qty=1.00 sl=384.15 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-07 09:15:00 | 365.00 | 387.18 | 386.98 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 11:15:00 | 369.75 | 386.86 | 386.83 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-04-07 12:15:00 | 369.80 | 386.69 | 386.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 12:15:00 | 369.80 | 386.69 | 386.74 | EMA200 below EMA400 |

### Cycle 4 — SELL (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 11:15:00 | 387.05 | 393.34 | 393.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 10:15:00 | 386.00 | 392.94 | 393.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 14:15:00 | 388.80 | 388.77 | 390.58 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-07-21 09:15:00 | 386.90 | 388.75 | 390.56 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-21 11:15:00 | 387.35 | 388.71 | 390.52 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 389.85 | 388.60 | 390.39 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 390.80 | 388.64 | 390.38 | SL hit (close>ema400) qty=1.00 sl=390.38 alert=retest1 |
| Cross detected — sustain check pending | 2025-07-24 10:15:00 | 386.60 | 388.74 | 390.36 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 12:15:00 | 384.95 | 388.66 | 390.30 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-08-19 09:15:00 | 387.00 | 383.46 | 386.08 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-08-19 10:15:00 | 387.45 | 383.50 | 386.08 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-19 13:15:00 | 384.70 | 383.59 | 386.09 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 15:15:00 | 385.90 | 383.63 | 386.09 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-09-05 13:15:00 | 391.85 | 382.34 | 384.34 | SL hit (close>static) qty=1.00 sl=391.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-05 13:15:00 | 391.85 | 382.34 | 384.34 | SL hit (close>static) qty=1.00 sl=391.75 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-08 14:15:00 | 386.90 | 383.02 | 384.61 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 386.95 | 383.10 | 384.64 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 1140m) |
| Cross detected — sustain check pending | 2025-09-09 11:15:00 | 387.35 | 383.19 | 384.67 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-09-09 12:15:00 | 388.70 | 383.24 | 384.69 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 392.10 | 383.63 | 384.84 | SL hit (close>static) qty=1.00 sl=391.75 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-29 12:15:00 | 386.30 | 389.78 | 388.22 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-09-29 13:15:00 | 389.70 | 389.78 | 388.23 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-03 09:15:00 | 383.00 | 389.73 | 388.33 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 11:15:00 | 382.30 | 389.59 | 388.28 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 386.50 | 388.78 | 387.94 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-08 12:15:00 | 383.20 | 388.42 | 387.79 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 14:15:00 | 381.85 | 388.29 | 387.74 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-10-09 13:15:00 | 383.10 | 388.03 | 387.62 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 15:15:00 | 383.30 | 387.94 | 387.58 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-10-10 13:15:00 | 383.20 | 387.79 | 387.51 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-10-10 14:15:00 | 385.10 | 387.76 | 387.50 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-13 09:15:00 | 382.60 | 387.68 | 387.46 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 11:15:00 | 381.75 | 387.56 | 387.40 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-10-14 10:15:00 | 382.50 | 387.25 | 387.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 10:15:00 | 382.50 | 387.25 | 387.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 10:15:00 | 382.50 | 387.25 | 387.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 10:15:00 | 382.50 | 387.25 | 387.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 382.50 | 387.25 | 387.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 381.40 | 387.14 | 387.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 12:15:00 | 387.40 | 386.75 | 386.99 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 12:15:00 | 387.40 | 386.75 | 386.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 387.40 | 386.75 | 386.99 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-17 13:15:00 | 385.90 | 386.78 | 386.99 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-10-17 14:15:00 | 388.60 | 386.80 | 387.00 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-29 13:15:00 | 382.60 | 389.11 | 388.23 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 15:15:00 | 382.95 | 388.98 | 388.17 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-10-30 11:15:00 | 385.35 | 388.87 | 388.13 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-10-30 13:15:00 | 386.75 | 388.82 | 388.11 | ENTRY2 sustain failed after 120m |
| Stop hit — per-position SL triggered | 2025-10-30 14:15:00 | 387.65 | 388.81 | 388.11 | SL hit (close>static) qty=1.00 sl=387.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-04 09:15:00 | 379.30 | 388.81 | 388.16 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:15:00 | 379.20 | 388.61 | 388.07 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-11-06 13:15:00 | 374.35 | 387.52 | 387.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 13:15:00 | 374.35 | 387.52 | 387.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 372.95 | 387.38 | 387.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 12:15:00 | 386.80 | 385.73 | 386.56 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 12:15:00 | 386.80 | 385.73 | 386.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 386.80 | 385.73 | 386.56 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-11-13 12:15:00 | 384.95 | 385.76 | 386.55 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 14:15:00 | 383.40 | 385.71 | 386.52 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 388.20 | 385.70 | 386.48 | SL hit (close>static) qty=1.00 sl=387.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-18 14:15:00 | 384.10 | 385.83 | 386.50 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 380.80 | 385.76 | 386.45 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 1140m) |
| Cross detected — sustain check pending | 2025-12-12 10:15:00 | 384.40 | 380.69 | 382.73 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 12:15:00 | 383.45 | 380.74 | 382.74 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-12-17 15:15:00 | 384.40 | 381.24 | 382.78 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 384.50 | 381.27 | 382.79 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 385.25 | 381.31 | 382.80 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-12-19 09:15:00 | 383.95 | 381.54 | 382.87 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-12-19 10:15:00 | 384.55 | 381.57 | 382.88 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-19 12:15:00 | 384.20 | 381.63 | 382.90 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-12-19 13:15:00 | 385.10 | 381.66 | 382.91 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-12-23 09:15:00 | 397.40 | 382.15 | 383.09 | SL hit (close>static) qty=1.00 sl=387.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 09:15:00 | 397.40 | 382.15 | 383.09 | SL hit (close>static) qty=1.00 sl=387.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 09:15:00 | 397.40 | 382.15 | 383.09 | SL hit (close>static) qty=1.00 sl=387.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-10-07 11:15:00 | 483.20 | 2024-10-07 14:15:00 | 479.35 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest1 | 2025-02-03 11:15:00 | 371.70 | 2025-03-06 12:15:00 | 383.05 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest1 | 2025-02-06 09:15:00 | 379.20 | 2025-03-06 12:15:00 | 383.05 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-03-11 09:15:00 | 372.65 | 2025-03-17 13:15:00 | 385.00 | STOP_HIT | 1.00 | -3.31% |
| SELL | retest2 | 2025-04-07 11:15:00 | 369.75 | 2025-04-07 12:15:00 | 369.80 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest1 | 2025-07-21 11:15:00 | 387.35 | 2025-07-23 09:15:00 | 390.80 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-07-24 12:15:00 | 384.95 | 2025-09-05 13:15:00 | 391.85 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-08-19 15:15:00 | 385.90 | 2025-09-05 13:15:00 | 391.85 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-09-09 09:15:00 | 386.95 | 2025-09-10 11:15:00 | 392.10 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-10-03 11:15:00 | 382.30 | 2025-10-14 10:15:00 | 382.50 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2025-10-08 14:15:00 | 381.85 | 2025-10-14 10:15:00 | 382.50 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-10-09 15:15:00 | 383.30 | 2025-10-14 10:15:00 | 382.50 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2025-10-13 11:15:00 | 381.75 | 2025-10-14 10:15:00 | 382.50 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-10-29 15:15:00 | 382.95 | 2025-10-30 14:15:00 | 387.65 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-11-04 11:15:00 | 379.20 | 2025-11-06 13:15:00 | 374.35 | STOP_HIT | 1.00 | 1.28% |
| SELL | retest2 | 2025-11-13 14:15:00 | 383.40 | 2025-11-17 09:15:00 | 388.20 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-11-19 09:15:00 | 380.80 | 2025-12-23 09:15:00 | 397.40 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest2 | 2025-12-12 12:15:00 | 383.45 | 2025-12-23 09:15:00 | 397.40 | STOP_HIT | 1.00 | -3.64% |
| SELL | retest2 | 2025-12-18 09:15:00 | 384.50 | 2025-12-23 09:15:00 | 397.40 | STOP_HIT | 1.00 | -3.36% |
